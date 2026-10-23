"""Host-RAM copies of a model's GPU tensors, so an idle model can leave the GPU and come back in about a second.

By Siddid Soni, 2026-10 (distributed under this repo's LTX-2 Community License).

    mirror = Mirror("cuda", model)     # first park of a model built on the GPU: copies its weights to host RAM
    mirror.park()                      # the GPU copy goes (the tensors point at the host copy)
    mirror.unpark()                    # back on the GPU, in place: the model object and its tensors are the same
    mirror.release()                   # before the model is dropped: gives the host RAM back

    mirror = Mirror("cpu", submodule)  # a model built on the CPU (no GPU memory used): park() keeps one host copy
    mirror.park()

Ported from Konvey (konvey/speech/mirror.py). Every tensor of the
given modules on ``device`` is copied into one pageable slab registered with ``cudaHostRegister`` (fast async
copies). Measured on an RTX 3070 Ti laptop: Gemma 5.8 GB and the DiT 6.6 GB park in 0.3 s and unpark in 0.5 s,
the first host copy takes about 1 s, and release returns the RAM (tensors from ``pin_memory=True`` stay in torch's
host cache, and a pinned slab is rounded up to 8 GiB). The weights must not change while parked, and park drops
the GPU copy only if nothing else holds it (a CUDA graph, a cache, an output with a grad_fn): run inference under
``torch.inference_mode()``.
"""

from __future__ import annotations

import gc
import logging

import torch

log = logging.getLogger(__name__)

ALIGN = 512  # every tensor starts on an aligned offset of the slab, so a typed view of it is valid


class Mirror:
    """Host copies of the ``device`` tensors the modules own (parameters, buffers and the tables of bitsandbytes
    4-bit quant states), kept in one slab. ``pin`` registers the slab as page-locked memory (default: when CUDA
    is available)."""

    def __init__(self, device: str, *modules: torch.nn.Module, pin: bool | None = None):
        if not modules:
            raise ValueError("Mirror needs at least one module")
        self.device = device
        self.parked = False
        self.registered = False
        seen, self.tensors = set(), []

        def add(t):
            if isinstance(t, torch.Tensor) and t.device.type == device and id(t) not in seen:
                seen.add(id(t))
                self.tensors.append(t)

        for m in (m for module in modules for m in module.modules()):
            for t in (*m._parameters.values(), *m._buffers.values()):
                add(t)
                qs = getattr(t, "quant_state", None)  # bitsandbytes 4-bit: its tables are tensors too
                while qs is not None:
                    for name in ("absmax", "code", "offset"):
                        add(getattr(qs, name, None))
                    qs = getattr(qs, "state2", None)
        offsets, size = [], 0
        for t in self.tensors:
            offsets.append(size)
            size += -(-t.numel() * t.element_size() // ALIGN) * ALIGN
        self.nbytes = size
        self.slab = torch.zeros(size, dtype=torch.uint8)  # pageable, so release() really gives it back
        if (torch.cuda.is_available() if pin is None else pin) and size:
            rc = torch.cuda.cudart().cudaHostRegister(self.slab.data_ptr(), size, 0)  # page-locked: fast async copies
            self.registered = int(rc) == 0
            if not self.registered:  # still correct, only slower
                log.warning("cudaHostRegister failed (%s) for %.1f GB; copies will use pageable memory", rc, size / 1e9)
        self.host = [self.slab[o:o + t.numel() * t.element_size()].view(t.dtype).view(t.shape)
                     for t, o in zip(self.tensors, offsets)]
        for h, t in zip(self.host, self.tensors):
            h.copy_(t, non_blocking=True)
        self._sync()

    def _sync(self) -> None:
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    def park(self) -> None:
        """Point every tensor at its host copy; the GPU copy (or, for a CPU-built model, the first CPU copy) goes."""
        if self.parked:
            return
        for t, h in zip(self.tensors, self.host):
            t.data = h
        self.parked = True
        gc.collect()
        torch.cuda.empty_cache()

    def unpark(self) -> None:
        """Copy every tensor to the GPU, in place; the host copy stays for the next park."""
        if not self.parked:
            return
        for t, h in zip(self.tensors, self.host):
            t.data = h.to("cuda", non_blocking=True)
        self._sync()
        self.parked = False

    def release(self) -> None:
        """Give the host RAM back. Call it before the model is dropped; a parked model keeps working until then."""
        if self.slab is not None and self.registered:
            torch.cuda.cudart().cudaHostUnregister(self.slab.data_ptr())
        self.registered = False
        self.host, self.tensors, self.slab = [], [], None


class DiskMirror:
    """One stage's weights in exactly one place at a time: the GPU, host RAM (a page-locked slab), or an SSD file.

        store = DiskMirror(path, *modules)   # modules built on the GPU, or on the CPU (then to_ram() first)
        store.save()                          # write the file once (background thread), reused across runs
        store.drop()                          # GPU -> nothing: the file holds the bytes, so leaving costs no copy
        store.prefetch()                      # SSD -> RAM in a background thread, while the GPU works on something else
        store.upload()                        # RAM -> GPU, in place; the RAM copy is released

    Only the stage being prefetched holds host RAM, so the GPU model and one parked model never both sit in RAM.
    Use under torch.inference_mode(); the weights must not change after the first save.
    """

    def __init__(self, path, *modules: torch.nn.Module):
        import threading
        from pathlib import Path

        self.path = Path(path)
        seen, self.tensors = set(), []

        def add(t):
            if isinstance(t, torch.Tensor) and id(t) not in seen:
                seen.add(id(t))
                self.tensors.append(t)

        for m in (m for module in modules for m in module.modules()):
            for t in (*m._parameters.values(), *m._buffers.values()):
                add(t)
                qs = getattr(t, "quant_state", None)  # bitsandbytes 4-bit: its tables are tensors too
                while qs is not None:
                    for name in ("absmax", "code", "offset"):
                        add(getattr(qs, name, None))
                    qs = getattr(qs, "state2", None)
        self.shapes = [(t.shape, t.dtype) for t in self.tensors]
        self.offsets, size = [], 0
        for t in self.tensors:
            self.offsets.append(size)
            size += -(-t.numel() * t.element_size() // ALIGN) * ALIGN
        self.nbytes = size
        self.where = self.tensors[0].device.type if self.tensors else "cpu"   # "cuda", "cpu" (built), "ram", "disk"
        self.slab = None
        self.registered = False
        self._lock = threading.Lock()
        self._saver = self._loader = None

    # -- host slab --------------------------------------------------------------------------------------------------

    def _views(self, slab):
        return [slab[o:o + s.numel() * dt.itemsize].view(dt).view(s)
                for (s, dt), o in zip(self.shapes, self.offsets)]

    def _register(self) -> None:
        if torch.cuda.is_available() and self.nbytes:
            self.registered = int(torch.cuda.cudart().cudaHostRegister(self.slab.data_ptr(), self.nbytes, 0)) == 0

    def _release(self) -> None:
        if self.slab is not None and self.registered:
            torch.cuda.cudart().cudaHostUnregister(self.slab.data_ptr())
        self.registered = False
        self.slab = None

    def to_ram(self) -> None:
        """A stage built on the CPU: move its tensors into one page-locked slab, one tensor at a time, so the build
        and the slab never both hold the whole stage."""
        if self.where != "cpu":
            return
        self.slab = torch.empty(self.nbytes, dtype=torch.uint8)
        for t, v in zip(self.tensors, self._views(self.slab)):
            v.copy_(t)
            t.data = v
        gc.collect()
        self._register()
        self.where = "ram"

    # -- SSD file -------------------------------------------------------------------------------------------------

    def _file_ok(self) -> bool:
        return self.path.exists() and self.path.stat().st_size == self.nbytes

    def _write(self) -> None:
        with torch.inference_mode():
            if self._file_ok():
                return
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(".tmp")
            with open(tmp, "wb") as f:
                f.truncate(self.nbytes)
                for t, o in zip(self.tensors, self.offsets):
                    f.seek(o)
                    f.write(t.detach().contiguous().view(-1).view(torch.uint8).cpu().numpy().tobytes())
            tmp.replace(self.path)

    def save(self, background: bool = True) -> None:
        """Write the stage to its file once (skipped when a file of the right size is already there)."""
        import threading

        if self._saver is None:
            self._saver = threading.Thread(target=self._write, daemon=True)
            self._saver.start()
        if not background:
            self._saver.join()

    def drop(self) -> None:
        """GPU -> nothing: the weights are only in the file now."""
        if self.where == "disk":
            return
        self.save(background=False)
        if not self._file_ok():   # the write failed: keep the weights rather than lose them
            self._saver = None
            raise IOError(f"could not write {self.path}; the stage stays where it is")
        for t in self.tensors:
            t.data = torch.empty(0, dtype=t.dtype, device=t.device)
        self._release()
        self.where = "disk"
        gc.collect()
        torch.cuda.empty_cache()

    def _read(self) -> None:
        with self._lock:
            slab = torch.empty(self.nbytes, dtype=torch.uint8)
            buf, done = memoryview(slab.numpy()), 0
            with open(self.path, "rb") as f:
                while done < self.nbytes:
                    n = f.readinto(buf[done:])
                    if not n:
                        raise IOError(f"{self.path} is shorter than {self.nbytes} bytes")
                    done += n
            self.slab = slab
            self._register()

    def prefetch(self) -> None:
        """SSD -> RAM in the background (no-op unless the stage is only on disk)."""
        import threading

        if self.where == "disk" and self._loader is None:
            self._loader = threading.Thread(target=self._read, daemon=True)
            self._loader.start()

    def upload(self) -> None:
        """-> GPU, in place (waits for a prefetch still reading); the RAM copy is released."""
        if self.where == "cuda":
            return
        if self.where == "disk":
            self.prefetch()
            self._loader.join()
            self._loader = None
        for t, v in zip(self.tensors, self._views(self.slab)):
            t.data = v.to("cuda", non_blocking=True)
        self._sync()
        self._release()
        self.where = "cuda"

    @staticmethod
    def _sync() -> None:
        torch.cuda.synchronize()
