"""

By Siddid Soni, 2026-10 (distributed under this repo's LTX-2 Community License).
Apple Silicon 服务（mlx-16 / mlx-24 档）：Gemma 3 12B 4-bit 走 MLX（mlx-lm，mlx-community/gemma-3-12b-it-4bit，
~8 GB），其余 DramaBox 原样走 PyTorch MPS。Gemma 的每层隐状态转成 torch 张量后送进嵌入处理器。

  mlx-24 (≥24G 统一内存)  Gemma 常驻（~8 GB）+ DiT bf16（6.6 GB）+ 其余
  mlx-16 (16G)            Gemma 只在编码新提示词时加载，编完即释放；DiT 常驻。嵌入有缓存，重复提示词不再加载 Gemma
"""
from __future__ import annotations

import time
from pathlib import Path


from config import DTYPE, MLX_GEMMA_DIR, TIER
from inference_server import TTSServer, _free, log
from text_encoders import MlxGemma, build_embeddings_processor


class MacTTSServer(TTSServer):
    """TTSServer with Gemma in MLX and everything else on MPS."""

    EMBED_KEY = "mlx-4bit"
    ALWAYS_TRIM = True   # mlx-lm takes no padding mask: real tokens only

    def __init__(self, checkpoint: str | None = None, full_checkpoint: str | None = None,
                 gemma_root: str | None = None, device: str = "mps", dtype: str = DTYPE, tier: str = TIER):
        self.tier = tier
        self._gemma = None
        self._embedder = None
        super().__init__(checkpoint=checkpoint, full_checkpoint=full_checkpoint,
                         gemma_root=gemma_root or str(MLX_GEMMA_DIR), device=device, dtype=dtype)

    def _load_all(self):
        if not (Path(self.gemma_root) / "config.json").exists():
            raise FileNotFoundError(
                f"MLX Gemma not found in {self.gemma_root}. Download it with:\n"
                f"  huggingface-cli download mlx-community/gemma-3-12b-it-4bit --local-dir {self.gemma_root}")
        t0 = time.time()
        self._embedder = build_embeddings_processor(self.full_checkpoint, self.gemma_root, self.dtype, self.device)
        log.info(f"  嵌入处理器 (MPS): {time.time() - t0:.1f}s")
        if self.tier == "mlx-24":
            t0 = time.time()
            self._gemma = MlxGemma(self.gemma_root, self.dtype, self.device)
            log.info(f"  Gemma (MLX, 常驻): {time.time() - t0:.1f}s")
        self._load_audio_stage()

    def _encode_uncached(self, todo: list[str]) -> None:
        gemma = self._gemma
        if gemma is None:
            t0 = time.time()
            gemma = MlxGemma(self.gemma_root, self.dtype, self.device)
            log.info(f"  Gemma (MLX) 加载: {time.time() - t0:.1f}s")
        try:
            for p in todo:
                hs, mask = gemma.encode(p)
                self._store_embed(p, self._audio_encoding(self._embedder, hs, mask))
                del hs
        finally:
            if gemma is not self._gemma:
                gemma.close()
                _free()

