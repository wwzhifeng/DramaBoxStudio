"""语速调节（变速不变调）。

用 PyAV 内置的 libav `atempo` 滤镜在进程内完成，不依赖外部 ffmpeg 二进制，
随整合包自包含、离线可用。atempo 单次支持 0.5~2.0，超出自动串联多级。
"""
from __future__ import annotations

import math
from pathlib import Path


def _atempo_chain(rate: float) -> list[float]:
    """atempo 单次只支持 0.5~2.0，超出时拆成多级串联。"""
    rate = max(0.25, min(4.0, float(rate)))
    factors = []
    r = rate
    while r > 2.0:
        factors.append(2.0); r /= 2.0
    while r < 0.5:
        factors.append(0.5); r /= 0.5
    factors.append(round(r, 4))
    return factors


def change_speed(in_path: str, rate: float, out_path: str | None = None) -> str:
    """把 in_path 的语速改为 rate 倍（>1 加速、<1 减速），音调不变。

    rate 约等于 1 时原样返回。返回处理后的文件路径。
    """
    if rate is None or abs(float(rate) - 1.0) < 0.02:
        return in_path

    import av

    if out_path is None:
        p = Path(in_path)
        out_path = str(p.with_name(f"{p.stem}_x{float(rate):.2f}{p.suffix}"))

    in_container = av.open(in_path)
    in_stream = in_container.streams.audio[0]
    cc = in_stream.codec_context

    out_container = av.open(out_path, mode="w")
    out_stream = out_container.add_stream("pcm_s16le", rate=cc.rate)

    graph = av.filter.Graph()
    src = graph.add_abuffer(
        format=cc.format.name,
        sample_rate=cc.rate,
        layout=cc.layout.name,
        time_base=in_stream.time_base,
    )
    prev = src
    for f in _atempo_chain(rate):
        node = graph.add("atempo", f"{f}")
        prev.link_to(node)
        prev = node
    sink = graph.add("abuffersink")
    prev.link_to(sink)
    graph.configure()

    def _drain():
        while True:
            try:
                out_frame = sink.pull()
            except (av.error.BlockingIOError, av.error.EOFError):
                break
            out_frame.pts = None
            for packet in out_stream.encode(out_frame):
                out_container.mux(packet)

    for frame in in_container.decode(audio=0):
        frame.pts = None
        src.push(frame)
        _drain()
    # flush
    src.push(None)
    _drain()
    for packet in out_stream.encode(None):
        out_container.mux(packet)

    out_container.close()
    in_container.close()
    return out_path
