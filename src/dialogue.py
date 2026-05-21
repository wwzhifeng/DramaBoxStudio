"""对话工坊 — 剧本解析 + 批量生成 + 拼接。"""
import logging
import re
import time
from pathlib import Path

import numpy as np
import torch
from scipy.io.wavfile import read as wav_read, write as wav_write

from config import OUTPUT_DIR

log = logging.getLogger(__name__)

_LINE_RE = re.compile(r"^(.+?)[：:]\s*(.+)$")


def parse_script(text: str) -> list[dict]:
    """解析剧本，返回 [{character, line}, ...]。

    支持格式：
        张三：你好啊！
        李四: 好久不见。
        张三："这是对话内容"
        （无角色前缀的行合并到上一句）

    返回的 character 可能为 None（无法识别角色时）。
    """
    results: list[dict] = []
    for raw in text.strip().splitlines():
        line = raw.strip()
        if not line:
            continue
        m = _LINE_RE.match(line)
        if m:
            char = (m.group(1) or "").strip()
            content = m.group(2).strip()
        else:
            char = ""
            content = line

        # 去掉首尾引号
        if content.startswith('"') and content.endswith('"'):
            content = content[1:-1]
        if content.startswith("""") and content.endswith(""""):
            content = content[1:-1]

        if not content:
            continue

        character = char if char else None
        # 如果无角色，合并到上一句
        if character is None and results:
            results[-1]["line"] += content
        else:
            results.append({"character": character or "旁白", "line": content})

    return results


def generate_dialogue(
    lines: list[dict],
    voice_map: dict[str, str],
    tts,
    progress_cb=None,
) -> str | None:
    """逐句生成 → 拼接 → 返回完整音频路径。

    lines:       parse_script() 的输出
    voice_map:   {角色名: 参考音频路径}
    tts:         TTSServer 实例
    progress_cb: 可选回调 (current, total, character, line)
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    segments: list[np.ndarray] = []
    sr = 24000
    total = len(lines)

    for i, item in enumerate(lines):
        char = item["character"]
        line = item["line"]
        ref = voice_map.get(char) if char else None

        prompt = f'{char} says, "{line}"' if char else line

        log.info(f"[{i+1}/{total}] {char}: {line[:40]}...")
        if progress_cb:
            progress_cb(i + 1, total, char or "", line)

        waveform, sample_rate = tts.generate(prompt=prompt, voice_ref=ref)
        wav = waveform.cpu().float()
        if wav.dim() == 3:
            wav = wav.squeeze(0)
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
        wav_np = wav.squeeze(0).numpy()
        segments.append(wav_np)
        sr = sample_rate

        # 句间小停顿 0.3s
        silence = np.zeros(int(sr * 0.3), dtype=np.float32)
        segments.append(silence)

    if not segments:
        return None

    merged = np.concatenate(segments)
    merged_int16 = (merged * 32767).clip(-32768, 32767).astype(np.int16)
    out_path = OUTPUT_DIR / f"dialogue_{int(time.time())}.wav"
    wav_write(str(out_path), sr, merged_int16)
    log.info(f"对话生成完成: {out_path} ({len(segments)//2} 句, {len(merged)/sr:.1f}s)")
    return str(out_path)
