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

# ── 情感预设：中文标签（UI 显示）→ 英文导演指令（喂模型）──────────
# 模型(DramaBox)情感表演用英文自然语言效果最好，故指令走英文；UI 标签保持中文。
# 想调某个情绪的强弱/效果，只改右边这句英文即可（生成后用耳朵听着调）。
EMOTION_PRESETS: dict[str, str] = {
    "正常": "语气平静自然地说道",
    "开心": "心情愉悦、语气温暖，带着藏不住的笑意爽朗地说道",
    "兴奋": "情绪饱满、语速加快、铿锵有力地大声说道",
    "愤怒": "怒火中烧、咬牙切齿，猛地拔高音量厉声怒喝道",
    "悲伤": "声音哽咽、字字沉重，强忍着泪意悲伤地说道",
    "恐惧": "声音发颤、气息急促不稳，惊恐不安地说道",
    "温柔": "放轻了声音、语气绵软温暖，柔声细语地说道",
    "深情": "饱含深情、语气真挚动人，缱绻款款地诉说道",
    "惊讶": "猛地一愣、语气陡然加重，难以置信地沉声说道",
    "大笑": "再也忍俊不禁、爽朗放声大笑着说道",
    "冷漠": "语气平淡疏离、不带一丝起伏，冷冷地说道",
    "讽刺": "嘴角噙着冷笑、阴阳怪气地拉长腔调讥讽道",
    "严肃": "神色一沉、语气凝重，一字一顿地郑重说道",
    "疲惫": "声音低沉乏力、有气无力，疲惫地缓缓叹道",
    "命令": "语气强硬、不容置疑，居高临下地厉声命令道",
    # 天生偏女性化的语气，UI 标注（女声），避免用在男声音色上变声
    "撒娇（女声）": "嗲声嗲气、尾音上扬并娇软地拖长，撒娇耍赖地说道",
    "娇媚（女声）": "声音甜糯柔媚、缠绵婉转，带着三分娇、七分媚，眼波流转般妩媚动人地低声说道",
    "媚惑（女声）": "嗓音低柔微哑、黏糯缠绵，字字慢条斯理地勾着尾音，含着一抹勾魂摄魄的笑意，魅惑撩人地低声说道",
    "性感低语（女声）": "把声音压到极低极软、几乎全是温热的气声，像凑在耳边、欲语还休地轻吐每个字，慵懒而极尽性感地呢喃道",
}
# 供 UI 下拉使用的标签列表（中文）
EMOTION_LABELS: list[str] = list(EMOTION_PRESETS.keys())


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
        for _q in ('"', "'", '“', '”', '‘', '’', '＂', '「', '」'):
            if content.startswith(_q):
                content = content[len(_q):]
            if content.endswith(_q):
                content = content[:-len(_q)]

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
        emotion = item.get("emotion") or "正常"
        ref = voice_map.get(char) if char else None

        # 官方格式：导演指令在引号外（不会被念出），台词在双引号内（原样念）
        directive = EMOTION_PRESETS.get(emotion, EMOTION_PRESETS["正常"])
        prompt = f'{directive}，"{line}"'

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
