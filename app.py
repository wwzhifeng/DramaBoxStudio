"""
DramaBox Studio — AI 配音工作室
基于 Resemble AI / Lightricks LTX-2.3
"""
import logging
import os
import sys
import time
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(APP_DIR))
sys.path.insert(0, str(APP_DIR / "src"))

from inference_server import TTSServer
from config import (GRADIO_PORT, DIT_CHECKPOINT, AUDIO_COMPONENTS, GEMMA_DIR,
                    OUTPUT_DIR, VOICES_DIR, CFG_SCALE, STG_SCALE, STEPS, SEED)
from voice_library import list_voices, save_voice, delete_voice
from dialogue import parse_script, EMOTION_LABELS

# 逐句情感：UI 最多为前 N 句提供情感选择器（超出按"正常"生成）
MAX_EMOTION_LINES = 30

import gradio as gr

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("DramaBox")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
VOICES_DIR.mkdir(parents=True, exist_ok=True)

# ── 加载模型（全局单例）──────────────────────────────
log.info("DramaBox Studio 启动中，加载模型...")
tts = TTSServer(
    checkpoint=str(DIT_CHECKPOINT),
    full_checkpoint=str(AUDIO_COMPONENTS),
    gemma_root=str(GEMMA_DIR),
    device="cuda",
    dtype="bf16",
)
log.info("模型就绪。")

# ══════════════════════════════════════════════════
# UI — 暗色影棚风
# ══════════════════════════════════════════════════

CSS = """
/* ═══════════════════════════════════════════════════
   DramaBox Studio · 暗夜鎏金 (Midnight Gold)
   纯外观样式 — 不含任何布局/定位/遮罩/flex容器规则
   ═══════════════════════════════════════════════════ */
:root {
    --bg:        #0E0E10;
    --surface:   #16161A;
    --surface-2: #1C1C21;
    --line:      rgba(255,255,255,.065);
    --line-2:    rgba(255,255,255,.11);
    --txt:       #EDEBE4;
    --txt-dim:   #9C9A90;
    --txt-mute:  #66645C;
    --gold:      #C9A24B;
    --gold-2:    #E2C277;
    --gold-soft: rgba(201,162,75,.10);
    --gold-line: rgba(201,162,75,.30);
    --r:    16px;
    --r-sm: 10px;
    --shadow: 0 1px 2px rgba(0,0,0,.5), 0 10px 34px rgba(0,0,0,.30);
}

/* ── 全局基调 ── */
body, .gradio-container {
    background: var(--bg) !important;
    color: var(--txt) !important;
    font-family: "Inter", "Microsoft YaHei", "PingFang SC", system-ui, sans-serif !important;
    letter-spacing: .005em !important;
}
/* 默认就放大到 125%（等价于浏览器缩放 125%）；加在 body 上避免横向溢出 */
body { zoom: 1.25; }
/* 全宽利用 16:9 屏，超宽屏封顶；紧凑垂直间距以求接近一屏 */
.gradio-container {
    max-width: 1680px !important;
    width: 100% !important;
    margin: 0 auto !important;
    padding: 12px 30px 8px !important;
}

/* ── 卡片：组件块统一面板 ── */
.block, .form {
    background: var(--surface) !important;
    border: 1px solid var(--line) !important;
    border-radius: var(--r) !important;
    box-shadow: var(--shadow) !important;
}

/* ── Tab 栏：杂志感金色下划线 ── */
.tabs > .tab-nav {
    border-bottom: 1px solid var(--line) !important;
    margin-bottom: 16px !important; gap: 2px !important;
}
.tabs > .tab-nav > button {
    background: transparent !important; color: var(--txt-dim) !important;
    border: none !important; border-bottom: 2px solid transparent !important;
    border-radius: 0 !important; margin: 0 !important;
    font-size: 15px !important; font-weight: 600 !important;
    padding: 13px 26px !important;
    transition: color .2s ease, border-color .2s ease !important;
}
.tabs > .tab-nav > button:hover { color: var(--txt) !important; }
.tabs > .tab-nav > button.selected {
    color: var(--gold) !important;
    border-bottom-color: var(--gold) !important;
}

/* ── 输入框 / 文本域 / 数字 ── */
textarea, input[type="text"], input[type="number"],
.gradio-container input[type="text"], .gradio-container textarea {
    background: var(--surface-2) !important; color: var(--txt) !important;
    border: 1px solid var(--line) !important; border-radius: var(--r-sm) !important;
    font-size: 14px !important; line-height: 1.6 !important;
    transition: border-color .2s ease, background .2s ease !important;
}
textarea::placeholder, input::placeholder { color: var(--txt-mute) !important; }
textarea:focus, input:focus {
    border-color: var(--gold-line) !important; background: #202024 !important;
    outline: none !important;
}

/* ── 主按钮：鎏金 ── */
button.primary, .gr-button-primary {
    background: linear-gradient(180deg, var(--gold-2), var(--gold)) !important;
    color: #1A1509 !important; font-weight: 700 !important;
    border: none !important; border-radius: var(--r-sm) !important;
    font-size: 15px !important; letter-spacing: .02em !important;
    box-shadow: 0 2px 10px rgba(201,162,75,.18) !important;
    transition: filter .18s ease, transform .12s ease !important;
}
button.primary:hover, .gr-button-primary:hover { filter: brightness(1.08); transform: translateY(-1px); }
button.primary:active { transform: translateY(0); }

/* ── 次按钮：描边幽灵 ── */
button.secondary, .gr-button-secondary {
    background: transparent !important; color: var(--txt-dim) !important;
    border: 1px solid var(--line-2) !important; border-radius: var(--r-sm) !important;
    font-size: 14px !important; font-weight: 600 !important;
    transition: color .18s ease, border-color .18s ease !important;
}
button.secondary:hover, .gr-button-secondary:hover {
    color: var(--gold-2) !important; border-color: var(--gold-line) !important;
}

/* ── 标签：暗底 + 浅字（主要靠上面的 theme.set()，这里做兜底）── */
.gradio-container {
    --block-label-background-fill: #1C1C21 !important;
    --block-label-text-color: #D2CFC6 !important;
    --block-label-border-color: #2A2A30 !important;
    /* 组件标题(角色/音色/台词等标签) = 按钮同款哑光金 + 深字（这才是橙色的真正来源） */
    --block-title-background-fill: #C9A24B !important;
    --block-title-text-color: #1A1509 !important;
    --block-title-border-color: #C9A24B !important;
    --block-info-text-color: var(--txt-mute) !important;
}
.block-label, span.block-label, .block .block-label, [class*="block_label"], [class*="block-label"] {
    background: #1C1C21 !important;
    color: #D2CFC6 !important;
    border: 1px solid #2A2A30 !important; box-shadow: none !important;
    font-weight: 600 !important; letter-spacing: .02em !important;
}
.block-label svg, .block-label .icon { color: var(--gold) !important; opacity: .85; }
.label-text { color: var(--txt-dim) !important; }

/* ── 徽标 ── */
.badge {
    display: inline-block; background: var(--gold-soft); color: var(--gold);
    border: 1px solid var(--gold-line); border-radius: 999px;
    padding: 3px 13px; font-size: 11.5px; font-weight: 700;
}

/* ── Banner ── */
.studio-banner {
    background: linear-gradient(120deg, #1A1712 0%, var(--surface) 55%) !important;
    border: 1px solid var(--line);
    border-left: 3px solid var(--gold);
    border-radius: var(--r); padding: 18px 28px; margin-bottom: 32px;
    display: flex; align-items: center; gap: 18px;
    box-shadow: var(--shadow);
}
.studio-banner .brand-mark {
    font-size: 30px; line-height: 1;
    width: 58px; height: 58px; flex: 0 0 58px;
    display: flex; align-items: center; justify-content: center;
    background: var(--gold-soft); border: 1px solid var(--gold-line);
    border-radius: 14px;
}
.studio-banner .logo-text {
    font-size: 26px; font-weight: 800; color: var(--txt);
    letter-spacing: .01em; margin-bottom: 4px;
}
.studio-banner .logo-text .accent { color: var(--gold); }
.studio-banner .sub { color: var(--txt-mute); font-size: 13px; letter-spacing: .04em; }

/* ── Markdown 说明文字 ── */
.prose, .prose p, .md { color: var(--txt-dim) !important; font-size: 13.5px !important; line-height: 1.75 !important; }
.prose strong, .md strong { color: var(--txt) !important; }
.prose code, .md code {
    background: var(--surface-2) !important; color: var(--gold-2) !important;
    border: 1px solid var(--line) !important; border-radius: 5px !important;
    padding: 1px 7px !important; font-size: 12.5px !important;
}

/* ── DataFrame（仅配色，不动结构） ── */
.gr-dataframe table th, table thead th {
    background: var(--surface-2) !important; color: var(--gold) !important;
    font-weight: 700 !important; letter-spacing: .03em !important;
    border-color: var(--line) !important;
}
.gr-dataframe table td, table tbody td {
    color: var(--txt-dim) !important; border-color: var(--line) !important;
}

/* ── 滑块 ── */
input[type="range"] { accent-color: var(--gold) !important; }

/* ── Accordion ── */
.label-wrap > span { color: var(--txt) !important; font-weight: 600 !important; }

/* ── 进度条 ── */
.progress-bar, .gr-progress .progress-bar { background: var(--gold) !important; }

/* ── Examples（胶囊标签） ── */
.gr-samples-table td, .examples td {
    background: var(--surface-2) !important; color: var(--txt-dim) !important;
    border: 1px solid var(--line) !important; border-radius: 999px !important;
    transition: color .18s ease, border-color .18s ease !important;
}
.gr-samples-table td:hover, .examples td:hover {
    color: var(--gold) !important; border-color: var(--gold-line) !important;
}

/* ── 滚动条 ── */
*::-webkit-scrollbar { width: 10px; height: 10px; }
*::-webkit-scrollbar-track { background: var(--bg); }
*::-webkit-scrollbar-thumb { background: #2B2B30; border-radius: 999px; border: 2px solid var(--bg); }
*::-webkit-scrollbar-thumb:hover { background: #3A3A40; }

/* ── 底部 ── */
footer { visibility: hidden; }

/* ── 自定义页脚（署名 + 链接） ── */
.studio-footer {
    margin-top: 30px; padding: 18px 28px;
    background: linear-gradient(120deg, var(--surface) 60%, #1A1712 100%);
    border: 1px solid var(--line); border-top: 2px solid var(--gold);
    border-radius: var(--r); box-shadow: var(--shadow);
    display: flex; align-items: center; justify-content: space-between;
    flex-wrap: wrap; gap: 16px;
}
.studio-footer .ft-left { color: var(--txt-dim); font-size: 13px; letter-spacing: .05em; }
.studio-footer .ft-name {
    color: var(--gold); font-weight: 800; font-size: 15px;
    letter-spacing: .08em; margin-left: 4px;
}
.studio-footer .ft-links { display: flex; gap: 12px; flex-wrap: wrap; }
.studio-footer .ft-links a {
    color: var(--txt-dim); text-decoration: none;
    font-size: 13px; font-weight: 600; letter-spacing: .03em;
    padding: 9px 20px; border-radius: 999px;
    border: 1px solid var(--line-2);
    transition: color .18s ease, border-color .18s ease, background .18s ease;
}
.studio-footer .ft-links a:hover {
    color: #1A1509; background: var(--gold); border-color: var(--gold);
}
"""


# ── 本地内嵌 Inter 字体（自包含 / 离线 / 不依赖外网或系统字体） ──
def _build_font_css() -> str:
    """读取 assets/fonts 下的 Inter woff2，base64 内嵌为 @font-face。
    随包分发，离线可用；用 data URI 而非文件路径，避免任何机器上的路径/服务问题。"""
    import base64
    font_dir = APP_DIR / "assets" / "fonts"
    faces = []
    for weight in (400, 600, 700, 800):
        fp = font_dir / f"inter-latin-{weight}-normal.woff2"
        if not fp.exists():
            continue
        b64 = base64.b64encode(fp.read_bytes()).decode("ascii")
        faces.append(
            "@font-face{font-family:'Inter';font-style:normal;font-weight:%d;"
            "font-display:swap;src:url(data:font/woff2;base64,%s) format('woff2')}" % (weight, b64)
        )
    return "\n".join(faces)


CSS = _build_font_css() + "\n" + CSS

# ── 预设示例 ─────────────────────────────────────────
_VOICES_DIR = APP_DIR / "assets" / "voices"

EXAMPLES: list[tuple[str, str, str]] = [
    ("黑帮大佬·阴冷威胁", str(_VOICES_DIR / "male_harvey_keitel.mp3"),
     '一个上了年纪的黑帮大佬，声音沙哑低沉、不动声色却暗藏杀气，'
     '"你以为，走进这扇门，还能活着出去？" '
     '他低低地笑了几声，笑里藏刀，"年轻人……是谁给你的胆子。"'),
    ("暴怒训斥·拍案而起", str(_VOICES_DIR / "male_samuel_j.mp3"),
     '一个男人强压着怒火，一字一顿、咬字极重，"我，再问你，最后一遍——东西在哪儿。" '
     '话音未落他猛地一拍桌子，声音轰然炸开，"别逼我亲自动手！"'),
    ("脱口秀·笑到窒息", str(_VOICES_DIR / "male_conan.mp3"),
     '一个脱口秀主持人倒吸一口凉气，难以置信地喊，"不是吧！这话你也敢说？" '
     '紧接着他再也忍不住，放声狂笑，"哈哈哈哈——我的天，我要笑死了，喘不上气了！"'),
    ("深夜电台·治愈低语", str(_VOICES_DIR / "male_old_movie.wav"),
     '一个低沉磁性的男声，慢悠悠地、像贴着耳边轻语，"夜深了，还没睡的朋友，这首歌，送给此刻有心事的你。" '
     '他顿了顿，声音愈发温柔，"别撑着了……今晚，好好睡一觉吧。"'),
    ("热血少年·绝地觉醒", str(_VOICES_DIR / "male_arnie.mp3"),
     '一个少年的声音起初还在发颤、带着哭腔，"我……我真的做得到吗？" '
     '忽然像是下定了决心，声音一字一句地坚定起来，最后嘶吼出声，"够了！这一次——我绝不再逃！"'),
    ("醉酒老爸·深夜倾诉", str(_VOICES_DIR / "male_petergriffin.wav"),
     '一个中年男人喝多了，舌头有些打结、声音疲惫又哽咽，"儿子啊……爸这辈子没本事，让你跟着受委屈了。" '
     '他重重地叹了口气，"你可……一定要比爸强啊。"'),
    ("宫斗娘娘·笑里藏刀", str(_VOICES_DIR / "female_shadowheart.wav"),
     '一个女人柔声细语、语气温婉得体，"妹妹这身衣裳，真是好看。" '
     '话锋陡然一转，声音里渗出阴冷的笑意，"只可惜啊……怕是没机会再穿第二回了。"'),
    ("甜妹撒娇·黏人精（女声）", str(_VOICES_DIR / "female_american.wav"),
     '一个年轻女人嗲声嗲气、尾音软软地往上勾，"哎呀人家不管啦～你今天答应过要陪我的嘛！" '
     '她故意把声音拖得长长的撒娇，"就去嘛就去嘛，好不好嘛～"'),
]


# ── 回调函数 ─────────────────────────────────────────

def on_generate(prompt: str, audio_ref, cfg: float, stg: float,
                dur_mult: float, gen_dur: float, ref_dur: float, seed: int,
                speed: float = 1.0):
    """单句生成。参考音频来自下方播放器（选音色会载入这里，或直接上传）。speed=语速倍率。"""
    if not prompt or not prompt.strip():
        raise gr.Error("请输入提示词。")
    t0 = time.time()
    ref_path = audio_ref if audio_ref and os.path.exists(str(audio_ref)) else None
    output = str(OUTPUT_DIR / f"output_{int(time.time())}.wav")
    tts.generate_to_file(
        prompt=prompt, output=output, voice_ref=ref_path,
        cfg_scale=cfg, stg_scale=stg,
        duration_multiplier=dur_mult, seed=int(seed),
        gen_duration=float(gen_dur), ref_duration=float(ref_dur),
    )
    # 语速调节（变速不变调，PyAV atempo，自包含离线）
    if speed and abs(float(speed) - 1.0) >= 0.02:
        try:
            from audio_speed import change_speed
            output = change_speed(output, float(speed))
        except Exception as e:
            log.warning(f"语速调节失败，输出原速: {e}")
    dur = time.time() - t0
    log.info(f"生成: {dur:.1f}s → {output}")
    return output


def on_save_voice(name: str, audio) -> list:
    if not name or not name.strip():
        raise gr.Error("请输入音色名称。")
    if audio is None:
        raise gr.Error("请上传参考音频。")
    save_voice(name.strip(), str(audio))
    return refresh_voice_table()


def on_delete_voice(name: str) -> list:
    if not name or not name.strip():
        raise gr.Error("请输入要删除的音色名称。")
    if not delete_voice(name.strip()):
        raise gr.Error(f"未找到音色「{name}」。")
    return refresh_voice_table()


def refresh_voice_table() -> list:
    voices = list_voices()
    if not voices:
        return []
    return [[v["name"], f'{v["duration_seconds"]:.1f}s', v["created_at"]] for v in voices]


def get_voice_choices() -> list:
    """返回 [(name, path)] 供 dropdown 使用。"""
    return [(v["name"], v["path"]) for v in list_voices()]


def on_quick_save_voice(name: str, audio_ref):
    """在快速生成页把上传的参考音频存为音色，并刷新音色下拉。"""
    if not name or not name.strip():
        raise gr.Error("请先填写音色名称。")
    if not audio_ref:
        raise gr.Error("请先上传参考音频，再保存。")
    if save_voice(name.strip(), str(audio_ref)) is None:
        raise gr.Error("保存失败，请检查音频文件。")
    return gr.update(choices=get_voice_choices())


def on_refresh_voices():
    """重新加载音色下拉（用于另一标签页保存后的同步）。"""
    return gr.update(choices=get_voice_choices())


def on_parse_script(text: str):
    """解析剧本 → 更新角色分配 UI。"""
    if not text or not text.strip():
        raise gr.Error("请粘贴剧本内容。")
    lines = parse_script(text)
    if not lines:
        raise gr.Error("未识别到有效对话。")
    chars = list(dict.fromkeys(item["character"] for item in lines if item["character"]))
    choices = get_voice_choices()
    updates = []
    for i in range(8):
        if i < len(chars):
            updates.append(gr.update(visible=True, value=chars[i], label=f"角色 {i+1}"))
        else:
            updates.append(gr.update(visible=False, value=""))
    # 自动给每个角色分配库里的一个音色（轮流，避免都用同一个；留空会走内置默认嗓音）
    voice_paths = [c[1] for c in choices]
    dropdown_updates = []
    for i in range(8):
        if i < len(chars):
            default_voice = voice_paths[i % len(voice_paths)] if voice_paths else None
            dropdown_updates.append(gr.update(visible=True, choices=choices, value=default_voice))
        else:
            dropdown_updates.append(gr.update(visible=False, choices=choices))
    # 逐句情感下拉：为前 MAX_EMOTION_LINES 句各显示一个，标签带上台词预览
    emotion_updates = []
    for i in range(MAX_EMOTION_LINES):
        if i < len(lines):
            spk = lines[i]["character"] or "旁白"
            text = lines[i]["line"]
            preview = text[:16] + ("…" if len(text) > 16 else "")
            emotion_updates.append(gr.update(visible=True, value="正常",
                                             label=f"{i+1}. {spk}：{preview}"))
        else:
            emotion_updates.append(gr.update(visible=False))
    title_update = gr.update(visible=True)
    # store parsed data in state
    parsed_json = [{"character": item["character"], "line": item["line"]} for item in lines]
    status = f"已识别 {len(lines)} 句台词 · {len(chars)} 个角色"
    if len(lines) > MAX_EMOTION_LINES:
        status += f"（逐句情感最多设置前 {MAX_EMOTION_LINES} 句，其余按「正常」生成）"
    return updates + dropdown_updates + emotion_updates + [title_update, parsed_json, status]


def on_generate_dialogue(parsed_lines, speed, *dropdown_values):
    """批量生成对话。speed=语速倍率；dropdown_values = 前 8 个音色下拉 + 后 N 个逐句情感下拉。"""
    if not parsed_lines:
        raise gr.Error("请先解析剧本。")
    voices = dropdown_values[:8]
    emotions = dropdown_values[8:]
    # 角色 → 音色
    voice_map = {}
    chars = list(dict.fromkeys(item["character"] for item in parsed_lines if item["character"]))
    for i, char in enumerate(chars):
        if i < len(voices) and voices[i]:
            voice_map[char] = voices[i]
    # 把每句的情感并进台词数据（超出选择器范围的按"正常"）
    lines_with_emotion = []
    for i, item in enumerate(parsed_lines):
        emo = emotions[i] if i < len(emotions) and emotions[i] else "正常"
        lines_with_emotion.append({**item, "emotion": emo})
    from dialogue import generate_dialogue

    def progress_cb(cur, total, character, line):
        log.info(f"[{cur}/{total}] {character}: {line[:30]}...")

    out = generate_dialogue(lines_with_emotion, voice_map, tts, progress_cb)
    if out is None:
        raise gr.Error("生成失败。")
    # 语速调节（变速不变调）
    if speed and abs(float(speed) - 1.0) >= 0.02:
        try:
            from audio_speed import change_speed
            out = change_speed(out, float(speed))
        except Exception as e:
            log.warning(f"语速调节失败，输出原速: {e}")
    return out


# ══════════════════════════════════════════════════
with gr.Blocks(
    title="DramaBox Studio — AI 配音工作室",
    theme=gr.themes.Soft(
        primary_hue="amber",
        secondary_hue="gray",
        neutral_hue="gray",
        font=gr.themes.Font("Inter"),
        font_mono=gr.themes.Font("ui-monospace"),
    ).set(
        block_label_background_fill="#1C1C21",
        block_label_text_color="#D2CFC6",
        block_label_border_color="#2A2A30",
        # 组件标题(角色/音色/台词等标签) → 用按钮同款哑光金 + 深字（替换刺眼橙）
        block_title_background_fill="#C9A24B",
        block_title_text_color="#1A1509",
        block_title_border_color="#C9A24B",
        block_info_text_color="#66645C",
    ),
    css=CSS,
    analytics_enabled=False,
) as app:

    # ── Header ──
    gr.HTML(
        '<div class="studio-banner">'
        '<div class="brand-mark">🎭</div>'
        '<div class="brand-text">'
        '<div class="logo-text">DramaBox <span class="accent">Studio</span></div>'
        '<div class="sub">基于 LTX-2.3 · Resemble AI DramaBox · 纯本地 AI 配音工作室</div>'
        '</div>'
        '</div>'
    )

    with gr.Tabs(elem_classes=["tabs"]) as tabs:

        # ══════════════════════════════════════════
        # Tab 1: 快速生成
        # ══════════════════════════════════════════
        with gr.TabItem("🎤 快速生成", id="tab_quick"):
            gr.Markdown(
                "输入场景提示词，可选上传参考音频（10 秒以上），生成带表情的语音。"
                "**格式：** 对话放 `\"双引号\"` 内，动作描述放引号外。"
            )

            with gr.Row():
                with gr.Column(scale=3):
                    prompt_box = gr.Textbox(
                        label="场景提示词",
                        placeholder='A woman speaks warmly, "Hello, how are you today?" '
                                    'She laughs, "Hahaha, it is so good to see you!"',
                        lines=6,
                    )
                    with gr.Row():
                        quick_voice = gr.Dropdown(
                            label="选择已保存音色（选中会载入下方，可试听）",
                            choices=get_voice_choices(), value=None,
                            interactive=True, scale=5,
                        )
                        refresh_voice_btn = gr.Button("🔄 刷新", variant="secondary", scale=1)
                    audio_ref = gr.Audio(
                        label="参考音频 · 可试听（选上方音色会载入这里，也可直接上传 10s+）",
                        type="filepath",
                    )
                    with gr.Row():
                        quick_voice_name = gr.Textbox(
                            label="把上传的音频存为音色", placeholder="给音色起个名，例如：女主角",
                            scale=4,
                        )
                        quick_save_btn = gr.Button("💾 保存音色", variant="secondary", scale=1)
                    gen_btn = gr.Button("✦ 生成", variant="primary", size="lg")

                with gr.Column(scale=2):
                    with gr.Accordion("⚙️ 推理设置", open=False):
                        cfg_slider = gr.Slider(1.0, 10.0, value=CFG_SCALE, step=0.5, label="CFG 引导强度")
                        stg_slider = gr.Slider(0.0, 5.0, value=STG_SCALE, step=0.5, label="STG 跳过令牌引导")
                        dur_slider = gr.Slider(0.8, 2.0, value=1.1, step=0.05,
                                               label="时长倍率（目标时长 = 0 时生效）")
                        gen_dur_slider = gr.Slider(0.0, 60.0, value=0.0, step=1.0,
                                                   label="目标时长（秒）— 0 = 自动")
                        ref_dur_slider = gr.Slider(3.0, 30.0, value=10.0, step=1.0, label="参考音频时长（秒）")
                        speed_slider = gr.Slider(0.5, 2.0, value=1.0, step=0.05,
                                                 label="语速（1=原速，>1 加速，<1 减速；变速不变调）")
                        seed_input = gr.Number(value=SEED, label="随机种子", precision=0)
                    audio_out = gr.Audio(label="生成结果", type="filepath")

            gen_btn.click(
                on_generate,
                inputs=[prompt_box, audio_ref, cfg_slider, stg_slider,
                        dur_slider, gen_dur_slider, ref_dur_slider, seed_input, speed_slider],
                outputs=[audio_out],
            )

            # 选中已保存音色 → 自动载入到参考音频播放器（可试听）
            quick_voice.change(lambda p: p, inputs=[quick_voice], outputs=[audio_ref])
            # 把当前参考音频存为音色 / 刷新音色下拉
            quick_save_btn.click(on_quick_save_voice,
                                 inputs=[quick_voice_name, audio_ref], outputs=[quick_voice])
            refresh_voice_btn.click(on_refresh_voices, outputs=[quick_voice])

            gr.Examples(
                label="📋 点击示例快速体验",
                examples=[
                    [name, prompt, voice_path, CFG_SCALE, STG_SCALE, 1.1, 0.0, 10.0, SEED]
                    for name, voice_path, prompt in EXAMPLES
                ],
                example_labels=[name for name, _, _ in EXAMPLES],
                inputs=[gr.Textbox(visible=False, label="场景"),
                        prompt_box, audio_ref,
                        cfg_slider, stg_slider, dur_slider, gen_dur_slider,
                        ref_dur_slider, seed_input],
                outputs=[audio_out],
                fn=lambda _n, prompt, ref, cfg, stg, dur, gd, rd, sd: on_generate(
                    prompt, ref, cfg, stg, dur, gd, rd, sd),
                cache_examples=False, run_on_click=False, examples_per_page=20,
            )

        # ══════════════════════════════════════════
        # Tab 2: 对话工坊
        # ══════════════════════════════════════════
        with gr.TabItem("🎬 对话工坊", id="tab_dialogue"):
            gr.Markdown(
                "粘贴剧本 → 自动识别角色并**从音色库自动分配声音**（可下拉更换）→ **逐句挑选情感** → 一键生成。\n\n"
                "**剧本格式：** `角色名：台词` 或 `角色名:台词`，每行一句。"
            )

            with gr.Row():
                with gr.Column(scale=1):
                    script_input = gr.Textbox(
                        label="剧本内容",
                        placeholder='张三：好久不见啊，兄弟！\n李四：是啊，都快十年了。\n张三：你还记得那次我们一起...',
                        lines=12,
                    )
                    with gr.Row():
                        parse_btn = gr.Button("🔍 解析角色", variant="primary")
                    parse_status = gr.Markdown("")

                with gr.Column(scale=1):
                    gr.Markdown("**角色 → 音色分配**")
                    char_names = []
                    voice_drops = []
                    for i in range(8):
                        with gr.Row():
                            cn = gr.Textbox(label=f"角色 {i+1}", visible=False, interactive=False, scale=1)
                            vd = gr.Dropdown(label="音色", visible=False, interactive=True, scale=1)
                            char_names.append(cn)
                            voice_drops.append(vd)

                    emotion_title = gr.Markdown(
                        "**🎭 逐句情感** · 每句台词挑一个情绪（默认正常）"
                        "　|　标〔女声〕的请只用于女性音色，否则可能变声", visible=False)
                    emotion_drops = []
                    for i in range(MAX_EMOTION_LINES):
                        ed = gr.Dropdown(
                            choices=EMOTION_LABELS, value="正常",
                            label=f"第 {i+1} 句", visible=False, interactive=True,
                        )
                        emotion_drops.append(ed)

                    dlg_speed = gr.Slider(0.5, 2.0, value=1.0, step=0.05,
                                          label="语速（1=原速，>1 加速，<1 减速；变速不变调）")
                    dialogue_gen_btn = gr.Button("🎬 生成完整对话", variant="primary", visible=False)
                    dialogue_out = gr.Audio(label="完整对话音频", type="filepath")

            # State to hold parsed lines between parse and generate
            parsed_state = gr.State([])

            all_char_outputs = char_names + voice_drops
            parse_btn.click(
                on_parse_script,
                inputs=[script_input],
                outputs=all_char_outputs + emotion_drops + [emotion_title, parsed_state, parse_status],
            ).then(
                lambda: gr.update(visible=True),
                outputs=[dialogue_gen_btn],
            )

            dialogue_gen_btn.click(
                on_generate_dialogue,
                inputs=[parsed_state, dlg_speed] + voice_drops + emotion_drops,
                outputs=[dialogue_out],
            )

    # ── Footer ──
    gr.HTML(
        '<div class="studio-footer">'
        '<div class="ft-left">整合包制作 ·<span class="ft-name">王知风</span></div>'
        '<div class="ft-links">'
        '<a href="https://wangzhifeng.vip/" target="_blank" rel="noopener">更多 AI 工具</a>'
        '<a href="https://wangzhifeng.vip/" target="_blank" rel="noopener">详细教程</a>'
        '</div>'
        '</div>'
    )

if __name__ == "__main__":
    port = int(os.environ.get("GRADIO_SERVER_PORT", str(GRADIO_PORT)))
    app.queue(max_size=10).launch(
        server_name="0.0.0.0", server_port=port,
        inbrowser=True,
    )
