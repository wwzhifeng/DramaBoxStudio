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
from dialogue import parse_script

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

/* ── 标签：全部去掉橙色填充，统一成低调暗色文字 ── */
.gradio-container {
    --block-label-background-fill: transparent !important;
    --block-label-text-color: var(--txt-dim) !important;
    --block-label-border-color: transparent !important;
    --block-title-text-color: var(--txt-dim) !important;
    --block-info-text-color: var(--txt-mute) !important;
}
label, label span, .label-text,
.block label, .block label span,
.block-label, span.block-label, .block .block-label, [class*="block_label"] {
    background: transparent !important;
    color: var(--txt-dim) !important;
    border: none !important; box-shadow: none !important;
    font-size: 12.5px !important; font-weight: 600 !important;
    letter-spacing: .03em !important;
}
.block-label svg, .block-label .icon, label svg { color: var(--gold) !important; opacity: .8; }

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
    ("反派独白", str(_VOICES_DIR / "male_harvey_keitel.mp3"),
     'A shadowy villain speaks with cold menace, "You have entered my domain, mortal." '
     'He chuckles darkly, "Such arrogance will be your undoing."'),
    ("脱口秀主持人大笑", str(_VOICES_DIR / "male_conan.mp3"),
     'A talk show host gasps with shock, "No! You did NOT just say that!" '
     'He bursts into uncontrollable laughter, "Hahaha! Oh my god!"'),
    ("温柔低语", str(_VOICES_DIR / "female_shadowheart.wav"),
     'A woman speaks tenderly, "It has been a long day, my love." '
     'She whispers, "Close your eyes. I am right here."'),
    ("老式电台主播", str(_VOICES_DIR / "male_old_movie.wav"),
     'A radio host settles into a warm tone, "Good evening everyone, '
     'and welcome back to the show. We have a wonderful lineup tonight."'),
    ("少年英雄", str(_VOICES_DIR / "male_arnie.mp3"),
     'A young warrior speaks with a trembling voice, "I... I do not know if I can do this." '
     'His voice steadies with fire, "No more running. I WILL fight!"'),
    ("疲惫父亲", str(_VOICES_DIR / "male_petergriffin.wav"),
     'An exhausted father speaks with fraying patience, "Sweetie, daddy is asking very nicely." '
     'He sighs deeply, "Ohhhh my goodness."'),
]


# ── 回调函数 ─────────────────────────────────────────

def on_generate(prompt: str, audio_ref, cfg: float, stg: float,
                dur_mult: float, gen_dur: float, ref_dur: float, seed: int):
    """单句生成。"""
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
    dropdown_updates = []
    for i in range(8):
        if i < len(chars):
            dropdown_updates.append(gr.update(visible=True, choices=choices))
        else:
            dropdown_updates.append(gr.update(visible=False, choices=choices))
    # store parsed data in state
    parsed_json = [{"character": item["character"], "line": item["line"]} for item in lines]
    return updates + dropdown_updates + [parsed_json, f"已识别 {len(lines)} 句台词 · {len(chars)} 个角色"]


def on_generate_dialogue(parsed_lines, *dropdown_values):
    """批量生成对话。"""
    if not parsed_lines:
        raise gr.Error("请先解析剧本。")
    voice_map = {}
    chars = list(dict.fromkeys(item["character"] for item in parsed_lines if item["character"]))
    for i, char in enumerate(chars):
        if i < len(dropdown_values) and dropdown_values[i]:
            voice_map[char] = dropdown_values[i]
    from dialogue import generate_dialogue

    def progress_cb(cur, total, character, line):
        log.info(f"[{cur}/{total}] {character}: {line[:30]}...")

    out = generate_dialogue(parsed_lines, voice_map, tts, progress_cb)
    if out is None:
        raise gr.Error("生成失败。")
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
                    audio_ref = gr.Audio(label="参考音频（可选）", type="filepath")
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
                        seed_input = gr.Number(value=SEED, label="随机种子", precision=0)
                    audio_out = gr.Audio(label="生成结果", type="filepath")

            gen_btn.click(
                on_generate,
                inputs=[prompt_box, audio_ref, cfg_slider, stg_slider,
                        dur_slider, gen_dur_slider, ref_dur_slider, seed_input],
                outputs=[audio_out],
            )

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
        # Tab 2: 音色库
        # ══════════════════════════════════════════
        with gr.TabItem("📚 音色库", id="tab_voices"):
            gr.Markdown("管理你的声音角色。上传 10 秒以上的参考音频，给角色命名，对话工坊可直接选用。")

            with gr.Row():
                with gr.Column(scale=2):
                    voice_audio = gr.Audio(label="上传参考音频（10s+）", type="filepath")
                    voice_name = gr.Textbox(label="角色名称", placeholder="例如：张三、李四、女主角")
                    save_btn = gr.Button("保存音色", variant="primary")
                    delete_name = gr.Textbox(label="删除音色（输入名称）", placeholder="输入要删除的角色名称")
                    delete_btn = gr.Button("删除", variant="secondary")

                with gr.Column(scale=3):
                    voice_table = gr.Dataframe(
                        headers=["名称", "时长", "创建时间"],
                        datatype=["str", "str", "str"],
                        col_count=(3, "fixed"),
                        label="已保存的音色",
                        value=refresh_voice_table(),
                        interactive=False,
                    )

            save_btn.click(on_save_voice, inputs=[voice_name, voice_audio],
                          outputs=[voice_table])
            delete_btn.click(on_delete_voice, inputs=[delete_name],
                           outputs=[voice_table])

        # ══════════════════════════════════════════
        # Tab 3: 对话工坊
        # ══════════════════════════════════════════
        with gr.TabItem("🎬 对话工坊", id="tab_dialogue"):
            gr.Markdown(
                "粘贴剧本，自动识别角色，分配音色，一键生成完整对话音频。\n\n"
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

                    dialogue_gen_btn = gr.Button("🎬 生成完整对话", variant="primary", visible=False)
                    dialogue_out = gr.Audio(label="完整对话音频", type="filepath")

            # State to hold parsed lines between parse and generate
            parsed_state = gr.State([])

            all_char_outputs = char_names + voice_drops
            parse_btn.click(
                on_parse_script,
                inputs=[script_input],
                outputs=all_char_outputs + [parsed_state, parse_status],
            ).then(
                lambda: gr.update(visible=True),
                outputs=[dialogue_gen_btn],
            )

            dialogue_gen_btn.click(
                on_generate_dialogue,
                inputs=[parsed_state] + voice_drops,
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
