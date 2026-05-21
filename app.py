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
/* ── 全局基调 ── */
body, .gradio-container {
    background: #0b0b14 !important;
    color: #e0e0e8 !important;
    font-family: "Microsoft YaHei", "PingFang SC", system-ui, sans-serif !important;
}
.gradio-container { max-width: 1200px !important; margin: 0 auto !important; }

/* ── Tab 栏 ── */
.tabs > .tab-nav > button {
    background: #111122 !important; color: #888 !important;
    border: 1px solid #1a1a30 !important; border-radius: 10px 10px 0 0 !important;
    font-size: 15px !important; padding: 12px 32px !important; margin-right: 4px !important;
    font-weight: 600 !important; transition: all .2s !important;
}
.tabs > .tab-nav > button.selected {
    background: #1a1a2e !important; color: #f59e0b !important;
    border-color: #f59e0b40 !important; border-bottom-color: transparent !important;
    box-shadow: 0 -2px 12px #f59e0b20 !important;
}

/* ── 卡片 ── */
.card, .gr-box, .gr-form, .panel {
    background: #12121f !important; border: 1px solid #1f1f35 !important;
    border-radius: 14px !important; box-shadow: 0 4px 24px #00000040 !important;
    padding: 20px !important; margin-bottom: 16px !important;
}

/* ── 输入框 ── */
textarea, input[type="text"], input[type="number"], .wrap .scroll-hide {
    background: #0d0d18 !important; color: #e0e0e8 !important;
    border: 1px solid #232340 !important; border-radius: 10px !important;
    font-size: 14px !important; padding: 12px !important;
}
textarea:focus, input:focus { border-color: #f59e0b60 !important; box-shadow: 0 0 12px #f59e0b18 !important; }

/* ── 按钮 ── */
button.primary, .gr-button-primary {
    background: linear-gradient(135deg, #f59e0b, #d97706) !important;
    color: #0b0b14 !important; font-weight: 700 !important;
    border: none !important; border-radius: 10px !important;
    font-size: 15px !important; padding: 12px 28px !important;
    box-shadow: 0 4px 16px #f59e0b30 !important; transition: all .2s !important;
}
button.primary:hover, .gr-button-primary:hover {
    transform: translateY(-1px); box-shadow: 0 6px 24px #f59e0b50 !important;
}
button.secondary, .gr-button-secondary {
    background: #1a1a2e !important; color: #ccc !important;
    border: 1px solid #2a2a40 !important; border-radius: 10px !important;
    font-size: 14px !important; padding: 10px 20px !important;
}

/* ── 标签 / 徽标 ── */
label, .label-text { color: #aaa !important; font-size: 13px !important; font-weight: 600 !important; }
.badge {
    display: inline-block; background: #f59e0b18; color: #f59e0b;
    border: 1px solid #f59e0b30; border-radius: 6px;
    padding: 4px 12px; font-size: 12px; font-weight: 700; margin-right: 6px;
}

/* ── Banner ── */
.studio-banner {
    background: linear-gradient(135deg, #1a1030 0%, #0f1a2e 50%, #101a20 100%);
    border: 1px solid #232340; border-left: 4px solid #f59e0b;
    border-radius: 14px; padding: 18px 24px; margin-bottom: 20px;
    color: #d0d0e0; font-size: 13px; line-height: 1.6;
    display: flex; align-items: center; gap: 16px;
}
.studio-banner .logo-text { font-size: 22px; font-weight: 800; color: #f59e0b; }
.studio-banner .sub { color: #888; font-size: 12px; }

/* ── Audio 组件 ── */
.audio-controls { background: #0d0d18 !important; border-radius: 10px !important; }

/* ── DataFrame ── */
.gr-dataframe { font-size: 13px !important; }
.gr-dataframe th { background: #1a1a2e !important; color: #f59e0b !important; }
.gr-dataframe td { background: #12121f !important; color: #ccc !important; }

/* ── 滑块 ── */
.gr-slider input[type="range"] { accent-color: #f59e0b; }

/* ── Accordion ── */
.gr-accordion { background: #12121f !important; border: 1px solid #1f1f35 !important; border-radius: 14px !important; }

/* ── 进度条 ── */
.gr-progress .progress-bar { background: #f59e0b !important; }

/* ── 底部 ── */
footer { visibility: hidden; }
"""

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
        secondary_hue="slate",
        neutral_hue="slate",
        font=gr.themes.GoogleFont("Inter"),
    ),
    css=CSS,
    analytics_enabled=False,
) as app:

    # ── Header ──
    gr.HTML(
        '<div class="studio-banner">'
        '<div><span class="logo-text">🎭 DramaBox Studio</span><br>'
        '<span class="sub">基于 LTX-2.3 · Resemble AI DramaBox · 纯本地 AI 配音</span></div>'
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
                cache_examples=False, run_on_click=True, examples_per_page=20,
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
                        label="已保存的音色",
                        value=[],
                        interactive=False,
                    )

            # 更新初始音色列表
            initial_voices = list_voices()
            if initial_voices:
                voice_table.value = [[v["name"], f'{v["duration_seconds"]:.1f}s', v["created_at"]]
                                     for v in initial_voices]

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

if __name__ == "__main__":
    port = int(os.environ.get("GRADIO_SERVER_PORT", str(GRADIO_PORT)))
    app.queue(max_size=10).launch(
        server_name="0.0.0.0", server_port=port,
        inbrowser=True,
    )
