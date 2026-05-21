"""
DramaBox — 表情 TTS 语音克隆（中文界面）
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
from config import GRADIO_PORT, DIT_CHECKPOINT, AUDIO_COMPONENTS, GEMMA_DIR, OUTPUT_DIR, VOICES_DIR

import gradio as gr

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── 加载模型 ──────────────────────────────────────────────────────────────────
log.info("加载 DramaBox 模型（首次启动需等待）...")
tts = TTSServer(
    checkpoint=str(DIT_CHECKPOINT),
    full_checkpoint=str(AUDIO_COMPONENTS),
    gemma_root=str(GEMMA_DIR),
    device="cuda",
    dtype="bf16",
)
log.info("TTSServer 就绪。")

# ── 预设示例 ──────────────────────────────────────────────────────────────────
_VOICES_DIR = APP_DIR / "assets" / "voices"

EXAMPLES: list[tuple[str, str, str]] = [
    (
        "反派独白",
        str(_VOICES_DIR / "male_harvey_keitel.mp3"),
        'A shadowy villain speaks with cold menace, "You have entered my domain, mortal." '
        'He chuckles darkly, "Such arrogance will be your undoing." '
        'His voice rises with fury, "Kneel, or be destroyed where you stand!"',
    ),
    (
        "脱口秀主持人大笑",
        str(_VOICES_DIR / "male_conan.mp3"),
        'A talk show host gasps with shock, "No! You did NOT just say that!" '
        'He bursts into uncontrollable laughter, "Hahaha! Oh my god, oh my god!" '
        'He wheezes, "I cannot, I literally cannot breathe right now!"',
    ),
    (
        "温柔晚安低语",
        str(_VOICES_DIR / "female_shadowheart.wav"),
        'A woman speaks tenderly, "It has been a long day, my love." '
        'She whispers, "Close your eyes. I am right here." '
        'She hums quietly, "Mmmm-mmm. Sleep now."',
    ),
    (
        "老式电台主播",
        str(_VOICES_DIR / "male_old_movie.wav"),
        'A radio host clears his throat, "Excuse me, pardon that." '
        'He settles into a warm, professional tone, "Good evening everyone, '
        'and welcome back to the show. We have got a wonderful lineup tonight."',
    ),
    (
        "猫娘咯咯笑",
        str(_VOICES_DIR / "female_american.wav"),
        'A playful girl already mid-giggle, "Hehehe, oh my gosh you should see your face!" '
        'She gasps for air between giggles, "Oh my, hehe, oh my, I cannot stop!" '
        'She tries to compose herself, "Ahhhhh okay okay okay, I will stop, I promise."',
    ),
    (
        "少年英雄颤抖的勇气",
        str(_VOICES_DIR / "male_arnie.mp3"),
        'A young warrior speaks with a trembling voice, "I... I do not know if I can do this." '
        'He takes a shaky breath, "But someone has to try." '
        'His voice steadies with growing fire, "No more running. I WILL fight!"',
    ),
    (
        "疲惫父亲失去耐心",
        str(_VOICES_DIR / "male_petergriffin.wav"),
        'An exhausted father speaks with fraying patience, "Sweetie, daddy is asking very nicely." '
        'He sighs deeply, "Ohhhh my goodness." '
        'He puts on an overly cheerful voice, "Hey buddy! Look at the shiny thing!" '
        'Then he laughs helplessly, "Hahaha, I am losing my mind."',
    ),
    (
        "自信满满播报员",
        str(_VOICES_DIR / "male_samuel_j.mp3"),
        'A confident announcer speaks proudly, "And now, the moment you have all been waiting for." '
        'He chuckles knowingly, "Heheh, trust me, this one is going to blow you away."',
    ),
]


def on_generate(prompt: str, audio_ref, cfg: float, stg: float, dur_mult: float,
                gen_dur: float, ref_dur: float, seed: int):
    if not prompt or not prompt.strip():
        raise gr.Error("请输入提示词。")
    t0 = time.time()
    ref_path = audio_ref if audio_ref and os.path.exists(str(audio_ref)) else None
    output = str(OUTPUT_DIR / f"output_{int(time.time())}.wav")
    tts.generate_to_file(
        prompt=prompt,
        output=output,
        voice_ref=ref_path,
        cfg_scale=cfg, stg_scale=stg,
        duration_multiplier=dur_mult, seed=int(seed),
        gen_duration=float(gen_dur),
        ref_duration=float(ref_dur),
    )
    elapsed = time.time() - t0
    log.info(f"生成完成: {elapsed:.1f}s → {output}")
    return output


# ── UI ────────────────────────────────────────────────────────────────────────
_BANNER_CSS = """
.prompt-box textarea { font-size: 14px !important; line-height: 1.5 !important; }
.ltx-banner {
    background: linear-gradient(90deg, #1a1f3a 0%, #2a1f3a 100%);
    border-left: 4px solid #ff6b35;
    padding: 10px 16px;
    margin: 0 0 12px 0;
    border-radius: 6px;
    color: #e8e8f0;
    font-size: 13px;
    line-height: 1.5;
}
.ltx-banner a { color: #ff9a6c; font-weight: 600; text-decoration: none; }
.ltx-banner a:hover { text-decoration: underline; }
.ltx-banner strong { color: #ffffff; }
"""

with gr.Blocks(
    title="DramaBox — 表情 TTS 语音克隆",
    theme=gr.themes.Default(),
    css=_BANNER_CSS,
    analytics_enabled=False,
) as app:
    gr.Markdown("# DramaBox — 表情 TTS 语音克隆")
    gr.HTML(
        '<div class="ltx-banner">'
        ' 基于 <a href="https://github.com/Lightricks/LTX-2">LTX-2.3</a> 音频分支，'
        '<strong>Resemble AI</strong> 的 DramaBox IC-LoRA 微调模型。'
        ' 纯本地运行，无需联网。'
        '</div>'
    )
    gr.Markdown(
        "输入场景提示词，可选上传参考音频（10秒以上），生成带表情的语音。\n\n"
        "**格式：** 对话放 `\"双引号\"` 内，动作描述放引号外。"
        "拟声词（`\"Hahaha\"`、`\"Mmmm\"`、`\"Ugh\"`）放引号内；"
        "命名动作（`她叹了口气。`、`他清了清嗓子。`）放引号外。"
    )

    with gr.Row():
        with gr.Column(scale=3):
            prompt_box = gr.Textbox(
                label="场景提示词",
                placeholder=EXAMPLES[0][2],
                lines=6, elem_classes=["prompt-box"],
            )
            audio_ref = gr.Audio(
                label="参考音频（可选，建议 10 秒以上）",
                type="filepath",
            )
            gen_btn = gr.Button("生成", variant="primary", size="lg")

        with gr.Column(scale=2):
            with gr.Accordion("推理设置", open=False):
                cfg_slider = gr.Slider(1.0, 10.0, value=2.5, step=0.5, label="CFG 引导强度")
                stg_slider = gr.Slider(0.0, 5.0, value=1.5, step=0.5, label="STG 跳过令牌引导")
                dur_slider = gr.Slider(0.8, 2.0, value=1.1, step=0.05,
                                       label="时长倍率（目标时长 = 0 时生效）")
                gen_dur_slider = gr.Slider(0.0, 60.0, value=0.0, step=1.0,
                                           label="目标时长（秒）— 0 = 自动估算")
                ref_dur_slider = gr.Slider(3.0, 30.0, value=10.0, step=1.0,
                                           label="参考音频时长（秒）")
                seed_input = gr.Number(value=42, label="随机种子", precision=0)
            audio_out = gr.Audio(label="生成音频", type="filepath")
            with gr.Accordion("写作指南", open=False):
                gr.Markdown(
                    "**结构：** `<角色描述>，\"<对话>\" <动作> \"<更多对话>\"`\n\n"
                    "**引号内**（模型会说出来）：\n"
                    "- 对话：`\"你好，最近怎么样？\"`\n"
                    "- 拟声词：`\"Hahaha\"`、`\"Hehehe\"`、`\"Mmmmm\"`、`\"Ugh\"`\n\n"
                    "**引号外**（舞台指令）：\n"
                    "- `她深深叹了口气。`、`他紧张地咽了口唾沫。`、`一阵长久的沉默。`\n"
                    "- `她的声音沙哑了。`、`他清了清嗓子。`\n\n"
                    "**避免放引号内：** 清嗓（Ahem）、噗（Pfft）、叹气（Sigh）、"
                    "倒抽气（Gasp）、咳嗽（Cough）—— 模型会按字面读出来。"
                )

    gen_btn.click(
        on_generate,
        inputs=[prompt_box, audio_ref, cfg_slider, stg_slider,
                dur_slider, gen_dur_slider, ref_dur_slider, seed_input],
        outputs=[audio_out],
    )

    gr.Examples(
        label="点击任意一行即可生成示例",
        examples=[
            [name, prompt, voice_path, 2.5, 1.5, 1.1,
             30.0 if name.startswith("30s") else 0.0, 10.0, 42]
            for name, voice_path, prompt in EXAMPLES
        ],
        example_labels=[name for name, _, _ in EXAMPLES],
        inputs=[gr.Textbox(visible=False, label="场景"),
                prompt_box, audio_ref,
                cfg_slider, stg_slider, dur_slider, gen_dur_slider,
                ref_dur_slider, seed_input],
        outputs=[audio_out],
        fn=lambda _name, prompt, ref, cfg, stg, dur, gen_dur, ref_dur, seed: on_generate(
            prompt, ref, cfg, stg, dur, gen_dur, ref_dur, seed),
        cache_examples=False,
        run_on_click=True,
        examples_per_page=20,
    )

if __name__ == "__main__":
    port = int(os.environ.get("GRADIO_SERVER_PORT", str(GRADIO_PORT)))
    app.queue(max_size=10).launch(
        server_name="0.0.0.0", server_port=port,
        ssr_mode=False,
        show_api=False,
        inbrowser=True,
    )
