#!/usr/bin/env python3
"""DramaBox — Gradio demo (warm server).

Loads the warm TTSServer once, then handles requests at ~2.5 s each. All
generated audio is invisibly watermarked with Resemble Perth before being
returned to the user.
"""
import logging
import os
import sys
import tempfile
import time

import gradio as gr
import spaces

# Local src import.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
from inference_server import TTSServer  # noqa: E402
from model_downloader import get_all_paths  # noqa: E402


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logging.info("Fetching DramaBox checkpoints from HuggingFace (cached after first run)...")
PATHS = get_all_paths()  # CPU-side download is fine outside the GPU window

# Lazy-loaded inside the @spaces.GPU function (no GPU available at import time on ZeroGPU).
_TTS: TTSServer | None = None


def _ensure_tts() -> TTSServer:
    global _TTS
    if _TTS is None:
        logging.info("Loading DramaBox warm server (Gemma + DiT + VAE + Decoder)...")
        _TTS = TTSServer(
            checkpoint=PATHS["transformer"],
            full_checkpoint=PATHS["audio_components"],
            gemma_root=PATHS["gemma_root"],
            device="cuda",
            dtype=os.environ.get("LTX_DTYPE", "bf16"),
            compile_model=False,                  # torch.compile breaks under ZeroGPU's brief GPU windows
            bnb_4bit=True,                        # unsloth Gemma is pre-quantized
        )
        logging.info("TTSServer ready.")
    return _TTS


# ── Example prompts (shown as click-to-fill chips in the UI) ─────────────────
EXAMPLES: list[tuple[str, str]] = [
    (
        "Villain monologue",
        'A shadowy villain speaks with cold menace, "You have entered my domain, mortal." '
        'He chuckles darkly, "Such arrogance will be your undoing." '
        'His voice rises with fury, "Kneel, or be destroyed where you stand!"'
    ),
    (
        "Talk-show host wheeze-laugh",
        'A talk show host gasps with shock, "No! You did NOT just say that!" '
        'He bursts into uncontrollable laughter, "Hahaha! Oh my god, oh my god!" '
        'He wheezes, "I cannot, I literally cannot breathe right now!"'
    ),
    (
        "Tender goodnight whisper",
        'A woman speaks tenderly, "It has been a long day, my love." '
        'She whispers, "Close your eyes. I am right here." '
        'She hums quietly, "Mmmm-mmm. Sleep now."'
    ),
    (
        "Old-school radio anchor",
        'A radio host clears his throat, "Excuse me, pardon that." '
        'He settles into a warm, professional tone, "Good evening everyone, '
        'and welcome back to the show. We have got a wonderful lineup tonight."'
    ),
    (
        "Catgirl uncontrollable giggling",
        'A playful girl already mid-giggle, "Hehehe, oh my gosh you should see your face!" '
        'She gasps for air between giggles, "Oh my, hehe, oh my, I cannot stop!" '
        'She tries to compose herself, "Ahhhhh okay okay okay, I will stop, I promise."'
    ),
    (
        "Hero stammering courage",
        'A young warrior speaks with a trembling voice, "I... I do not know if I can do this." '
        'He takes a shaky breath, "But someone has to try." '
        'His voice steadies with growing fire, "No more running. I WILL fight!"'
    ),
    (
        "Exhausted dad, fraying patience",
        'An exhausted father speaks with fraying patience, "Sweetie, daddy is asking very nicely." '
        'He sighs deeply, "Ohhhh my goodness." '
        'He puts on an overly cheerful voice, "Hey buddy! Look at the shiny thing!" '
        'Then he laughs helplessly, "Hahaha, I am losing my mind."'
    ),
    (
        "Smug-confident announcer",
        'A confident announcer speaks proudly, "And now, the moment you have all been waiting for." '
        'He chuckles knowingly, "Heheh, trust me, this one is going to blow you away."'
    ),
]


@spaces.GPU(duration=120)
def on_generate(prompt: str, audio_ref, cfg: float, stg: float, dur_mult: float, seed: int):
    if not prompt or not prompt.strip():
        raise gr.Error("Prompt is empty.")
    t0 = time.time()
    tts = _ensure_tts()
    ref_path = audio_ref if audio_ref and os.path.exists(str(audio_ref)) else None
    output = tempfile.mktemp(suffix=".wav", prefix="dramabox_")
    tts.generate_to_file(
        prompt=prompt,
        output=output,
        voice_ref=ref_path,
        cfg_scale=cfg, stg_scale=stg,
        duration_multiplier=dur_mult, seed=int(seed),
    )
    elapsed = time.time() - t0
    logging.info(f"Generated in {elapsed:.2f}s -> {output}")
    return output


# ── UI ──────────────────────────────────────────────────────────────────────
with gr.Blocks(
    title="DramaBox — Expressive TTS",
    theme=gr.themes.Default(),
    css=".prompt-box textarea { font-size: 14px !important; line-height: 1.5 !important; }",
    analytics_enabled=False,
) as app:
    gr.Markdown("# 🎭 DramaBox — Expressive TTS with Voice Cloning")
    gr.Markdown(
        "Write a scene prompt, optionally upload a 10-second voice reference, "
        "and generate. Audio is automatically watermarked with "
        "[Resemble Perth](https://github.com/resemble-ai/Perth).\n\n"
        "**Tips:** put dialogue inside `\"double quotes\"`, scene directions outside. "
        "Phonetic sounds (`\"Hahaha\"`, `\"Mmmm\"`, `\"Ugh\"`) go inside quotes; named "
        "actions (`She sighs.`, `He clears his throat.`) go outside."
    )

    with gr.Row():
        with gr.Column(scale=3):
            prompt_box = gr.Textbox(
                label="Scene prompt",
                placeholder=EXAMPLES[0][1],
                lines=6, elem_classes=["prompt-box"],
            )
            example_chooser = gr.Dropdown(
                choices=[e[0] for e in EXAMPLES],
                label="Load an example prompt", interactive=True, value=None,
            )
            audio_ref = gr.Audio(
                label="Voice reference (optional, 10+ seconds)",
                type="filepath",
            )
            gen_btn = gr.Button("Generate", variant="primary", size="lg")

        with gr.Column(scale=2):
            with gr.Accordion("Inference settings", open=False):
                cfg_slider = gr.Slider(1.0, 10.0, value=2.5, step=0.5, label="CFG scale")
                stg_slider = gr.Slider(0.0, 5.0, value=1.5, step=0.5, label="STG scale")
                dur_slider = gr.Slider(0.8, 2.0, value=1.1, step=0.05, label="Duration ×")
                seed_input = gr.Number(value=42, label="Seed", precision=0)
            audio_out = gr.Audio(label="Generated audio", type="filepath")
            with gr.Accordion("Prompt writing guide", open=False):
                gr.Markdown(
                    "**Structure:** `<speaker description>, \"<dialogue>\" <action> \"<more dialogue>\"`\n\n"
                    "**Inside quotes** (model speaks them):\n"
                    "- Dialogue: `\"Hello, how are you?\"`\n"
                    "- Phonetic sounds: `\"Hahaha\"`, `\"Hehehe\"`, `\"Mmmmm\"`, `\"Ugh\"`, `\"Argh\"`\n\n"
                    "**Outside quotes** (stage directions):\n"
                    "- `She sighs deeply.`, `He gulps nervously.`, `A long pause.`\n"
                    "- `Her voice cracks.`, `He clears his throat.`\n\n"
                    "**Avoid inside quotes:** Ahem, Pfft, Sigh, Gasp, Cough — the model speaks them literally."
                )

    def _load_example(choice: str):
        if not choice:
            return gr.update()
        for name, prompt in EXAMPLES:
            if name == choice:
                return prompt
        return gr.update()

    example_chooser.change(_load_example, inputs=[example_chooser], outputs=[prompt_box])
    gen_btn.click(
        on_generate,
        inputs=[prompt_box, audio_ref, cfg_slider, stg_slider, dur_slider, seed_input],
        outputs=[audio_out],
    )


if __name__ == "__main__":
    # HF Spaces routes external traffic to container port 7860 by default.
    # Defaulting to 7861 caused the gateway to return 500 for every external request.
    port = int(os.environ.get("GRADIO_SERVER_PORT", "7860"))
    app.queue(max_size=10).launch(
        server_name="0.0.0.0", server_port=port,
        share=os.environ.get("GRADIO_SHARE", "0") == "1",
        ssr_mode=False,                       # Gradio 5 SSR + ZeroGPU fork has known race conditions
        show_api=False,                       # don't auto-derive Python schemas (caused bool-iter / dict-cache crashes)
    )
