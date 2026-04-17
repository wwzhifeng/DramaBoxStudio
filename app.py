#!/usr/bin/env python3
"""
LTX-2.3 TTS Voice Cloning — Gradio Demo (Warm Server)

All models kept in GPU memory. ~2.5s per generation instead of ~30s.

Flow:
  1. User writes plain text + optional style/emotion direction
  2. (Optional) User uploads voice reference → Gemini analyzes voice profile
  3. Click "Generate Prompt" → Gemini creates a scene prompt
  4. User reviews/edits the prompt
  5. Click "Generate Audio" → warm TTS server generates in ~2.5s
"""
import os
import sys
import json
import tempfile
import logging
import time
from pathlib import Path

import gradio as gr

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

os.environ["GRADIO_TEMP_DIR"] = "/tmp/gradio_tmp"
os.makedirs("/tmp/gradio_tmp", exist_ok=True)

# ── Gemini ──────────────────────────────────────────────────────────────────
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
gemini_client = None
if GEMINI_API_KEY:
    from google import genai
    from google.genai import types
    gemini_client = genai.Client(api_key=GEMINI_API_KEY)

# ── Warm TTS Server ────────────────────────────────────────────────────────
APP_DIR = Path(__file__).parent
sys.path.insert(0, str(APP_DIR / "src"))
sys.path.insert(0, str(APP_DIR / "ltx2"))

from inference_server import TTSServer

logging.info("Loading warm TTS server (all models to GPU, bf16, Gemma 4-bit)...")
tts = TTSServer(device="cuda", dtype="bf16", compile_model=False, bnb_4bit=True)
logging.info("TTS server ready!")

# ── Prompt engineering ──────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are a prompt engineer for LTX-2.3, an audio-only text-to-speech model.
The model generates speech from scene descriptions. Your job is to convert plain text
into an expressive scene prompt that controls how the model speaks.

## Prompt structure

A good prompt has this pattern:
  <speaker description>, "<dialogue>" <action direction> "<more dialogue>"

Example:
  A woman speaks with cold fury in a measured, low voice, "I have told you a thousand times." Her voice sharpens with rising anger, "Do you honestly think I enjoy repeating myself?!" She lets out a mocking laugh, "Hahaha, how utterly pathetic."

## Rules (CRITICAL — breaking these produces bad audio)

1. ALL spoken words MUST be inside double quotes: "like this"
2. Phonetic sounds that WORK inside quotes (model produces the actual sound):
   - Laughs: "Hahaha" "Hehehe" (NEVER separated like "Ha ha ha")
   - Sounds: "Mmmmm" "Ugh" "Argh" "Ahhh" "Hmm"
3. Non-vocal actions go OUTSIDE quotes as stage directions:
   - She sighs deeply. He gulps nervously. A long pause. She takes a shaky breath.
   - She scoffs. He clears his throat. She snorts dismissively.
4. Words that the model speaks LITERALLY (never put inside quotes):
   - "Ahem" — model says "ahem" instead of clearing throat. Use: She clears her throat.
   - "Pfft" — model says "pfft" literally. Use: She scoffs dismissively.
   - "Sigh" — model says "sigh". Use: She sighs.
   - "Gasp" — model says "gasp". Use: She gasps.
   - "Cough" — model says "cough". Use: He coughs.
5. The prompt MUST end with the last closing quote mark. Do NOT add anything after the final quote (no "Clean studio recording", no "Warm intimate recording", nothing).
6. Match the speaker description gender/age/accent to the voice reference if provided
7. Keep speech flowing naturally — connect emotional beats within utterances

## Speaker description patterns (use at the start)

- "A woman speaks warmly" / "A man speaks with gravelly authority"
- "A young girl speaks excitedly" / "An elderly man speaks slowly"
- Include tone: seductive, cheerful, menacing, trembling, casual, professional
- Include delivery: whispers, shouts, speaks rapidly, speaks with a thick accent

## Action directions (between quotes)

- Physical: He slams his fist. She leans in close. He takes a drag of a cigarette.
- Emotional: Her voice cracks. His tone shifts to warmth. She steadies herself.
- Vocal: He coughs. She clears her throat. He burps. She pants heavily.
- Pauses: A long silence. She pauses to collect herself. He trails off.

## What NOT to do

- Do NOT add environment/recording descriptions at the end
- Do NOT use single quotes for dialogue (use double quotes only)
- Do NOT separate laugh sounds: "Ha ha ha" → use "Hahaha"
- Do NOT put non-speech words in quotes: "Sigh" "Ahem" "Gasp" "Pfft" "Cough" — model says them literally
- Do NOT make the prompt too short — the model needs enough context for expressiveness
- Do NOT change the meaning of the user's text, only enhance the delivery"""

VOICE_PROFILE_PROMPT = """Listen to this audio clip carefully. Create a brief voice profile.

Return ONLY a JSON object:
```json
{
  "gender": "Male/Female",
  "age": "child/young/middle-aged/elderly",
  "accent": "American/British/etc",
  "tone": "deep/medium/high-pitched, gravelly/smooth/breathy",
  "style": "formal/casual/energetic/calm/dramatic",
  "summary": "A one-line description, e.g.: A warm middle-aged woman with a slight British accent and a calm, soothing delivery"
}
```"""


# ── Gemini functions ────────────────────────────────────────────────────────

def analyze_voice(audio_path: str) -> dict:
    if not gemini_client or not audio_path:
        return {}
    try:
        audio_data = Path(audio_path).read_bytes()
        ext = Path(audio_path).suffix.lower()
        mime = {".wav": "audio/wav", ".mp3": "audio/mp3", ".flac": "audio/flac"}.get(ext, "audio/wav")
        resp = gemini_client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[types.Content(role="user", parts=[
                types.Part.from_bytes(data=audio_data, mime_type=mime),
                types.Part.from_text(text=VOICE_PROFILE_PROMPT),
            ])],
            config=types.GenerateContentConfig(temperature=0.3),
        )
        text = resp.candidates[0].content.parts[0].text
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0]
        elif "```" in text:
            text = text.split("```")[1].split("```")[0]
        return json.loads(text.strip())
    except Exception as e:
        logging.error(f"Voice analysis failed: {e}")
        return {"error": str(e)}


def generate_prompt(text: str, style: str = "", voice_profile: dict = None) -> str:
    if not gemini_client:
        return f'A person speaks clearly, "{text}"'

    profile_ctx = ""
    if voice_profile and "summary" in voice_profile:
        profile_ctx = f"""
Voice reference profile:
- {voice_profile.get('summary', '')}
- Gender: {voice_profile.get('gender', 'Unknown')}
- Age: {voice_profile.get('age', 'Unknown')}
- Tone: {voice_profile.get('tone', 'Unknown')}
- Style: {voice_profile.get('style', 'Unknown')}
- Accent: {voice_profile.get('accent', 'Unknown')}

The speaker description MUST match this voice profile."""

    style_ctx = f"\nUser's style/emotion direction: {style.strip()}" if style.strip() else ""

    user_msg = f"""{SYSTEM_PROMPT}

{profile_ctx}
{style_ctx}

Convert this text into a TTS scene prompt. Preserve the exact words but add speaker description, acting directions, and emotional delivery.

Text:
{text}

Return ONLY the prompt. No explanation, no markdown formatting."""

    try:
        resp = gemini_client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[types.Content(role="user", parts=[types.Part.from_text(text=user_msg)])],
            config=types.GenerateContentConfig(temperature=0.7),
        )
        prompt = resp.candidates[0].content.parts[0].text.strip()
        if prompt.startswith("```"):
            lines = prompt.split("\n")
            prompt = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
        return prompt.strip()
    except Exception as e:
        logging.error(f"Prompt generation failed: {e}")
        return f'A person speaks clearly, "{text}"'


# ── Gradio handlers ─────────────────────────────────────────────────────────

def on_analyze_voice(audio_ref):
    if audio_ref is None:
        return ""
    profile = analyze_voice(audio_ref)
    if "error" in profile:
        return f"Error: {profile['error']}"
    return json.dumps(profile, indent=2)


def on_generate_prompt(text, style, voice_profile_json, audio_ref):
    if not text.strip():
        raise gr.Error("Enter some text first")
    voice_profile = None
    if voice_profile_json.strip():
        try:
            voice_profile = json.loads(voice_profile_json)
        except json.JSONDecodeError:
            pass
    return generate_prompt(text, style, voice_profile)


def on_generate_audio(prompt, audio_ref, cfg, stg, dur_mult, seed):
    if not prompt.strip():
        raise gr.Error("Generate or write a prompt first")

    t0 = time.time()
    ref_path = audio_ref if audio_ref and os.path.exists(str(audio_ref)) else None
    output = tempfile.mktemp(suffix=".wav", prefix="ltx_tts_")
    tts.generate_to_file(
        prompt=prompt,
        output=output,
        voice_ref=ref_path,
        cfg_scale=cfg, stg_scale=stg,
        duration_multiplier=dur_mult, seed=int(seed),
    )
    elapsed = time.time() - t0
    logging.info(f"Gradio request completed in {elapsed:.2f}s")
    return output


# ── UI ──────────────────────────────────────────────────────────────────────

with gr.Blocks(
    title="LTX-2.3 TTS Voice Cloning",
    theme=gr.themes.Default(),
    css=".prompt-box textarea { font-size: 14px !important; line-height: 1.5 !important; }"
) as app:

    gr.Markdown("# LTX-2.3 TTS — IC-LoRA Voice Cloning")
    gr.Markdown("Write what you want to say + how you want it said. Gemini creates a scene prompt. Edit it, then generate audio. **~2.5s per generation.**")

    with gr.Row():
        with gr.Column(scale=3):
            gr.Markdown("### Step 1: What to say")
            text_input = gr.Textbox(
                label="Text",
                placeholder="Hello everyone, welcome back to the show! I have some exciting news to share with you today.",
                lines=3,
            )
            style_input = gr.Textbox(
                label="Style / emotion direction (plain English)",
                placeholder="e.g. excited and energetic, like a podcast host who just got great news. Include some laughing.",
                lines=2,
            )

            gr.Markdown("### Step 2 (optional): Voice reference")
            audio_ref = gr.Audio(label="Upload voice reference (10+ seconds)", type="filepath")
            voice_profile = gr.Textbox(
                label="Voice profile (auto-filled when you upload audio)",
                lines=4, interactive=True,
                placeholder='Upload a voice clip above, or describe manually:\ne.g. {"gender": "Female", "age": "young", "tone": "bright, energetic"}',
            )

            gr.Markdown("### Step 3: Generate & edit prompt")
            gen_prompt_btn = gr.Button("Generate Prompt", variant="secondary", size="lg")
            prompt_editor = gr.Textbox(
                label="Scene prompt (edit this before generating audio)",
                lines=6, interactive=True, elem_classes=["prompt-box"],
                placeholder='Click "Generate Prompt" or write your own:\n\nA woman speaks excitedly, "Hello everyone!" She laughs, "Hahaha, this is amazing!"',
            )

            gr.Markdown("### Step 4: Generate audio")
            gen_audio_btn = gr.Button("Generate Audio", variant="primary", size="lg")

        with gr.Column(scale=2):
            with gr.Accordion("Inference Settings", open=False):
                cfg_slider = gr.Slider(1.0, 10.0, value=2.5, step=0.5, label="CFG Scale")
                stg_slider = gr.Slider(0.0, 5.0, value=1.5, step=0.5, label="STG Scale")
                dur_slider = gr.Slider(0.8, 2.0, value=1.1, step=0.05, label="Duration multiplier")
                seed_input = gr.Number(value=42, label="Seed", precision=0)

            audio_output = gr.Audio(label="Generated Audio", type="filepath")

            with gr.Accordion("Prompt writing guide", open=False):
                gr.Markdown("""
**Structure:** `<speaker desc>, "<speech>" <action> "<more speech>"`

**Dialogue** in double quotes: `"Hello, how are you?"`

**Sounds** inside quotes as one word: `"Hahaha"` `"Hehehe"` `"Mmmmm"` `"Ugh"` `"Argh"`

**Actions** outside quotes: `She sighs.` `He gulps.` `She scoffs.` `He clears his throat.`

**Never inside quotes:** Ahem, Pfft, Sigh, Gasp, Cough (model speaks them literally)

**End prompt** at last closing quote. No trailing descriptions.
                """)

    # ── Events ──────────────────────────────────────────────────
    audio_ref.change(fn=on_analyze_voice, inputs=[audio_ref], outputs=[voice_profile])
    gen_prompt_btn.click(
        fn=on_generate_prompt,
        inputs=[text_input, style_input, voice_profile, audio_ref],
        outputs=[prompt_editor],
    )
    gen_audio_btn.click(
        fn=on_generate_audio,
        inputs=[prompt_editor, audio_ref, cfg_slider, stg_slider, dur_slider, seed_input],
        outputs=[audio_output],
    )


if __name__ == "__main__":
    port = int(os.environ.get("GRADIO_SERVER_PORT", "7861"))
    app.queue(max_size=10)  # Queue up to 10 concurrent users
    app.launch(server_name="0.0.0.0", server_port=port, share=True)
