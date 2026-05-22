<p align="center">
  <img src="assets/Dramabox.png" alt="DramaBox Studio" width="720"/>
</p>

# DramaBox Studio — AI Voice Studio (Community Edition)

[![License](https://img.shields.io/badge/license-LTX--2_Community-blue)](LICENSE)
[![Base Model](https://img.shields.io/badge/base-LTX--2.3-orange)](https://github.com/Lightricks/LTX-2)
[![Upstream](https://img.shields.io/badge/upstream-ResembleAI%2FDramaBox-purple)](https://github.com/resemble-ai/DramaBox)

A community distribution of **Resemble AI's DramaBox** — expressive, prompt-driven TTS with voice cloning. The prompt controls speaker identity, emotion, delivery, laughs, sighs, pauses and transitions; a 10-second voice reference clones the target timbre.

**Built for Chinese users. Optimized for low-VRAM GPUs. Batteries included.**

---

## What's Different from Upstream

| Feature | Upstream | DramaBox Studio |
|---|---|---|
| VRAM peak | ~24 GB | **~8 GB** (8 GB cards supported) |
| VRAM config | Manual | **Auto-detect** (3 tiers) |
| UI | Default Gradio | **Midnight Gold** dark cinematic theme |
| Voice library | None | **Upload / name / list / delete** |
| Dialogue workshop | None | **Script → auto-parse characters → batch generate** |
| Prompt examples | 2 English | **8 Chinese-scene examples** with English direction |
| Prompt helpers | None | **One-click speaker tags + 18 non-verbal cues + quote wrapper** |
| Language | English prompts | English direction + Chinese dialogue supported |

---

## Quick Start

### Prerequisites

- Windows 10 / 11
- NVIDIA GPU with **8 GB+ VRAM**
- NVIDIA driver installed

### Option 1: Integration Package (Recommended)

Download the all-in-one package (embedded Python + models included):

> [Download link — coming soon]

Unzip, double-click `DramaBoxStudio.bat`, open `http://localhost:7860`.

### Option 2: Run from Source

```bash
# 1. Clone
git clone https://github.com/wangzhifeng/DramaBoxStudio.git
cd DramaBoxStudio

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch (models auto-download on first run)
python app.py
```

First launch downloads ~16 GB of model weights from HuggingFace. Subsequent launches load instantly.

---

## Prompt Writing

```
<speaker description in English>, "<dialogue>" <action direction> "<more dialogue>"
```

**Inside quotes** — spoken aloud:
- Dialogue: `"Hello, how are you?"`
- Vocal sounds (one word): `"Hahaha"` `"Hehehe"` `"Mmmmm"` `"Ugh"`

**Outside quotes** — performed, not spoken:
- `She sighs deeply.` · `A long pause.` · `He clears his throat.`
- `Her voice cracks.` · `He gulps nervously.` · `She drops to a whisper.`

**Never put inside quotes:** `Sigh` `Gasp` `Cough` `Ahem` `Pfft` — the model will speak them literally.

**Tips:**
- Match speaker description gender/age to the voice reference
- Break long dialogue with stage directions between segments
- Use the built-in cue dropdown (18 non-verbal actions) to insert at cursor
- Select Chinese text and click `「」→""` to wrap it in quotes

---

## Inference Settings

| Parameter | Default | Effect |
|---|---|---|
| `cfg_scale` | 2.5 | Lower = more natural, higher = more text-faithful |
| `stg_scale` | 1.5 | Skip-token guidance for expressiveness |
| `duration_multiplier` | 1.1 | 10% breathing room on auto-estimated length |
| `steps` | 30 | Euler flow matching |
| Speed | 0.5–2.0x | Tempo change without pitch shift |

---

## Features in Development

- [ ] Structured prompt builder (fill-in-the-blank UI)
- [ ] Streaming audio playback
- [ ] Batch export with subtitle (.srt) generation
- [ ] Chinese LoRA fine-tune for native-level Mandarin prosody

---

## License

This project is a derivative of [ResembleAI/DramaBox](https://github.com/resemble-ai/DramaBox), which is an IC-LoRA fine-tune of [Lightricks/LTX-2.3](https://github.com/Lightricks/LTX-2). Distributed under the **LTX-2 Community License Agreement** — see [LICENSE](LICENSE).

**Attribution:**
- Original DramaBox by [Resemble AI](https://www.resemble.ai/learn/models/dramabox)
- Base model LTX-2.3 by [Lightricks](https://github.com/Lightricks/LTX-2)
- Community modifications by **Wang Zhifeng** ([wangzhifeng.vip](https://wangzhifeng.vip/))

Model weights are auto-downloaded from [ResembleAI/Dramabox](https://huggingface.co/ResembleAI/Dramabox) on HuggingFace and are **not** included in this repository.

---

## Disclaimer

This is an unofficial community distribution. For the official DramaBox, visit [ResembleAI/DramaBox](https://github.com/resemble-ai/DramaBox).

DramaBox outputs are automatically watermarked with [Resemble Perth](https://github.com/resemble-ai/Perth) when the watermark feature is enabled. Please comply with the [LTX-2 Acceptable Use Policy](LICENSE) when using generated audio.
