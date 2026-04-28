# Dramabox - Expressive TTS with Voice Cloning

Prompt-driven TTS with voice cloning built on a 3.3B Diffusion Transformer with flow matching.

## Folder Structure

```
DramaBox/
├── src/
│   ├── inference.py          # TTS inference with voice cloning
│   ├── inference_server.py   # Warm server (~2.5s per generation)
│   ├── audio_conditioning.py # Reference audio conditioning
│   └── model_downloader.py   # Auto-download models from HuggingFace
├── patches/
│   ├── attention.py          # dtype fix for mask allocation
│   └── guiders.py            # Per-token CFG clamping
├── assets/
│   └── silence_latent_frame.pt
├── evals/
│   ├── eval_short.txt        # 30 short prompts (~5-15s)
│   ├── eval_long.txt         # 15 long prompts (~20-37s)
│   └── eval_expressive.txt   # 15 expressive prompts (laughs, sighs, stammers)
├── scripts/
│   ├── inference.sh          # Inference wrapper
│   └── eval.sh               # Evaluation runner
├── app.py                    # Gradio demo app
├── ltx2/                     # LTX-2 dependency packages
└── README.md
```

## Models

Models auto-download from [ResembleAI/Dramabox](https://huggingface.co/ResembleAI/Dramabox) on HuggingFace.

| Model | Size | Description |
|-------|------|-------------|
| `dramabox-dit-v1.safetensors` | 6.6 GB | DiT transformer |
| `dramabox-audio-components.safetensors` | 2.7 GB | Audio VAE + vocoder + text projection |
| [unsloth/gemma-3-12b-it-bnb-4bit](https://huggingface.co/unsloth/gemma-3-12b-it-bnb-4bit) | ~8 GB | Text encoder (auto-downloaded) |

**VRAM**: ~24 GB peak | **Speed**: ~2.5s per generation (warm server, H100)

## Quick Start

### Warm Server (recommended, ~2.5s per request)

```python
from src.inference_server import TTSServer

server = TTSServer(device="cuda")

server.generate_to_file(
    prompt='A woman speaks warmly, "Hello, how are you today?" She laughs, "Hahaha, it is so good to see you!"',
    output="output.wav",
    voice_ref="reference.wav",  # optional, 10+ seconds
)
```

### Gradio App

```bash
GEMINI_API_KEY=your_key CUDA_VISIBLE_DEVICES=4 python app.py
```

### CLI Inference

```bash
python src/inference.py \
  --voice-sample reference.wav \
  --prompt 'A woman speaks warmly, "Hello, how are you today?"' \
  --output output.wav \
  --cfg-scale 2.5 --stg-scale 1.5
```

### Evaluation

```bash
bash scripts/eval.sh --eval expressive --output eval_results/
```

## Inference Settings

| Parameter | Default | Notes |
|-----------|---------|-------|
| cfg-scale | 2.5 | Lower = more natural, higher = more text following |
| stg-scale | 1.5 | Skip-token guidance |
| rescale | 0 | No rescaling |
| modality | 1 | No modality guidance |
| duration-multiplier | 1.1 | 10% breathing room |
| steps | 30 | Euler flow matching |

## Prompt Writing Guide

**Structure:** `<speaker description>, "<dialogue>" <action direction> "<more dialogue>"`

### What works inside quotes (model produces actual sounds)
- Laughs: `"Hahaha"` `"Hehehe"` (always one word, never separated)
- Sounds: `"Mmmmm"` `"Ugh"` `"Argh"` `"Ahhh"` `"Hmm"`

### What goes outside quotes (stage directions)
- `She sighs deeply.` `He gulps nervously.` `A long pause.`
- `Her voice cracks.` `He clears his throat.` `She scoffs.`

### Never inside quotes (model speaks them literally)
- Ahem, Pfft, Sigh, Gasp, Cough

### Tips
- Match gender/age in speaker description to voice reference
- Break long dialogue into segments with acting directions between them
- End prompt at the last closing quote mark (no trailing descriptions)

## Watermarking

Every audio output from `inference.py` and `inference_server.TTSServer.generate_to_file` is automatically watermarked with [Resemble Perth](https://github.com/resemble-ai/Perth) — an imperceptible neural watermark that survives MP3 compression, audio editing, and common manipulations while maintaining nearly 100% detection accuracy.

```python
import perth, librosa
wav, sr = librosa.load("output.wav", sr=None, mono=True)
detector = perth.PerthImplicitWatermarker()
print(detector.get_watermark(wav, sample_rate=sr))   # confidence ≈ 1.0 for our outputs
```

Pass `--no-watermark` to `inference.py` (or `watermark=False` to `generate_to_file`) to disable for debugging.

## Training

DramaBox is an IC-LoRA fine-tune of the LTX-2.3 22B audio-only branch. To train your own:

```bash
# 1. Preprocess raw (audio, transcript) pairs → audio_latents/ + conditions/
python src/preprocess.py \
  --dataset-type manifest \
  --index your_data.jsonl \
  --output-dir /path/to/preprocessed/ \
  --checkpoint dramabox-audio-components.safetensors \
  --gemma-root /path/to/gemma-3-12b-it-bnb-4bit/

# 2. Edit configs/training_args.example.yaml → your data paths

# 3. Launch (uses HuggingFace accelerate)
bash scripts/train.sh \
  --config configs/training_args.example.yaml \
  --gpus 0,1,2,3,4,5,6 \
  --train-val-gpu 7
```

| Script | Purpose |
|---|---|
| `src/preprocess.py` | Encode audio (Audio VAE) + text (Gemma) into training-ready `.pt` files |
| `src/train.py` | IC-LoRA training loop with peft, accelerate multi-GPU, periodic validation |
| `src/validate.py` | Spawned by `train.py` at each save step; runs the warm validator on a held-out prompt set |
| `scripts/train.sh` | YAML-config wrapper around `accelerate launch src/train.py` |

LoRA targets the audio branch only: `audio_attn1.{to_q,to_k,to_v,to_out.0}` + `audio_ff.{net.0.proj,net.2}` × 48 transformer blocks (288 LoRA pairs total). Default rank 128 / alpha 128 / dropout 0.1, cosine LR schedule from 1e-4 with 500-step warmup over 10k steps.

## Language

English.

## License

Built on [LTX-2](https://github.com/Lightricks/LTX-2) by Lightricks. Distributed under the LTX-2 Community License Agreement — see [`LICENSE`](LICENSE).
