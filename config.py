"""
DramaBox 配置 — 所有路径相对项目根，离线可用。
启动时自动检测 GPU 显存，按档位选择量化/加载策略。

Modified by Siddid Soni, 2026-10: CUDA / Apple Silicon tiers and the DRAMABOX_* settings.
"""
import logging
import os
import warnings
from pathlib import Path

# ── 屏蔽第三方库噪音 ──────────────────────────────
os.environ.setdefault("TORCH_DISTRIBUTED_DEBUG", "OFF")
os.environ.setdefault("TQDM_DISABLE", "1")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")   # MPS 缺的算子回落 CPU
warnings.filterwarnings("ignore")
for _name in ("transformers", "diffusers", "torch.distributed", "torch._dynamo",
              "torch.distributed.elastic.multiprocessing", "ltx_pipelines", "ltx_core"):
    logging.getLogger(_name).setLevel(logging.ERROR)

# 一刀切：屏蔽所有含 "Uninitialized" 的垃圾日志
_UninitializedFilter = logging.Filter("UninitializedFilter")
_UninitializedFilter.filter = lambda r: "Uninitialized" not in r.getMessage()
logging.getLogger().addFilter(_UninitializedFilter)

ROOT = Path(__file__).resolve().parent

# ── 路径 ──────────────────────────────────────────
MODELS_DIR     = ROOT / "models"
CHECKPOINT_DIR = MODELS_DIR / "checkpoints"
GEMMA_DIR      = MODELS_DIR / "gemma"
VOICES_DIR     = MODELS_DIR / "voices"
VOICES_JSON    = MODELS_DIR / "voices.json"
OUTPUT_DIR     = ROOT / "output"
TMP_DIR        = ROOT / "tmp"

# ── 模型文件 ──────────────────────────────────────
DIT_CHECKPOINT   = CHECKPOINT_DIR / "dramabox-dit-v1.safetensors"
AUDIO_COMPONENTS = CHECKPOINT_DIR / "dramabox-audio-components.safetensors"
FULL_CHECKPOINT  = AUDIO_COMPONENTS

# ── 服务端口 ──────────────────────────────────────
GRADIO_PORT = 7860
API_PORT    = 9880

# ── 推理参数 ──────────────────────────────────────
DTYPE       = "bf16"
CFG_SCALE   = 2.5
STG_SCALE   = 1.5
STEPS       = 30
SEED        = 42

# ── Prompt embedding cache (keyed by Gemma variant + prompt) ──
EMBED_CACHE_DIR = ROOT / "cache" / "prompt-embeds"

# Gemma runs on the real tokens only and the hidden states are zero-padded to 1024 afterwards (same result, the
# padding is masked out downstream). 0 = Konvey/upstream behaviour: Gemma runs all 1024 padded tokens. MLX always trims.
TRIM_PADDING = os.environ.get("DRAMABOX_TRIM_PADDING", "1") != "0"

# Low-VRAM tiers: where a stage waits while another uses the GPU.
#   both   every stage kept in host RAM (Konvey) — fastest swaps (~0.5 s), most RAM (~17 GB on 8 GB cards)
#   ssd    one big model in RAM at a time: each stage's weights are written once to cache/park/ (~14 GB SSD); a stage
#          leaving the GPU is just freed, and the next one is read SSD -> RAM in the background while the GPU works.
#          At startup Gemma is built on the GPU while the audio stage is built on the CPU into RAM.
PARK_MODES = ("both", "ssd")

# What happens to Gemma's output before the embeddings processor (all tiers). On low it waits in host RAM between
# stages; on balanced / performance / mlx Gemma and the processor share the GPU, so nothing is stored, real behaves
# like full, and the project modes lower the GPU peak while encoding (the stock feature extractor briefly holds ~5
# full-size 368 MB copies; project-exact holds ~1, project only the real tokens). Peaks below are the low tier's.
#   full           what Gemma returns (Konvey): 0.37 GB per prompt with trim off, real tokens only with trim on
#   real           only the real tokens (~0.37 MB per token), zero-padded again on the GPU — identical result.
#                  Default: lowest Gemma-stage GPU peak (5.75 GB trim on / 6.31 GB trim off), safe on desktop cards
#                  that also drive monitors
#   project        the feature extractor's projection runs on the GPU next to Gemma on the real tokens, so only
#                  ~4 KB per token is stored (6.50 GB peak); ~0.5-1.4% rounding differences vs Konvey
#   project-exact  same, but on the padded 1024 rows like Konvey: bit-identical with trim off. 7.0 GB peak (7.47 GB
#                  reserved with trim off) — fits a headless 8 GB card; a prompt that runs out of memory falls back
#                  to project
GEMMA_OUTPUT_MODES = ("full", "real", "project", "project-exact")
GEMMA_OUTPUT = os.environ.get("DRAMABOX_GEMMA_OUTPUT", "real")
if GEMMA_OUTPUT not in GEMMA_OUTPUT_MODES:
    raise ValueError(f"DRAMABOX_GEMMA_OUTPUT must be one of {GEMMA_OUTPUT_MODES}, got {GEMMA_OUTPUT!r}")
PARK_DIR = ROOT / "cache" / "park"
PARK = os.environ.get("DRAMABOX_PARK", "both")
if PARK not in PARK_MODES:
    raise ValueError(f"DRAMABOX_PARK must be one of {PARK_MODES}, got {PARK!r}")

# ── Apple Silicon: Gemma 3 12B 4-bit for MLX (mlx-community/gemma-3-12b-it-4bit) ──
MLX_GEMMA_DIR = Path(os.environ.get("DRAMABOX_MLX_GEMMA", MODELS_DIR / "gemma-mlx"))

# ══════════════════════════════════════════════════
# 硬件自检 + 自动档位 (override: DRAMABOX_TIER=low|balanced|performance|mlx-16|mlx-24)
#
#   low          CUDA <11 GB   Gemma (embed table on CPU) / embeddings processor / DiT take turns on the GPU
#   balanced     CUDA 11-20 GB text stage and audio stage take turns, idle one parked in pinned host RAM
#   performance  CUDA >=20 GB  everything resident on the GPU
#   mlx-16       Mac <24 GB    Gemma in MLX, loaded only to encode new prompts; DiT on MPS
#   mlx-24       Mac >=24 GB   Gemma in MLX resident; DiT on MPS
# ══════════════════════════════════════════════════

TIERS = ("low", "balanced", "performance", "mlx-16", "mlx-24")


def _detect_backend() -> tuple[str, float]:
    """(backend, memory GB): CUDA VRAM, or Apple Silicon unified memory."""
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda", torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        if torch.backends.mps.is_available():
            import subprocess
            mem = int(subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True).stdout)
            return "mps", mem / (1024 ** 3)
    except Exception:
        pass
    return "cpu", 0.0


BACKEND, _mem = _detect_backend()
DEVICE = {"cuda": "cuda", "mps": "mps"}.get(BACKEND, "cpu")

if BACKEND == "mps":
    TIER = "mlx-24" if _mem >= 24 else "mlx-16"
elif _mem >= 20:
    TIER = "performance"
elif _mem >= 11:
    TIER = "balanced"
else:
    TIER = "low"
TIER = os.environ.get("DRAMABOX_TIER", TIER)
if TIER not in TIERS:
    raise ValueError(f"DRAMABOX_TIER must be one of {TIERS}, got {TIER!r}")

_tier_msg = {
    "low":         "低显存档（Gemma / 嵌入处理器 / DiT 轮流上 GPU）",
    "balanced":    "均衡档（文本与音频两阶段轮流上 GPU，闲置阶段驻留内存）",
    "performance": "性能档（全 BF16，模型常驻）",
    "mlx-16":      "Mac 16G 档（MLX Gemma 按需加载，DiT 走 MPS）",
    "mlx-24":      "Mac 24G+ 档（MLX Gemma 常驻，DiT 走 MPS）",
}[TIER]
_tier_msg = f"{BACKEND.upper()} {_mem:.1f}G → {_tier_msg}"
if BACKEND == "cpu":
    _tier_msg = "未检测到 CUDA / MPS 设备 → 无法运行"

LOW_VRAM         = TIER in ("low", "balanced")
USE_INT8         = os.environ.get("DRAMABOX_INT8") == "1"   # performance 档可选 torchao INT8（默认关）
USE_BITSANDBYTES = True   # Gemma 4bit 量化加载 (CUDA)
USE_COMPILE      = False  # torch.compile（Windows 缺 Triton，不可用）

# 打印档位
if __name__ != "__main__":
    logging.getLogger(__name__).info(_tier_msg)
