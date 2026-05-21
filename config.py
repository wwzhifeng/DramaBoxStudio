"""
DramaBox 配置 — 所有路径相对项目根，离线可用。
启动时自动检测 GPU 显存，按档位选择量化/加载策略。
"""
import logging
import os
import warnings
from pathlib import Path

# ── 屏蔽第三方库噪音 ──────────────────────────────
os.environ.setdefault("TORCH_DISTRIBUTED_DEBUG", "OFF")
os.environ.setdefault("TQDM_DISABLE", "1")
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
DEVICE      = "cuda"
DTYPE       = "bf16"
CFG_SCALE   = 2.5
STG_SCALE   = 1.5
STEPS       = 30
SEED        = 42

# ══════════════════════════════════════════════════
# 显存自检 + 自动档位
# ══════════════════════════════════════════════════

def _detect_vram_gb() -> float:
    """返回可用 GPU 总显存（GB），检测失败返回 0。"""
    try:
        import torch
        if not torch.cuda.is_available():
            return 0.0
        prop = torch.cuda.get_device_properties(0)
        return prop.total_memory / (1024 ** 3)
    except Exception:
        return 0.0


# 自动选定策略（可在 import 后覆写）
_vram = _detect_vram_gb()

if _vram >= 20:
    USE_INT8  = False
    LOW_VRAM  = False
    _tier_msg = f"显存 {_vram:.1f}G → 性能档（全 BF16，模型常驻）"
elif _vram >= 10:
    USE_INT8  = True
    LOW_VRAM  = True
    _tier_msg = f"显存 {_vram:.1f}G → 均衡档（INT8 + Gemma CPU 搬运）"
elif _vram >= 6:
    USE_INT8  = True
    LOW_VRAM  = True
    _tier_msg = f"显存 {_vram:.1f}G → 低显存档（编码时 DiT 也卸 CPU，峰值 ~8G）"
elif _vram > 0:
    USE_INT8  = True
    LOW_VRAM  = True
    _tier_msg = f"显存 {_vram:.1f}G → 极限档（可能 OOM）"
else:
    USE_INT8  = True
    LOW_VRAM  = True
    _tier_msg = "未检测到 CUDA 设备 → 默认低显存模式（可能无法运行）"

USE_BITSANDBYTES = True   # Gemma 4bit 量化加载
USE_COMPILE      = False  # torch.compile（Windows 缺 Triton，不可用）

# 打印档位
if __name__ != "__main__":
    import logging
    logging.getLogger(__name__).info(_tier_msg)
