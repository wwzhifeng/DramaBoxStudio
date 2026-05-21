"""
DramaBox TTS 引擎 — 模型热加载，一次加载多次推理。

基于 Resemble AI 官方 DramaBox，改造为离线运行：
  - 路径统一走 config.py，不硬编码
  - 自动检测显存，按档位选择量化/加载策略
  - 低显存模式：Gemma 存 CPU RAM，编码时搬上 GPU，编完搬回
  - 无 Perth 水印（可选开启）
"""
from __future__ import annotations

import json
import logging
import os
import sys
import time
from pathlib import Path

import torch
import torchaudio

APP_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(APP_DIR))
sys.path.insert(0, str(APP_DIR / "ltx2"))
sys.path.insert(0, str(APP_DIR / "src"))

from config import (
    DEVICE, DTYPE, CFG_SCALE, STG_SCALE, STEPS, SEED,
    USE_INT8, USE_BITSANDBYTES, LOW_VRAM,
    DIT_CHECKPOINT, AUDIO_COMPONENTS, FULL_CHECKPOINT, GEMMA_DIR,
)

from audio_conditioning import AudioConditionByReferenceLatent
from ltx_core.components.noisers import GaussianNoiser
from ltx_core.components.patchifiers import AudioPatchifier
from ltx_core.components.guiders import MultiModalGuider, MultiModalGuiderParams
from ltx_core.components.schedulers import LTX2Scheduler
from ltx_core.components.diffusion_steps import EulerDiffusionStep
from ltx_core.loader import DummyRegistry
from ltx_core.loader.single_gpu_model_builder import SingleGPUModelBuilder as Builder
from ltx_core.loader.sd_ops import SDOps
from ltx_core.model.transformer.model import LTXModel, LTXModelType, X0Model
from ltx_core.model.transformer.rope import LTXRopeType
from ltx_core.model.transformer.text_projection import create_caption_projection
from ltx_core.model.transformer.attention import AttentionFunction
from ltx_core.model.model_protocol import ModelConfigurator
from ltx_core.tools import AudioLatentTools
from ltx_core.types import Audio, AudioLatentShape, VideoPixelShape
from ltx_core.model.audio_vae import encode_audio as vae_encode_audio
from ltx_pipelines.utils.blocks import AudioConditioner, AudioDecoder, PromptEncoder
from ltx_pipelines.utils.media_io import decode_audio_from_file
from ltx_pipelines.utils.denoisers import GuidedDenoiser
from ltx_pipelines.utils.samplers import euler_denoising_loop
from safetensors import safe_open

DEFAULT_NEG = "worst quality, inconsistent, robotic, distorted, noise, static, muffled, unclear, unnatural, monotone"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def estimate_duration(prompt: str, multiplier: float = 1.1) -> float:
    from inference import estimate_speech_duration
    return max(3.0, round(estimate_speech_duration(prompt) * multiplier, 1))


def auto_rescale_for_cfg(cfg: float) -> float:
    if cfg <= 2.0:
        return 0.0
    if cfg <= 3.0:
        return 0.6 * (cfg - 2.0)
    if cfg <= 4.0:
        return 0.6 + 0.2 * (cfg - 3.0)
    if cfg <= 8.0:
        return 0.8
    return min(1.0, 0.8 + 0.1 * (cfg - 8.0))


class TTSServer:
    """LTX-2.3 音频 DiT + Gemma 文本编码器的热加载 TTS 服务。"""

    def __init__(
        self,
        checkpoint: str | None = None,
        full_checkpoint: str | None = None,
        gemma_root: str | None = None,
        device: str = DEVICE,
        dtype: str = DTYPE,
        low_vram: bool = LOW_VRAM,
    ):
        self.checkpoint = checkpoint or str(DIT_CHECKPOINT)
        self.full_checkpoint = full_checkpoint or str(FULL_CHECKPOINT)
        self.gemma_root = gemma_root or str(GEMMA_DIR)
        self.device = torch.device(device)
        self.dtype = torch.float16 if dtype == "fp16" else torch.bfloat16
        self.patchifier = AudioPatchifier(patch_size=1)
        self._low_vram = low_vram

        self._prompt_encoder = None
        self._velocity_model = None
        self._audio_conditioner = None
        self._audio_decoder = None
        self._perth = None

        log.info("DramaBox TTSServer 加载中...")
        t0 = time.time()
        self._load_all()

        if USE_INT8:
            self.apply_int8()

        log.info(f"全部模型加载完成 ({time.time() - t0:.1f}s) — 就绪")

    def _load_all(self):
        # 1. Prompt 编码器 (Gemma + embeddings processor)
        t0 = time.time()
        self._prompt_encoder = PromptEncoder(
            checkpoint_path=self.full_checkpoint,
            gemma_root=self.gemma_root,
            dtype=self.dtype, device=self.device,
            warm=True, use_bnb_4bit=USE_BITSANDBYTES, audio_only=True,
        )
        log.info(f"  PromptEncoder: {time.time() - t0:.1f}s")

        # 低显存模式：Gemma 编码完立即卸到 CPU，再加载 DiT+VAE
        if self._low_vram:
            self._gemma_to_cpu()

        # 2. Audio conditioner (VAE 编码器)
        t0 = time.time()
        self._audio_conditioner = AudioConditioner(
            checkpoint_path=self.full_checkpoint,
            dtype=self.dtype, device=self.device, warm=True,
        )
        log.info(f"  AudioConditioner: {time.time() - t0:.1f}s")

        # 3. Transformer (DiT 扩散主干)
        t0 = time.time()
        self._velocity_model = self._build_transformer()
        n_params = sum(p.numel() for p in self._velocity_model.parameters()) / 1e9
        vram_gb = sum(p.numel() * p.element_size() for p in self._velocity_model.parameters()) / 1e9
        log.info(f"  Transformer: {time.time() - t0:.1f}s ({n_params:.1f}B 参数, {vram_gb:.1f}GB {self.dtype})")

        # 4. Audio decoder (VAE 解码器 + Vocoder)
        t0 = time.time()
        self._audio_decoder = AudioDecoder(
            checkpoint_path=self.full_checkpoint,
            dtype=self.dtype, device=self.device, warm=True,
        )
        log.info(f"  AudioDecoder: {time.time() - t0:.1f}s")

    def _build_transformer(self) -> torch.nn.Module:
        with safe_open(self.checkpoint, framework="pt") as f:
            cfg = json.loads(f.metadata()["config"])
        t = cfg.get("transformer", {})

        class AudioOnlyConfigurator(ModelConfigurator[LTXModel]):
            @classmethod
            def from_config(cls, cfg):
                t = cfg.get("transformer", {})
                cp = None
                if not t.get("caption_proj_before_connector", False):
                    with torch.device("meta"):
                        cp = create_caption_projection(t, audio=True)
                return LTXModel(
                    model_type=LTXModelType.AudioOnly,
                    audio_num_attention_heads=t.get("audio_num_attention_heads", 32),
                    audio_attention_head_dim=t.get("audio_attention_head_dim", 64),
                    audio_in_channels=t.get("audio_in_channels", 128),
                    audio_out_channels=t.get("audio_out_channels", 128),
                    num_layers=t.get("num_layers", 48),
                    audio_cross_attention_dim=t.get("audio_cross_attention_dim", 2048),
                    norm_eps=t.get("norm_eps", 1e-6),
                    attention_type=AttentionFunction(t.get("attention_type", "default")),
                    positional_embedding_theta=10000.0,
                    audio_positional_embedding_max_pos=[20.0],
                    timestep_scale_multiplier=t.get("timestep_scale_multiplier", 1000),
                    use_middle_indices_grid=t.get("use_middle_indices_grid", True),
                    rope_type=LTXRopeType(t.get("rope_type", "interleaved")),
                    double_precision_rope=t.get("frequencies_precision", False) == "float64",
                    apply_gated_attention=t.get("apply_gated_attention", False),
                    audio_caption_projection=cp,
                    cross_attention_adaln=t.get("cross_attention_adaln", False),
                )

        audio_sd_ops = SDOps("AO").with_matching(prefix="model.diffusion_model.").with_replacement(
            "model.diffusion_model.", "")
        builder = Builder(
            model_path=self.checkpoint,
            model_class_configurator=AudioOnlyConfigurator,
            model_sd_ops=audio_sd_ops,
            registry=DummyRegistry(),
        )
        return builder.build(device=self.device, dtype=self.dtype).to(self.device).eval()

    # ── 低显存模式：Gemma 存 CPU RAM，按需搬上 GPU ──

    def _gemma_to_cpu(self) -> None:
        """将 Gemma 搬回 CPU RAM（保留 Python 对象，不释放）。"""
        pe = self._prompt_encoder
        if pe is not None and hasattr(pe, '_warm_text_encoder') and pe._warm_text_encoder is not None:
            pe._warm_text_encoder.to('cpu')
            torch.cuda.empty_cache()
            log.info(f"Gemma → CPU, GPU 显存: {torch.cuda.memory_allocated(self.device) / 1e9:.1f}GB")

    def _gemma_to_gpu(self) -> None:
        """将 Gemma 从 CPU RAM 搬回 GPU 用于编码。"""
        pe = self._prompt_encoder
        if pe is not None and hasattr(pe, '_warm_text_encoder') and pe._warm_text_encoder is not None:
            pe._warm_text_encoder.to(self.device)

    # ── 生成 ──────────────────────────────────────

    @torch.inference_mode()
    def generate(
        self,
        prompt: str,
        voice_ref: str | None = None,
        cfg_scale: float = CFG_SCALE,
        stg_scale: float = STG_SCALE,
        duration_multiplier: float = 1.1,
        seed: int = SEED,
        ref_duration: float = 10.0,
        rescale_scale: str | float = "auto",
        gen_duration: float = 0.0,
    ) -> tuple[torch.Tensor, int]:
        t_total = time.time()

        if gen_duration > 0:
            gen_dur = float(gen_duration)
        else:
            gen_dur = estimate_duration(prompt, duration_multiplier)

        fps = 25.0
        n_frames = int(round(gen_dur * fps)) + 1
        n_frames = ((n_frames - 1 + 4) // 8) * 8 + 1
        pixel_shape = VideoPixelShape(batch=1, frames=n_frames, height=64, width=64, fps=fps)
        target_shape = AudioLatentShape.from_video_pixel_shape(pixel_shape)
        audio_tools = AudioLatentTools(patchifier=self.patchifier, target_shape=target_shape)

        # Prompt encode — low-VRAM: 搬运 Gemma GPU→CPU，避免磁盘重载
        prompts = [prompt, DEFAULT_NEG] if cfg_scale > 1.0 else [prompt]
        if self._low_vram:
            self._gemma_to_gpu()
        ctx = self._prompt_encoder(prompts, streaming_prefetch_count=None)
        if self._low_vram:
            self._gemma_to_cpu()
        a_ctx = ctx[0].audio_encoding
        a_ctx_neg = ctx[1].audio_encoding if cfg_scale > 1.0 else None

        state = audio_tools.create_initial_state(device=self.device, dtype=self.dtype)
        gen = torch.Generator(device=self.device).manual_seed(seed)
        noiser = GaussianNoiser(generator=gen)
        state = noiser(state, noise_scale=1.0)

        if voice_ref and os.path.exists(voice_ref):
            voice = decode_audio_from_file(voice_ref, self.device, 0.0, ref_duration)
            w = voice.waveform
            if w.dim() == 2:
                if w.shape[0] == 1:
                    w = w.repeat(2, 1)
                w = w.unsqueeze(0)
            elif w.dim() == 3 and w.shape[1] == 1:
                w = w.repeat(1, 2, 1)
            target_samples = int(ref_duration * voice.sampling_rate)
            if w.shape[-1] < target_samples:
                w = w.repeat(1, 1, (target_samples // w.shape[-1]) + 1)
            w = w[..., :target_samples]
            peak = w.abs().max()
            if peak > 0:
                w = w * (10 ** (-4.0 / 20) / peak)
            voice = Audio(waveform=w, sampling_rate=voice.sampling_rate)
            ref_latent = self._audio_conditioner(lambda enc: vae_encode_audio(voice, enc, None))
            cond = AudioConditionByReferenceLatent(latent=ref_latent.to(self.device, self.dtype), strength=1.0)
            state = cond.apply_to(state, audio_tools)

        resc = auto_rescale_for_cfg(cfg_scale) if rescale_scale == "auto" else float(rescale_scale)
        guider = MultiModalGuider(
            params=MultiModalGuiderParams(
                cfg_scale=cfg_scale, stg_scale=stg_scale,
                stg_blocks=[29], rescale_scale=resc, modality_scale=1.0,
            ),
            negative_context=a_ctx_neg,
        )
        denoiser = GuidedDenoiser(v_context=None, a_context=a_ctx, video_guider=None, audio_guider=guider)
        sigmas = LTX2Scheduler().execute(steps=STEPS, latent=state.latent).to(self.device)

        x0 = X0Model(self._velocity_model)
        _, audio_state = euler_denoising_loop(
            sigmas=sigmas, video_state=None, audio_state=state,
            stepper=EulerDiffusionStep(), transformer=x0, denoiser=denoiser,
        )

        audio_state = audio_tools.clear_conditioning(audio_state)
        audio_state = audio_tools.unpatchify(audio_state)

        latent = audio_state.latent
        if latent.shape[2] > 513:
            f0, f1 = 511, 514
            n = f1 - f0
            patched = latent.clone()
            for f in (512, 513):
                t = (f - f0) / n
                patched[:, :, f, :] = (1.0 - t) * latent[:, :, f0, :] + t * latent[:, :, f1, :]
            latent = patched

        decoded = self._audio_decoder(latent)
        dur = decoded.waveform.shape[-1] / decoded.sampling_rate
        log.info(f"生成完成: {time.time() - t_total:.1f}s → {dur:.1f}s 音频")
        return decoded.waveform, decoded.sampling_rate

    def generate_to_file(self, prompt: str, output: str, voice_ref: str | None = None,
                         watermark: bool = False, **kwargs) -> str:
        """生成并保存为文件。"""
        waveform, sr = self.generate(prompt, voice_ref=voice_ref, **kwargs)
        import numpy as np
        wav_cpu = waveform.cpu().float()
        if wav_cpu.dim() == 3:
            wav_cpu = wav_cpu.squeeze(0)
        if watermark:
            try:
                import perth
                if not self._perth:
                    self._perth = perth.PerthImplicitWatermarker()
                mono = wav_cpu.mean(dim=0).numpy() if wav_cpu.shape[0] > 1 else wav_cpu[0].numpy()
                mono_wm = self._perth.apply_watermark(mono, sample_rate=sr)
                mono_wm_t = torch.from_numpy(np.asarray(mono_wm, dtype=np.float32)).unsqueeze(0)
                wav_cpu = mono_wm_t if wav_cpu.shape[0] == 1 else mono_wm_t.repeat(wav_cpu.shape[0], 1)
            except Exception as e:
                log.warning(f"Perth 水印跳过 ({e})")
        from scipy.io.wavfile import write as wav_write
        wav_np = wav_cpu.squeeze(0).numpy().T
        wav_np_int16 = (wav_np * 32767).clip(-32768, 32767).astype(np.int16)
        wav_write(output, sr, wav_np_int16)
        log.info(f"已保存: {output}")
        return output

    # ── 量化 ──────────────────────────────────────

    def apply_int8(self) -> None:
        """对 DiT 应用 INT8 量化。"""
        try:
            from torchao.quantization import quantize_, Int8WeightOnlyConfig
            quantize_(self._velocity_model, Int8WeightOnlyConfig())
            vram = torch.cuda.memory_allocated(self.device) / 1e9
            log.info(f"INT8 已应用，GPU 显存: {vram:.2f} GB")
        except ImportError:
            log.warning("torchao 未安装，跳过 INT8 量化")
        except Exception as e:
            log.warning(f"INT8 量化失败 ({e})，保持 BF16 运行")
