"""
DramaBox TTS 引擎 — 模型热加载，一次加载多次推理。

Modified by Siddid Soni, 2026-10: VRAM tiers, low-VRAM staged server, prompt / voice caches, Gemma output modes.

基于 Resemble AI 官方 DramaBox，改造为离线运行：
  - 路径统一走 config.py，不硬编码
  - 自动检测显存，按档位选服务（create_server）：
      performance (≥20G)  TTSServer         全部常驻 GPU
      balanced    (16G)   LowVramTTSServer  文本 / 音频两阶段轮流上 GPU，闲置阶段驻留内存
      low         (8G)    LowVramTTSServer  Gemma / 嵌入处理器 / 音频三阶段轮流上 GPU
      mlx-16 / mlx-24     MacTTSServer      Gemma 走 MLX，其余走 MPS（mac_server.py）
  - 提示词嵌入缓存（内存 + 磁盘），负向提示词与重复提示词只编码一次
  - 无 Perth 水印（可选开启）

低显存档移植自 Konvey（scripts/dramabox_lowvram.py, konvey/speech/dramabox.py, konvey/speech/mirror.py）。
"""
from __future__ import annotations

import os

# Before torch is imported: without it, decoding right after a denoise can fail on fragmentation (Konvey).
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import gc  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import logging  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import torch  # noqa: E402
import torchaudio  # noqa: E402

APP_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(APP_DIR))
sys.path.insert(0, str(APP_DIR / "ltx2"))
sys.path.insert(0, str(APP_DIR / "src"))

from config import (  # noqa: E402
    DEVICE, DTYPE, CFG_SCALE, STG_SCALE, STEPS, SEED, TIER, TRIM_PADDING, PARK, GEMMA_OUTPUT,
    USE_INT8, USE_BITSANDBYTES, EMBED_CACHE_DIR, PARK_DIR,
    DIT_CHECKPOINT, AUDIO_COMPONENTS, FULL_CHECKPOINT, GEMMA_DIR,
)

from audio_conditioning import AudioConditionByReferenceLatent  # noqa: E402
from ltx_core.components.noisers import GaussianNoiser  # noqa: E402
from ltx_core.components.patchifiers import AudioPatchifier  # noqa: E402
from ltx_core.components.guiders import MultiModalGuider, MultiModalGuiderParams  # noqa: E402
from ltx_core.components.schedulers import LTX2Scheduler  # noqa: E402
from ltx_core.components.diffusion_steps import EulerDiffusionStep  # noqa: E402
from ltx_core.loader import DummyRegistry  # noqa: E402
from ltx_core.loader.single_gpu_model_builder import SingleGPUModelBuilder as Builder  # noqa: E402
from ltx_core.loader.sd_ops import SDOps  # noqa: E402
from ltx_core.model.transformer.model import LTXModel, LTXModelType, X0Model  # noqa: E402
from ltx_core.model.transformer.rope import LTXRopeType  # noqa: E402
from ltx_core.model.transformer.text_projection import create_caption_projection  # noqa: E402
from ltx_core.model.transformer.attention import AttentionFunction  # noqa: E402
from ltx_core.model.model_protocol import ModelConfigurator  # noqa: E402
from ltx_core.tools import AudioLatentTools  # noqa: E402
from ltx_core.types import Audio, AudioLatentShape, VideoPixelShape  # noqa: E402
from ltx_core.model.audio_vae import encode_audio as vae_encode_audio  # noqa: E402
from ltx_pipelines.utils.blocks import AudioConditioner, AudioDecoder, PromptEncoder  # noqa: E402
from ltx_core.text_encoders.gemma.embeddings_processor import convert_to_additive_mask  # noqa: E402
from ltx_pipelines.utils.media_io import decode_audio_from_file  # noqa: E402
from ltx_pipelines.utils.denoisers import GuidedDenoiser  # noqa: E402
from ltx_pipelines.utils.samplers import euler_denoising_loop  # noqa: E402
from safetensors import safe_open  # noqa: E402

from text_encoders import audio_features_exact, encode_trimmed, left_pad  # noqa: E402

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


def _free() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif torch.backends.mps.is_available():
        torch.mps.empty_cache()


def _gpu_gb() -> float:
    return torch.cuda.memory_allocated() / 2**30 if torch.cuda.is_available() else 0.0


class TTSServer:
    """性能档（≥20G）：LTX-2.3 音频 DiT + Gemma 文本编码器全部常驻 GPU 的热加载 TTS 服务。"""

    EMBED_KEY = "bnb4"   # Gemma 变体，写进嵌入缓存键（不同 Gemma 的嵌入不能混用）
    ALWAYS_TRIM = False  # True：无视 DRAMABOX_TRIM_PADDING，总是只跑真实 token

    def __init__(
        self,
        checkpoint: str | None = None,
        full_checkpoint: str | None = None,
        gemma_root: str | None = None,
        device: str = DEVICE,
        dtype: str = DTYPE,
        gemma_output: str = GEMMA_OUTPUT,
    ):
        self.checkpoint = checkpoint or str(DIT_CHECKPOINT)
        self.full_checkpoint = full_checkpoint or str(FULL_CHECKPOINT)
        self.gemma_root = gemma_root or str(GEMMA_DIR)
        self.device = torch.device(device)
        self.dtype = torch.float16 if dtype == "fp16" else torch.bfloat16
        self.patchifier = AudioPatchifier(patch_size=1)
        self.trim = self.ALWAYS_TRIM or TRIM_PADDING
        self.gemma_output = gemma_output

        self._prompt_encoder = None
        self._velocity_model = None
        self._audio_conditioner = None
        self._audio_decoder = None
        self._perth = None
        self._embeds: dict[str, torch.Tensor] = {}       # prompt → audio_encoding（CPU）
        self._ref_latents: dict[tuple, torch.Tensor] = {}  # (参考音频, mtime, 时长) → latent（CPU）

        log.info(f"DramaBox {type(self).__name__} 加载中（{TIER}，padding trim {'开' if self.trim else '关'}）...")
        t0 = time.time()
        self._load_all()
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

        self._load_audio_stage()

        if USE_INT8:
            self.apply_int8()

    def _load_audio_stage(self, device: torch.device | None = None):
        device = device or self.device
        # 2. Audio conditioner (VAE 编码器)
        t0 = time.time()
        self._audio_conditioner = AudioConditioner(
            checkpoint_path=self.full_checkpoint,
            dtype=self.dtype, device=device, warm=True,
        )
        log.info(f"  AudioConditioner: {time.time() - t0:.1f}s")

        # 3. Transformer (DiT 扩散主干)
        t0 = time.time()
        self._velocity_model = self._build_transformer(device)
        n_params = sum(p.numel() for p in self._velocity_model.parameters()) / 1e9
        vram_gb = sum(p.numel() * p.element_size() for p in self._velocity_model.parameters()) / 1e9
        log.info(f"  Transformer: {time.time() - t0:.1f}s ({n_params:.1f}B 参数, {vram_gb:.1f}GB {self.dtype})")

        # 4. Audio decoder (VAE 解码器 + Vocoder)
        t0 = time.time()
        self._audio_decoder = AudioDecoder(
            checkpoint_path=self.full_checkpoint,
            dtype=self.dtype, device=device, warm=True,
        )
        log.info(f"  AudioDecoder: {time.time() - t0:.1f}s")

    def _build_transformer(self, device: torch.device | None = None) -> torch.nn.Module:
        device = device or self.device
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
        return builder.build(device=device, dtype=self.dtype).to(device).eval()

    # ── 提示词嵌入缓存 ────────────────────────────────

    def _embed_path(self, prompt: str) -> Path:
        key = f"{self.EMBED_KEY}{'-trim' if self.trim else ''}\0{prompt}"
        return EMBED_CACHE_DIR / f"{hashlib.sha256(key.encode()).hexdigest()[:16]}.pt"

    def _cached_embed(self, prompt: str) -> torch.Tensor | None:
        emb = self._embeds.get(prompt)
        if emb is None:
            path = self._embed_path(prompt)
            if path.exists():
                try:
                    emb = self._embeds[prompt] = torch.load(path, map_location="cpu")
                except Exception as e:
                    log.warning(f"嵌入缓存读取失败，重新编码 ({e})")
        return emb

    def _store_embed(self, prompt: str, audio_encoding: torch.Tensor) -> None:
        emb = self._embeds[prompt] = audio_encoding.detach().to("cpu")
        try:
            EMBED_CACHE_DIR.mkdir(parents=True, exist_ok=True)
            torch.save(emb, self._embed_path(prompt))
        except Exception as e:
            log.warning(f"嵌入缓存写入失败 ({e})")

    @torch.inference_mode()
    def encode(self, prompts: list[str]) -> None:
        """编码尚未缓存的提示词。批量生成前先把全部提示词传进来，低显存档就只切换一次阶段。"""
        todo = [p for p in dict.fromkeys(prompts) if self._cached_embed(p) is None]
        if not todo:
            return
        t0 = time.time()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        self._encode_uncached(todo)
        peak = f"，GPU 峰值 {torch.cuda.max_memory_allocated() / 2**30:.2f} GB" if torch.cuda.is_available() else ""
        log.info(f"编码 {len(todo)} 条提示词: {time.time() - t0:.1f}s{peak}（{self.gemma_output}）")

    def _encode_uncached(self, todo: list[str]) -> None:
        pe = self._prompt_encoder
        for p in todo:
            te = pe._warm_text_encoder
            hs, mask = encode_trimmed(te, p) if self.trim else te.encode(p)
            self._store_embed(p, self._audio_encoding(pe._warm_embeddings_processor, hs, mask))
            del hs

    # ── Gemma 输出 → 嵌入（DRAMABOX_GEMMA_OUTPUT；Gemma 与嵌入处理器同在 GPU 时 real 等同 full） ──

    def _audio_encoding(self, embedder, hs, mask) -> torch.Tensor:
        """Gemma 隐状态（左填充到 1024）→ audio encoding。project 模式先投影再过 connector，GPU 瞬时占用更低。"""
        if not self.gemma_output.startswith("project"):
            return embedder.process_hidden_states(hs, mask).audio_encoding
        return self._connect(embedder.audio_connector, self._projection(embedder.feature_extractor, hs, mask), mask)

    def _projection(self, feature_extractor, hs, mask) -> torch.Tensor:
        """The feature extractor's audio projection for the real tokens ([1, n, 2048]). project-exact runs it on the
        padded rows like Konvey (bit-identical); project, or project-exact out of GPU memory, on the real rows."""
        n = int(mask.sum())
        feats = self._exact_features(feature_extractor, hs, mask) if self.gemma_output == "project-exact" else None
        if feats is None:
            ones = torch.ones(1, n, dtype=torch.int64, device=hs[0].device)
            _, feats = feature_extractor(tuple(h[:, -n:] for h in hs), ones, "left")
        return feats[:, -n:]   # padded rows are replaced by the connector's registers anyway

    @staticmethod
    def _exact_features(feature_extractor, hs, mask) -> torch.Tensor | None:
        """project-exact; None when it runs out of GPU memory (→ project for this prompt)."""
        try:
            return audio_features_exact(feature_extractor, hs, mask)
        except torch.OutOfMemoryError:
            pass
        # past the except block, so the failed frame's tensors are gone before the retry
        _free()
        log.warning("project-exact 显存不足，本条改用 project（显存占用更低，有微小舍入差异）")
        return None

    @staticmethod
    def _connect(connector, feats, mask) -> torch.Tensor:
        """The rest of process_hidden_states() after the feature extractor (audio-only create_embeddings)."""
        if feats.shape[1] < mask.shape[1]:
            (feats,), _ = left_pad((feats,), feats.shape[1], mask.shape[1])
        enc, _ = connector(feats, convert_to_additive_mask(mask, feats.dtype))
        return enc

    # ── 阶段钩子（低显存档覆写） ───────────────────────

    def _audio_on(self) -> None:
        """确保音频阶段（DiT + VAE）在 GPU 上。"""

    def _ref_latent(self, voice_ref: str, ref_duration: float) -> torch.Tensor:
        """参考音频 → VAE latent，按 (路径, mtime, 时长) 缓存：同一音色只编码一次。"""
        key = (voice_ref, os.path.getmtime(voice_ref), float(ref_duration))
        if key not in self._ref_latents:
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
            latent = self._audio_conditioner(lambda enc: vae_encode_audio(voice, enc, None))
            self._ref_latents[key] = latent.to("cpu")
        return self._ref_latents[key].to(self.device, self.dtype)

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

        # Prompt encode — 命中缓存则不碰 Gemma；低显存档只在有新提示词时才切换阶段
        use_cfg = cfg_scale > 1.0
        self.encode([prompt, DEFAULT_NEG] if use_cfg else [prompt])
        self._audio_on()
        a_ctx = self._cached_embed(prompt).to(self.device, self.dtype)
        a_ctx_neg = self._cached_embed(DEFAULT_NEG).to(self.device, self.dtype) if use_cfg else None

        state = audio_tools.create_initial_state(device=self.device, dtype=self.dtype)
        gen = torch.Generator(device=self.device).manual_seed(seed)
        noiser = GaussianNoiser(generator=gen)
        state = noiser(state, noise_scale=1.0)

        if voice_ref and os.path.exists(voice_ref):
            ref_latent = self._ref_latent(voice_ref, ref_duration)
            cond = AudioConditionByReferenceLatent(latent=ref_latent, strength=1.0)
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
        """对 DiT 应用 INT8 量化（仅性能档，DRAMABOX_INT8=1 开启）。"""
        try:
            from torchao.quantization import quantize_, Int8WeightOnlyConfig
            quantize_(self._velocity_model, Int8WeightOnlyConfig())
            log.info(f"INT8 已应用，GPU 显存: {_gpu_gb():.2f} GB")
        except ImportError:
            log.warning("torchao 未安装，跳过 INT8 量化")
        except Exception as e:
            log.warning(f"INT8 量化失败 ({e})，保持 BF16 运行")


class LowVramTTSServer(TTSServer):
    """8G / 16G 档（移植自 Konvey）：各阶段轮流上 GPU，闲置阶段驻留在锁页内存（Mirror：卸下 ~0.3s，装回 ~0.5s）。

      balanced (16G)  text  = Gemma 4bit + 嵌入处理器 (~10G)   | audio = DiT bf16 + VAE 编/解码器 + vocoder (~7G)
      low      (8G)   gemma = SplitGemma，词嵌入表查表在 CPU (~5.8G) | embed = 嵌入处理器 | audio = 同上

    全部计算都在 GPU 上。文本侧在首次遇到新提示词时才构建；嵌入有缓存，GPU 只在有新提示词时切换阶段。
    DRAMABOX_PARK=ssd：同一时刻内存里只有一个大模型，其余在 SSD（见 config.py）。
    """

    def __init__(self, *args, tier: str = TIER, park: str = PARK, **kwargs):
        self.tier = tier
        self.park = park
        # low + project*: the projection rides in the Gemma stage, the connector in the embed stage
        self.project = tier == "low" and kwargs.get("gemma_output", GEMMA_OUTPUT).startswith("project")
        self._split = tier == "low"
        self._gemma = None       # SplitGemma（low）
        self._embedder = None    # 嵌入处理器（low）
        self._mirrors: dict = {}
        self._stores: dict = {}   # ssd: stage → DiskMirror
        self._on: set[str] = set()
        super().__init__(*args, **kwargs)

    def _load_all(self):
        """只加载音频阶段；文本侧首次使用时再构建（ssd 模式见 _load_ssd）。"""
        if self.park == "ssd":
            return self._load_ssd()
        self._load_audio_stage()
        self._on = {"audio"}
        _free()
        log.info(f"  音频阶段: GPU {_gpu_gb():.1f} GB（DRAMABOX_PARK={self.park}）")

    def _stage_modules(self, stage: str) -> list:
        if stage == "text":
            pe = self._prompt_encoder
            return [pe._warm_text_encoder, pe._warm_embeddings_processor]
        if stage == "gemma":
            return self._gemma.gpu_modules() + ([self._embedder.feature_extractor] if self.project else [])
        if stage == "embed":
            return [self._embedder.audio_connector] if self.project else [self._embedder]
        return [self._velocity_model, self._audio_conditioner._warm_encoder,
                self._audio_decoder._warm_decoder, self._audio_decoder._warm_vocoder]

    def _build(self, stage: str) -> None:
        from text_encoders import SplitGemma, build_embeddings_processor

        t0 = time.time()
        if stage == "text":
            self._prompt_encoder = PromptEncoder(
                checkpoint_path=self.full_checkpoint, gemma_root=self.gemma_root,
                dtype=self.dtype, device=self.device,
                warm=True, use_bnb_4bit=USE_BITSANDBYTES, audio_only=True,
            )
        elif stage == "gemma":
            self._gemma = SplitGemma(self.gemma_root, self.dtype)
            if self.project and self.park != "ssd":   # ssd: the CPU-side build makes the processor
                self._embedder = build_embeddings_processor(
                    self.full_checkpoint, self.gemma_root, self.dtype, torch.device("cpu"))
                self._embedder.feature_extractor.to(self.device)
        elif stage == "embed":
            if self.project:
                self._embedder.audio_connector.to(self.device)
            else:
                self._embedder = build_embeddings_processor(
                    self.full_checkpoint, self.gemma_root, self.dtype, self.device)
        _free()
        log.info(f"  {stage} 阶段构建: {time.time() - t0:.1f}s, GPU {_gpu_gb():.1f} GB")

    # ── ssd：同一时刻内存里只有一个大模型 ─────────────────

    # 某阶段上 GPU 后，后台从 SSD 预读到内存的阶段（low：gemma → embed → audio；balanced：text ↔ audio）
    PREFETCH = {"gemma": ("embed", "audio"), "embed": ("audio",), "audio": ("gemma", "embed", "text"),
                "text": ("audio",)}

    def _store_path(self, stage: str) -> Path:
        files = [self.checkpoint, self.full_checkpoint] + ([] if stage == "audio" else [self.gemma_root])
        sig = [self.tier, stage, str(self.dtype), "project" if self.project and stage != "audio" else ""] + [f"{f}:{os.stat(f).st_mtime_ns}:{os.stat(f).st_size}" for f in files]
        return PARK_DIR / f"{stage}-{hashlib.sha256(chr(0).join(sig).encode()).hexdigest()[:16]}.bin"

    def _new_store(self, stage: str):
        from mirror import DiskMirror
        store = self._stores[stage] = DiskMirror(self._store_path(stage), *self._stage_modules(stage))
        return store

    def _load_ssd(self):
        """文本侧（Gemma）在 GPU 上构建，同时音频阶段（low 还有嵌入处理器）在 CPU 上构建、直接放进内存。"""
        import threading

        cpu = torch.device("cpu")
        cpu_stages = ["audio"] + (["embed"] if self._split else [])
        errors = []

        def build_on_cpu():
            from text_encoders import build_embeddings_processor
            try:
                with torch.no_grad():   # not inference_mode: the weights are moved to the GPU outside it later
                    t0 = time.time()
                    self._load_audio_stage(cpu)
                    if self._split:
                        self._embedder = build_embeddings_processor(
                            self.full_checkpoint, self.gemma_root, self.dtype, cpu)
                    for st in cpu_stages:
                        self._new_store(st).to_ram()
                        self._stores[st].save()
                    log.info(f"  CPU 侧 {cpu_stages} 构建进内存: {time.time() - t0:.1f}s")
            except BaseException as e:  # noqa: BLE001 — re-raised on the main thread
                errors.append(e)

        worker = threading.Thread(target=build_on_cpu, daemon=True)
        worker.start()
        text = "gemma" if self._split else "text"
        self._build(text)
        worker.join()
        if errors:
            raise errors[0]
        if self.project:   # the projection rides with Gemma; the connector stays in the embed stage
            self._embedder.feature_extractor.to(self.device)
        self._new_store(text).save()   # GPU → SSD in the background, before the first swap needs it
        self._on = {text}
        log.info(f"  ssd 模式就绪: {text} 在 GPU，{cpu_stages} 在内存，GPU {_gpu_gb():.1f} GB")

    def _use_ssd(self, stage: str) -> None:
        t0 = time.time()
        for s in list(self._on):
            self._stores[s].drop()
            self._on.discard(s)
        self._stores[stage].upload()
        self._on.add(stage)
        for nxt in self.PREFETCH[stage]:
            if nxt in self._stores:
                self._stores[nxt].prefetch()
        log.info(f"  → {stage} 阶段上 GPU: {time.time() - t0:.1f}s")

    def _use(self, stage: str) -> None:
        """把 stage 搬上 GPU，其余阶段先卸到内存。"""
        from mirror import Mirror

        if stage in self._on:
            return
        if self.park == "ssd":
            return self._use_ssd(stage)
        t0 = time.time()
        for s in list(self._on):
            if s not in self._mirrors:
                self._mirrors[s] = Mirror("cuda", *self._stage_modules(s))
            self._mirrors[s].park()
            self._on.discard(s)
        if stage in self._mirrors:
            self._mirrors[stage].unpark()
            log.info(f"  → {stage} 阶段上 GPU: {time.time() - t0:.1f}s")
        else:
            self._build(stage)
        self._on.add(stage)

    def _audio_on(self) -> None:
        self._use("audio")

    def _gemma_output(self, prompt: str):
        """low: Gemma (+ the projection in project modes) on one prompt → what waits in host RAM, and the mask.
        Its own function so the prompt's GPU tensors are freed before the next prompt is encoded."""
        hs, mask = self._gemma.encode(prompt, trim=self.trim)
        n = int(mask.sum()) if self.gemma_output != "full" or self.trim else mask.shape[1]
        if not self.project:
            return tuple(h[:, -n:].cpu() for h in hs), mask.cpu()
        return (self._projection(self._embedder.feature_extractor, hs, mask).cpu(),), mask.cpu()

    def _encode_uncached(self, todo: list[str]) -> None:
        if not self._split:
            self._use("text")
            super()._encode_uncached(todo)
            return
        # low: Gemma 的输出先暂存内存，Gemma 卸下后嵌入处理器再上 GPU（存什么见 DRAMABOX_GEMMA_OUTPUT）
        self._use("gemma")
        torch.cuda.reset_peak_memory_stats()
        raw, stored = [], 0
        for p in todo:
            out, mask = self._gemma_output(p)
            stored += sum(t.numel() * t.element_size() for t in out)
            raw.append((out, mask))
        log.info(f"  Gemma 阶段 GPU 峰值 {torch.cuda.max_memory_allocated() / 2**30:.2f} GB，"
                 f"暂存 {stored / 2**20:.0f} MB（{len(todo)} 条，{self.gemma_output}）")
        self._use("embed")
        for p, (out, mask) in zip(todo, raw):
            out = tuple(t.to(self.device) for t in out)
            mask = mask.to(self.device)
            if self.project:
                enc = self._connect(self._embedder.audio_connector, out[0], mask)
            else:
                if out[0].shape[1] < mask.shape[1]:
                    out, _ = left_pad(out, out[0].shape[1], mask.shape[1])
                enc = self._embedder.process_hidden_states(out, mask).audio_encoding
            self._store_embed(p, enc)
        del raw

def create_server(**kwargs) -> TTSServer:
    """按档位选服务：mlx-* → MacTTSServer，low / balanced → LowVramTTSServer，performance → TTSServer。"""
    if TIER.startswith("mlx"):
        from mac_server import MacTTSServer
        return MacTTSServer(**kwargs)
    if TIER in ("low", "balanced"):
        return LowVramTTSServer(**kwargs)
    return TTSServer(**kwargs)
