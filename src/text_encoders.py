"""Gemma front ends for the VRAM tiers, all producing what the embeddings processor wants: Gemma's 49 hidden states

By Siddid Soni, 2026-10 (distributed under this repo's LTX-2 Community License).
(embeddings, 47 layer outputs, final norm) left-padded to 1024 tokens, plus the binary mask.

  SplitGemma     8 GB CUDA tier: bnb 4-bit Gemma with the 2 GB embedding table looked up on the CPU (~5.8 GB VRAM).
                 Ported from Konvey (konvey/speech/dramabox.py, _SplitGemma).
  MlxGemma       Apple Silicon: Gemma 3 12B 4-bit in MLX (mlx-lm), hidden states converted to torch.
  build_embeddings_processor()
                 The audio-only embeddings processor alone, without loading Gemma (Konvey's PromptEncoder trick).

Padding trim (config.TRIM_PADDING, always on for MLX): only the real tokens go through Gemma, and the 1024-token left
padding is added to the hidden states afterwards, as zeros. Same result: the feature extractor zeroes padded positions
and the connector replaces them with its learned registers, and RoPE only sees relative positions, so the real tokens'
states do not depend on how much padding sits before them (Gemma 3's sliding window is 1024, so nothing falls out of
it either). Gemma does ~1/5 to 1/10 of the work for a typical prompt.
"""
from __future__ import annotations

import gc
import threading

import torch

from ltx_core.text_encoders.gemma.tokenizer import LTXVGemmaTokenizer
from ltx_core.utils import find_matching_file

MAX_TOKENS = 1024
_PATCH_LOCK = threading.Lock()


def build_embeddings_processor(checkpoint_path: str, gemma_root: str, dtype: torch.dtype, device: torch.device):
    """The warm audio-only embeddings processor (video parts stripped). PromptEncoder builds it right after loading
    Gemma; with the Gemma load skipped, only the processor is built. gemma_root must hold tokenizer.model,
    preprocessor_config.json and model*.safetensors (both the bnb and the mlx-community checkpoints do)."""
    from ltx_pipelines.utils import blocks

    with _PATCH_LOCK:
        load = blocks.PromptEncoder._load_bnb_4bit_encoder
        blocks.PromptEncoder._load_bnb_4bit_encoder = lambda self, root: None
        try:
            pe = blocks.PromptEncoder(checkpoint_path=checkpoint_path, gemma_root=gemma_root, dtype=dtype,
                                      device=device, warm=True, use_bnb_4bit=True, audio_only=True)
        finally:
            blocks.PromptEncoder._load_bnb_4bit_encoder = load
    return pe._warm_embeddings_processor


def real_token_ids(tokenizer: LTXVGemmaTokenizer, text: str) -> list[int]:
    """The ids tokenize_with_weights() gives for ``text``, without the padding."""
    return tokenizer.tokenizer(text.strip(), truncation=True, max_length=tokenizer.max_length)["input_ids"]


def left_pad(hidden_states, n: int, max_length: int = MAX_TOKENS):
    """Hidden states of ``n`` real tokens -> (states left-padded with zeros to max_length, binary mask), as if Gemma
    had run on the padded sequence."""
    hidden = tuple(torch.nn.functional.pad(h, (0, 0, max_length - n, 0)) for h in hidden_states)
    mask = torch.zeros(1, max_length, dtype=torch.int64, device=hidden[0].device)
    mask[:, max_length - n:] = 1
    return hidden, mask


def encode_trimmed(text_encoder, text: str):
    """ltx's GemmaTextEncoder.encode (the warm bnb 4-bit Gemma in PromptEncoder), on the real tokens only."""
    ids = torch.tensor([real_token_ids(text_encoder.tokenizer, text)], device=text_encoder.model.device)
    out = text_encoder.model.model(input_ids=ids, attention_mask=torch.ones_like(ids), output_hidden_states=True)
    return left_pad(out.hidden_states, ids.shape[1], text_encoder.tokenizer.max_length)


def audio_features_exact(feature_extractor, hidden_states, mask, chunk: int = 128) -> torch.Tensor:
    """FeatureExtractorV2's audio branch on the padded rows, bit-identical to calling the feature extractor, in far
    less memory: the per-token RMS norm runs a chunk of tokens at a time into one buffer (the stock version holds ~5
    full-size copies), and the projection runs once over all rows, so cuBLAS sees the same matrix shape."""
    import math

    fe = feature_extractor
    t, d, layers = hidden_states[0].shape[1], hidden_states[0].shape[2], len(hidden_states)
    lin = fe.audio_aggregate_embed
    scale = math.sqrt(lin.out_features / fe.embedding_dim)
    buf = torch.empty(1, t, d * layers, dtype=hidden_states[0].dtype, device=hidden_states[0].device)
    for a in range(0, t, chunk):
        enc = torch.stack([h[:, a:a + chunk] for h in hidden_states], dim=-1)   # [1, c, D, L]
        var = torch.mean(enc ** 2, dim=2, keepdim=True)
        normed = (enc * torch.rsqrt(var + 1e-6)).reshape(1, enc.shape[1], d * layers)
        keep = mask[:, a:a + chunk].bool().unsqueeze(-1)
        normed = torch.where(keep, normed, torch.zeros_like(normed))
        buf[:, a:a + chunk] = normed * scale
        del enc, var, normed
    return lin(buf)


class SplitGemma:
    """bnb-4bit Gemma with only the decoder layers (+ the small vision tower) on the GPU. The embedding table is
    looked up on the CPU and fed in as inputs_embeds (left as "cpu" in the device_map, accelerate would stream
    it to the GPU on every call); lm_head is never needed for encoding."""

    def __init__(self, gemma_root: str, dtype: torch.dtype):
        from accelerate.hooks import AlignDevicesHook
        from safetensors import safe_open
        from transformers import AutoConfig, Gemma3ForConditionalGeneration

        config = AutoConfig.from_pretrained(gemma_root)
        # The language model's parts are mapped one by one: mapped whole to the GPU, accelerate would place all its
        # children there first, the table included (a 2 GB spike while loading). The vision tower is packed 4-bit in
        # this checkpoint, so it can't be offloaded (offloaded modules load unquantized). The checkpoint's
        # quantization_config wins over a passed one, so patch the offload flag in.
        lm = "model.language_model."
        device_map = {lm + "layers": 0, lm + "norm": 0, lm + "rotary_emb": 0, lm + "rotary_emb_local": 0,
                      lm + "embed_tokens": "cpu", "model.vision_tower": 0, "model.multi_modal_projector": "cpu",
                      "lm_head": "cpu"}
        config.quantization_config["llm_int8_enable_fp32_cpu_offload"] = True
        self.model = Gemma3ForConditionalGeneration.from_pretrained(
            gemma_root, config=config, device_map=device_map, dtype=dtype).eval()
        shard = find_matching_file(gemma_root, "model-00001-of-*.safetensors")
        with safe_open(str(shard), framework="pt") as f:
            self.embed = f.get_tensor("language_model.model.embed_tokens.weight").to(dtype)
        self.embed_scale = torch.tensor(config.text_config.hidden_size ** 0.5, dtype=dtype)
        # accelerate keeps a GPU copy of the (tied) table in a tied-params cache every hook shares: drop the unused
        # modules and clear that cache (detaching their hooks would copy the weights back first).
        self.model.model.language_model.embed_tokens = torch.nn.Identity()
        self.model.lm_head = torch.nn.Identity()
        for m in self.model.modules():
            hook = getattr(m, "_hf_hook", None)
            for h in getattr(hook, "hooks", [hook]):
                if isinstance(h, AlignDevicesHook) and h.tied_params_map:
                    h.tied_params_map.clear()
        self.tokenizer = LTXVGemmaTokenizer(str(find_matching_file(gemma_root, "tokenizer.model").parent), MAX_TOKENS)
        gc.collect()
        torch.cuda.empty_cache()

    def gpu_modules(self) -> list:
        """What lives on the GPU (and is parked in host RAM between prompts): the decoder layers, norm, rotary
        embeddings and the vision tower. The embedding table (2 GB) and the tokenizer stay in host RAM."""
        lm = self.model.model.language_model
        rotary = [getattr(lm, n) for n in ("rotary_emb", "rotary_emb_local") if getattr(lm, n, None) is not None]
        return [lm.layers, lm.norm, *rotary, self.model.model.vision_tower]

    def encode(self, text: str, trim: bool = False):
        """ltx's GemmaTextEncoder.encode, with the embedding lookup on the CPU (``trim``: real tokens only)."""
        if trim:
            ids = torch.tensor([real_token_ids(self.tokenizer, text)])
            embeds = (torch.nn.functional.embedding(ids, self.embed) * self.embed_scale).to("cuda")
            mask = torch.ones(ids.shape, dtype=torch.int64, device="cuda")
            out = self.model.model(inputs_embeds=embeds, attention_mask=mask, output_hidden_states=True)
            return left_pad(out.hidden_states, ids.shape[1], self.tokenizer.max_length)
        pairs = self.tokenizer.tokenize_with_weights(text)["gemma"]
        ids = torch.tensor([[t[0] for t in pairs]])
        mask = torch.tensor([[w[1] for w in pairs]], device="cuda")
        embeds = (torch.nn.functional.embedding(ids, self.embed) * self.embed_scale).to("cuda")
        out = self.model.model(inputs_embeds=embeds, attention_mask=mask, output_hidden_states=True)
        return out.hidden_states, mask


class _Capture:
    """Stands in for one MLX decoder layer: records the hidden state going in, then runs the layer."""

    def __init__(self, layer, seen: list):
        self.layer, self.seen = layer, seen

    def __call__(self, x, *args, **kwargs):
        self.seen.append(x)
        return self.layer(x, *args, **kwargs)


class MlxGemma:
    """Gemma 3 12B (mlx-community/gemma-3-12b-it-4bit, ~8 GB) in MLX, returning every layer's hidden state the way
    transformers' output_hidden_states does: the input to each decoder layer, then the final norm's output. Always
    trimmed (mlx-lm takes no padding mask): real tokens only, padded afterwards."""

    def __init__(self, path: str, dtype: torch.dtype, device: torch.device):
        from mlx_lm import load

        self.model, _ = load(path)
        self.text = self._text_model(self.model)
        self.tokenizer = LTXVGemmaTokenizer(path, MAX_TOKENS)
        self.dtype, self.device = dtype, device

    @staticmethod
    def _text_model(model):
        """The module with embed_tokens, layers and norm (mlx-lm nests it differently across Gemma 3 versions)."""
        lm = getattr(model, "language_model", None)
        for m in (getattr(lm, "model", None), lm, getattr(model, "model", None), model):
            if m is not None and all(hasattr(m, a) for a in ("embed_tokens", "layers", "norm")):
                return m
        raise RuntimeError(f"no Gemma text model found in {type(model).__name__}")

    def encode(self, text: str):
        import mlx.core as mx
        import numpy as np

        ids = real_token_ids(self.tokenizer, text)
        seen: list = []
        layers = self.text.layers
        self.text.layers = [_Capture(layer, seen) for layer in layers]
        try:
            last = self.text(mx.array([ids]))
        finally:
            self.text.layers = layers
        states = [*seen, last]
        mx.eval(states)
        # numpy has no bfloat16: through float32, then to the torch dtype on the device
        hidden = [torch.from_numpy(np.array(h.astype(mx.float32))).to(self.device, self.dtype) for h in states]
        return left_pad(hidden, len(ids), self.tokenizer.max_length)

    def close(self) -> None:
        import mlx.core as mx

        self.model = self.text = None
        gc.collect()
        (getattr(mx, "clear_cache", None) or mx.metal.clear_cache)()
