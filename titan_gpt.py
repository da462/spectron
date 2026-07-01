# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn.functional as F
from torch import nn
from dataclasses import dataclass

# Prefer the legacy TorchTitan helper when it exists. Newer TorchTitan checkouts
# moved attention internals under torchtitan.models.common and no longer expose
# torchtitan.models.attention, so keep a local SDPA fallback for this repo's
# default causal path.
try:
    from torchtitan.models.attention import build_attention, init_attention_mask
except ModuleNotFoundError:
    class _CausalSDPAAttention(nn.Module):
        def forward(
            self,
            q: torch.Tensor,
            k: torch.Tensor,
            v: torch.Tensor,
        ) -> torch.Tensor:
            return F.scaled_dot_product_attention(q, k, v, is_causal=True)

    def build_attention(use_flex_attn: bool, attn_mask_type: str) -> nn.Module:
        if use_flex_attn:
            raise ImportError(
                "This TorchTitan checkout does not expose "
                "torchtitan.models.attention. Run without --use_flex_attn or "
                "install a TorchTitan version that provides the legacy helper."
            )
        if attn_mask_type != "causal":
            raise ValueError(
                f"Local SDPA fallback only supports causal masks, got {attn_mask_type!r}"
            )
        return _CausalSDPAAttention()

    def init_attention_mask(*args, **kwargs) -> None:
        return None

# --- Model Args ---

@dataclass
class TitanModelArgs:
    vocab_size: int = 50257
    n_layers: int = 12
    n_heads: int = 12
    dim: int = 768
    max_seq_len: int = 1024
    n_kv_heads: int | None = None
    multiple_of: int = 256
    ffn_dim_multiplier: float | None = None
    norm_eps: float = 1e-5
    rope_theta: float = 10000.0
    use_flex_attn: bool = False
    attn_mask_type: str = "causal"
    depth_init: bool = False


# --- Rotary Embedding Helpers from torchtitan ---

def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0) -> torch.Tensor:
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    return torch.polar(torch.ones_like(freqs), freqs)

def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    ndim = x.ndim
    assert ndim > 1
    seqlen = x.shape[1]
    freqs_cis = freqs_cis[0:seqlen]
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)

def apply_rotary_emb(xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


# --- Attention and FeedForward Blocks ---

def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        torch.unsqueeze(x, dim=3)
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
    )

class Attention(nn.Module):
    def __init__(self, model_args: TitanModelArgs):
        super().__init__()
        self.n_heads = model_args.n_heads
        self.n_kv_heads = model_args.n_heads if model_args.n_kv_heads is None else model_args.n_kv_heads
        self.n_rep = self.n_heads // self.n_kv_heads
        self.head_dim = model_args.dim // model_args.n_heads
        self.wq = nn.Linear(model_args.dim, model_args.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(model_args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(model_args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(model_args.n_heads * self.head_dim, model_args.dim, bias=False)
        self.sdpa = build_attention(model_args.use_flex_attn, model_args.attn_mask_type)

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor):
        bs, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
        xq = xq.view(bs, seqlen, -1, self.head_dim)
        xk = xk.view(bs, seqlen, -1, self.head_dim)
        xv = xv.view(bs, seqlen, -1, self.head_dim)
        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)
        keys = repeat_kv(xk, self.n_rep)
        values = repeat_kv(xv, self.n_rep)
        xq = xq.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)
        output = self.sdpa(xq, keys, values)
        output = output.transpose(1, 2).contiguous().view(bs, seqlen, -1)
        return self.wo(output)

class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, multiple_of: int, ffn_dim_multiplier: float | None):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class TransformerBlock(nn.Module):
    def __init__(self, layer_id: int, model_args: TitanModelArgs):
        super().__init__()
        self.attention = Attention(model_args)
        self.feed_forward = FeedForward(
            dim=model_args.dim,
            hidden_dim=4 * model_args.dim,
            multiple_of=model_args.multiple_of,
            ffn_dim_multiplier=model_args.ffn_dim_multiplier,
        )
        self.attention_norm = nn.RMSNorm(model_args.dim, eps=model_args.norm_eps)
        self.ffn_norm = nn.RMSNorm(model_args.dim, eps=model_args.norm_eps)
        self.masked = False
        self.backward_nograd = False  # when True and masked, preserve forward value with identity gradient
        # Debug counters for mask behavior verification
        self.debug_skip_calls = 0
        self.debug_nograd_calls = 0

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor):
        if self.masked:
            if self.backward_nograd:
                # Compute full block output without tracking, then stitch identity gradient:
                # y = x + (y_full - x).detach()  => forward equals y_full; d y / d x = I; no param grads
                with torch.no_grad():
                    attn_out = self.attention(self.attention_norm(x), freqs_cis)
                    h = x + attn_out
                    y_full = h + self.feed_forward(self.ffn_norm(h))
                # Debug: record that the no-grad masked branch was taken
                self.debug_nograd_calls += 1
                return x + (y_full - x).detach()
            # Debug: record that the skip branch was taken
            self.debug_skip_calls += 1
            return x
        h = x + self.attention(self.attention_norm(x), freqs_cis)
        out = h + self.feed_forward(self.ffn_norm(h))
        return out


# --- Main Transformer Model ---

class TitanGPT(nn.Module):
    def __init__(self, model_args: TitanModelArgs):
        super().__init__()
        self.model_args = model_args
        self.tok_embeddings = nn.Embedding(model_args.vocab_size, model_args.dim)
        self.layers = torch.nn.ModuleDict()
        for layer_id in range(model_args.n_layers):
            self.layers[str(layer_id)] = TransformerBlock(layer_id, model_args)
        self.norm = nn.RMSNorm(model_args.dim, eps=model_args.norm_eps)
        self.output = nn.Linear(model_args.dim, model_args.vocab_size, bias=False)
        self.register_buffer("freqs_cis", self._precompute_freqs_cis(), persistent=False)

    def _precompute_freqs_cis(self) -> torch.Tensor:
        return precompute_freqs_cis(
            self.model_args.dim // self.model_args.n_heads,
            self.model_args.max_seq_len,
            self.model_args.rope_theta,
        )

    def forward(self, tokens: torch.Tensor, input_batch: torch.Tensor | None = None, eos_id: int | None = None):
        if self.model_args.use_flex_attn:
            init_attention_mask(input_batch if input_batch is not None else tokens, eos_id=eos_id)

        h = self.tok_embeddings(tokens)
        freqs_cis = self.freqs_cis.to(h.device)
        for layer in self.layers.values():
            h = layer(h, freqs_cis)
        h = self.norm(h)
        return self.output(h)

    def set_mask_from_vector(self, mask_vec):
        assert len(mask_vec) == len(self.layers), "Mask vector length must match number of transformer blocks"
        for i, (block, val) in enumerate(zip(self.layers.values(), mask_vec)):
            if isinstance(val, torch.Tensor):
                # Ensure Python bool to avoid device sync/ambiguity in conditionals
                val_int = int(val.item())
            else:
                val_int = int(val)
            block.masked = (val_int == 0)
            # Ensure forward-skip mode never uses backward_nograd path
            block.backward_nograd = False

    def set_mask_with_mode(self, mask_vec, mode: str | None = None):
        assert len(mask_vec) == len(self.layers), "Mask vector length must match number of transformer blocks"
        for i, (block, val) in enumerate(zip(self.layers.values(), mask_vec)):
            if isinstance(val, torch.Tensor):
                val_int = int(val.item())
            else:
                val_int = int(val)
            masked = (val_int == 0)
            block.masked = masked
            block.backward_nograd = (mode == 'backward_nograd') and masked

    def get_mask_debug_counters(self):
        skip = []
        nograd = []
        for layer in self.layers.values():
            skip.append(int(getattr(layer, 'debug_skip_calls', 0)))
            nograd.append(int(getattr(layer, 'debug_nograd_calls', 0)))
        return {'skip_calls': skip, 'nograd_calls': nograd}

    def reset_mask_debug_counters(self):
        for layer in self.layers.values():
            if hasattr(layer, 'debug_skip_calls'):
                layer.debug_skip_calls = 0
            if hasattr(layer, 'debug_nograd_calls'):
                layer.debug_nograd_calls = 0
