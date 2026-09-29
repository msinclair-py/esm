import functools

import einops
import torch
import torch.nn.functional as F
from torch import nn

from esm.layers.rotary import RotaryEmbedding, TritonRotaryEmbedding

try:
    from flash_attn import flash_attn_varlen_qkvpacked_func
except (ImportError, RuntimeError):
    flash_attn_varlen_qkvpacked_func = None  # ty:ignore[invalid-assignment]

try:
    # FlexAttention: a device-agnostic fused-attention path. Runs on CUDA and, as of
    # PyTorch 2.9, on Intel XPU (Aurora) via the Triton-XPU backend — unlike flash_attn,
    # which is CUDA-only. Requires torch.compile for performance (eager is correct but slow).
    from torch.nn.attention.flex_attention import create_block_mask, flex_attention
except ImportError:
    create_block_mask = None  # ty:ignore[invalid-assignment]
    flex_attention = None  # ty:ignore[invalid-assignment]


class MultiHeadAttention(nn.Module):
    def __init__(
        self, d_model: int, n_heads: int, bias: bool = False, qk_layernorm: bool = True
    ):
        super().__init__()

        self.d_model = d_model
        self.n_heads = n_heads

        self.d_head = self.d_model // self.n_heads
        self.layernorm_qkv = nn.Sequential(
            nn.LayerNorm(d_model), nn.Linear(d_model, d_model * 3, bias=bias)
        )
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)

        if qk_layernorm:
            self.q_ln = nn.LayerNorm(d_model, bias=bias)
            self.k_ln = nn.LayerNorm(d_model, bias=bias)
        else:
            self.q_ln = nn.Identity()
            self.k_ln = nn.Identity()

        self.rotary = RotaryEmbedding(d_model // n_heads)

    def _apply_rotary(self, q: torch.Tensor, k: torch.Tensor):
        q = q.unflatten(-1, (self.n_heads, self.d_head))
        k = k.unflatten(-1, (self.n_heads, self.d_head))
        q, k = self.rotary(q, k)
        q = q.flatten(-2, -1)
        k = k.flatten(-2, -1)
        return q, k

    def forward(
        self, x, seq_id, output_attentions: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        qkv_BLD3 = self.layernorm_qkv(x)
        query_BLD, key_BLD, value_BLD = torch.chunk(qkv_BLD3, 3, dim=-1)
        query_BLD, key_BLD = (
            self.q_ln(query_BLD).to(query_BLD.dtype),
            self.k_ln(key_BLD).to(query_BLD.dtype),
        )
        query_BLD, key_BLD = self._apply_rotary(query_BLD, key_BLD)

        reshaper = functools.partial(
            einops.rearrange, pattern="b s (h d) -> b h s d", h=self.n_heads
        )

        query_BHLD, key_BHLD, value_BHLD = map(
            reshaper, (query_BLD, key_BLD, value_BLD)
        )

        if seq_id is not None:
            # Where True, enable participation in attention.
            mask_BLL = seq_id.unsqueeze(-1) == seq_id.unsqueeze(-2)
            mask_BHLL = mask_BLL.unsqueeze(1)
        else:
            mask_BHLL = None

        if output_attentions:
            scale = self.d_head**-0.5
            attn_scores = torch.einsum("bhld,bhsd->bhls", query_BHLD, key_BHLD) * scale
            if mask_BHLL is not None:
                attn_scores = attn_scores.masked_fill(~mask_BHLL, float("-inf"))
            attn_weights = torch.softmax(attn_scores, dim=-1)
            context_BHLD = torch.einsum("bhls,bhsd->bhld", attn_weights, value_BHLD)
        else:
            attn_weights = None
            if mask_BHLL is None:
                # Shortcut, if we don't use attention biases then torch
                # will autoselect flashattention as the implementation
                context_BHLD = F.scaled_dot_product_attention(
                    query_BHLD, key_BHLD, value_BHLD
                )
            else:
                context_BHLD = F.scaled_dot_product_attention(
                    query_BHLD, key_BHLD, value_BHLD, mask_BHLL
                )

        context_BLD = einops.rearrange(context_BHLD, "b h s d -> b s (h d)")

        return self.out_proj(context_BLD), attn_weights


class FlashMultiHeadAttention(MultiHeadAttention):
    def __init__(
        self, d_model: int, n_heads: int, bias: bool = False, qk_layernorm: bool = True
    ):
        super().__init__(
            d_model=d_model, n_heads=n_heads, bias=bias, qk_layernorm=qk_layernorm
        )

        # Flash attention rotary.
        self.rotary = TritonRotaryEmbedding(d_model // n_heads)

    def forward(
        self, x, seq_id, output_attentions: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if output_attentions:
            raise ValueError(
                "FlashMultiHeadAttention does not support output_attentions=True. "
                "Use MultiHeadAttention instead."
            )
        assert seq_id.dtype == torch.bool

        seqlens = seq_id.sum(dim=-1, dtype=torch.int32)
        cu_seqlens = F.pad(torch.cumsum(seqlens, dim=0, dtype=torch.int32), (1, 0))
        max_seqlen = seqlens.max().item()

        qkv_ND3 = self.layernorm_qkv(x)

        query_ND, key_ND, value_ND = torch.chunk(qkv_ND3, 3, dim=-1)
        query_ND, key_ND = (
            self.q_ln(query_ND).to(query_ND.dtype),
            self.k_ln(key_ND).to(query_ND.dtype),
        )

        qkv_N3D = torch.stack([query_ND, key_ND, value_ND], dim=1)
        qkv_N3HD = einops.rearrange(
            qkv_N3D, pattern="n a (h d) -> n a h d", h=self.n_heads
        )
        qkv_N3HD = self.rotary(qkv_N3HD, cu_seqlens, max_seqlen)

        assert flash_attn_varlen_qkvpacked_func is not None
        context_NHD = flash_attn_varlen_qkvpacked_func(
            qkv_N3HD, cu_seqlens, max_seqlen, softmax_scale=self.d_head**-0.5
        )
        context_ND = einops.rearrange(context_NHD, "n h d -> n (h d)")

        return self.out_proj(context_ND), None


def build_sequence_block_mask(seq_id: torch.Tensor):
    """BlockMask that lets a token attend only within its own sequence id.

    Mirrors ``MultiHeadAttention``'s ``mask_BLL = seq_id_i == seq_id_j`` (works for
    integer sequence ids or a boolean real-vs-pad mask). Cheap blocks that are fully
    masked are skipped by FlexAttention, recovering most of the savings that flash's
    ``cu_seqlens`` packing gave. ``seq_id`` has shape ``[B, L]``.

    NOTE: this is the same for every layer in a forward pass, so build it once and
    reuse it rather than per-layer. Wrapping this call in ``torch.compile`` makes the
    mask construction itself fast.
    """
    assert create_block_mask is not None
    B, L = seq_id.shape

    def mask_mod(b, h, q_idx, kv_idx):
        return seq_id[b, q_idx] == seq_id[b, kv_idx]

    # H=None broadcasts the mask across heads.
    return create_block_mask(mask_mod, B, None, L, L, device=seq_id.device)


class FlexMultiHeadAttention(MultiHeadAttention):
    """Multi-head attention built on PyTorch FlexAttention.

    Numerically equivalent to ``MultiHeadAttention``'s masked SDPA path, but routes
    the attention core through ``flex_attention`` so one implementation runs on CUDA
    and Intel XPU (Aurora) without the CUDA-only ``flash_attn``/Triton dependency.

    Unlike ``FlashMultiHeadAttention``, rotary is applied *before* attention using the
    base class's pure-torch ``RotaryEmbedding`` (FlexAttention handles only the
    attention core), so this path also drops ``flash_attn.ops.triton.rotary``.

    For performance, wrap the enclosing model in ``torch.compile`` (optionally with
    ``mode="max-autotune"``); eager FlexAttention is correct but slow.
    """

    def __init__(
        self, d_model: int, n_heads: int, bias: bool = False, qk_layernorm: bool = True
    ):
        super().__init__(
            d_model=d_model, n_heads=n_heads, bias=bias, qk_layernorm=qk_layernorm
        )
        assert flex_attention is not None, (
            "FlexMultiHeadAttention requires torch>=2.5 (Intel XPU support needs "
            "torch>=2.9). torch.nn.attention.flex_attention could not be imported."
        )

    def forward(
        self, x, seq_id, output_attentions: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if output_attentions:
            raise ValueError(
                "FlexMultiHeadAttention does not support output_attentions=True. "
                "Use MultiHeadAttention instead."
            )

        qkv_BLD3 = self.layernorm_qkv(x)
        query_BLD, key_BLD, value_BLD = torch.chunk(qkv_BLD3, 3, dim=-1)
        query_BLD, key_BLD = (
            self.q_ln(query_BLD).to(query_BLD.dtype),
            self.k_ln(key_BLD).to(query_BLD.dtype),
        )
        query_BLD, key_BLD = self._apply_rotary(query_BLD, key_BLD)

        reshaper = functools.partial(
            einops.rearrange, pattern="b s (h d) -> b h s d", h=self.n_heads
        )
        query_BHLD, key_BHLD, value_BHLD = map(
            reshaper, (query_BLD, key_BLD, value_BLD)
        )

        block_mask = build_sequence_block_mask(seq_id) if seq_id is not None else None

        # flex_attention's default softmax scale is d_head**-0.5, matching SDPA.
        context_BHLD = flex_attention(
            query_BHLD, key_BHLD, value_BHLD, block_mask=block_mask
        )
        context_BLD = einops.rearrange(context_BHLD, "b h s d -> b s (h d)")

        return self.out_proj(context_BLD), None
