"""Parity tests: FlexMultiHeadAttention vs. the reference masked-SDPA path.

FlexAttention is the device-agnostic replacement for flash_attn (CUDA-only). It runs
on CUDA and, as of PyTorch 2.9, on Intel XPU (Aurora). These tests assert that the
flex path is numerically equivalent to MultiHeadAttention given identical weights and
inputs, so it is a safe substitute for the flash/SDPA path.

The tests skip gracefully when FlexAttention is unavailable or unsupported on the
device at hand (e.g. older torch, or a CPU-only lint env without flex CPU support).
"""

import pytest
import torch

from esm.layers.attention import (
    FlexMultiHeadAttention,
    MultiHeadAttention,
    flex_attention,
)

B, L = 2, 16
D_MODEL = 64
N_HEADS = 4


def _pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch, "xpu", None) is not None and torch.xpu.is_available():
        return torch.device("xpu")
    return torch.device("cpu")


def _flex_runs_on(device: torch.device) -> bool:
    """Probe whether flex_attention actually executes on this device/torch build."""
    if flex_attention is None:
        return False
    try:
        q = torch.randn(1, 1, L, 16, device=device)
        flex_attention(q, q, q)
        return True
    except Exception:
        return False


requires_flex = pytest.mark.skipif(
    not _flex_runs_on(_pick_device()),
    reason="FlexAttention is not available/usable on this device.",
)


def _make_pair(device: torch.device, qk_layernorm: bool):
    """A flex module and a reference module sharing identical weights."""
    torch.manual_seed(0)
    ref = MultiHeadAttention(
        D_MODEL, N_HEADS, qk_layernorm=qk_layernorm
    ).to(device).eval()
    flex = FlexMultiHeadAttention(
        D_MODEL, N_HEADS, qk_layernorm=qk_layernorm
    ).to(device).eval()
    # Both subclass MultiHeadAttention and use the same pure-torch RotaryEmbedding,
    # so their parameter layouts match exactly.
    flex.load_state_dict(ref.state_dict())
    return flex, ref


@requires_flex
@pytest.mark.parametrize("qk_layernorm", [True, False])
def test_flex_matches_sdpa_no_mask(qk_layernorm):
    device = _pick_device()
    flex, ref = _make_pair(device, qk_layernorm)
    x = torch.randn(B, L, D_MODEL, device=device)

    with torch.no_grad():
        out_flex, _ = flex(x, seq_id=None)
        out_ref, _ = ref(x, seq_id=None)

    torch.testing.assert_close(out_flex, out_ref, atol=1e-4, rtol=1e-4)


@requires_flex
def test_flex_matches_sdpa_with_sequence_packing():
    """Block-diagonal mask from packed sequence ids must match the SDPA mask path."""
    device = _pick_device()
    flex, ref = _make_pair(device, qk_layernorm=True)
    x = torch.randn(B, L, D_MODEL, device=device)
    # Two packed documents per row (ids 0 and 1): tokens only attend within their doc.
    seq_id = torch.zeros(B, L, dtype=torch.long, device=device)
    seq_id[:, L // 2 :] = 1

    with torch.no_grad():
        out_flex, _ = flex(x, seq_id=seq_id)
        out_ref, _ = ref(x, seq_id=seq_id)

    torch.testing.assert_close(out_flex, out_ref, atol=1e-4, rtol=1e-4)


@requires_flex
def test_flex_matches_sdpa_with_bool_padding_mask():
    """A boolean real-vs-pad mask (as ESMC builds) must match the SDPA path."""
    device = _pick_device()
    flex, ref = _make_pair(device, qk_layernorm=True)
    x = torch.randn(B, L, D_MODEL, device=device)
    seq_id = torch.ones(B, L, dtype=torch.bool, device=device)
    seq_id[:, -3:] = False  # last 3 positions are padding

    with torch.no_grad():
        out_flex, _ = flex(x, seq_id=seq_id)
        out_ref, _ = ref(x, seq_id=seq_id)

    # Only the non-padding positions are consumed downstream; compare those.
    torch.testing.assert_close(
        out_flex[:, :-3], out_ref[:, :-3], atol=1e-4, rtol=1e-4
    )


@requires_flex
def test_flex_rejects_output_attentions():
    device = _pick_device()
    flex, _ = _make_pair(device, qk_layernorm=True)
    x = torch.randn(B, L, D_MODEL, device=device)
    with pytest.raises(ValueError, match="output_attentions"):
        flex(x, seq_id=None, output_attentions=True)
