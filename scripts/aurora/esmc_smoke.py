#!/usr/bin/env python
"""Minimal ESMC smoke test for Aurora Intel Max GPUs (XPU).

Validates the XPU-compat patches end-to-end on a real device:
  - ``get_default_device()`` selects ``xpu`` (esm/utils/misc.py)
  - ``autocast_context()`` enables bf16 autocast on ``xpu`` (not just cuda)
  - the non-flash SDPA attention path runs on XPU (``use_flash_attn=False``)

Two stages:
  1. Tiny architectural smoke — builds a 2-layer ESMC from scratch (no weight
     download) and runs a forward on the selected device. Always runs; this alone
     exercises every patched code path.
  2. Real-checkpoint smoke — loads ESMC_300M (or --model) and folds a synthetic
     sequence of length --length, reporting latency and peak memory. Skipped
     gracefully if weights can't be fetched (compute nodes need the ALCF proxy or a
     pre-populated HF cache; see the PBS script).

Usage:
    python esmc_smoke.py [--model esmc_300m] [--length 256] [--no-real]
"""

from __future__ import annotations

import argparse
import sys
import time

import torch

_AA = "ACDEFGHIKLMNPQRSTVWY"


def banner(msg: str) -> None:
    print(f"\n=== {msg} ===", flush=True)


def sync(device: torch.device) -> None:
    if device.type == "xpu":
        torch.xpu.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()


def reset_peak(device: torch.device) -> None:
    if device.type == "xpu":
        torch.xpu.reset_peak_memory_stats()
    elif device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()


def peak_gb(device: torch.device) -> float:
    if device.type == "xpu":
        return torch.xpu.max_memory_allocated() / 1e9
    if device.type == "cuda":
        return torch.cuda.max_memory_allocated() / 1e9
    return float("nan")


def device_report() -> bool:
    import os

    banner("environment")
    print("torch:", torch.__version__)
    for k in ("ZE_FLAT_DEVICE_HIERARCHY", "ZE_AFFINITY_MASK"):
        print(f"{k}:", os.environ.get(k, "(unset)"))
    has_xpu = hasattr(torch, "xpu") and torch.xpu.is_available()
    print("xpu available:", has_xpu)
    if has_xpu:
        print("xpu device count:", torch.xpu.device_count())
        for i in range(torch.xpu.device_count()):
            p = torch.xpu.get_device_properties(i)
            print(f"  xpu:{i} {p.name}  total_memory={p.total_memory / 1e9:.1f} GB")
    return has_xpu


def tiny_smoke(device: torch.device) -> None:
    banner("stage 1: tiny architectural smoke (no download)")
    from esm.models.esmc import ESMC
    from esm.tokenization.sequence_tokenizer import EsmSequenceTokenizer
    from esm.utils.constants import esm3 as C

    torch.manual_seed(0)
    model = (
        ESMC(
            d_model=64,
            n_heads=4,
            n_layers=2,
            tokenizer=EsmSequenceTokenizer(),
            use_flash_attn=False,  # SDPA fallback path — what runs on XPU today
        )
        .to(device)
        .eval()
    )
    if device.type != "cpu":
        model = model.to(torch.bfloat16)

    tokens = torch.randint(
        C.SEQUENCE_STANDARD_AA_MIN_TOKEN,
        C.SEQUENCE_STANDARD_AA_MAX_TOKEN,
        (1, 32),
        device=device,
    )
    reset_peak(device)
    with torch.no_grad():
        out = model(sequence_tokens=tokens)
    sync(device)

    logits = out.sequence_logits
    assert logits.shape[:2] == (1, 32), logits.shape
    assert torch.isfinite(logits.float()).all(), "non-finite logits!"
    print(
        f"OK tiny forward: logits {tuple(logits.shape)} {logits.dtype} on {device} "
        f"(peak {peak_gb(device):.3f} GB)"
    )


def real_smoke(device: torch.device, model_name: str, length: int) -> None:
    banner(f"stage 2: {model_name} real-checkpoint smoke (L={length})")
    from esm.models.esmc import ESMC
    from esm.sdk.api import ESMProtein, LogitsConfig

    try:
        # Exercises from_pretrained (device select + bf16 cast + the fixed loader).
        model = ESMC.from_pretrained(model_name, device=device).eval()
    except Exception as e:  # noqa: BLE001 - want a clean skip on any load failure
        print(f"SKIP: could not load {model_name} ({type(e).__name__}: {e})")
        print("      (compute nodes need the ALCF proxy or a pre-cached HF_HOME)")
        return

    # Deterministic synthetic sequence of the requested length.
    seq = "".join(_AA[i % len(_AA)] for i in range(length))
    protein = ESMProtein(sequence=seq)

    reset_peak(device)
    t0 = time.time()
    with torch.no_grad():
        tensor = model.encode(protein)
        out = model.logits(
            tensor, LogitsConfig(sequence=True, return_embeddings=True)
        )
    sync(device)
    dt = time.time() - t0

    emb = out.embeddings
    assert emb is not None and torch.isfinite(emb.float()).all(), "non-finite embeddings!"
    print(
        f"OK {model_name}: L={length} forward in {dt:.2f}s, "
        f"emb {tuple(emb.shape)} {emb.dtype}, peak {peak_gb(device):.2f} GB"
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="esmc_300m")
    ap.add_argument("--length", type=int, default=256)
    ap.add_argument("--no-real", action="store_true", help="skip stage 2")
    args = ap.parse_args()

    device_report()

    try:
        # get_default_device() is itself one of the patches under test.
        from esm.utils.misc import get_default_device
    except ImportError as e:
        print(
            f"\nERROR: could not import the esm package ({e}).\n"
            "Install it into an env layered on Aurora's `frameworks` module:\n"
            "    module load frameworks\n"
            "    python -m venv --system-site-packages $REPO/.venv-aurora\n"
            "    source $REPO/.venv-aurora/bin/activate\n"
            "    pip install -e $REPO\n",
            file=sys.stderr,
        )
        return 1

    device = get_default_device()
    banner(f"selected device: {device}")
    if device.type == "cpu":
        print("WARNING: no accelerator detected — running on CPU (login node?).")

    tiny_smoke(device)
    if not args.no_real:
        real_smoke(device, args.model, args.length)

    banner("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
