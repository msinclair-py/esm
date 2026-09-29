#!/usr/bin/env python
"""ESMFold2 serial-path smoke test for Aurora Intel Max GPUs (XPU).

Validates the autocast-gate fixes in the transformers fork on real weights: with the
hardcoded ``device_type="cuda"`` autocast contexts rewired to ``_device_autocast``, the
folding trunk + diffusion must run in bf16 on XPU (not silently fp32) and produce finite
all-atom coordinates without crashing.

Path exercised: ESMFold2Model.from_pretrained("biohub/ESMFold2") (auto-loads ESMC-6B) →
infer_protein(seq) → forward through FoldingTrunk (autocast @ modeling_esmfold2.py:223),
the LM precision context (:479), the atom embedder (:936), and the shared z-transition /
diffusion-Kabsch autocasts in modeling_esmfold2_common.py.

The Triton fused kernels are gated on tensor.is_cuda, so on XPU the reference (eager) path
runs — slower, but the point here is correctness, not speed. Pin to one tile and keep the
sequence short / loops minimal.

Usage:
    python esmfold2_smoke.py [--seq <AA>] [--length 48] [--loops 1] [--steps 4]
"""

from __future__ import annotations

import argparse
import sys
import time

import torch

_AA = "ACDEFGHIKLMNPQRSTVWY"
_REPO = "biohub/ESMFold2"


def banner(msg: str) -> None:
    print(f"\n=== {msg} ===", flush=True)


def device_report() -> None:
    import os

    banner("environment")
    print("torch:", torch.__version__)
    for k in ("ZE_FLAT_DEVICE_HIERARCHY", "ZE_AFFINITY_MASK"):
        print(f"{k}:", os.environ.get(k, "(unset)"))
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        for i in range(torch.xpu.device_count()):
            p = torch.xpu.get_device_properties(i)
            print(f"  xpu:{i} {p.name}  total_memory={p.total_memory / 1e9:.1f} GB")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", default=None, help="amino-acid sequence (overrides --length)")
    ap.add_argument("--length", type=int, default=48)
    ap.add_argument("--loops", type=int, default=1, help="trunk refinement loops")
    ap.add_argument("--steps", type=int, default=4, help="diffusion ODE steps")
    args = ap.parse_args()

    device_report()

    try:
        from esm.utils.misc import get_default_device
    except ImportError as e:
        print(f"\nERROR: esm not importable ({e}); see run_esmc_smoke.pbs for env setup.")
        return 1
    device = get_default_device()
    banner(f"selected device: {device}")
    if device.type == "cpu":
        print("WARNING: no accelerator detected (login node?).")

    try:
        from transformers.models.esmfold2.modeling_esmfold2 import ESMFold2Model
    except Exception as e:  # noqa: BLE001
        print(f"SKIP: could not import ESMFold2Model ({type(e).__name__}: {e})")
        return 0

    banner(f"loading {_REPO} (+ auto-loading ESMC-6B; ~27 GB first time)")
    t0 = time.time()
    try:
        model = ESMFold2Model.from_pretrained(_REPO).to(device).eval()
    except Exception as e:  # noqa: BLE001
        print(f"SKIP: could not load {_REPO} ({type(e).__name__}: {e})")
        print("      (compute nodes need the ALCF proxy or a pre-cached HF_HOME)")
        return 0
    model.set_kernel_backend(None)  # reference path; fused Triton is CUDA-only anyway
    print(f"loaded in {time.time() - t0:.1f}s")

    seq = args.seq or "".join(_AA[i % len(_AA)] for i in range(args.length))
    banner(f"folding L={len(seq)} (loops={args.loops}, steps={args.steps})")

    if device.type == "xpu":
        torch.xpu.reset_peak_memory_stats()
    t0 = time.time()
    out = model.infer_protein(
        seq,
        num_loops=args.loops,
        num_diffusion_samples=1,
        num_sampling_steps=args.steps,
    )
    if device.type == "xpu":
        torch.xpu.synchronize()
    dt = time.time() - t0

    coords = out["sample_atom_coords"]
    plddt = out.get("plddt")
    finite = torch.isfinite(coords.float()).all().item()
    peak = (
        torch.xpu.max_memory_allocated() / 1e9 if device.type == "xpu" else float("nan")
    )
    print(
        f"OK folded L={len(seq)} in {dt:.1f}s | coords {tuple(coords.shape)} {coords.dtype} "
        f"| finite={finite} | peak {peak:.1f} GB"
    )
    if plddt is not None:
        print(f"   mean pLDDT: {plddt.float().mean().item():.3f}")
    assert finite, "non-finite coordinates!"

    banner("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
