#!/usr/bin/env python
"""ESMFold2 README example, adapted to run on Aurora's Intel Max GPUs (XPU).

This is the upstream README multi-entity example (a methyltransferase + two DNA
strands with a modified base + a SAH ligand → 1mht), changed only where the
stock example assumes NVIDIA CUDA:

  * Device selection goes through ``esm.utils.misc.get_default_device()``
    (xpu → cuda → cpu) instead of a hardcoded ``.cuda()``. The bf16 autocast
    inside the fork is already device-aware (see the autocast-gate fixes in
    TODO.md), so no forward-path change is needed.
  * The fused Triton kernels are CUDA-only (gated on ``tensor.is_cuda``), so on
    XPU we explicitly select the eager reference path via
    ``model.set_kernel_backend(None)`` — correct, just unfused/slower.
  * Adds an environment + peak-memory report, matching the sibling smoke scripts.

Run on a compute node (see scripts/aurora/run_esmc_smoke.pbs for the env: load
`frameworks`, the venv, and put `$REPO/transformers/src` first on PYTHONPATH so
the ESMFold2 *fork* is importable — stock transformers lacks `models.esmfold2`).

Usage:
    python readme_example.py [--loops 20] [--steps 100] [--chunk-size N] [--out FILE]
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import torch

from esm.models.esmfold2 import (
    DNAInput,
    ESMFold2InputBuilder,
    LigandInput,
    Modification,
    ProteinInput,
    StructurePredictionInput,
)
from esm.utils.misc import get_default_device

HHAI_SEQ = (
    "MIEIKDKQLTGLRFIDLFAGLGGFRLALESCGAECVYSNEWDKYAQEVYEMNFGEKPEGDITQVNEKTIPDH"
    "DILCAGFPCQAFSISGKQKGFEDSRGTLFFDIARIVREKKPKVVFMENVKNFASHDNGNTLEVVKNTMNELD"
    "YSFHAKVLNALDYGIPQKRERIYMICFRNDLNIQNFQFPKPFELNTFVKDLLLPDSEVEHLVIDRKDLVMTN"
    "QEIEQTTPKTVRLGIVGKGGQGERIYSTRGIAITLSAYGGGIFAKTGGYLVNGKTRKLHPRECARVMGYPDS"
    "YKVHPSTSQAYKQFGNSVVINVLQYIAYNIGSSLNFKPY"
)


def banner(msg: str) -> None:
    print(f"\n=== {msg} ===", flush=True)


def device_report() -> None:
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
    ap.add_argument("--loops", type=int, default=20, help="trunk refinement loops")
    ap.add_argument("--steps", type=int, default=100, help="diffusion sampling steps")
    ap.add_argument(
        "--chunk-size",
        type=int,
        default=None,
        help="folding-trunk chunk size; lower (32/16) cuts O(L^2) peak memory on a "
        "single 64 GB tile for large complexes (default: model's own setting)",
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="1mht_pred.cif")
    args = ap.parse_args()

    device_report()

    device = get_default_device()
    banner(f"selected device: {device}")
    if device.type == "cpu":
        print("WARNING: no accelerator detected (login node?); this will be very slow.")

    # The ESMFold2 model lives in the transformers *fork* (stock transformers has
    # no models.esmfold2); import it lazily so the device report still prints if
    # the fork isn't on PYTHONPATH yet.
    try:
        from transformers.models.esmfold2.modeling_esmfold2 import ESMFold2Model
    except Exception as e:  # noqa: BLE001
        print(f"ERROR: could not import ESMFold2Model ({type(e).__name__}: {e})")
        print("       Put $REPO/transformers/src first on PYTHONPATH (the fork).")
        return 1

    banner("loading biohub/ESMFold2 (+ auto-loading ESMC-6B; ~27 GB first time)")
    t0 = time.time()
    model = ESMFold2Model.from_pretrained("biohub/ESMFold2").to(device).eval()
    # Fused Triton kernels are CUDA-only; force the eager reference path on XPU.
    model.set_kernel_backend(None)
    if args.chunk_size is not None:
        model.set_chunk_size(args.chunk_size)
    print(f"loaded in {time.time() - t0:.1f}s")

    spi = StructurePredictionInput(
        sequences=[
            ProteinInput(id="A", sequence=HHAI_SEQ),
            DNAInput(
                id="B",
                sequence="GATAGCGCTATC",
                modifications=[Modification(position=5, ccd="C36")],
            ),
            DNAInput(
                id="C",
                sequence="TGATAGCGCTATC",
                modifications=[Modification(position=6, ccd="C36")],
            ),
            LigandInput(id="L", ccd=["SAH"]),
        ]
    )

    banner(f"folding (loops={args.loops}, steps={args.steps}, seed={args.seed})")
    if device.type == "xpu":
        torch.xpu.reset_peak_memory_stats()
    t0 = time.time()
    result = ESMFold2InputBuilder().fold(
        model,
        spi,
        num_loops=args.loops,
        num_sampling_steps=args.steps,
        num_diffusion_samples=1,
        seed=args.seed,
    )
    if device.type == "xpu":
        torch.xpu.synchronize()
    dt = time.time() - t0

    peak = (
        torch.xpu.max_memory_allocated() / 1e9 if device.type == "xpu" else float("nan")
    )
    print(
        f"OK folded in {dt:.1f}s | pLDDT mean: {float(result.plddt.mean()):.3f}, "
        f"pTM: {float(result.ptm):.3f}, ipTM: {float(result.iptm):.3f} | peak {peak:.1f} GB"
    )

    with open(args.out, "w") as f:
        f.write(result.complex.to_mmcif())
    print(f"wrote {args.out}")

    banner("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
