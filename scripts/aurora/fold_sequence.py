#!/usr/bin/env python
"""Single-sequence ESMFold2 folding helper, device-agnostic (XPU / CUDA / CPU).

The "fast path": fold one protein chain from its amino-acid sequence and write a
structure file. No MSA, no ligands — for the single-sequence speedup ESMFold2 is
designed for. For multi-entity complexes or MSA-guided folding, see
scripts/aurora/readme_example.py.

Two output paths:
  * default (mmCIF) goes through ``ESMFold2InputBuilder().fold()`` so you also get
    pLDDT / pTM confidence metrics.
  * ``--pdb`` uses the lighter ``model.infer_protein_as_pdb()`` fast path (no pTM).

Aurora notes (mirrors the sibling scripts):
  * Device via ``esm.utils.misc.get_default_device()`` (xpu → cuda → cpu), never a
    hardcoded ``.cuda()``.
  * On XPU/CPU the fused Triton kernels are CUDA-only, so we force the eager
    reference path with ``model.set_kernel_backend(None)``.
  * ESMFold2Model lives in the transformers *fork*; put ``$REPO/transformers/src``
    first on PYTHONPATH (see scripts/aurora/run_esmc_smoke.pbs for the full env).

Usage:
    python fold_sequence.py --seq MKTAYIAKQR... [--out pred.cif]
    python fold_sequence.py --fasta target.fasta --loops 20 --steps 100
    python fold_sequence.py --length 64 --pdb --out pred.pdb   # random AA, fast path
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import torch


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


def load_sequence(args: argparse.Namespace) -> str:
    if args.seq:
        return "".join(args.seq.split()).upper()
    if args.fasta:
        from esm.utils.parsing import read_sequences

        entries = list(read_sequences(args.fasta))
        if not entries:
            raise SystemExit(f"no sequences found in {args.fasta}")
        if len(entries) > 1:
            print(f"NOTE: {len(entries)} records in {args.fasta}; folding the first.")
        return entries[0].sequence.upper()
    # Deterministic pseudo-random sequence (Date/random are fine in a plain script).
    aa = "ACDEFGHIKLMNPQRSTVWY"
    return "".join(aa[i % len(aa)] for i in range(args.length))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    src = ap.add_mutually_exclusive_group()
    src.add_argument("--seq", default=None, help="amino-acid sequence to fold")
    src.add_argument("--fasta", default=None, help="FASTA file; folds the first record")
    ap.add_argument("--length", type=int, default=48, help="random-sequence length if no --seq/--fasta")
    ap.add_argument("--loops", type=int, default=20, help="trunk refinement loops")
    ap.add_argument("--steps", type=int, default=100, help="diffusion sampling steps")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--chunk-size", type=int, default=None, help="folding-trunk chunk size; lower (32/16) cuts peak memory")
    ap.add_argument("--pdb", action="store_true", help="use the infer_protein_as_pdb fast path (PDB, no pTM)")
    ap.add_argument("--out", default=None, help="output file (default pred.cif, or pred.pdb with --pdb)")
    args = ap.parse_args()

    out_path = args.out or ("pred.pdb" if args.pdb else "pred.cif")

    device_report()

    from esm.utils.misc import get_default_device

    device = get_default_device()
    banner(f"selected device: {device}")
    if device.type == "cpu":
        print("WARNING: no accelerator detected (login node?); this will be very slow.")

    # ESMFold2Model lives in the transformers *fork*; import lazily so the device
    # report still prints if the fork isn't on PYTHONPATH yet.
    try:
        from transformers.models.esmfold2.modeling_esmfold2 import ESMFold2Model
    except Exception as e:  # noqa: BLE001
        print(f"ERROR: could not import ESMFold2Model ({type(e).__name__}: {e})")
        print("       Put $REPO/transformers/src first on PYTHONPATH (the fork).")
        return 1

    seq = load_sequence(args)
    banner(f"loading biohub/ESMFold2 (+ auto-loading ESMC-6B; ~27 GB first time)")
    t0 = time.time()
    model = ESMFold2Model.from_pretrained("biohub/ESMFold2").to(device).eval()
    if device.type != "cuda":
        model.set_kernel_backend(None)  # fused Triton kernels are CUDA-only
    if args.chunk_size is not None:
        model.set_chunk_size(args.chunk_size)
    print(f"loaded in {time.time() - t0:.1f}s")

    banner(f"folding L={len(seq)} (loops={args.loops}, steps={args.steps}, seed={args.seed})")
    if device.type == "xpu":
        torch.xpu.reset_peak_memory_stats()
    t0 = time.time()

    if args.pdb:
        # Fast path: PDB string straight from the model, no confidence metrics.
        pdb = model.infer_protein_as_pdb(
            seq,
            num_loops=args.loops,
            num_sampling_steps=args.steps,
            num_diffusion_samples=1,
        )
        result_str = pdb
        plddt = ptm = iptm = None
    else:
        from esm.models.esmfold2 import (
            ESMFold2InputBuilder,
            ProteinInput,
            StructurePredictionInput,
        )

        spi = StructurePredictionInput(sequences=[ProteinInput(id="A", sequence=seq)])
        result = ESMFold2InputBuilder().fold(
            model,
            spi,
            num_loops=args.loops,
            num_sampling_steps=args.steps,
            num_diffusion_samples=1,
            seed=args.seed,
        )
        result_str = result.complex.to_mmcif()
        plddt = float(result.plddt.mean()) if result.plddt is not None else None
        ptm, iptm = result.ptm, result.iptm

    if device.type == "xpu":
        torch.xpu.synchronize()
    dt = time.time() - t0
    peak = torch.xpu.max_memory_allocated() / 1e9 if device.type == "xpu" else float("nan")

    metrics = f"pLDDT {plddt:.3f}" if plddt is not None else "pLDDT n/a (--pdb fast path)"
    if ptm is not None:
        metrics += f" | pTM {ptm:.3f}"
    print(f"OK folded L={len(seq)} in {dt:.1f}s | {metrics} | peak {peak:.1f} GB")

    with open(out_path, "w") as f:
        f.write(result_str)
    print(f"wrote {out_path}")

    banner("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
