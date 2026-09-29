#!/usr/bin/env python
"""Embed the binder sequences in a results CSV with ESMC, grouped by system.

Reads a CSV (default ``.results.csv``), groups rows by the ``system`` column and
writes one ``.npz`` per unique system into ``--out-dir``. Each ``.npz`` holds:

    ids    : [N]            binder_id for each row (np.str_)
    pooled : [N, d_model]   masked-mean ESMC embedding (float32)

The model is loaded once and reused across all systems. The actual forward pass,
length-sorted batching and masked mean-pooling are reused from the sibling
``embed_sequences.py`` so there's a single embedding code path.

Device/env handling matches the other Aurora scripts (xpu -> cuda -> cpu, bf16
off-CPU, non-flash SDPA). Run under the same env as the smoke scripts: load
`frameworks`, activate the venv, and have a populated HF_HOME.

Usage:
    # default: .results.csv -> results_embeddings/{nipah,langya,mevh}.npz
    python embed_results_csv.py

    # explicit columns / model / output
    python embed_results_csv.py --csv .results.csv --model esmc_300m \
        --system-col system --id-col binder_id --seq-col binder_seq \
        --out-dir results_embeddings --batch-size 32
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time

import numpy as np

# Reuse the embedding machinery from the sibling script.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from embed_sequences import banner, embed_batch, make_batches  # noqa: E402


def read_grouped(
    path: str, system_col: str, id_col: str, seq_col: str
) -> dict[str, list[tuple[str, str]]]:
    """CSV -> {system: [(id, sequence), ...]} with empties dropped, ids deduped.

    Within a system, the first occurrence of a duplicate id wins (a warning is
    printed); empty/blank sequences are skipped.
    """
    groups: dict[str, list[tuple[str, str]]] = {}
    seen: dict[str, set[str]] = {}
    with open(path, newline="") as fh:
        reader = csv.DictReader(fh)
        for col in (system_col, id_col, seq_col):
            if col not in (reader.fieldnames or []):
                raise SystemExit(
                    f"ERROR: column {col!r} not in {path} "
                    f"(have: {reader.fieldnames})"
                )
        for row in reader:
            system = (row[system_col] or "").strip()
            name = (row[id_col] or "").strip()
            seq = (row[seq_col] or "").strip().upper()
            if not system or not name or not seq:
                continue
            grp = groups.setdefault(system, [])
            s = seen.setdefault(system, set())
            if name in s:
                print(f"  WARNING: duplicate id {name!r} in {system!r}; keeping first")
                continue
            s.add(name)
            grp.append((name, seq))
    return groups


def safe_filename(name: str) -> str:
    """Make a system label safe to use as a filename stem."""
    return "".join(c if (c.isalnum() or c in "-_.") else "_" for c in name)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", default=".results.csv", help="input CSV (default .results.csv)")
    ap.add_argument("--system-col", default="system", help="column to group by")
    ap.add_argument("--id-col", default="binder_id", help="column used as the embedding id")
    ap.add_argument("--seq-col", default="binder_seq", help="column with the sequence")
    ap.add_argument("--model", default="esmc_300m", help="esmc_300m | esmc_600m")
    ap.add_argument("--out-dir", default="results_embeddings", help="output directory")
    ap.add_argument("--batch-size", type=int, default=32, help="sequences per batch")
    ap.add_argument(
        "--max-tokens",
        type=int,
        default=16384,
        help="cap on padded tokens (batch_size * longest) to bound memory",
    )
    ap.add_argument(
        "--overwrite",
        action="store_true",
        help="re-embed systems whose .npz already exists (default: skip them)",
    )
    args = ap.parse_args()

    if not os.path.exists(args.csv):
        ap.error(f"CSV not found: {args.csv}")

    groups = read_grouped(args.csv, args.system_col, args.id_col, args.seq_col)
    if not groups:
        ap.error("no (system, id, sequence) rows found in CSV")

    os.makedirs(args.out_dir, exist_ok=True)
    banner(f"{len(groups)} system(s) from {args.csv}")
    for system, recs in sorted(groups.items()):
        print(f"  {system}: {len(recs)} sequence(s)")

    try:
        from esm.models.esmc import ESMC
        from esm.utils.misc import get_default_device
    except ImportError as e:
        print(
            f"\nERROR: could not import the esm package ({e}).\n"
            "Activate the Aurora env first (see scripts/aurora/run_esmc_smoke.pbs):\n"
            "    module load frameworks\n"
            "    source $REPO/.venv-aurora-uv/bin/activate\n",
            file=sys.stderr,
        )
        return 1

    device = get_default_device()
    banner(f"selected device: {device}")
    if device.type == "cpu":
        print("WARNING: no accelerator detected (login node?); this will be slow.")

    banner(f"loading {args.model}")
    t0 = time.time()
    try:
        model = ESMC.from_pretrained(
            args.model, device=device, use_flash_attn=False
        ).eval()
    except Exception as e:  # noqa: BLE001 - clean message on any load failure
        print(f"ERROR: could not load {args.model} ({type(e).__name__}: {e})")
        print("       Compute nodes need the ALCF proxy or a pre-cached HF_HOME.")
        return 1
    print(f"loaded in {time.time() - t0:.1f}s")

    for system, records in sorted(groups.items()):
        out_path = os.path.join(args.out_dir, f"{safe_filename(system)}.npz")
        if os.path.exists(out_path) and not args.overwrite:
            print(f"\nSKIP {system}: {out_path} exists (use --overwrite)")
            continue

        batches = make_batches(records, args.batch_size, args.max_tokens)
        banner(f"{system}: embedding {len(records)} seq(s) in {len(batches)} batch(es)")
        ids: list[str] = []
        pooled: list[np.ndarray] = []
        t0 = time.time()
        for bi, batch in enumerate(batches):
            names = [n for n, _ in batch]
            seqs = [s for _, s in batch]
            vecs, _ = embed_batch(model, seqs, device)
            for j, name in enumerate(names):
                ids.append(name)
                pooled.append(vecs[j].numpy())
            if bi % 20 == 0 or bi == len(batches) - 1:
                done = len(ids)
                rate = done / max(time.time() - t0, 1e-6)
                print(
                    f"  batch {bi + 1}/{len(batches)}  "
                    f"{done}/{len(records)} done ({rate:.1f} seq/s)",
                    flush=True,
                )

        mat = np.stack(pooled)  # [N, d_model]
        np.savez(out_path, ids=np.array(ids), pooled=mat)
        print(
            f"  -> {out_path}  pooled {mat.shape}  in {time.time() - t0:.1f}s"
        )

    banner("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
