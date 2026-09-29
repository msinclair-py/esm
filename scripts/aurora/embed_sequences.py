#!/usr/bin/env python
"""Embed protein sequences with ESMC on Aurora's Intel Max GPUs (XPU).

A worked "how to get embeddings out of ESMC" example that also scales to large
FASTA files. The core of a single forward pass is the standard SDK dance:

    protein = ESMProtein(sequence=...)          # wrap the raw string
    tensor  = model.encode(protein)             # str -> token ids (adds BOS/EOS)
    out     = model.logits(                      # forward pass
        tensor, LogitsConfig(return_embeddings=True)
    )
    emb = out.embeddings                         # [1, L+2, d_model], bf16

``out.embeddings`` is the last hidden layer, shaped [B, L+2, d_model] because the
tokenizer wraps each sequence with BOS + EOS. For a per-protein vector we take a
masked mean over the residue axis, excluding BOS/EOS/pad tokens.

For throughput this script batches: sequences are sorted by length and packed
into padded batches (``--batch-size`` sequences, capped by ``--max-tokens`` so a
few long sequences can't blow up O(L) memory). The model derives an attention
mask from the pad token, so padded positions don't leak into the result — the
batched mean matches the one-at-a-time mean to bf16 noise (~1e-3).

Two output modes:
  * default (in-memory): writes a single ``.npz`` with a ``pooled`` [N, d_model]
    matrix and ``ids``; add ``--save-residues`` to also store per-residue arrays
    (only sensible for small N).
  * ``--shard-dir DIR`` (recommended for large N): streams pooled vectors to
    ``DIR/pooled_*.npz`` shards as it goes, and on restart skips any id already
    present in a shard. Crash-safe and resumable. ``--merge`` consolidates the
    shards into ``DIR/pooled_all.npz`` at the end.

Device handling matches the sibling Aurora scripts: ``get_default_device()``
picks xpu -> cuda -> cpu, weights cast to bf16 off-CPU, non-flash SDPA attention
(``use_flash_attn=False``) since flash-attn isn't built for XPU. Run with the
same env as the smoke scripts (see scripts/aurora/run_esmc_smoke.pbs): load
`frameworks`, the venv, and a populated HF_HOME.

Usage:
    # a couple of inline sequences -> embeddings.npz
    python embed_sequences.py --seq MKTAYIAKQR --seq GAVLIPFWMS

    # a big FASTA, pooled vectors, restartable shards, then merge
    python embed_sequences.py --fasta proteins.fasta --model esmc_600m \
        --batch-size 32 --shard-dir proteins_emb --merge
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
import time

import numpy as np
import torch


def banner(msg: str) -> None:
    print(f"\n=== {msg} ===", flush=True)


def sync(device: torch.device) -> None:
    if device.type == "xpu":
        torch.xpu.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()


# --------------------------------------------------------------------------- IO


def read_fasta(path: str) -> list[tuple[str, str]]:
    """Minimal FASTA reader -> list of (id, sequence). No external deps."""
    records: list[tuple[str, str]] = []
    name: str | None = None
    chunks: list[str] = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if name is not None:
                    records.append((name, "".join(chunks)))
                name = line[1:].split()[0] or f"seq{len(records)}"
                chunks = []
            else:
                chunks.append(line)
    if name is not None:
        records.append((name, "".join(chunks)))
    return records


def collect_sequences(args: argparse.Namespace) -> list[tuple[str, str]]:
    records: list[tuple[str, str]] = []
    if args.fasta:
        records.extend(read_fasta(args.fasta))
    for i, s in enumerate(args.seq or []):
        records.append((f"seq{i}", s.strip().upper()))
    # Drop empties and de-duplicate ids (first wins), warning on collisions.
    seen: set[str] = set()
    out: list[tuple[str, str]] = []
    for name, seq in records:
        if not seq:
            print(f"  SKIP {name}: empty sequence")
            continue
        if name in seen:
            print(f"  WARNING: duplicate id {name!r}; keeping first occurrence")
            continue
        seen.add(name)
        out.append((name, seq))
    return out


def done_ids_from_shards(shard_dir: str) -> set[str]:
    """Ids already embedded in existing shards (for resumable runs)."""
    done: set[str] = set()
    for path in sorted(glob.glob(os.path.join(shard_dir, "pooled_[0-9]*.npz"))):
        with np.load(path, allow_pickle=False) as z:
            done.update(z["ids"].tolist())
    return done


def next_shard_index(shard_dir: str) -> int:
    existing = glob.glob(os.path.join(shard_dir, "pooled_[0-9]*.npz"))
    return len(existing)


# ---------------------------------------------------------------------- batching


def make_batches(
    records: list[tuple[str, str]], batch_size: int, max_tokens: int
) -> list[list[tuple[str, str]]]:
    """Length-sorted, padding-minimizing batches.

    Sorting by length means each padded batch wastes little on padding. A batch
    is flushed at ``batch_size`` sequences, or earlier if adding the next one
    would push padded tokens (n * longest) past ``max_tokens``.
    """
    ordered = sorted(records, key=lambda r: len(r[1]))
    batches: list[list[tuple[str, str]]] = []
    cur: list[tuple[str, str]] = []
    for rec in ordered:
        longest = max(len(rec[1]), max((len(r[1]) for r in cur), default=0))
        if cur and (
            len(cur) >= batch_size or (len(cur) + 1) * longest > max_tokens
        ):
            batches.append(cur)
            cur = []
        cur.append(rec)
    if cur:
        batches.append(cur)
    return batches


def residue_mask(tokens: torch.Tensor, tok) -> torch.Tensor:
    """Boolean [B, L] mask of real amino acids (excludes pad, BOS/cls, EOS)."""
    mask = torch.ones_like(tokens, dtype=torch.bool)
    for special in (tok.pad_token_id, tok.cls_token_id, tok.eos_token_id):
        if special is not None:
            mask &= tokens != special
    return mask


def embed_batch(
    model, seqs: list[str], device: torch.device
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """Forward a padded batch -> (pooled [B, d_model], list of per-residue [L_i, d])."""
    from esm.utils.misc import autocast_context

    tokens = model._tokenize(seqs)  # [B, Lmax], padded, on model device
    with torch.no_grad(), autocast_context(device):
        out = model(sequence_tokens=tokens)  # pad-derived attention mask inside
    sync(device)

    emb = out.embeddings  # [B, Lmax, d_model]
    mask = residue_mask(tokens, model.tokenizer)  # [B, Lmax]
    mf = mask.unsqueeze(-1).to(emb.dtype)
    pooled = (emb * mf).sum(1) / mf.sum(1).clamp(min=1)  # [B, d_model]

    residues = [emb[i, mask[i]].float().cpu() for i in range(emb.shape[0])]
    return pooled.float().cpu(), residues


# --------------------------------------------------------------------------- run


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seq", action="append", help="inline sequence (repeatable)")
    ap.add_argument("--fasta", help="FASTA file of sequences to embed")
    ap.add_argument("--model", default="esmc_300m", help="esmc_300m | esmc_600m")
    ap.add_argument("--batch-size", type=int, default=16, help="sequences per batch")
    ap.add_argument(
        "--max-tokens",
        type=int,
        default=16384,
        help="cap on padded tokens (batch_size * longest) to bound memory",
    )
    ap.add_argument("--out", default="embeddings.npz", help="output .npz (in-memory mode)")
    ap.add_argument(
        "--shard-dir",
        help="stream pooled vectors to resumable shards here (recommended for large N)",
    )
    ap.add_argument(
        "--flush-every",
        type=int,
        default=2048,
        help="shard-dir mode: write a shard after this many new sequences",
    )
    ap.add_argument(
        "--merge",
        action="store_true",
        help="shard-dir mode: also write a consolidated pooled_all.npz at the end",
    )
    ap.add_argument(
        "--save-residues",
        action="store_true",
        help="in-memory mode: also store per-residue arrays (small N only)",
    )
    args = ap.parse_args()

    records = collect_sequences(args)
    if not records:
        ap.error("no sequences given; use --seq SEQ (repeatable) and/or --fasta FILE")

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

    # Resumable shard mode: skip ids already on disk before touching the GPU.
    sharded = args.shard_dir is not None
    if sharded:
        os.makedirs(args.shard_dir, exist_ok=True)
        if args.save_residues:
            print("NOTE: --save-residues is ignored in --shard-dir mode (pooled only)")
        done = done_ids_from_shards(args.shard_dir)
        if done:
            before = len(records)
            records = [(n, s) for n, s in records if n not in done]
            print(f"resuming: {len(done)} already embedded, {before - len(records)} skipped")
        if not records:
            print("nothing left to embed; all ids already present in shards.")
            if args.merge:
                merge_shards(args.shard_dir)
            return 0

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

    batches = make_batches(records, args.batch_size, args.max_tokens)
    banner(f"embedding {len(records)} sequence(s) in {len(batches)} batch(es)")

    # Accumulators (in-memory mode) / flush buffers (shard mode).
    all_ids: list[str] = []
    all_pooled: list[np.ndarray] = []
    residues: dict[str, np.ndarray] = {}
    buf_ids: list[str] = []
    buf_pooled: list[np.ndarray] = []
    shard_i = next_shard_index(args.shard_dir) if sharded else 0
    n_done = 0
    t0 = time.time()

    def flush_shard() -> None:
        nonlocal shard_i, buf_ids, buf_pooled
        if not buf_ids:
            return
        path = os.path.join(args.shard_dir, f"pooled_{shard_i:05d}.npz")
        np.savez(path, ids=np.array(buf_ids), pooled=np.stack(buf_pooled))
        print(f"  wrote {path} ({len(buf_ids)} seqs)")
        shard_i += 1
        buf_ids, buf_pooled = [], []

    for bi, batch in enumerate(batches):
        names = [n for n, _ in batch]
        seqs = [s for _, s in batch]
        pooled, res = embed_batch(model, seqs, device)
        for j, name in enumerate(names):
            vec = pooled[j].numpy()
            if sharded:
                buf_ids.append(name)
                buf_pooled.append(vec)
            else:
                all_ids.append(name)
                all_pooled.append(vec)
                if args.save_residues:
                    residues[f"residues/{name}"] = res[j].numpy()
        n_done += len(batch)
        if sharded and len(buf_ids) >= args.flush_every:
            flush_shard()
        if bi % 20 == 0 or bi == len(batches) - 1:
            rate = n_done / max(time.time() - t0, 1e-6)
            print(
                f"  batch {bi + 1}/{len(batches)}  "
                f"Lmax={max(len(s) for s in seqs)}  {n_done}/{len(records)} done  "
                f"({rate:.1f} seq/s)",
                flush=True,
            )

    dt = time.time() - t0

    if sharded:
        flush_shard()
        banner("done")
        print(f"embedded {n_done} new sequence(s) in {dt:.1f}s -> {args.shard_dir}/")
        if args.merge:
            merge_shards(args.shard_dir)
        return 0

    payload: dict[str, np.ndarray] = {
        "ids": np.array(all_ids),
        "pooled": np.stack(all_pooled),  # [N, d_model]
    }
    payload.update(residues)
    np.savez(args.out, **payload)
    banner("done")
    print(
        f"embedded {len(all_ids)} sequence(s) in {dt:.1f}s "
        f"({len(all_ids) / max(dt, 1e-6):.1f} seq/s); "
        f"pooled {payload['pooled'].shape} -> {args.out}"
    )
    return 0


def merge_shards(shard_dir: str) -> None:
    """Concatenate all shards into pooled_all.npz (last value wins on id clash)."""
    ids: list[str] = []
    mats: list[np.ndarray] = []
    for path in sorted(glob.glob(os.path.join(shard_dir, "pooled_[0-9]*.npz"))):
        with np.load(path, allow_pickle=False) as z:
            ids.extend(z["ids"].tolist())
            mats.append(z["pooled"])
    if not mats:
        print("merge: no shards found.")
        return
    pooled = np.concatenate(mats, axis=0)
    # De-dup ids keeping the last occurrence.
    keep: dict[str, int] = {name: i for i, name in enumerate(ids)}
    order = sorted(keep.values())
    out_ids = np.array([ids[i] for i in order])
    out_mat = pooled[order]
    out = os.path.join(shard_dir, "pooled_all.npz")
    np.savez(out, ids=out_ids, pooled=out_mat)
    print(f"merged {len(out_ids)} unique sequence(s) -> {out} {out_mat.shape}")


if __name__ == "__main__":
    sys.exit(main())
