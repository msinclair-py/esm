# TODO — Intel Max GPU (Aurora / XPU) compatibility

Aurora's Intel Data Center GPU Max (Ponte Vecchio) uses the PyTorch `xpu` backend
(oneAPI/SYCL, via Intel Extension for PyTorch). This repo was written for NVIDIA CUDA.
This file tracks the porting work.

## Done

- [x] **Device-agnostic accelerator selection & autocast.** Added `get_default_device()`,
  `is_accelerator()`, and `autocast_context()` to `esm/utils/misc.py`. Rewired
  `esm/models/esmc.py`, `esm/models/esm3.py` (device default + bf16 autocast gating, which
  was hardcoded to `device.type == "cuda"`) and `esm/models/vqvae.py` (replaced the
  deprecated CUDA-namespaced `torch.cuda.amp.autocast` with the device-generic API).
- [x] **`FlexMultiHeadAttention`** (`esm/layers/attention.py`) — device-agnostic
  replacement for the CUDA-only `flash_attn`/Triton path. Uses `flex_attention` +
  `build_sequence_block_mask()` (same-sequence masking, matching the SDPA path) and the
  pure-torch rotary. Wired behind a `use_flex_attn` flag through `UnifiedTransformerBlock`
  (`esm/layers/blocks.py`) and `TransformerStack` (`esm/layers/transformer_stack.py`).
- [x] **Parity test** `tests/oss_pytests/test_flex_attention_parity.py` — asserts flex
  output matches the masked-SDPA reference (no mask / packed sequences / bool padding mask)
  to 1e-4; auto-selects xpu→cuda→cpu and skips if FlexAttention is unusable. Passing on
  Aurora's torch 2.10 stack (validated on the **CPU** path on a login node; exercises XPU
  on a compute node).

## To do

- [ ] **End-to-end `use_flex_attn` hookup.** Add a `use_flex_attn` param to `ESMC.__init__`
  (`esm/models/esmc.py:57`) and the ESM3 / `pretrained.py` model builders, threading it to
  `TransformerStack`. No forward-path change needed: flex uses padded `[B,L,D]` + block mask,
  so `_use_flash_attn` stays `False` and the existing non-flash branch (no `unpad_input`/
  `pad_input`) is correct.
- [ ] **`torch.compile` for performance.** Eager FlexAttention is correct but slow. Wrap the
  model (or the attention) in `torch.compile(mode="max-autotune")` on XPU. Build
  `build_sequence_block_mask()` once per forward pass, not per layer.
- [ ] **flash_attn default behavior on XPU.** The `use_flash_attn=True` defaults degrade
  gracefully via try/except when `flash_attn` is absent, but consider auto-detecting
  (flash importable AND device is cuda) so the default is correct on both backends without
  changing CUDA behavior. For now, pass `use_flash_attn=False` (SDPA fallback) or
  `use_flex_attn=True` on Aurora.
- [ ] **XPU RNG seeding** in `esm/models/esmfold2/processor.py:29-42` — currently uses
  `torch.cuda.{get,set}_rng_state_all` / `manual_seed_all`, guarded by
  `torch.cuda.is_available()`, so ESMFold2 diffusion sampling is unseeded on XPU. Add
  `torch.xpu.*` equivalents (reproducibility only, not a crash).
### ESMFold2 `transformers` fork (audited 2026-06-24)

The fork adds `src/transformers/models/esmfold2/` (trunk + diffusion + a Triton kernel
library + a tensor-parallel `distributed/` layer) and `models/esmc/`. Audited a local
clone at `./transformers`. Design is fallback-friendly: every accelerated path is
`try/except` + runtime-gated, so ESMFold2 **functionally runs on XPU by degrading** — but
every fast path is CUDA-gated (so XPU gets the slow path), and multi-GPU is a hard blocker.

- [x] **bf16 autocast gating fixed (all 7 sites).** Hardcoded `device_type="cuda"` /
  `enabled=*.is_cuda` autocast contexts left the trunk in fp32 on XPU. Added a
  `_device_autocast(device, *, enabled=True, dtype=bf16)` helper to
  `modeling_esmfold2_common.py` (targets the tensor's real device → bf16 on cuda+xpu, no-op
  on cpu; `enabled=False` force-disables with the correct device_type so it actually turns
  off an XPU autocast). Rewired all sites: `modeling_esmfold2.py` (trunk `:223`,
  `_lm_precision_context` `:479` now takes a device, atom embedder `:936`),
  `modeling_esmfold2_common.py` (z-transitions `:1391`, two Kabsch force-fp32 regions
  `:1890/:1907`), `modeling_esmfold2_experimental.py` (atom embedder `:890`). Verified:
  parses clean, no new ruff errors vs. baseline, helper probes to `device_type=xpu,
  enabled=True` on Aurora. TE fp8 path stays NVIDIA-gated (`TE_AVAILABLE`), unaffected.
- [ ] **flash_attn → SDPA fallback already works** (`modeling_esmfold2_common.py:26-41`,
  attn forward `:578-624`, falls back to `F.scaled_dot_product_attention` at `:624`).
  Optionally wire the FlexAttention path here too for speed.
- [ ] **Triton fused kernels never fire on XPU.** `kernels/` (fused pair-bias, dropout-
  residual, LN-Lin-SwiGLU, trimul) are gated by `_fused_active()` requiring `tensor.is_cuda`
  (`modeling_esmfold2_common.py:91`; see also `:2709`). On XPU → unfused eager. To accelerate:
  (a) port kernels to the Triton-XPU backend, AND (b) widen the `is_cuda` gates to
  `is_cuda or is_xpu`. Large effort; do only if eager XPU perf is insufficient.
- [ ] **cuequivariance_torch** (`modeling_esmfold2_common.py:43-55`) — NVIDIA-only optional
  accel backend; gated by `CUE_AVAILABLE`, unused on XPU. No action (won't be selected).
- [x] **Distributed / tensor-parallel = NOT NEEDED for inference** (`distributed/` dir).
  Verified the `distributed/` package is **not on the ESMFold2 inference path**: none of
  `modeling_esmfold2{,_common,_experimental}.py` import it (0 refs each), and the cookbook
  entry point (`ESMFold2ExperimentalModel`) imports only `.configuration_esmfold2` +
  `.modeling_esmfold2_common`. Moreover the package is **un-importable as shipped** — every
  file imports via `from projects.huggingface.transformers.models.esmfold2.distributed...`,
  an internal Biohub monorepo path with no `projects` package in the fork. It's vestigial
  tensor-parallel code, not a blocker. The NCCL-vs-XCCL question is therefore **moot for
  running ESMFold2**. (For the record, if TP is ever wanted: NCCL is NVIDIA-only and
  unavailable on Aurora — `is_nccl_available()==False`, `is_xccl_available()==True` on the
  torch 2.10 build — so the XPU collective backend would be **XCCL**, and `backend_for_device`
  would need `"xpu": "xccl"`. But that path would be a from-scratch build-out regardless.)
- [~] **6B single-GPU memory sizing — analyzed; needs on-node OOM sweep.**
  Aurora PVC Max 1550 = 128 GB HBM2e as **2 tiles × 64 GB**. Env here has
  `ZE_FLAT_DEVICE_HIERARCHY=FLAT` (Aurora default) → **one `xpu` device = one 64 GB tile**,
  not 128 GB. The model is tuned for an 80 GB GPU.
  - **Weights ~15 GB bf16** (ESMC 6B `lm_d_model=2560 × lm_num_layers=80` ≈ 12.8 GB + trunk/
    heads) — fit a 64 GB tile easily; not the bottleneck.
  - **O(L²) folding activations are the limit.** Pair tensor `[L,L,d_pair=256]` (~1.1 GB @
    L=1438, ~2 GB @ L=2000) + trimul/OPM temporaries. Fork comments: default `chunk_size=64`
    "fits L≈2k on 80 GB"; "~76 GB peak at chunk=128, L=1438"; no-chunk OOMs past ~600.
  - **Verdict:** short/medium L (≲~1000) fit one 64 GB tile fine; large complexes (L≳~1400)
    at default chunk=64 exceed 64 GB → OOM. Options: (1) lower `model.set_chunk_size(32/16)`
    (single tile, slower — simplest); (2) `ZE_FLAT_DEVICE_HIERARCHY=COMPOSITE` → 128 GB
    device, but relies on implicit cross-tile scaling (validate on Aurora stack); (3) NOT
    tensor-parallel (the `distributed/` path is dead/un-wired).
  - **TODO:** run an empirical OOM sweep on a compute node — L vs chunk_size vs tile/composite
    — to pin the exact length threshold for single-tile folding.

## On-node validation

- [x] **ESMC validated on Aurora XPU (2026-06-24, debug compute node).** One PVC tile
  reports `total_memory=68.7 GB` (confirms 64 GiB/tile, FLAT mode). Results:
  - Stage 1 tiny ESMC forward: bf16 logits on `xpu`, finite — confirms
    `get_default_device()`→xpu + bf16 autocast + SDPA attention path.
  - Stage 2 real **ESMC_300M**: L=256 forward in 0.72s, emb (1,258,960), finite, peak 0.71 GB.
  - **ESMC_6B** (6.35B params): loads in 8s, bf16, on one tile; L=256 → 1.36s / 13.0 GB,
    L=512 → 0.88s / 13.2 GB. **Comfortably fits a single 64 GB tile** — weights (~12.7 GB
    bf16) dominate, activations tiny (ESMC is a plain LM with SDPA attention; the O(L²)
    pair-tensor memory concern is ESMFold2's folding trunk, not ESMC). 6B headroom is huge.
  - Env setup that worked: uv venv with `include-system-site-packages=true` +
    `pip install -e .` (NOT `uv pip install` — uv ignores system site-packages and would
    install a PyPI CUDA torch + nvidia-* wheels, clobbering the XPU torch; pip sees the
    frameworks torch/transformers-fork/accelerate as already-satisfied and skips them).
  - `ZE_AFFINITY_MASK` must be a plain index (`0`) in FLAT mode — `0.0` selects nothing and
    makes xpu unavailable. PBS script fixed accordingly.
- [x] **esm↔huggingface_hub loader bug fixed (orthogonal to XPU).**
  `ESMC.from_pretrained` → `huggingface_hub.load_torch_model(model, data_root(...))` failed on
  hub >=0.36 (defaults to `safe=True` safetensors + filename pattern, handed the snapshot root,
  while the weights are one pickle `.pth` under `data/weights/`; checkpoint itself valid, 308
  tensors). Would fail identically on CUDA. Fixed in `esm/pretrained.py`: dropped the
  `load_torch_model` import; added `_load_esmc_state_dict(model_key, device)` and made all
  three ESMC builders (300M/600M/6B) `load_state_dict(..., assign=True)` like the ESM3 builders.
  The loader handles **both** repo layouts (they differ by size!):
    - 300M/600M: single pickled `.pth` under `data/weights/` → `torch.load`.
    - 6B: HF sharded safetensors at the snapshot root (`model-0000N-of-00006.safetensors` +
      index) packaged for the transformers `ESMCForMaskedLM` wrapper → load shards, then remap
      keys to esm-native `ESMC` (strip `esmc.` prefix, rename `lm_head.` → `sequence_head.`).
  Uses **strict** `load_state_dict`, which caught the head rename — the old `load_torch_model(...,
  strict=False)` would have silently left the 6B output head randomly initialized. Verified on
  XPU: `from_pretrained('esmc_300m'|'esmc_6b')` both load and run (6B = 6.35B params, peak 13 GB).
  Smoke test uses the plain `from_pretrained` path.
- [ ] **Run the ESMC XPU smoke test on a compute node.** Scripts drafted:
  `scripts/aurora/esmc_smoke.py` (stage 1: tiny ESMC forward, no download, exercises
  get_default_device + bf16 autocast + SDPA path; stage 2: ESMC_300M real-checkpoint forward
  with latency + peak-mem report) and `scripts/aurora/run_esmc_smoke.pbs` (debug queue, 1 node,
  loads `frameworks`, single-tile `ZE_AFFINITY_MASK=0.0`). Validated on a login node: lints
  clean, graceful actionable exit when esm deps absent, PBS `bash -n` OK. **One-time setup
  needed:** the bare `frameworks` module lacks esm's deps — layer a venv
  (`python -m venv --system-site-packages $REPO/.venv-aurora && pip install -e $REPO`) and
  pre-cache HF weights from a login node. Submit: `qsub scripts/aurora/run_esmc_smoke.pbs`.
- [x] **ESMFold2 serial path validated on Aurora XPU (2026-06-24).** `scripts/aurora/esmfold2_smoke.py`:
  `ESMFold2Model.from_pretrained("biohub/ESMFold2")` (auto-loads ESMC-6B) → `infer_protein(seq)`.
  Folded L=48 in 4.4s, coords (1,416,3), **finite**, peak **14.0 GB** on one tile. Confirms the
  autocast-gate fixes work: FoldingTrunk + diffusion + LM precision context ran in bf16 on XPU
  without crashing (flash→SDPA, fused-Triton→eager, cuequivariance→off all fell back). Load was
  slow (230s — reads ~25 GB + CPU→XPU move). Notes:
    - Requires the **fork** `transformers` on PYTHONPATH (`$REPO/transformers/src` first); stock
      transformers 4.57.6 lacks `models.esmfold2`. The ESMC smoke didn't need this (esm-native).
    - ESMFold2 LM repo is `biohub/ESMC-6B` (25 GB), a *separate* repo from `biohub/esmc-6b-2024-12`.
    - mean pLDDT 0.229 on a synthetic poly-AA seq (expected — not a real fold).
  - [x] **Folding quality confirmed on a real protein.** Ubiquitin (76 aa, loops=8, steps=20):
    folded in 6.2s, coords (1,608,3), finite, **mean pLDDT 0.803** — a high-confidence, correct
    fold. The model cleanly separates real (0.80) from synthetic (0.23) input, confirming the
    bf16 autocast numerics on XPU are *correct*, not just finite. This is the gold-standard
    validation of the autocast-gate fixes.
- [ ] **ESMFold2 env in PBS.** Add `$REPO/transformers/src` to PYTHONPATH (and an esmfold2 smoke
  invocation) in `run_esmc_smoke.pbs` or a sibling script, so the fork path is reproducible in batch.
- [ ] **Cosmetic:** `.cuda()` in a docstring example (`modeling_esmfold2.py:7`); SVD
  `driver="gesvd" if H32.is_cuda else None` (`:1754`) falls back to the default driver on XPU.

**Bottom line:** single-XPU ESMFold2 inference should run today via the fallbacks, but in the
slowest mode (SDPA + unfused + fp32). Fixing the autocast gate (item 1) to enable bf16 is the
single biggest XPU win. Fused-kernel and multi-XPU performance require real porting work
(Triton-XPU + XCCL distributed backend).
- [ ] **Packaging.** `pyproject.toml` pins `torch>=2.2.0`; FlexAttention on XPU needs
  **torch >= 2.9** (XPU build / matching IPEX). pixi platforms are `linux-64`/`osx-arm64`
  with no Intel/IPEX deps — the oneAPI PyTorch stack comes from Aurora's modules, not this
  lockfile.
- [ ] **Numerical validation at scale.** Confirm bf16 FlexAttention vs. flash/SDPA logit
  parity on real ESMC checkpoints (300M/600M/6B; all use d_head=64) before trusting the
  flex path in production.
