#!/usr/bin/env bash
#
# Build a uv venv for ESM on Aurora that REUSES the system XPU PyTorch stack.
#
#   module load frameworks          # provides python 3.12 + torch (XPU build)
#   scripts/aurora/setup_uv_venv.sh
#
# Override paths without editing:
#   REPO=/path/to/esm VENV=/path/to/.venv scripts/aurora/setup_uv_venv.sh
#
# Why this is not just `uv pip install -e .`:
#
#   The `frameworks` module ships a torch built against Intel's oneAPI/XPU stack
#   (torch 2.10 + IPEX), plus numpy/scipy/etc. We must NOT let uv replace any of
#   those: a fresh PyPI torch would be the CUDA build (wrong backend, multi-GB of
#   nvidia-* wheels), and a fresh numpy/scipy can break ABI with the system torch.
#
#   uv's resolver does NOT treat --system-site-packages contents as "already
#   installed" -- `uv pip install biotite` happily pulls a second numpy into the
#   venv. So the strategy is:
#     1. venv with --system-site-packages  -> sees system torch/numpy at runtime
#     2. install esm + the transformers FORK with --no-deps  (no resolution)
#     3. resolve the remaining PyPI deps, then install ONLY the ones the system
#        doesn't already provide, with --no-deps and the GPU vendor wheels removed.
#
# Idempotent: re-running recreates the venv from scratch.

set -euo pipefail

REPO="${REPO:-/lus/flare/projects/FRAME-IDP/msinclair/software/esm}"
VENV="${VENV:-$REPO/.venv-aurora-uv}"
FORK="$REPO/transformers"   # local clone of the Biohub transformers fork (has esmfold2/esmc)

command -v uv  >/dev/null || { echo "ERROR: uv not found (module load frameworks?)" >&2; exit 1; }
SYS_PY="$(command -v python3)"

# --- 1. sanity: the system interpreter must carry the XPU torch ----------------
"$SYS_PY" - <<'PY'
import importlib.util, sys
if importlib.util.find_spec("torch") is None:
    sys.exit("ERROR: system python has no torch -- run `module load frameworks` first.")
import torch
xpu = getattr(getattr(torch, "xpu", None), "is_available", lambda: False)()
print(f"  system torch {torch.__version__}  (xpu available: {xpu})")
print(f"  at {torch.__file__}")
PY

# --- 2. fresh venv that can see the system site-packages ----------------------
echo ">> creating venv at $VENV (--system-site-packages)"
rm -rf "$VENV"
uv venv --system-site-packages --python "$SYS_PY" "$VENV"
# shellcheck disable=SC1091
source "$VENV/bin/activate"

# --- 3. the two installs that must bypass dependency resolution ---------------
# transformers FORK first: editable so the venv copy shadows the system's stock
# transformers (4.57.x), which lacks the esmfold2/esmc model code esm needs.
echo ">> installing transformers fork (--no-deps, editable): $FORK"
uv pip install --no-deps -e "$FORK"
echo ">> installing esm (--no-deps, editable): $REPO"
uv pip install --no-deps -e "$REPO"

# --- 4. work out which PyPI deps are genuinely missing ------------------------
# (a) top-level runtime deps from pyproject that the system does NOT already
#     provide, excluding torch (system) and transformers (fork, installed above).
REQS_IN="$(mktemp)"; CLOSURE="$(mktemp)"; INSTALL="$(mktemp)"
trap 'rm -f "$REQS_IN" "$CLOSURE" "$INSTALL"' EXIT

python - "$REPO/pyproject.toml" > "$REQS_IN" <<'PY'
import sys, tomllib, importlib.metadata as md, re
SKIP = {"torch", "transformers"}  # provided by system / the fork
deps = tomllib.load(open(sys.argv[1], "rb"))["project"]["dependencies"]
for spec in deps:
    name = re.split(r'[\s=<>!~;@\[]', spec.strip())[0].lower().replace("_", "-")
    if name in SKIP:
        continue
    try:                       # already importable in the venv (via system)?
        md.version(name)
    except md.PackageNotFoundError:
        print(spec)
PY

if [[ -s "$REQS_IN" ]]; then
    echo ">> resolving transitive closure of missing deps"
    uv pip compile --quiet "$REQS_IN" -o "$CLOSURE"

    # (b) filter the closure: drop anything the system already provides, and drop
    #     the GPU vendor wheels (torch's CUDA deps) -- the system XPU torch is the
    #     runtime, so nvidia-*/cuda-*/triton are both wrong and huge.
    python - "$CLOSURE" > "$INSTALL" <<'PY'
import sys, re, importlib.metadata as md
DENY = re.compile(r'^(torch|triton|nvidia[-_].*|cuda[-_].*)$', re.I)
for line in open(sys.argv[1]):
    line = line.split("#", 1)[0].strip()
    if not line:
        continue
    name = re.split(r'[\s=<>!~;@\[]', line)[0]
    if DENY.match(name):
        continue
    try:
        md.version(name)        # already provided by the system -> reuse it
    except md.PackageNotFoundError:
        print(line)
PY

    echo ">> installing $(grep -c . "$INSTALL") missing packages (--no-deps):"
    sed 's/^/     /' "$INSTALL"
    uv pip install --no-deps -r "$INSTALL"
else
    echo ">> no missing top-level deps; system provides them all"
fi

# --- 5. verify ----------------------------------------------------------------
echo ">> verifying the environment"
python - <<'PY'
import importlib, torch
import esm                                   # the package installed here
import transformers
from transformers.models import esmfold2      # the fork-only model package
print(f"  esm                ok  ({esm.__file__})")
print(f"  transformers       {transformers.__version__}  ({transformers.__file__})")
print(f"  transformers fork  ok  (esmfold2 importable)")
print(f"  torch              {torch.__version__}")
PY

echo
echo "Done. Activate with:  source $VENV/bin/activate"
