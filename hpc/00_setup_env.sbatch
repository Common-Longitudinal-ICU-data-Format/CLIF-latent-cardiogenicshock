#!/bin/bash
#SBATCH --job-name=clifcs-setup
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/slurm-%x-%j.out
#SBATCH --error=logs/slurm-%x-%j.out
#
# Stage 0: build the virtualenv. Run this ONCE, before anything else.
#
# WHY THIS IS A BATCH JOB AND NOT SOMETHING YOU TYPE
#
# The venv has to be built somewhere that looks like where the work will run:
# the login node usually has no GPU and often a different CPU generation, and
# the last line of this script verifies CUDA, which can only pass on a node
# that has a card. Requesting --gres=gpu:1 for a pip install feels wasteful
# and is the point -- it is the only way the check means anything.
#
# This script deliberately does NOT source _common.sh. That activates the
# venv, which does not exist yet.
#
#     sbatch hpc/00_setup_env.sbatch
#
# To do the same thing interactively instead -- useful the first time, when
# you want to see pip fail in real time -- request a shell and run the body
# by hand:
#
#     srun --partition=gpu --gres=gpu:1 --cpus-per-task=4 \
#          --mem=32G --time=01:00:00 --pty bash
#
# Those are alternatives. `srun --pty bash` asks for an interactive terminal
# and must never appear inside a batch script.

set -euo pipefail

# --- EDIT THESE to match `module avail` on this cluster -----------------
PYTHON_MODULE="${PYTHON_MODULE:-python/3.11}"
CUDA_MODULE="${CUDA_MODULE:-cuda/12.1}"
VENV_HOME="${VENV_HOME:-$HOME/venvs/clif-cs}"
# ------------------------------------------------------------------------

cd "$(dirname "$0")/.."
mkdir -p logs

module purge
module load "$PYTHON_MODULE"
module load "$CUDA_MODULE"

echo "=============================================================="
echo " node     $(hostname)"
echo " python   $(command -v python)  $(python --version 2>&1)"
echo " venv     $VENV_HOME"
echo "=============================================================="

python - <<'PY'
import sys
if sys.version_info[:2] != (3, 11):
    raise SystemExit(
        f"need Python 3.11, got {sys.version.split()[0]}. The pinned "
        "coreopsis stack is built against it. Fix PYTHON_MODULE."
    )
print("  python version ok")
PY

if [ -d "$VENV_HOME" ]; then
    echo "note: $VENV_HOME already exists; reusing it."
    echo "      delete it first if you want a clean build."
else
    python -m venv "$VENV_HOME"
fi
# shellcheck disable=SC1091
source "$VENV_HOME/bin/activate"

python -m pip install --upgrade pip

# Order matters. cotorra depends on torch/torchvision/torchaudio WITHOUT
# pins, so installing requirements.txt after the CUDA wheels would quietly
# pull CPU-only builds from PyPI over the top of them. Requirements first,
# CUDA torch second, and never the other way round.
python -m pip install -r requirements.txt
python -m pip install \
    torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
    --index-url https://download.pytorch.org/whl/cu121

echo
echo "--- verification ---"
python - <<'PY'
import sys
import torch

print(f"  torch              {torch.__version__}")
print(f"  cuda available     {torch.cuda.is_available()}")

problems = []
if "+cu" not in torch.__version__:
    problems.append(
        f"torch {torch.__version__} is a CPU build. Something reinstalled it "
        "over the cu121 wheels -- rerun the two pip install lines in order."
    )
if not torch.cuda.is_available():
    problems.append(
        "no CUDA device visible. If this job did get a GPU, the CUDA module "
        "is wrong or does not match the cu121 wheels."
    )
else:
    print(f"  gpu                {torch.cuda.get_device_name(0)}")
    print(f"  vram               "
          f"{torch.cuda.get_device_properties(0).total_memory / 2**30:.1f} GiB")

# the pinned stack, which is what makes embeddings comparable between sites
for name in ("cocoa_tokenizer", "cotorra", "transformers", "polars", "clifpy"):
    try:
        module = __import__(name)
        print(f"  {name:<18} {getattr(module, '__version__', 'ok')}")
    except ImportError as exc:
        problems.append(f"{name} did not import: {exc}")

if problems:
    print()
    for problem in problems:
        print(f"  FAIL  {problem}")
    sys.exit(1)
print()
print("  environment ok")
PY

echo
echo "Next:  bash hpc/00_preflight.sh      (login node, checks paths.yaml)"
echo "Then:  sbatch hpc/01_cohort.sbatch"
