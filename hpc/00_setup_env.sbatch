#!/bin/bash
#SBATCH --job-name=clifcs-setup
#SBATCH --partition=tier1q
#SBATCH --cpus-per-task=4
#SBATCH --mem=16000M
#SBATCH --time=01:00:00
#SBATCH --output=logs/slurm-%x-%j.out
#SBATCH --error=logs/slurm-%x-%j.out
#
# Stage 0: build the virtualenv. Run this ONCE, before anything else.
#
# WHY THIS IS A BATCH JOB AND NOT SOMETHING YOU TYPE
#
# The venv has to be built somewhere that looks like where the work will run --
# the login node is often a different CPU generation, and some clusters do not
# let you spend twenty minutes of pip on it at all.
#
# NO GPU. An earlier version asked for a GPU so the verification below could
# call torch.cuda.is_available(), and argued that spending one on a pip install
# was the only way to make the check mean anything. That was wrong twice over:
#
#   * the check that actually matters is the `+cu121` string on the installed
#     wheel, which catches cotorra pulling CPU torch over the CUDA build. That
#     is a property of what pip resolved, not of the node, and it passes or
#     fails identically on a CPU node.
#   * torch.cuda.is_available() is already asserted at the top of
#     04_embed.sbatch, the only job that needs a card. Checking it here too
#     gated the whole venv on a scarce 6-node partition to learn something the
#     job that needs it re-learns anyway.
#
# And tier 1 is now entirely CPU: 01-03 never load the GEM weights. A site with
# no GPU can run the whole of its contribution, so it must be able to build the
# venv too.
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
#     srun --partition=tier1q --cpus-per-task=4 \
#          --mem=16000M --time=01:00:00 --pty bash
#
# Those are alternatives. `srun --pty bash` asks for an interactive terminal
# and must never appear inside a batch script.

set -euo pipefail

# --- EDIT THESE to match `module avail` on this cluster -----------------
# Empty by default, and on Rush CRI it should stay that way. The cluster
# has python/3.10.5 and python/3.12.3 and Lmod answers "exist but cannot
# be loaded as requested" for both -- they sit behind a toolchain
# prerequisite in a hierarchical module tree. None of that matters: the
# venv is built on a standalone CPython that uv installed, and activating
# the venv is what puts the right interpreter on PATH.
PYTHON_MODULE="${PYTHON_MODULE:-}"
# The interpreter the venv is created FROM. Must be 3.11 or newer.
BOOTSTRAP_PYTHON="${BOOTSTRAP_PYTHON:-python3}"
VENV_HOME="${VENV_HOME:-/gpfs/data/pearson-lab/PERSONAL/hhieromnimon/venvs/clif-cs}"
# Where pip keeps downloaded wheels. Defaults NEXT TO THE VENV rather than to
# ~/.cache/pip, because the two together are what exhaust a home quota: this
# install is torch plus a full NVIDIA stack, and it lands twice (see the
# install-order note below), so the cache alone runs to several GB.
PIP_CACHE_DIR="${PIP_CACHE_DIR:-$(dirname "$VENV_HOME")/pip-cache}"
export PIP_CACHE_DIR
# ------------------------------------------------------------------------

# `$(dirname "$0")` DOES NOT WORK IN A BATCH JOB. Slurm copies the script
# into a per-job spool directory and runs it from there, so $0 is
# /var/spool/slurmd/job<id>/<script> and the repo is nowhere near it. The
# first version of these scripts sourced that path, failed, and carried on
# regardless -- because `set -e` lived inside the file that had not loaded.
# The job then ran with no module load, no venv activation, and an empty
# PHENOTYPE_FLAGS, which silently refits the phenotype models instead of
# applying the shared ones.
#
# SLURM_SUBMIT_DIR is where sbatch was invoked, which the docs say is the
# repository root. Overridable, and checked rather than assumed.
set -euo pipefail

REPO_HOME="${REPO_HOME:-${SLURM_SUBMIT_DIR:-$PWD}}"
if [ ! -f "$REPO_HOME/requirements.txt" ]; then
    echo "cannot find requirements.txt under REPO_HOME=$REPO_HOME" >&2
    echo "Submit from the repository root, or set REPO_HOME." >&2
    exit 1
fi
cd "$REPO_HOME"
mkdir -p logs

module purge
if [ -n "$PYTHON_MODULE" ]; then
    module load "$PYTHON_MODULE"
fi

echo "=============================================================="
echo " node     $(hostname)"
echo " python   $(command -v "$BOOTSTRAP_PYTHON")  $("$BOOTSTRAP_PYTHON" --version 2>&1)"
echo " venv     $VENV_HOME"
echo " pip cache $PIP_CACHE_DIR"
echo "=============================================================="

"$BOOTSTRAP_PYTHON" - <<'PY'
import sys
# cotorra 26.6.3 and cocoa-tokenizer 26.6.2 both declare requires_python
# >=3.11, so pip refuses to install them on anything older. The repo was
# developed on 3.11; Rush CRI offers 3.10.5 and 3.12.3, which makes 3.12
# the only workable choice rather than a preference.
if sys.version_info < (3, 11):
    raise SystemExit(
        f"need Python 3.11 or newer, got {sys.version.split()[0]}. "
        "cotorra and cocoa-tokenizer declare requires_python >=3.11. "
        "Point BOOTSTRAP_PYTHON at a newer one -- uv can install it:\n"
        "  uv python install 3.11\n"
        "  sbatch --export=ALL,BOOTSTRAP_PYTHON=$(uv python find 3.11) hpc/00_setup_env.sbatch"
    )
print(f"  python version ok ({sys.version.split()[0]})")
PY

# A home directory is the wrong place for this and the failure is ugly: pip
# installs most of a 10+ GB dependency set, hits [Errno 122] Disk quota
# exceeded somewhere in the middle, and leaves a venv that looks built and
# imports about half of what it needs.
case "$VENV_HOME" in
    "$HOME"/*)
        echo "=================================================================="
        echo "WARNING: the venv is going into your HOME directory."
        echo "  $VENV_HOME"
        echo ""
        echo "This install needs roughly 10-15 GB: torch, a full NVIDIA runtime,"
        echo "and the cu121 wheels that replace it. Home quotas are usually far"
        echo "smaller than that, and the failure mode is [Errno 122] Disk quota"
        echo "exceeded part-way through, leaving a venv that looks built and"
        echo "imports about half of what it needs."
        echo ""
        echo "Put it on project or scratch storage instead -- set VENV_HOME in"
        echo "hpc/_common.sh, or pass it per job:"
        echo ""
        echo "  sbatch --export=ALL,VENV_HOME=/project/path/venvs/clif-cs hpc/00_setup_env.sbatch"
        echo ""
        echo "Check what you have:  quota -s     (or: lfs quota -h \$HOME)"
        echo "=================================================================="
        ;;
esac

if [ -d "$VENV_HOME" ]; then
    echo "note: $VENV_HOME already exists; reusing it."
    echo "      delete it first if you want a clean build."
    echo "      REUSING A VENV THAT FAILED PART-WAY IS NOT SAFE -- a quota or"
    echo "      network failure leaves it half-populated, and pip will report"
    echo "      'Requirement already satisfied' for packages that are broken."
else
    "$BOOTSTRAP_PYTHON" -m venv "$VENV_HOME"
fi
# shellcheck disable=SC1091
source "$VENV_HOME/bin/activate"

python -m pip install --upgrade pip

# WHICH TIER. Default is `site`, which is the 200 MB CPU-only set every
# contributing site needs and nothing more. Only pass `embed` or `all` if this
# machine is actually running 02_site_embeddings.py -- that is where the 10-15
# GB and the GPU requirement come from, because one line of cotorra pulls
# torch, transformers, datasets, lightgbm, optuna, xgboost and wandb.
#
#   sbatch --export=ALL,TIER=embed  hpc/00_setup_env.sbatch
#
#   site    tier 1 only. No torch, no CUDA, no GPU.          requirements-site.txt
#   embed   tier 1 + tier 2. Needs a GPU to be useful.       requirements-embed.txt
#   pooled  the coordinating analysis. No torch.             requirements-pooled.txt
#   all     every role on one machine.                       requirements.txt
TIER="${TIER:-site}"
case "$TIER" in
    site)   REQUIREMENTS=requirements-site.txt   ; NEEDS_TORCH=0 ;;
    embed)  REQUIREMENTS=requirements-embed.txt  ; NEEDS_TORCH=1 ;;
    pooled) REQUIREMENTS=requirements-pooled.txt ; NEEDS_TORCH=0 ;;
    all)    REQUIREMENTS=requirements.txt        ; NEEDS_TORCH=1 ;;
    *)      echo "TIER must be one of: site embed pooled all (got '$TIER')" >&2
            exit 1 ;;
esac
echo " tier     $TIER  ->  $REQUIREMENTS"

python -m pip install -r "$REQUIREMENTS"

# Order matters for the torch tiers. cotorra depends on
# torch/torchvision/torchaudio WITHOUT pins, so installing the requirements
# after the CUDA wheels would quietly pull CPU-only builds over the top of
# them. Requirements first, CUDA torch second, and never the other way round.
if [ "$NEEDS_TORCH" = "1" ]; then
    python -m pip install \
        torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
        --index-url https://download.pytorch.org/whl/cu121
fi

echo
echo "--- verification ---"
TIER="$TIER" python - <<'PY'
import importlib
import os
import sys

tier = os.environ.get("TIER", "site")
problems = []

# --- always: what tier 1 cannot run without
BASE = ("cocoa_tokenizer", "clifpy", "polars", "duckdb", "omegaconf", "sklearn")
EXTRA = {
    "site":   (),
    "embed":  ("cotorra", "transformers", "accelerate"),
    "pooled": ("matplotlib", "umap", "pacmap", "statsmodels"),
    "all":    ("cotorra", "transformers", "accelerate",
               "matplotlib", "umap", "pacmap", "statsmodels"),
}[tier]

for name in BASE + EXTRA:
    try:
        module = importlib.import_module(name)
        print(f"  {name:<18} {getattr(module, '__version__', 'ok')}")
    except ImportError as exc:
        problems.append(f"{name} did not import: {exc}")

# --- torch, only for the tiers that installed it
if tier in ("embed", "all"):
    try:
        import torch
    except ImportError as exc:
        problems.append(f"torch did not import: {exc}")
    else:
        print(f"  torch              {torch.__version__}")
        print(f"  cuda available     {torch.cuda.is_available()}")
        if "+cu" not in torch.__version__:
            problems.append(
                f"torch {torch.__version__} is a CPU build. Something "
                "reinstalled it over the cu121 wheels -- rerun the two pip "
                "install lines in order."
            )
        if torch.cuda.is_available():
            print(f"  gpu                {torch.cuda.get_device_name(0)}")
            print("  vram               "
                  f"{torch.cuda.get_device_properties(0).total_memory / 2**30:.1f} GiB")
        else:
            # NOT a failure: this job runs on a CPU partition by design. The
            # assertion lives at the top of 04_embed.sbatch, the only job
            # that needs a card.
            print("  (no CUDA device -- expected on a CPU node; 04_embed asserts it)")
else:
    print(f"  torch              not installed (tier {tier!r} does not need it)")

if problems:
    print()
    for problem in problems:
        print(f"  FAIL  {problem}")
    sys.exit(1)
print()
print(f"  environment ok for tier {tier!r}")
PY
echo "Next:  bash hpc/00_preflight.sh      (login node, checks paths.yaml)"
echo "Then:  sbatch hpc/01_cohort.sbatch"
echo
echo "CUDA was NOT verified here -- this job runs on a CPU partition, and"
echo "tier 1 (jobs 01-03) never loads the GEM weights. The card is checked at"
echo "the top of hpc/04_embed.sbatch, which is where one is first needed."
