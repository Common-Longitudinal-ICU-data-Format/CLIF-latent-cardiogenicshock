# Sourced by every job script here. Edit this file, not the five below.
#
# Everything cluster-specific lives in one place, so porting to another HPCC
# is one file rather than five headers.

# Partitions are set per job in each sbatch header, not here, because they
# differ by job. On Rush CRI:
#
#   express    4 nodes,   32 CPU, 128 GB, 6 h  -- short jobs
#   tier1q     147 nodes, 32 CPU, 128 GB, 10 d -- the long CPU jobs (default)
#   tier2q     42 nodes,  32 CPU, 512 GB, 10 d -- the prepare job lives here,
#              because collate was OOM-killed at 96 GB and tier1q caps at 128
#   tier3q     7 nodes,   32 CPU, 1.25 TB, 10 d -- escalate here if it still is
#   gpuq       6 nodes,   32+ CPU, 512+ GB     -- embed and env setup
#
# The lab-owned GPU partitions (pearsonq and the rest) are deliberately NOT
# used here. gpuq is the shared one.
#
# MEMORY IS CAPPED PER CPU, not per job. 8 CPU with 64 GB was rejected as
# "Requested node configuration is not available" on a partition whose nodes
# have 32 CPU and 128 GB -- which is not a shortage, it is 128000/32 = 4000
# MB per CPU being enforced. Every job here asks for exactly 4000 MB per
# CPU, written in M rather than G: Slurm's G is 1024 MB, so --mem=16G on
# 4 CPUs is 4096 MB/CPU and is refused by 96 MB. To get more memory, ask
# for more CPUs; raising --mem alone will be rejected at submission.
#
# There is NO cuda module on this cluster, and none is needed: the pip
# torch cu121 wheels bundle their own CUDA runtime, and the GPU nodes carry
# the driver. Only a python module is loaded.
#
# Check with:  sinfo -o "%20P %5a %10l %6D %N"
#              sinfo -o "%20P %10G %N" | grep -i gpu

# --- EDIT THESE ---------------------------------------------------------
# Normally already resolved by the job script, from SLURM_SUBMIT_DIR,
# before it sourced this file. The default below is only for sourcing
# _common.sh by hand in an interactive shell.
REPO_HOME="${REPO_HOME:-/gpfs/data/pearson-lab/PERSONAL/hhieromnimon/CLIF-latent-cardiogenicshock}"
# NOT $HOME. The dependency set is 10-15 GB -- torch, a full NVIDIA runtime,
# and the cu121 wheels that replace it -- which exceeds a normal home quota.
# Installing there fails as [Errno 122] Disk quota exceeded somewhere in the
# middle of pip, leaving a venv that looks built and imports about half of
# what it needs.
VENV_HOME="${VENV_HOME:-/gpfs/data/pearson-lab/PERSONAL/hhieromnimon/venvs/clif-cs}"
# The python module the venv was BUILT FROM. Must match, because a venv is
# permanently bound to the interpreter that created it: `bin/python` is a
# symlink to that interpreter and `pyvenv.cfg` records its absolute path.
# Activating a venv whose base interpreter is not on PATH -- which is what a
# `module purge` with PYTHON_MODULE empty leaves you -- gives a venv that
# either fails to start or starts without the module's LD_LIBRARY_PATH.
#
# On Rush CRI `module load python/3.12.3` works, and 3.12.3 clears the >=3.11
# floor cotorra and cocoa-tokenizer declare. An earlier version of this
# comment said Lmod refused both pythons here and that a uv-installed
# standalone CPython was the only route; that is no longer true. Check with
# `module spider python` -- 3.10.5 is still too old, 3.12.3 loads.
#
# Leave it EMPTY only if the venv was built on an interpreter that needs no
# module: a system python new enough, or a standalone CPython from uv.
PYTHON_MODULE="${PYTHON_MODULE:-python/3.12.3}"
SITE="${SITE:-RUSH}"
# ------------------------------------------------------------------------

set -euo pipefail

module purge
if [ -n "$PYTHON_MODULE" ]; then
    module load "$PYTHON_MODULE"
fi

if [ ! -f "$VENV_HOME/bin/activate" ]; then
    echo "no virtualenv at $VENV_HOME. Build it with:" >&2
    echo "  sbatch hpc/00_setup_env.sbatch" >&2
    exit 1
fi
source "$VENV_HOME/bin/activate"
cd "$REPO_HOME"
mkdir -p logs

# The three shared phenotype models, applied rather than refit. 01 now
# defaults to exactly these files, so passing them is belt and braces rather
# than mandatory -- but it keeps the invocation self-documenting in the job
# log. A site must NOT refit: its own clusters make "phenotype II", "beta"
# and "group A" mean something different here than everywhere else, and every
# pooled breakdown becomes uninterpretable. They are in git, under models/.
PHENOTYPE_FLAGS=(
    --phenotype-model models/phenotype_model.json
    --seymour-model   models/seymour_model.json
    --bhavani-model   models/bhavani_model.json
)

echo "=============================================================="
echo " job      ${SLURM_JOB_NAME:-interactive} (${SLURM_JOB_ID:-none})"
echo " node     $(hostname)"
echo " commit   $(git rev-parse --short HEAD) $(git describe --always --dirty 2>/dev/null || true)"
echo " python   $(python --version 2>&1)"
echo " started  $(date -Is)"
echo "=============================================================="
