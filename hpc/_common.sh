# Sourced by every job script here. Edit this file, not the five below.
#
# Everything cluster-specific lives in one place, so porting to another HPCC
# is one file rather than five headers.

# Partitions are set per job in each sbatch header, not here, because they
# differ by job. On Rush CRI:
#
#   express    4 nodes,   32 CPU, 128 GB, 6 h  -- short jobs
#   tier1q     147 nodes, 32 CPU, 128 GB, 10 d -- the long CPU jobs (default)
#   tier2q     42 nodes,  32 CPU, 512 GB, 10 d -- escalate here on OOM
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
REPO_HOME="${REPO_HOME:-$HOME/CLIF-latent-cardiogenicshock}"
VENV_HOME="${VENV_HOME:-$HOME/venvs/clif-cs}"
PYTHON_MODULE="${PYTHON_MODULE:-python/3.12.3}"
SITE="${SITE:-RUSH}"
# ------------------------------------------------------------------------

set -euo pipefail

module purge
module load "$PYTHON_MODULE"

source "$VENV_HOME/bin/activate"
cd "$REPO_HOME"
mkdir -p logs

# The three phenotype models are REQUIRED at a contributing site. Without
# them the site refits its own clusters and "phenotype II", "beta" and
# "group A" stop meaning the same thing at each site, which makes every
# pooled breakdown uninterpretable. They are in git, under models/.
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
