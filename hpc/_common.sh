# Sourced by every job script here. Edit this file, not the five below.
#
# Everything cluster-specific lives in one place, so porting to another HPCC
# is one file rather than five headers.

# --- EDIT THESE ---------------------------------------------------------
REPO_HOME="${REPO_HOME:-$HOME/CLIF-latent-cardiogenicshock}"
VENV_HOME="${VENV_HOME:-$HOME/venvs/clif-cs}"
PYTHON_MODULE="${PYTHON_MODULE:-python/3.11}"
CUDA_MODULE="${CUDA_MODULE:-cuda/12.1}"
SITE="${SITE:-RUSH}"
# ------------------------------------------------------------------------

set -euo pipefail

module purge
module load "$PYTHON_MODULE"
if [ "${NEEDS_GPU:-0}" = "1" ]; then
    module load "$CUDA_MODULE"
fi

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
