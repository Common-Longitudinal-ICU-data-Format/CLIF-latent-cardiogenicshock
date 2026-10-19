#!/bin/bash
#SBATCH --job-name=clifcs-prepare
#SBATCH --partition=tier1q
#SBATCH --cpus-per-task=24
#SBATCH --mem=96000M
#SBATCH --time=12:00:00
#SBATCH --output=logs/slurm-%x-%j.out
#SBATCH --error=logs/slurm-%x-%j.out
#
# Stage 2 of 4: subset -> derive -> collate -> tokenize -> winnow. All CPU.
#
# No GPU is requested because nothing here touches one. Running these stages
# inside the GPU allocation works and wastes most of it.
#
# `derive` is the memory peak of the whole pipeline -- clifpy is pandas-based.
# If this job is OOM-killed, raise --mem and resume with
#
#     python code/01_site_aggregate.py --from-stage derive
#
# rather than starting over. The stages are resumable and their outputs are
# on scratch.

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
if [ ! -f "$REPO_HOME/hpc/_common.sh" ]; then
    echo "cannot find hpc/_common.sh under REPO_HOME=$REPO_HOME" >&2
    echo "Submit from the repository root, or set REPO_HOME:" >&2
    echo "  sbatch --export=ALL,REPO_HOME=/path/to/repo <script>" >&2
    exit 1
fi
source "$REPO_HOME/hpc/_common.sh"

for stage in subset derive collate tokenize winnow; do
    echo ""
    echo "--- stage: $stage  ($(date -Is)) ---"
    python code/01_site_aggregate.py --stage "$stage"
done

echo ""
echo "Tokenized and winnowed. Next: hpc/03_embed.sbatch (GPU)."
echo "If the GPU queue is deep, check the scratch purge policy first --"
echo "these intermediates have to survive until that job starts."
