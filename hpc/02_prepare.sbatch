#!/bin/bash
#SBATCH --job-name=clifcs-prepare
#SBATCH --partition=tier2q
#SBATCH --cpus-per-task=32
#SBATCH --mem=256000M
#SBATCH --time=12:00:00
#SBATCH --output=logs/slurm-%x-%j.out
#SBATCH --error=logs/slurm-%x-%j.out
#
# Stage 2 of 3 for TIER 1: subset -> derive -> collate -> tokenize -> winnow.
# All CPU, and no GEM weights -- `tokenize` loads the frozen TOKENIZER, which
# is a different file and does not need a GPU.
#
# No GPU is requested because nothing here touches one. Running these stages
# inside the GPU allocation works and wastes most of it.
#
# MEMORY. Sized by what actually happened rather than by guess: `collate`
# was OOM-killed at 96 GB on a 5,677-patient cohort, so this asks for 256 GB
# on tier2q, whose nodes have 512 GB. tier1q was the obvious home for a long
# CPU job and cannot be: its nodes top out at 128 GB.
#
# `derive` (clifpy, pandas-based) and `collate` (cocoa, building the
# processed dataset from ~10M vitals and ~6M assessment rows) are the two
# peaks. Neither scales with the patient count so much as with the raw row
# count, which is why a small cohort at a busy site is not a small job.
#
# If this is still OOM-killed, escalate to tier3q (1.25 TB nodes, 7 of them)
# and resume at the stage that died rather than starting over -- the stages
# are resumable and their outputs are on scratch:
#
#     sbatch --partition=tier3q --mem=512000M hpc/02_prepare.sbatch
#     python code/01_site_tables.py --from-stage collate
#
# If sbatch refuses this with "Requested node configuration is not
# available", the per-CPU memory cap on tier2q is lower than 8000 MB. Check
# with `scontrol show partition tier2q | grep -o 'MaxMemPerCPU=[^ ]*'` and
# divide.

# `$(dirname "$0")` DOES NOT WORK IN A BATCH JOB. Slurm copies the script
# into a per-job spool directory and runs it from there, so $0 is
# /var/spool/slurmd/job<id>/<script> and the repo is nowhere near it. The
# first version of these scripts sourced that path, failed, and carried on
# regardless -- because `set -e` lived inside the file that had not loaded.
# The job then ran with no module load, no venv activation, and an empty
# PHENOTYPE_FLAGS. That no longer refits silently -- 01 now applies the
# models shipped in models/ by default and makes you ask for a refit with
# --refit-phenotypes -- but a job running without its venv or modules is
# still broken, so the check stays.
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

# Each stage is run separately so the log says which one died. `set -e` in
# _common.sh stops the loop on the first failure -- which it did NOT do on
# the first Rush run, because _common.sh had not loaded, so an OOM-killed
# collate was followed by a tokenize that could not find its input and a
# winnow that could not find the tokenizer. Three tracebacks, one cause, and
# the only line that said so was Slurm's oom_kill notice at the very end.
for stage in subset derive collate tokenize winnow; do
    echo ""
    echo "--- stage: $stage  ($(date -Is)) ---"
    if ! python code/01_site_tables.py --stage "$stage"; then
        status=$?
        echo ""
        echo "=================================================================="
        echo "STAGE '$stage' FAILED (exit $status)."
        echo ""
        echo "If the job also reports an oom_kill event, it is a memory"
        echo "problem and not a data problem: raise --mem (and --cpus-per-task"
        echo "with it, the cap is per CPU) and resume at this stage rather"
        echo "than starting over --"
        echo ""
        echo "    python code/01_site_tables.py --from-stage $stage"
        echo ""
        echo "The earlier stages' outputs are already on scratch."
        echo "=================================================================="
        exit $status
    fi
done

echo ""
echo "Tokenized and winnowed. Next: hpc/03_tier1.sbatch (CPU) completes tier 1."
echo "If the GPU queue is deep, check the scratch purge policy first --"
echo "these intermediates have to survive until that job starts."
