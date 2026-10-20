#!/bin/bash
#SBATCH --job-name=clifcs-cohort
#SBATCH --partition=express
#SBATCH --cpus-per-task=8
#SBATCH --mem=32000M
#SBATCH --time=02:00:00
#SBATCH --output=logs/slurm-%x-%j.out
#SBATCH --error=logs/slurm-%x-%j.out
#
# Stage 1 of 3 for TIER 1: build the cohort, then STOP.
#
# This is its own job on purpose. A human has to read
# output/cohort/filter_ledger.csv before anything expensive runs. The two
# things that go wrong are an ICU definition that does not match this site's
# unit labels, and a table that is present but not populated. Neither is
# visible from an exit code.

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

python code/01_site_tables.py --stage cohort "${PHENOTYPE_FLAGS[@]}"

echo ""
echo "--------------------------------------------------------------------"
echo "STOP HERE. Before submitting hpc/02_prepare.sbatch, open"
echo ""
echo "    output/cohort/filter_ledger.csv"
echo ""
echo "and check that:"
echo "  (a) the ICU row is a plausible fraction of total hospitalizations"
echo "      for this site -- sites label units differently;"
echo "  (b) no criterion sub-count is implausibly large or implausibly zero."
echo "      A populated-but-meaningless table reads as a valid filter. UCMC's"
echo "      clif_ecmo_mcs names 82% of all hospitalizations because it is a"
echo "      concept table, not a record of mechanical support."
echo ""
echo "Also confirm the phenotype counts above came from the APPLIED models"
echo "and not a local refit."
echo "--------------------------------------------------------------------"
