#!/bin/bash
#SBATCH --job-name=clifcs-cohort
#SBATCH --partition=general
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/slurm-%x-%j.out
#SBATCH --error=logs/slurm-%x-%j.out
#
# Stage 1 of 4: build the cohort, then STOP.
#
# This is its own job on purpose. A human has to read
# output/cohort/filter_ledger.csv before anything expensive runs. The two
# things that go wrong are an ICU definition that does not match this site's
# unit labels, and a table that is present but not populated. Neither is
# visible from an exit code.

source "$(dirname "$0")/_common.sh"

python code/01_site_aggregate.py --stage cohort "${PHENOTYPE_FLAGS[@]}"

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
