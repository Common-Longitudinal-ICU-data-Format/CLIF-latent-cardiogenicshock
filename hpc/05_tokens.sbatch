#!/bin/bash
#SBATCH --job-name=clifcs-tokens
#SBATCH --partition=general
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=logs/slurm-%x-%j.out
#SBATCH --error=logs/slurm-%x-%j.out
#
# TIER 2 -- DO NOT RUN WITHOUT THE IRB APPROVAL AND DUA IN PLACE.
#
# The output is NOT de-identified. It carries the model's full input sequence
# per patient. Record numbers are replaced by a salted hash, timestamps are
# dropped and age is reduced to five-year bands, which narrows what travels;
# it does not make the file anonymous. Treat it as a limited data set at best
# and move it under the same approval you would use to share patient-level
# clinical data.
#
# Tier 1 is a complete contribution on its own. If the agreement does not
# cover patient-level transfer, do not submit this job -- the pooled analysis
# degrades rather than failing.
#
# No GPU, and it is quick: three seconds on a 34,142-patient cohort, because
# it reads the sequences the winnow stage already wrote. The 30 minutes here
# is slack, not an estimate.

source "$(dirname "$0")/_common.sh"

if [ "${I_HAVE_CONFIRMED_THE_DUA:-no}" != "yes" ]; then
    echo "Refusing to run."
    echo ""
    echo "Set I_HAVE_CONFIRMED_THE_DUA=yes to confirm that IRB approval and a"
    echo "DUA covering patient-level transfer are in place for this site:"
    echo ""
    echo "  sbatch --export=ALL,I_HAVE_CONFIRMED_THE_DUA=yes hpc/05_tokens.sbatch"
    echo ""
    echo "This gate exists because a flag on a command invites somebody to"
    echo "pass it without the question having been asked."
    exit 1
fi

python code/02_site_tokens.py

echo ""
echo "===================================================================="
echo "TIER 2 DELIVERABLE:  output/token_level_data_${SITE}/"
echo "  NOT de-identified. Roughly 10 MB per 10,000 patients."
echo "===================================================================="
