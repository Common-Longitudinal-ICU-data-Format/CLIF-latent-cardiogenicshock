#!/bin/bash
#SBATCH --job-name=clifcs-embeddings
#SBATCH --partition=express
#SBATCH --cpus-per-task=4
#SBATCH --mem=16000M
#SBATCH --time=00:30:00
#SBATCH --output=logs/slurm-%x-%j.out
#SBATCH --error=logs/slurm-%x-%j.out
#
# TIER 2 -- DO NOT RUN WITHOUT THE IRB APPROVAL AND DUA IN PLACE.
#
# Adds patient/ to the bundle 01 already wrote, so this site sends exactly
# ONE directory: output/upload_to_box_<SITE>/. Everything the pooled analysis
# needs is in it.
#
# The output is NOT de-identified. A 1024-d embedding is a learned function
# of the patient's entire recorded history. Record numbers become a salted
# hash, timestamps are dropped and age is reduced to five-year bands, which
# narrows what travels; it does not make the file anonymous. Treat it as a
# limited data set at best and move it under the same approval you would use
# to share patient-level clinical data.
#
# Tier 1 is a complete contribution on its own. If the agreement does not
# cover patient-level transfer, do not submit this job -- the pooled analysis
# degrades rather than failing, and a tier-1-only site still appears in every
# count table and in the phenotype-coherence comparison.
#
# No GPU: it reads the embeddings the embed stage already wrote and redacts
# them. 01 must have run first; the export refuses if the tier-1 manifest was
# written under a different model or a different salt.

source "$(dirname "$0")/_common.sh"

if [ "${I_HAVE_CONFIRMED_THE_DUA:-no}" != "yes" ]; then
    echo "Refusing to run."
    echo ""
    echo "Set I_HAVE_CONFIRMED_THE_DUA=yes to confirm that IRB approval and a"
    echo "DUA covering patient-level transfer are in place for this site:"
    echo ""
    echo "  sbatch --export=ALL,I_HAVE_CONFIRMED_THE_DUA=yes hpc/05_embeddings.sbatch"
    echo ""
    echo "This gate exists because a flag on a command invites somebody to"
    echo "pass it without the question having been asked."
    exit 1
fi

python code/02_site_embeddings.py

echo ""
echo "===================================================================="
echo "TIER 2 DELIVERABLE:  output/upload_to_box_${SITE}/  -- the WHOLE directory"
echo "  patient/ has been added to the tier-1 bundle. Send this one directory;"
echo "  nothing else needs to leave. Roughly 45 MB per 10,000 patients."
echo "  NOT de-identified."
echo "===================================================================="
