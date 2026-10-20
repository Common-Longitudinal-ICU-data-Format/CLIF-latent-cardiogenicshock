#!/bin/bash
#SBATCH --job-name=clifcs-tier2
#SBATCH --partition=express
#SBATCH --cpus-per-task=4
#SBATCH --mem=16000M
#SBATCH --time=00:30:00
#SBATCH --output=logs/slurm-%x-%j.out
#SBATCH --error=logs/slurm-%x-%j.out
#
# TIER 2 -- DO NOT RUN WITHOUT THE IRB APPROVAL AND DUA IN PLACE.
#
# Stage 2 of 2 for TIER 2: purity -> export.
#
# Adds patient/ and purity/ to the bundle 01 already wrote, so this site sends
# exactly ONE directory: output/upload_to_box_<SITE>/. Everything the pooled
# analysis needs is in it.
#
# `purity` is the phenotype-coherence measure -- how much of a patient's
# neighbourhood in this site's own GEM space carries the same phenotype,
# against chance. It used to sit in tier 1 and forced that script to need a
# GPU; it is here now, which is why a tier-1-only site no longer reports it.
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
# count table and in the token-prevalence comparison 04 reads.
#
# No GPU: it reads the embeddings job 04 already wrote, measures coherence in
# that space and redacts the rest. Both 01 and 04_embed must have run first;
# the export refuses if the tier-1 manifest was written under a different model
# or a different salt.

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

if [ "${I_HAVE_CONFIRMED_THE_DUA:-no}" != "yes" ]; then
    echo "Refusing to run."
    echo ""
    echo "Set I_HAVE_CONFIRMED_THE_DUA=yes to confirm that IRB approval and a"
    echo "DUA covering patient-level transfer are in place for this site:"
    echo ""
    echo "  sbatch --export=ALL,I_HAVE_CONFIRMED_THE_DUA=yes hpc/05_tier2.sbatch"
    echo ""
    echo "This gate exists because a flag on a command invites somebody to"
    echo "pass it without the question having been asked."
    exit 1
fi

python code/02_site_embeddings.py --from-stage purity --min-cell "${MIN_CELL:-11}"

echo ""
echo "===================================================================="
echo "TIER 2 DELIVERABLE:  output/upload_to_box_${SITE}/  -- the WHOLE directory"
echo "  patient/ has been added to the tier-1 bundle. Send this one directory;"
echo "  nothing else needs to leave. Roughly 45 MB per 10,000 patients."
echo "  NOT de-identified."
echo "===================================================================="
