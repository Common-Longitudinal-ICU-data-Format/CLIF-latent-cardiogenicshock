#!/bin/bash
#SBATCH --job-name=clifcs-export
#SBATCH --partition=general
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=logs/slurm-%x-%j.out
#SBATCH --error=logs/slurm-%x-%j.out
#
# Stage 4 of 4: purity -> trials -> export. CPU.
#
# The `trials` stage reads raw CLIF again -- diagnoses, vitals, medications --
# to compute trial eligibility here, because the coordinating site cannot:
# under role=pooled it has no raw data and paths.site_data raises. The elig_*
# columns in the bundle are what pooled 11 reads, so a bundle built from a
# stale commit silently carries fewer trials than the pooled run expects.

source "$(dirname "$0")/_common.sh"

python code/01_site_aggregate.py --from-stage purity --min-cell "${MIN_CELL:-11}"

echo ""
echo "===================================================================="
echo "TIER 1 DELIVERABLE:  output/upload_to_box_${SITE}/"
echo "  Aggregate only, small cells suppressed at --min-cell ${MIN_CELL:-11}."
echo "  This is what you send. Nothing patient-level is in it."
echo ""
echo "BACK UP THE SALT. It is the one unrecoverable file in this pipeline:"
echo "  cp output/patient_level_data/.id_salt ~/clif-cs-salt-backup-\$(date +%F)"
echo ""
echo "  Never send it. If it is lost, every future export carries pseudonyms"
echo "  that cannot be matched to the ones already sent, and this site's"
echo "  contribution can be replaced but not updated."
echo "===================================================================="
