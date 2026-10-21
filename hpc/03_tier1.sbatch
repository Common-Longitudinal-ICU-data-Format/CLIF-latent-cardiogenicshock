#!/bin/bash
#SBATCH --job-name=clifcs-tier1
#SBATCH --partition=tier1q
#SBATCH --cpus-per-task=16
#SBATCH --mem=64000M
#SBATCH --time=04:00:00
#SBATCH --output=logs/slurm-%x-%j.out
#SBATCH --error=logs/slurm-%x-%j.out
#
# Stage 3 of 3 for TIER 1: tokencounts -> trials -> export. CPU, no GPU.
#
# This is the last job a site has to run. After it, output/upload_to_box_<SITE>/
# is a complete contribution: patient flow, the phenotype breakdowns, the trial
# comparison and the token-prevalence tables. Nothing in it is patient-level and
# the GEM weights were never loaded, so a site with no GPU and no patient-level
# agreement stops here.
#
# The `tokencounts` stage counts how often each token appears at this site. It
# is what 04_site_effect.py draws its tornado from, and it is tokenizer-only --
# which is why it sits in tier 1, and why the comparison now exists for a site
# that shares nothing patient-level.
#
# The `trials` stage reads raw CLIF again -- diagnoses, vitals, medications --
# to compute eligibility, IQR fit and treatment concordance here, because the
# coordinating site cannot: under role=pooled it has no raw data and
# paths.site_data raises. It also writes trial_patient_anchor-*.parquet, which
# is what 02 exports those columns from, so a stale commit here silently
# carries fewer trials than the pooled run expects.

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

python code/01_site_tables.py --from-stage tokencounts --min-cell "${MIN_CELL:-11}"

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
echo ""
echo "TIER 1 IS COMPLETE AND IS A FULL CONTRIBUTION. Stop here unless your"
echo "IRB approval and DUA cover patient-level transfer. If they do:"
echo "    sbatch hpc/04_embed.sbatch     (GPU -- loads the GEM weights)"
echo "    sbatch hpc/05_tier2.sbatch     (CPU -- purity, then export)"
echo "===================================================================="
