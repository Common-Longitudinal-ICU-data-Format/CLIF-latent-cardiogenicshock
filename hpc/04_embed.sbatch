#!/bin/bash
#SBATCH --job-name=clifcs-embed
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32000M
#SBATCH --time=08:00:00
#SBATCH --output=logs/slurm-%x-%j.out
#SBATCH --error=logs/slurm-%x-%j.out
#
# Stage 1 of 2 for TIER 2: the only stage in the pipeline that needs a GPU.
#
# TIER 2 -- DO NOT RUN WITHOUT THE IRB APPROVAL AND DUA IN PLACE. This is where
# the GEM weights are loaded for the first time; tier 1 (jobs 01-03) never
# touches them. A site whose agreement does not cover patient-level transfer
# has no reason to submit this job at all.
#
# BATCH SIZE. The pipeline default of 8 is sized for an 8 GB card; this script
# defaults to 32 for a datacentre GPU. output_hidden_states keeps every layer,
# so memory scales with batch size and sequence length together and a single
# long-sequence batch is the usual OOM trigger. Halve it and resubmit:
#
#     sbatch --export=ALL,BATCH_SIZE=16 hpc/04_embed.sbatch
#
# WALLTIME. Measured on 34,142 patients across both anchors at batch size 8:
# 1 h 58 m wall, about 0.11-0.17 s per patient per anchor. Two anchors run, so
# budget roughly 0.3 s x n_patients and double it for headroom. The 8 h here is
# generous for a cohort of that size and should be trimmed to whatever the
# queue rewards.

NEEDS_GPU=1
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

python -c "
import torch, sys
if not torch.cuda.is_available():
    sys.exit(
        'no CUDA device visible inside the job. Load the CUDA module in '
        '_common.sh, and check that torch was not reinstalled over the '
        'cu121 wheels by a later pip install.'
    )
print(f'  gpu      {torch.cuda.get_device_name(0)}')
print(f'  vram     {torch.cuda.get_device_properties(0).total_memory/2**30:.1f} GiB')
print(f'  torch    {torch.__version__}')
"

python code/02_site_embeddings.py --stage embed --batch-size "${BATCH_SIZE:-32}"

echo ""
echo "Embeddings are in output/patient_level_data/ and STAY LOCAL."
echo "Next: hpc/05_tier2.sbatch (CPU) redacts and exports."
