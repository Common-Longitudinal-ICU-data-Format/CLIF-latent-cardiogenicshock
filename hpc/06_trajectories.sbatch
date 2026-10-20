#!/bin/bash
#SBATCH --job-name=clifcs-trajectories
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32000M
#SBATCH --time=24:00:00
#SBATCH --output=logs/slurm-%x-%j.out
#SBATCH --error=logs/slurm-%x-%j.out
#
# TRAJECTORIES -- the co-located analysis, NOT part of the federated tiers.
#
# This job is different in kind from 01-05. Those produce a bundle a site
# sends; this one runs where all three sites CLIF extracts sit on the same
# filesystem, pools them directly, and sends nothing. There is no export
# contract on it because nothing crosses a boundary.
#
# WHAT IT COSTS. The extract stage winnows and embeds every patient at every
# grid point: 26 points by default (-24h to +48h in 6h steps), so it is
# roughly 26x the work of one anchor in 04_embed. Budget accordingly --
# 04_embed takes about 2 h for 34k patients across two anchors, so three
# sites over 26 points is a long job and the 24 h here is not generous.
# Resumable: the extract stage skips grid points whose output already exists,
# so a job that runs out of walltime can simply be resubmitted.
#
# SCRATCH. The grid embeddings are the biggest thing this pipeline writes --
# about 100 MB per 1000 patients per grid point at 1024 float32, plus the
# winnowed token shards, which are larger. They go to scratch_home, not to
# output/. Check you have the room before submitting.
#
# ORDER OF OPERATIONS. 05 must have run with --save-map first. 09 loads that
# map and applies it to every grid point BEFORE projecting, because the map is
# affine on the 1024-d embedding and correcting PaCMAP coordinates instead
# would be meaningless. It deliberately does not fit its own map: refitting on
# the grid would weight the fit by whose grid is complete.
#
#     python code/05_site_correction.py --anchor onset --save-map
#     sbatch hpc/06_trajectories.sbatch
#
# Set CORRECTION=none to draw the uncorrected field beside it. The two are
# written to different directories, so neither overwrites the other.

NEEDS_GPU=1
# `$(dirname "$0")` DOES NOT WORK IN A BATCH JOB. Slurm copies the script
# into a per-job spool directory and runs it from there, so $0 is
# /var/spool/slurmd/job<id>/<script> and the repo is nowhere near it.
set -euo pipefail

REPO_HOME="${REPO_HOME:-${SLURM_SUBMIT_DIR:-$PWD}}"
if [ ! -f "$REPO_HOME/hpc/_common.sh" ]; then
    echo "cannot find hpc/_common.sh under REPO_HOME=$REPO_HOME" >&2
    echo "Submit from the repository root, or set REPO_HOME:" >&2
    echo "  sbatch --export=ALL,REPO_HOME=/path/to/repo hpc/06_trajectories.sbatch" >&2
    exit 1
fi
source "$REPO_HOME/hpc/_common.sh"

CORRECTION="${CORRECTION:-leace}"

if [ "$CORRECTION" != "none" ]; then
    MAP="output/patient_level_data/${MODEL_KEY:-GEM-all}/corrections/anchor-onset_${CORRECTION}.npz"
    if [ ! -f "$MAP" ]; then
        echo "=================================================================="
        echo "NO SAVED CORRECTION MAP at $MAP"
        echo ""
        echo "09 applies the map 05 fitted at the anchor; it does not fit its"
        echo "own. Run this first (CPU, minutes):"
        echo ""
        echo "    python code/05_site_correction.py --anchor onset --save-map"
        echo ""
        echo "Or draw the uncorrected field:"
        echo "    sbatch --export=ALL,CORRECTION=none hpc/06_trajectories.sbatch"
        echo "=================================================================="
        exit 1
    fi
fi

python -c "
import torch, sys
if not torch.cuda.is_available():
    sys.exit('no CUDA device visible inside the job.')
print(f'  gpu      {torch.cuda.get_device_name(0)}')
print(f'  vram     {torch.cuda.get_device_properties(0).total_memory/2**30:.1f} GiB')
"

echo ""
echo "--- stage: extract  ($(date -Is)) ---"
python code/09_trajectories.py --stage extract \
    --batch-size "${BATCH_SIZE:-32}" \
    --start-hours "${START_HOURS:--24}" \
    --end-hours "${END_HOURS:-48}" \
    --step-hours "${STEP_HOURS:-6}"

echo ""
echo "--- stage: plot  ($(date -Is)) ---"
python code/09_trajectories.py --stage plot \
    --correction "$CORRECTION" \
    --correction-anchor onset \
    --start-hours "${START_HOURS:--24}" \
    --end-hours "${END_HOURS:-48}" \
    --step-hours "${STEP_HOURS:-6}"

echo ""
echo "===================================================================="
echo "TRAJECTORIES DONE. Figures under:"
echo "  output/figures/<model>/trajectory/corrected-${CORRECTION}/"
echo ""
echo "The headline pair, drawn with identical bins and mask thresholds:"
echo "  trajectory_flow_cardiogenic_coded.png"
echo "  trajectory_flow_cardiogenic_cardiac_origin.png"
echo "and the Zweck-coloured scatter beside each. The log reports how far the"
echo "two definitions overlap -- they are two readings of one cohort, not two"
echo "independent findings."
echo ""
echo "Nothing here is exportable. It is patient-level and co-located by"
echo "design; no export contract was applied to any of it."
echo "===================================================================="
