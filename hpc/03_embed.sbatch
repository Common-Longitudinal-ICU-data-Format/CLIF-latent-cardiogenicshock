#!/bin/bash
#SBATCH --job-name=clifcs-embed
#SBATCH --partition=pearsonq
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=logs/slurm-%x-%j.out
#SBATCH --error=logs/slurm-%x-%j.out
#
# Stage 3 of 4: the only stage in the pipeline that needs a GPU.
#
# BATCH SIZE. The pipeline default of 8 is sized for an 8 GB card; this script
# defaults to 32 for a datacentre GPU. output_hidden_states keeps every layer,
# so memory scales with batch size and sequence length together and a single
# long-sequence batch is the usual OOM trigger. Halve it and resubmit:
#
#     sbatch --export=ALL,BATCH_SIZE=16 hpc/03_embed.sbatch
#
# WALLTIME. Measured on 34,142 patients across both anchors at batch size 8:
# 1 h 58 m wall, about 0.11-0.17 s per patient per anchor. Two anchors run, so
# budget roughly 0.3 s x n_patients and double it for headroom. The 8 h here is
# generous for a cohort of that size and should be trimmed to whatever the
# queue rewards.

NEEDS_GPU=1
source "$(dirname "$0")/_common.sh"

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

python code/01_site_aggregate.py --stage embed --batch-size "${BATCH_SIZE:-32}"

echo ""
echo "Embeddings are in output/patient_level_data/ and STAY LOCAL."
echo "Next: hpc/04_export.sbatch (CPU)."
