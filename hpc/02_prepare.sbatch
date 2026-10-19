#!/bin/bash
#SBATCH --job-name=clifcs-prepare
#SBATCH --partition=general
#SBATCH --cpus-per-task=16
#SBATCH --mem=192G
#SBATCH --time=12:00:00
#SBATCH --output=logs/slurm-%x-%j.out
#SBATCH --error=logs/slurm-%x-%j.out
#
# Stage 2 of 4: subset -> derive -> collate -> tokenize -> winnow. All CPU.
#
# No GPU is requested because nothing here touches one. Running these stages
# inside the GPU allocation works and wastes most of it.
#
# `derive` is the memory peak of the whole pipeline -- clifpy is pandas-based.
# If this job is OOM-killed, raise --mem and resume with
#
#     python code/01_site_aggregate.py --from-stage derive
#
# rather than starting over. The stages are resumable and their outputs are
# on scratch.

source "$(dirname "$0")/_common.sh"

for stage in subset derive collate tokenize winnow; do
    echo ""
    echo "--- stage: $stage  ($(date -Is)) ---"
    python code/01_site_aggregate.py --stage "$stage"
done

echo ""
echo "Tokenized and winnowed. Next: hpc/03_embed.sbatch (GPU)."
echo "If the GPU queue is deep, check the scratch purge policy first --"
echo "these intermediates have to survive until that job starts."
