#!/bin/bash
#SBATCH --job-name=nestedcv_folds
#SBATCH --partition=etileno
#SBATCH --time=2-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=99G
#SBATCH --array=1-5%2
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

set -euo pipefail
mkdir -p logs

source /home/santos.arthur/Nuno/meuambiente/bin/activate

# Evita oversubscription com BLAS dentro dos workers:
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

DATASET="Input_Total.xlsx"
ELM="elm.py"
ART="./nestedcv_artifacts"

python nestedcv_run_fold.py \
  --dataset "$DATASET" \
  --elm "$ELM" \
  --artifacts "$ART" \
  --fold "${SLURM_ARRAY_TASK_ID}" \
  --outer_splits 5 \
  --inner_splits 5 \
  --pearson_thresh 0.59 \
  --random_state 42 \
  --save_models

