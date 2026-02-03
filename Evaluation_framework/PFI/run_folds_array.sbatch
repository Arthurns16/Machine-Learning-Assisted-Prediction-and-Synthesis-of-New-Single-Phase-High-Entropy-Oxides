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

# Fold atual do job-array
FOLD="${SLURM_ARRAY_TASK_ID}"

echo "[INFO] Running outer fold ${FOLD} / 5"
echo "[INFO] CPUs allocated: ${SLURM_CPUS_PER_TASK:-unknown}"
echo "[INFO] Host: $(hostname)"
echo "[INFO] Start time: $(date)"

python nestedcv_run_fold_pfi_top10_pfi_parallel.py \
  --dataset "${DATASET}" \
  --elm "${ELM}" \
  --artifacts "${ART}" \
  --fold "${FOLD}" \
  --outer_splits 5 \
  --inner_splits 5 \
  --n_jobs 0 \
  --pfi_repeats 10 \
  --pfi_n_jobs 0

echo "[INFO] Done fold ${FOLD} at $(date)"

