#!/bin/bash
#SBATCH --job-name=nestedcv_finalize
#SBATCH --partition=etileno
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=99G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

# IMPORTANTE: o diretório "logs" precisa existir ANTES do SLURM tentar abrir o arquivo de output.
# Então crie fora (recomendado):  mkdir -p logs
# (Manter aqui também não atrapalha)
mkdir -p logs

source /home/santos.arthur/Nuno/meuambiente/bin/activate

DATASET="Input_Total.xlsx"
ELM="elm.py"
ART="./nestedcv_artifacts"

python -u nestedcv_finalize.py \
  --dataset "$DATASET" \
  --elm "$ELM" \
  --artifacts "$ART" \
  --outer_splits 5 \
  --pearson_thresh 0.59 \
  --random_state 42 \
  --final_feature_policy all_data \
  --calibrate_final \
  --cal_method sigmoid \
  --cal_cv 5

