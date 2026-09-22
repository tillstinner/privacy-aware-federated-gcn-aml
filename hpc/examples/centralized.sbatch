#!/usr/bin/env bash
#SBATCH --job-name=aml-centralized
#SBATCH --output=slurm-%x-%j.out
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1

set -euo pipefail

: "${DATA_ROOT:?Set DATA_ROOT to an AMLSim output directory}"
OUTPUT_DIR="${OUTPUT_DIR:-runs/centralized}"
PYTHON="${PYTHON:-python}"

"$PYTHON" -m federated_gcn_aml.experiments.run_gcn_amlsim \
  --data-root "$DATA_ROOT" \
  --output-dir "$OUTPUT_DIR" \
  --representation-version pass2 \
  --feature-boundary privacy_clean \
  --edge-weight-mode unit \
  --epochs "${EPOCHS:-1000}" \
  --hidden-channels "${HIDDEN_CHANNELS:-128}" \
  --num-layers 3 \
  --dropout "${DROPOUT:-0.2}" \
  --device cuda
