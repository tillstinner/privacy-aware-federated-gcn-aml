#!/usr/bin/env bash
#SBATCH --job-name=aml-fed-private
#SBATCH --output=slurm-%x-%j.out
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gres=gpu:1

set -euo pipefail

: "${DATA_ROOT:?Set DATA_ROOT to an AMLSim output directory}"
OUTPUT_DIR="${OUTPUT_DIR:-runs/federated-private}"
PYTHON="${PYTHON:-python}"

"$PYTHON" -m federated_gcn_aml.experiments.run_federated_amlsim \
  --data-root "$DATA_ROOT" \
  --output-dir "$OUTPUT_DIR" \
  --feature-boundary privacy_clean \
  --federated-mode fedgcn_feature_pretrain \
  --num-hops "${NUM_HOPS:-2}" \
  --feature-pretrain-transport he \
  --feature-pretrain-payload-protection none \
  --model-update-aggregation masked_fedavg \
  --rounds "${ROUNDS:-1000}" \
  --local-epochs "${LOCAL_EPOCHS:-8}" \
  --hidden-channels "${HIDDEN_CHANNELS:-128}" \
  --num-layers 3 \
  --dropout "${DROPOUT:-0.2}" \
  --device cuda
