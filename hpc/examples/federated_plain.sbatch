#!/usr/bin/env bash
#SBATCH --job-name=aml-fed-plain
#SBATCH --output=slurm-%x-%j.out
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=gpu:1

set -euo pipefail

: "${DATA_ROOT:?Set DATA_ROOT to an AMLSim output directory}"
OUTPUT_DIR="${OUTPUT_DIR:-runs/federated-plain}"
PYTHON="${PYTHON:-python}"

"$PYTHON" -m federated_gcn_aml.experiments.run_federated_amlsim \
  --data-root "$DATA_ROOT" \
  --output-dir "$OUTPUT_DIR" \
  --feature-boundary privacy_clean \
  --federated-mode fedgcn_feature_pretrain \
  --num-hops "${NUM_HOPS:-2}" \
  --feature-pretrain-transport plain \
  --feature-pretrain-payload-protection none \
  --model-update-aggregation plain_fedavg \
  --rounds "${ROUNDS:-1000}" \
  --local-epochs "${LOCAL_EPOCHS:-8}" \
  --hidden-channels "${HIDDEN_CHANNELS:-128}" \
  --num-layers 3 \
  --dropout "${DROPOUT:-0.2}" \
  --device cuda
