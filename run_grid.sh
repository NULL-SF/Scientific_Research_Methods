#!/usr/bin/env bash
set -euo pipefail

# Grid runner for train.sh
# - Define DATASETS (relative to data/) and MODELS (relative to models/)
# - Runs all combinations sequentially
# - OUTPUT_DIR auto-includes dataset+model; AUTO_SUFFIX=1 prevents overwrite

BASE_DIR="/home/kemove/wsy/PLM"
TRAIN_SH="${BASE_DIR}/train.sh"

# ====== EDIT THESE LISTS ======
# Example datasets: (use paths after data/)
DATASETS=(
  # "WN18RR_tc"
  # "FB15k-237_tc"
  "YAGO3-10_tc"
)

# Example models: (use paths after models/)
MODELS=(
  # "google-bert/bert-base-uncased"
  # "SpanBERT/spanbert-base-cased"
  # "allenai/scibert_scivocab_uncased"
  "google-bert/bert-base-cased"
)

# ====== OPTIONAL GLOBAL SETTINGS ======
EVAL_EVERY_STEPS=${EVAL_EVERY_STEPS:-50}
PLOT_METRICS=${PLOT_METRICS:-1}
SMOOTH_ALPHA=${SMOOTH_ALPHA:-0.2}
MAX_SEQ_LEN=${MAX_SEQ_LEN:-20}
TRAIN_BS=${TRAIN_BS:-32}
EVAL_BS=${EVAL_BS:-512}
LR=${LR:-5e-5}
EPOCHS=${EPOCHS:-1.0}
GRAD_ACC=${GRAD_ACC:-1}
NUM_WORKERS=${NUM_WORKERS:-16}
MP_CHUNKSIZE=${MP_CHUNKSIZE:-500}
BUILD_WORKERS=${BUILD_WORKERS:-16}
BUILD_CHUNKSIZE=${BUILD_CHUNKSIZE:-500}

timestamp() { date +%Y-%m-%dT%H:%M:%S; }

echo "[INFO] $(timestamp) Starting grid run with ${#DATASETS[@]} datasets x ${#MODELS[@]} models"

for ds in "${DATASETS[@]}"; do
  for model in "${MODELS[@]}"; do
    # Sanitize names for directory
    safe_ds="${ds//\//-}"
    safe_model="${model//\//-}"
    safe_model="${safe_model// /_}"
    OUTPUT_DIR="${BASE_DIR}/output_${safe_ds}_${safe_model}"

    DATA_DIR="${BASE_DIR}/data/${ds}"
    BERT_MODEL="${BASE_DIR}/models/${model}"

    echo "[INFO] $(timestamp) Running dataset='${ds}' model='${model}' -> OUTPUT_DIR='${OUTPUT_DIR}'"

    # Per-run overrides passed as env vars to train.sh
    DATA_DIR="${DATA_DIR}" \
    BERT_MODEL="${BERT_MODEL}" \
    OUTPUT_DIR="${OUTPUT_DIR}" \
    MAX_SEQ_LEN="${MAX_SEQ_LEN}" \
    TRAIN_BS="${TRAIN_BS}" \
    EVAL_BS="${EVAL_BS}" \
    LR="${LR}" \
    EPOCHS="${EPOCHS}" \
    GRAD_ACC="${GRAD_ACC}" \
    NUM_WORKERS="${NUM_WORKERS}" \
    MP_CHUNKSIZE="${MP_CHUNKSIZE}" \
    BUILD_WORKERS="${BUILD_WORKERS}" \
    BUILD_CHUNKSIZE="${BUILD_CHUNKSIZE}" \
    EVAL_EVERY_STEPS="${EVAL_EVERY_STEPS}" \
    PLOT_METRICS="${PLOT_METRICS}" \
    SMOOTH_ALPHA="${SMOOTH_ALPHA}" \
    AUTO_SUFFIX=1 \
    bash "${TRAIN_SH}"

    echo "[INFO] $(timestamp) Finished dataset='${ds}' model='${model}'"
  done
done

echo "[INFO] $(timestamp) Grid run complete."


