#!/usr/bin/env bash
set -euo pipefail

# Minimal training launcher for triple classification (KG-BERT style)
# This repo depends on pytorch-pretrained-bert (legacy). We pin versions to avoid API drift.

PYTHON_BIN=${PYTHON_BIN:-python3}
DATA_DIR=${DATA_DIR:-"./data/WN18RR_tc"}
OUTPUT_DIR=${OUTPUT_DIR:-"./output_WN18RR_E"}
# BERT_MODEL=${BERT_MODEL:-"./models/google-bert/bert-base-uncased"}
# BERT_MODEL=${BERT_MODEL:-"./models/distilbert/distilbert-base-uncased"}
# BERT_MODEL=${BERT_MODEL:-"./models/SpanBERT/spanbert-base-cased"}
BERT_MODEL=${BERT_MODEL:-"./models/google/electra-base-discriminator"}
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
DEBUG=${DEBUG:-0}
DEBUG_SAMPLES=${DEBUG_SAMPLES:-10}
OVERWRITE=${OVERWRITE:-0}
AUTO_SUFFIX=${AUTO_SUFFIX:-1}
CUDA_DEVICE=${CUDA_DEVICE:-0}
SKIP_INSTALL=${SKIP_INSTALL:-0}
EVAL_EVERY_STEPS=${EVAL_EVERY_STEPS:-50}
PLOT_METRICS=${PLOT_METRICS:-1}
METRICS_PREFIX=${METRICS_PREFIX:-metrics}
SMOOTH_ALPHA=${SMOOTH_ALPHA:-0.2}

EFFECTIVE_DATA_DIR="${DATA_DIR}"

# Handle existing OUTPUT_DIR
if [[ -d "${OUTPUT_DIR}" ]] && [[ -n "$(ls -A "${OUTPUT_DIR}" 2>/dev/null || true)" ]]; then
  if [[ "${OVERWRITE}" == "1" ]]; then
    rm -rf "${OUTPUT_DIR}"
    mkdir -p "${OUTPUT_DIR}"
  elif [[ "${AUTO_SUFFIX}" == "1" ]]; then
    ts=$(date +%Y%m%d_%H%M%S)
    OUTPUT_DIR="${OUTPUT_DIR}_${ts}"
    mkdir -p "${OUTPUT_DIR}"
    echo "OUTPUT_DIR existed and not empty. Using suffixed dir: ${OUTPUT_DIR}"
  else
    echo "ERROR: OUTPUT_DIR exists and is not empty. Set OVERWRITE=1 or AUTO_SUFFIX=1."
    exit 1
  fi
else
  mkdir -p "${OUTPUT_DIR}"
fi

# If DEBUG is enabled, create a tiny dataset with only DEBUG_SAMPLES per split
if [[ "${DEBUG}" == "1" ]]; then
  DEBUG_DATA_DIR=$(mktemp -d -t kgbert_debug.XXXXXX)
  # Copy everything except the split TSVs, then write truncated TSVs
  rsync -a --exclude 'train.tsv' --exclude 'dev.tsv' --exclude 'test.tsv' "${DATA_DIR}/" "${DEBUG_DATA_DIR}/" || true
  for split in train dev test; do
    if [[ -f "${DATA_DIR}/${split}.tsv" ]]; then
      head -n "${DEBUG_SAMPLES}" "${DATA_DIR}/${split}.tsv" > "${DEBUG_DATA_DIR}/${split}.tsv"
    fi
  done
  EFFECTIVE_DATA_DIR="${DEBUG_DATA_DIR}"
  echo "DEBUG mode ON: using ${DEBUG_SAMPLES} samples per split from ${EFFECTIVE_DATA_DIR}"
  # Force single-GPU to avoid legacy DataParallel issues on tiny batches
  export CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}"
  echo "DEBUG mode: using single GPU CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
fi

# Optional: create and use a venv if VENV_DIR is set
if [[ -n "${VENV_DIR:-}" ]]; then
  python3 -m venv "$VENV_DIR"
  # shellcheck disable=SC1090
  source "$VENV_DIR/bin/activate"
fi

# Always skip package installation; assume environment is pre-configured

${PYTHON_BIN} run_bert_triple_classifier.py \
  --task_name kg \
  --do_train --do_eval --do_predict \
  --data_dir "${EFFECTIVE_DATA_DIR}" \
  --bert_model "${BERT_MODEL}" \
  --max_seq_length "${MAX_SEQ_LEN}" \
  --train_batch_size "${TRAIN_BS}" \
  --learning_rate "${LR}" \
  --num_train_epochs "${EPOCHS}" \
  --output_dir "${OUTPUT_DIR}" \
  --gradient_accumulation_steps "${GRAD_ACC}" \
  --eval_batch_size "${EVAL_BS}" \
  --num_workers "${NUM_WORKERS}" \
  --mp_chunksize "${MP_CHUNKSIZE}" \
  --build_workers "${BUILD_WORKERS}" \
  --build_chunksize "${BUILD_CHUNKSIZE}" \
  --eval_every_steps "${EVAL_EVERY_STEPS}" \
  $( (( PLOT_METRICS )) && echo --plot_metrics || true ) \
  --metrics_file_prefix "${METRICS_PREFIX}" \
  --smooth_alpha "${SMOOTH_ALPHA}" \
  --disable_data_parallel

echo "Training finished. Metrics saved under ${OUTPUT_DIR}."

# Cleanup debug temp dir
if [[ "${DEBUG}" == "1" ]] && [[ -n "${DEBUG_DATA_DIR:-}" ]]; then
  rm -rf "${DEBUG_DATA_DIR}" || true
fi