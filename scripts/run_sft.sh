#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
train_jsonl="${TRAIN_JSONL:?Set TRAIN_JSONL to the filtered SFT corpus}"
output_dir="${OUTPUT_DIR:-$repo_root/.runtime/sft_run}"

export HF_HOME="${HF_HOME:-$repo_root/.runtime/hf_home}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$repo_root/.runtime/xdg_cache}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
mkdir -p "$output_dir" "$HF_HOME" "$XDG_CACHE_HOME"

args=(
  "$repo_root/sft/train_unsloth_sft.py"
  --dataset "$train_jsonl"
  --model-name "${MODEL_NAME:-Qwen/Qwen3.5-4B}"
  --output-dir "$output_dir"
  --max-seq-length "${MAX_SEQ_LENGTH:-40960}"
  --num-train-epochs "${NUM_TRAIN_EPOCHS:-3}"
  --per-device-train-batch-size "${PER_DEVICE_TRAIN_BATCH_SIZE:-1}"
  --gradient-accumulation-steps "${GRADIENT_ACCUMULATION_STEPS:-2}"
  --learning-rate "${LEARNING_RATE:-2e-4}"
  --warmup-steps "${WARMUP_STEPS:-20}"
  --save-steps "${SAVE_STEPS:-4730}"
  --save-total-limit "${SAVE_TOTAL_LIMIT:-4}"
  --logging-steps "${LOGGING_STEPS:-5}"
  --report-to "${REPORT_TO:-none}"
)

if [[ -n "${EVAL_JSONL:-}" ]]; then
  args+=(--eval-dataset "$EVAL_JSONL" --eval-steps "${EVAL_STEPS:-4730}")
fi
if [[ "${LOCAL_FILES_ONLY:-0}" == 1 ]]; then
  args+=(--local-files-only)
fi

torchrun --standalone --nproc_per_node="${NPROC_PER_NODE:-8}" "${args[@]}"
