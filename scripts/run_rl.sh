#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
verl_src="${VERL_SRC:-$repo_root/third_party/verl-rolling-bcb638-full}"
verl_recipe_src="${VERL_RECIPE_SRC:-$repo_root/third_party/verl-recipe-rolling-ba246-full}"
general_deps="$repo_root/containers/python_deps"
megatron_deps="$repo_root/containers/python_deps_megatron_bridge"

# Patched by Codex for MolLang training: Qwen3.5 Megatron needs both the
# Megatron-Bridge overlay and the general MolLang CUDA deps (FLA/causal-conv1d).
export PYTHONPATH="$megatron_deps:$general_deps:$verl_recipe_src:$repo_root:$verl_src:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export HF_HOME="${HF_HOME:-$repo_root/.runtime/hf_home}"
export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-1}"
export VLLM_USE_V1="${VLLM_USE_V1:-1}"
export VLLM_ALLREDUCE_USE_SYMM_MEM="${VLLM_ALLREDUCE_USE_SYMM_MEM:-0}"
export VLLM_DISABLE_LOG_STATS="${VLLM_DISABLE_LOG_STATS:-False}"
export VLLM_LOGGING_LEVEL="${VLLM_LOGGING_LEVEL:-INFO}"
export VLLM_ENGINE_ITERATION_TIMEOUT_S="${VLLM_ENGINE_ITERATION_TIMEOUT_S:-600}"
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS="${VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS:-1800}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
export RAY_TMPDIR="${RAY_TMPDIR:-/tmp/rlv_mg_${USER:-user}_${SLURM_JOB_ID:-local}}"
export MOLLANG_VERL_PATCH_WANDB_HISTORY="${MOLLANG_VERL_PATCH_WANDB_HISTORY:-1}"
export MOLLANG_VERL_PATCH_MEMORY_INSTRUMENTATION="${MOLLANG_VERL_PATCH_MEMORY_INSTRUMENTATION:-1}"
export MOLLANG_REWARD_CONFIG_JSON="${MOLLANG_REWARD_CONFIG_JSON:-{\"tanimoto_weight\":0.8,\"exact_weight\":0.4,\"format_weight\":0.1,\"reasoning_length_penalty_weight\":0.0}}"

max_prompt="${MAX_PROMPT_LENGTH:-2048}"
max_response="${MAX_RESPONSE_LENGTH:-32768}"
max_model_len="${MAX_MODEL_LEN:-$((max_prompt + max_response))}"
overlong_expected_tokens="${OVERLONG_EXPECTED_TOKENS:-8192}"
if (( max_response > overlong_expected_tokens )); then
  overlong_buffer_len="${OVERLONG_BUFFER_LEN:-$((max_response - overlong_expected_tokens))}"
else
  overlong_buffer_len="${OVERLONG_BUFFER_LEN:-0}"
fi

model_path="${MODEL_PATH:?Set MODEL_PATH to the merged SFT base model directory}"
run_name="${RUN_NAME:-langmoldiode_qwen35_dapo_$(date +%Y%m%d_%H%M%S)}"
output_dir="${OUTPUT_DIR:-$repo_root/runs/$run_name}"
log_path="${LOG_PATH:-$repo_root/logs/${run_name}.log}"
mkdir -p "$output_dir" "$repo_root/logs"
export WANDB_DIR="${WANDB_DIR:-$output_dir/wandb}"
mkdir -p "$WANDB_DIR"

train_file="${TRAIN_FILE:?Set TRAIN_FILE to the filtered verl training parquet}"
val_file="${VAL_FILE:?Set VAL_FILE to the combined validation parquet}"
lora_target_modules_json="${MEGATRON_LORA_TARGET_MODULES_JSON:-[\"language_model.decoder.layers.*.self_attention.linear_qkv\",\"language_model.decoder.layers.*.self_attention.linear_proj\",\"language_model.decoder.layers.*.mlp.linear_fc1\",\"language_model.decoder.layers.*.mlp.linear_fc2\"]}"

actor_use_dynamic_bsz="${ACTOR_USE_DYNAMIC_BSZ:-False}"
ppo_micro_batch="${PPO_MICRO_BATCH_SIZE_PER_GPU:-1}"
logprob_micro_batch="${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-1}"
ref_logprob_micro_batch="${REF_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-1}"
rollout_correction_bypass_mode="${ROLLOUT_CORRECTION_BYPASS_MODE:-False}"
rollout_is="${ROLLOUT_IS:-null}"
rollout_is_threshold="${ROLLOUT_IS_THRESHOLD:-2.0}"
rollout_is_batch_normalize="${ROLLOUT_IS_BATCH_NORMALIZE:-False}"
rollout_rs="${ROLLOUT_RS:-null}"
rollout_rs_threshold="${ROLLOUT_RS_THRESHOLD:-null}"

echo "[megatron-full] host=$(hostname)"
echo "[megatron-full] model_path=$model_path"
echo "[megatron-full] train=$train_file val=$val_file"
echo "[megatron-full] run_name=$run_name output=$output_dir log=$log_path"
echo "[megatron-full] fixed_bsz actor_dynamic=$actor_use_dynamic_bsz ppo_micro=$ppo_micro_batch logprob_micro=$logprob_micro_batch"
echo "[megatron-full] rollout n=${NUM_GENERATIONS:-8} train_batch=${TRAIN_BATCH_SIZE:-512} gen_batch=${GEN_BATCH_SIZE:-640} ppo_mini_batch=${PPO_MINI_BATCH_SIZE:-32}"
echo "[megatron-full] rollout_log_probs=True correction_bypass=$rollout_correction_bypass_mode rollout_is=$rollout_is rollout_is_threshold=$rollout_is_threshold rollout_is_batch_normalize=$rollout_is_batch_normalize rollout_rs=$rollout_rs rollout_rs_threshold=$rollout_rs_threshold"
echo "[megatron-full] vllm gpu_mem=${VLLM_GPU_MEMORY_UTILIZATION:-0.90} seqs=${VLLM_MAX_NUM_SEQS:-256} bt=${VLLM_MAX_NUM_BATCHED_TOKENS:-8192}"
echo "[megatron-full] val_before_train=${VAL_BEFORE_TRAIN:-True} test_freq=${TEST_FREQ:-10} save_freq=${SAVE_FREQ:-1}"

if [[ ! -f "$model_path/config.json" ]]; then
  echo "Missing merged SFT model at $model_path." >&2
  exit 1
fi
if [[ ! -f "$train_file" || ! -f "$val_file" ]]; then
  echo "Missing data files: train=$train_file val=$val_file" >&2
  exit 1
fi

cd "$verl_src"
python -m dapo.main_dapo \
  --config-name=dapo_megatron_trainer \
  algorithm.adv_estimator=grpo \
  algorithm.use_kl_in_reward=False \
  algorithm.rollout_correction.bypass_mode="$rollout_correction_bypass_mode" \
  algorithm.rollout_correction.rollout_is="$rollout_is" \
  algorithm.rollout_correction.rollout_is_threshold="$rollout_is_threshold" \
  algorithm.rollout_correction.rollout_is_batch_normalize="$rollout_is_batch_normalize" \
  algorithm.rollout_correction.rollout_rs="$rollout_rs" \
  algorithm.rollout_correction.rollout_rs_threshold="$rollout_rs_threshold" \
  data.train_files="$train_file" \
  data.val_files="$val_file" \
  data.prompt_key=prompt \
  data.truncation=error \
  data.max_prompt_length="$max_prompt" \
  data.max_response_length="$max_response" \
  data.train_batch_size="${TRAIN_BATCH_SIZE:-512}" \
  data.train_max_samples="${TRAIN_MAX_SAMPLES:--1}" \
  data.val_max_samples="${VAL_MAX_SAMPLES:--1}" \
  data.val_batch_size="${VAL_BATCH_SIZE:-null}" \
  data.filter_overlong_prompts=False \
  data.shuffle=True \
  data.seed=533 \
  data.trust_remote_code=True \
  ++data.apply_chat_template_kwargs.enable_thinking=true \
  ++data.gen_batch_size="${GEN_BATCH_SIZE:-640}" \
  ++algorithm.filter_groups.enable="${FILTER_GROUPS_ENABLE:-True}" \
  ++algorithm.filter_groups.metric="${FILTER_GROUPS_METRIC:-seq_final_reward}" \
  ++algorithm.filter_groups.max_num_gen_batches="${FILTER_GROUPS_MAX_NUM_GEN_BATCHES:-10}" \
  reward.custom_reward_function.path="$repo_root/rl/reward.py" \
  reward.custom_reward_function.name=compute_score \
  reward.reward_manager.name="${REWARD_MANAGER:-dapo}" \
  ++reward.reward_kwargs.overlong_buffer_cfg.enable="${OVERLONG_REWARD:-True}" \
  ++reward.reward_kwargs.overlong_buffer_cfg.len="$overlong_buffer_len" \
  ++reward.reward_kwargs.overlong_buffer_cfg.penalty_factor="${OVERLONG_PENALTY_FACTOR:-0.4}" \
  ++reward.reward_kwargs.overlong_buffer_cfg.log="${OVERLONG_LOG:-True}" \
  ++reward.reward_kwargs.max_resp_len="$max_response" \
  actor_rollout_ref.model.path="$model_path" \
  actor_rollout_ref.model.trust_remote_code=True \
  actor_rollout_ref.model.use_remove_padding=False \
  actor_rollout_ref.model.lora.rank="${LORA_RANK:-64}" \
  actor_rollout_ref.model.lora.alpha="${LORA_ALPHA:-128}" \
  actor_rollout_ref.model.lora.dtype=bfloat16 \
  actor_rollout_ref.model.lora.merge="${MEGATRON_LORA_MERGE:-True}" \
  actor_rollout_ref.model.lora.target_modules="$lora_target_modules_json" \
  actor_rollout_ref.actor.optim.lr="${ACTOR_LR:-1e-6}" \
  actor_rollout_ref.actor.optim.weight_decay="${WEIGHT_DECAY:-0.0}" \
  actor_rollout_ref.actor.optim.lr_warmup_steps="${WARMUP_STEPS:-20}" \
  actor_rollout_ref.actor.optim.lr_decay_style=constant \
  actor_rollout_ref.actor.ppo_mini_batch_size="${PPO_MINI_BATCH_SIZE:-32}" \
  actor_rollout_ref.actor.ppo_epochs="${PPO_EPOCHS:-1}" \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="$ppo_micro_batch" \
  actor_rollout_ref.actor.use_dynamic_bsz="$actor_use_dynamic_bsz" \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu="${ACTOR_MAX_TOKEN_LEN_PER_GPU:-34816}" \
  actor_rollout_ref.actor.use_kl_loss=False \
  actor_rollout_ref.actor.kl_loss_coef=0.0 \
  actor_rollout_ref.actor.entropy_coeff="${ENTROPY_COEFF:-0}" \
  actor_rollout_ref.actor.loss_agg_mode="${LOSS_AGG_MODE:-token-mean}" \
  actor_rollout_ref.actor.megatron.use_mbridge=True \
  actor_rollout_ref.actor.megatron.vanilla_mbridge=False \
  actor_rollout_ref.actor.megatron.use_remove_padding=False \
  actor_rollout_ref.actor.megatron.tensor_model_parallel_size="${TP:-2}" \
  actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=1 \
  actor_rollout_ref.actor.megatron.context_parallel_size=1 \
  actor_rollout_ref.actor.megatron.param_offload="${ACTOR_PARAM_OFFLOAD:-False}" \
  actor_rollout_ref.actor.megatron.optimizer_offload="${ACTOR_OPTIMIZER_OFFLOAD:-False}" \
  actor_rollout_ref.actor.megatron.grad_offload="${ACTOR_GRAD_OFFLOAD:-False}" \
  actor_rollout_ref.actor.megatron.dtype=bfloat16 \
  ++actor_rollout_ref.actor.megatron.override_transformer_config.attention_backend=flash \
  +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_method=uniform \
  +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_granularity=full \
  +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_num_layers=1 \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.mode=async \
  actor_rollout_ref.rollout.tensor_model_parallel_size="${ROLLOUT_TP:-1}" \
  actor_rollout_ref.rollout.gpu_memory_utilization="${VLLM_GPU_MEMORY_UTILIZATION:-0.90}" \
  actor_rollout_ref.rollout.n="${NUM_GENERATIONS:-8}" \
  actor_rollout_ref.rollout.temperature="${TEMPERATURE:-1.0}" \
  actor_rollout_ref.rollout.top_p="${TOP_P:-1.0}" \
  actor_rollout_ref.rollout.top_k="${TOP_K:--1}" \
  ++actor_rollout_ref.rollout.repetition_penalty="${REPETITION_PENALTY:-1.0}" \
  actor_rollout_ref.rollout.val_kwargs.n="${VAL_GENERATIONS:-1}" \
  actor_rollout_ref.rollout.val_kwargs.do_sample="${VAL_DO_SAMPLE:-True}" \
  actor_rollout_ref.rollout.val_kwargs.temperature="${VAL_TEMPERATURE:-1.0}" \
  actor_rollout_ref.rollout.val_kwargs.top_p="${VAL_TOP_P:-0.95}" \
  actor_rollout_ref.rollout.val_kwargs.top_k="${VAL_TOP_K:-20}" \
  actor_rollout_ref.rollout.max_model_len="$max_model_len" \
  actor_rollout_ref.rollout.max_num_batched_tokens="${VLLM_MAX_NUM_BATCHED_TOKENS:-8192}" \
  actor_rollout_ref.rollout.max_num_seqs="${VLLM_MAX_NUM_SEQS:-256}" \
  actor_rollout_ref.rollout.enable_chunked_prefill=True \
  actor_rollout_ref.rollout.free_cache_engine=True \
  actor_rollout_ref.rollout.enforce_eager="${VLLM_ENFORCE_EAGER:-False}" \
  actor_rollout_ref.rollout.disable_log_stats="$VLLM_DISABLE_LOG_STATS" \
  actor_rollout_ref.rollout.calculate_log_probs=True \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu="$logprob_micro_batch" \
  actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=False \
  actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu="${LOG_PROB_MAX_TOKEN_LEN_PER_GPU:-34816}" \
  ++actor_rollout_ref.rollout.engine_kwargs.vllm.disable_custom_all_reduce="${VLLM_DISABLE_CUSTOM_ALL_REDUCE:-True}" \
  ++actor_rollout_ref.rollout.engine_kwargs.vllm.gdn_prefill_backend="${VLLM_GDN_PREFILL_BACKEND:-triton}" \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu="$ref_logprob_micro_batch" \
  actor_rollout_ref.ref.log_prob_use_dynamic_bsz=False \
  actor_rollout_ref.ref.log_prob_max_token_len_per_gpu="${REF_LOG_PROB_MAX_TOKEN_LEN_PER_GPU:-34816}" \
  actor_rollout_ref.ref.megatron.use_mbridge=True \
  actor_rollout_ref.ref.megatron.vanilla_mbridge=False \
  actor_rollout_ref.ref.megatron.use_remove_padding=False \
  actor_rollout_ref.ref.megatron.tensor_model_parallel_size="${TP:-2}" \
  actor_rollout_ref.ref.megatron.pipeline_model_parallel_size=1 \
  actor_rollout_ref.ref.megatron.context_parallel_size=1 \
  trainer.critic_warmup=0 \
  trainer.logger="${TRAINER_LOGGER:-['console','wandb']}" \
  trainer.project_name="${PROJECT_NAME:-LangMolDiode}" \
  trainer.experiment_name="$run_name" \
  trainer.n_gpus_per_node="${GPUS_PER_NODE:-8}" \
  trainer.nnodes="${NNODES:-1}" \
  trainer.balance_batch="${BALANCE_BATCH:-True}" \
  trainer.val_before_train="${VAL_BEFORE_TRAIN:-True}" \
  trainer.test_freq="${TEST_FREQ:-10}" \
  trainer.save_freq="${SAVE_FREQ:-1}" \
  trainer.total_epochs="${TOTAL_EPOCHS:-1}" \
  trainer.total_training_steps="${TOTAL_TRAINING_STEPS:-315}" \
  trainer.default_local_dir="$output_dir" \
  trainer.default_hdfs_dir=null \
  trainer.rollout_data_dir="${ROLLOUT_DATA_DIR:-$output_dir/rollouts}" \
  trainer.validation_data_dir="${VALIDATION_DATA_DIR:-$output_dir/validation_generations}" \
  trainer.log_val_generations="${LOG_VAL_GENERATIONS:-16}" \
  trainer.resume_mode="${RESUME_MODE:-disable}" \
  trainer.resume_from_path="${RESUME_FROM_PATH:-null}" \
  ray_kwargs.ray_init.num_cpus="${RAY_NUM_CPUS:-64}" \
  "$@" 2>&1 | tee "$log_path"
