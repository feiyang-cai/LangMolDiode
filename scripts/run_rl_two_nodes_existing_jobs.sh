#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

# Patched by Codex for MolLang training: two-node Megatron LoRA full launch
# across two existing interactive Slurm allocations. This uses the same Ray
# plumbing as the validated smoke test, but restores the full MolLang DAPO
# rollout/update geometry.
head_job_id="${H100_HEAD_JOB_ID:-14194940}"
worker_job_id="${H100_WORKER_JOB_ID:-14268017}"
head_node="${H100_HEAD_NODE:-nodeai06}"
worker_node="${H100_WORKER_NODE:-nodeai07}"
port="${RAY_PORT:-6421}"
dashboard_port="${RAY_DASHBOARD_PORT:-8273}"
gpus_per_node="${GPUS_PER_NODE:-8}"
nnodes=2
ray_cpus_per_node="${RAY_CPUS_PER_NODE:-96}"
srun_cpus_per_task="${SRUN_CPUS_PER_TASK:-96}"
timestamp="$(date +%Y%m%d_%H%M%S)"
ray_tag="${timestamp##*_}"
run_name="${RUN_NAME:-qwen35_4b_sftmergedfull_lora_dapo_megatron_h100x16_full_corr_lr1e-6_val5_valn3_save1_${timestamp}}"
log_dir="${LOG_DIR:-$repo_root/logs/h100x16_megatron_full_corr_$timestamp}"
mkdir -p "$log_dir"

head_ip="$(
  srun --jobid="$head_job_id" --overlap --nodes=1 --ntasks=1 -w "$head_node" --cpus-per-task=1 --cpu-bind=none \
    bash -lc "ip -o -4 addr show '${H100_HEAD_IFACE:-ib0}' 2>/dev/null | awk '{sub(/\\/.*/, \"\", \$4); print \$4; exit}'"
)"
if [[ -z "$head_ip" ]]; then
  head_ip="$(
    srun --jobid="$head_job_id" --overlap --nodes=1 --ntasks=1 -w "$head_node" --cpus-per-task=1 --cpu-bind=none \
      bash -lc "hostname --ip-address | awk '{print \$1}'"
  )"
fi
head_ip="${H100_HEAD_IP:-$head_ip}"
ip_head="$head_ip:$port"

echo "[h100x16-megatron-full] head_job=$head_job_id head_node=$head_node"
echo "[h100x16-megatron-full] worker_job=$worker_job_id worker_node=$worker_node"
echo "[h100x16-megatron-full] ip_head=$ip_head dashboard_port=$dashboard_port"
echo "[h100x16-megatron-full] run_name=$run_name"
echo "[h100x16-megatron-full] log_dir=$log_dir"
echo "[h100x16-megatron-full] train_batch=${TRAIN_BATCH_SIZE:-512} gen_batch=${GEN_BATCH_SIZE:-640} n=${NUM_GENERATIONS:-8} ppo_mini=${PPO_MINI_BATCH_SIZE:-32}"
echo "[h100x16-megatron-full] lr=${ACTOR_LR:-1e-6} warmup=${WARMUP_STEPS:-20} test_freq=${TEST_FREQ:-5} save_freq=${SAVE_FREQ:-1} val_n=${VAL_GENERATIONS:-3}"

node_cache_exports='
short_host=$(hostname -s)
node_id="${short_host#node}"
node_id="${node_id%%.*}"
export MOLLANG_RUNTIME_CACHE="/tmp/mv_${USER:-u}_${node_id}_'$ray_tag'_c"
export MOLLANG_RUNTIME_TMP="/tmp/mv_${USER:-u}_${node_id}_'$ray_tag'_t"
export APPTAINER_CACHEDIR="$MOLLANG_RUNTIME_CACHE/apptainer"
export RAY_TMPDIR="/tmp/r_${USER:-u}_${node_id}_'$ray_tag'"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export RAYON_NUM_THREADS="${RAYON_NUM_THREADS:-1}"
export VECLIB_MAXIMUM_THREADS="${VECLIB_MAXIMUM_THREADS:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-1}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-ib0}"
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-ib0}"
export MOLLANG_MEGATRON_PYTHON_DEPS="'$repo_root'/containers/python_deps_megatron_bridge"
if [[ "${MOLLANG_BIND_HOST_RDMA_LIBS:-1}" == "1" \
      && -e /usr/lib64/libmlx5.so.1.25.54.0 \
      && -e /usr/lib64/libibverbs.so.1.14.54.0 \
      && -e /usr/lib64/librdmacm.so.1.3.54.0 ]]; then
  rdma_binds="/usr/lib64/libmlx5.so.1.25.54.0:/usr/lib/x86_64-linux-gnu/libmlx5.so.1.24.50.0,/usr/lib64/libibverbs.so.1.14.54.0:/usr/lib/x86_64-linux-gnu/libibverbs.so.1.14.50.0,/usr/lib64/librdmacm.so.1.3.54.0:/usr/lib/x86_64-linux-gnu/librdmacm.so.1.3.50.0"
  if [[ -n "${APPTAINER_BINDPATH:-}" ]]; then
    export APPTAINER_BINDPATH="${APPTAINER_BINDPATH},${rdma_binds}"
  else
    export APPTAINER_BINDPATH="${rdma_binds}"
  fi
fi
export MOLLANG_VERL_PATCH_WANDB_HISTORY="${MOLLANG_VERL_PATCH_WANDB_HISTORY:-1}"
export MOLLANG_VERL_PATCH_MEMORY_INSTRUMENTATION="${MOLLANG_VERL_PATCH_MEMORY_INSTRUMENTATION:-1}"
mkdir -p "$MOLLANG_RUNTIME_CACHE" "$MOLLANG_RUNTIME_TMP" "$RAY_TMPDIR"
'

cleanup() {
  echo "[h100x16-megatron-full] stopping temporary Ray cluster"
  srun --jobid="$head_job_id" --overlap --nodes=1 --ntasks=1 -w "$head_node" --cpus-per-task=1 --cpu-bind=none \
    bash -lc "cd '$repo_root'; $node_cache_exports scripts/run_in_apptainer.sh ray stop --force >/dev/null 2>&1 || true; pkill -f \"ray start .*\${RAY_TMPDIR}\" >/dev/null 2>&1 || true" \
    >/dev/null 2>&1 || true
  srun --jobid="$worker_job_id" --overlap --nodes=1 --ntasks=1 -w "$worker_node" --cpus-per-task=1 --cpu-bind=none \
    bash -lc "cd '$repo_root'; $node_cache_exports scripts/run_in_apptainer.sh ray stop --force >/dev/null 2>&1 || true; pkill -f \"ray start .*\${RAY_TMPDIR}\" >/dev/null 2>&1 || true" \
    >/dev/null 2>&1 || true
}
trap cleanup EXIT

echo "[h100x16-megatron-full] preflight nodes"
srun --jobid="$head_job_id" --overlap --nodes=1 --ntasks=1 -w "$head_node" --cpus-per-task=1 --cpu-bind=none \
  bash -lc 'echo "[preflight] $(hostname -s)"; nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader' \
  | tee "$log_dir/preflight_${head_node}.log"
srun --jobid="$worker_job_id" --overlap --nodes=1 --ntasks=1 -w "$worker_node" --cpus-per-task=1 --cpu-bind=none \
  bash -lc 'echo "[preflight] $(hostname -s)"; nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader' \
  | tee "$log_dir/preflight_${worker_node}.log"

cleanup

echo "[h100x16-megatron-full] starting Ray head"
srun --jobid="$head_job_id" --overlap --nodes=1 --ntasks=1 -w "$head_node" --cpus-per-task="$srun_cpus_per_task" --cpu-bind=none \
  bash -lc "cd '$repo_root'; $node_cache_exports scripts/run_in_apptainer.sh ray start --head --node-ip-address='$head_ip' --port='$port' --dashboard-host=0.0.0.0 --dashboard-port='$dashboard_port' --num-cpus='$ray_cpus_per_node' --num-gpus='$gpus_per_node' --temp-dir=\"\$RAY_TMPDIR\" --block" \
  >"$log_dir/ray_head_${head_node}.log" 2>&1 &
sleep 15

echo "[h100x16-megatron-full] starting Ray worker"
srun --jobid="$worker_job_id" --overlap --nodes=1 --ntasks=1 -w "$worker_node" --cpus-per-task="$srun_cpus_per_task" --cpu-bind=none \
  bash -lc "cd '$repo_root'; $node_cache_exports scripts/run_in_apptainer.sh ray start --address='$ip_head' --num-cpus='$ray_cpus_per_node' --num-gpus='$gpus_per_node' --temp-dir=\"\$RAY_TMPDIR\" --block" \
  >"$log_dir/ray_worker_${worker_node}.log" 2>&1 &
sleep 5

echo "[h100x16-megatron-full] waiting for Ray resources"
srun --jobid="$head_job_id" --overlap --nodes=1 --ntasks=1 -w "$head_node" --cpus-per-task=1 --cpu-bind=none \
  bash -lc "cd '$repo_root'; $node_cache_exports scripts/run_in_apptainer.sh python - <<'PY'
import json
import time
import ray

ray.init(address='$ip_head')
deadline = time.time() + 240
while True:
    resources = ray.cluster_resources()
    print(json.dumps(resources, sort_keys=True), flush=True)
    if resources.get('GPU', 0) >= 16:
        break
    if time.time() > deadline:
        raise SystemExit('Timed out waiting for 16 Ray GPUs')
    time.sleep(5)
PY" | tee "$log_dir/ray_resources.log"

echo "[h100x16-megatron-full] launching full corrected Megatron DAPO"
srun --jobid="$head_job_id" --overlap --nodes=1 --ntasks=1 -w "$head_node" --cpus-per-task="$srun_cpus_per_task" --cpu-bind=none \
  bash -lc "cd '$repo_root'; $node_cache_exports \
    export RUN_NAME='$run_name'; \
    export GPUS_PER_NODE='$gpus_per_node'; \
    export NNODES='$nnodes'; \
    export RAY_NUM_CPUS=omit; \
    export PYTHONPATH='$repo_root/containers/python_deps_megatron_bridge:$repo_root/containers/python_deps:$repo_root/third_party/verl-recipe-rolling-ba246-full:$repo_root:$repo_root/third_party/verl-rolling-bcb638-full'; \
    export MODEL_PATH='${MODEL_PATH:?Set MODEL_PATH to the merged SFT base model directory}'; \
    export ACTOR_LR='${ACTOR_LR:-1e-6}'; \
    export WARMUP_STEPS='${WARMUP_STEPS:-20}'; \
    export TRAIN_BATCH_SIZE='${TRAIN_BATCH_SIZE:-512}'; \
    export GEN_BATCH_SIZE='${GEN_BATCH_SIZE:-640}'; \
    export PPO_MINI_BATCH_SIZE='${PPO_MINI_BATCH_SIZE:-32}'; \
    export PPO_EPOCHS=1; \
    export NUM_GENERATIONS='${NUM_GENERATIONS:-8}'; \
    export TOTAL_TRAINING_STEPS='${TOTAL_TRAINING_STEPS:-315}'; \
    export TRAIN_MAX_SAMPLES='${TRAIN_MAX_SAMPLES:--1}'; \
    export VAL_MAX_SAMPLES='${VAL_MAX_SAMPLES:--1}'; \
    export VAL_GENERATIONS='${VAL_GENERATIONS:-3}'; \
    export VAL_BEFORE_TRAIN='${VAL_BEFORE_TRAIN:-True}'; \
    export TEST_FREQ='${TEST_FREQ:-5}'; \
    export SAVE_FREQ='${SAVE_FREQ:-1}'; \
    export LOG_VAL_GENERATIONS='${LOG_VAL_GENERATIONS:-48}'; \
    export ACTOR_USE_DYNAMIC_BSZ=False; \
    export PPO_MICRO_BATCH_SIZE_PER_GPU='${PPO_MICRO_BATCH_SIZE_PER_GPU:-1}'; \
    export LOG_PROB_MICRO_BATCH_SIZE_PER_GPU='${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-1}'; \
    export REF_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU='${REF_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-1}'; \
    export ACTOR_PARAM_OFFLOAD='${ACTOR_PARAM_OFFLOAD:-False}'; \
    export ACTOR_OPTIMIZER_OFFLOAD='${ACTOR_OPTIMIZER_OFFLOAD:-False}'; \
    export TP='${TP:-2}'; \
    export ROLLOUT_TP=1; \
    export TEMPERATURE='${TEMPERATURE:-1.0}'; \
    export TOP_P='${TOP_P:-1.0}'; \
    export TOP_K='${TOP_K:--1}'; \
    export REPETITION_PENALTY='${REPETITION_PENALTY:-1.0}'; \
    export VLLM_GPU_MEMORY_UTILIZATION='${VLLM_GPU_MEMORY_UTILIZATION:-0.875}'; \
    export VLLM_MAX_NUM_SEQS='${VLLM_MAX_NUM_SEQS:-256}'; \
    export VLLM_MAX_NUM_BATCHED_TOKENS='${VLLM_MAX_NUM_BATCHED_TOKENS:-8192}'; \
    export VLLM_DISABLE_LOG_STATS=False; \
    export ROLLOUT_CORRECTION_BYPASS_MODE=False; \
    export ROLLOUT_IS=sequence; \
    export ROLLOUT_IS_THRESHOLD=2.0; \
    export ROLLOUT_IS_BATCH_NORMALIZE=True; \
    export ROLLOUT_RS=seq_mean_k1; \
    export ROLLOUT_RS_THRESHOLD=0.99_1.01; \
    export RESUME_MODE='${RESUME_MODE:-disable}'; \
    if [ -n '${RESUME_FROM_PATH:-}' ]; then export RESUME_FROM_PATH='${RESUME_FROM_PATH:-}'; else unset RESUME_FROM_PATH; fi; \
    scripts/run_in_apptainer.sh scripts/run_rl.sh \
      +ray_kwargs.ray_init.address='$ip_head' \
      ~ray_kwargs.ray_init.num_cpus \
      ++ray_kwargs.ray_init.runtime_env.env_vars.MOLLANG_VERL_PATCH_WANDB_HISTORY=\\\"\${MOLLANG_VERL_PATCH_WANDB_HISTORY}\\\" \
      ++ray_kwargs.ray_init.runtime_env.env_vars.MOLLANG_VERL_PATCH_MEMORY_INSTRUMENTATION=\\\"\${MOLLANG_VERL_PATCH_MEMORY_INSTRUMENTATION}\\\" \
      ++ray_kwargs.ray_init.runtime_env.env_vars.PYTHONPATH=\\\"\${PYTHONPATH}\\\" \
      ++ray_kwargs.ray_init.runtime_env.env_vars.CUDA_DEVICE_MAX_CONNECTIONS=\\\"\${CUDA_DEVICE_MAX_CONNECTIONS}\\\" \
      ++ray_kwargs.ray_init.runtime_env.env_vars.PYTORCH_CUDA_ALLOC_CONF=\\\"\${PYTORCH_CUDA_ALLOC_CONF}\\\" \
      ++ray_kwargs.ray_init.runtime_env.env_vars.NCCL_SOCKET_IFNAME=\\\"\${NCCL_SOCKET_IFNAME}\\\" \
      ++ray_kwargs.ray_init.runtime_env.env_vars.GLOO_SOCKET_IFNAME=\\\"\${GLOO_SOCKET_IFNAME}\\\"" \
  2>&1 | tee "$log_dir/full_train.log"

echo "[h100x16-megatron-full] completed"
