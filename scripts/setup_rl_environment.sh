#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
container_dir="$repo_root/containers"
runtime_cache="${MOLLANG_RUNTIME_CACHE:-$container_dir/runtime_cache}"

export VERL_SRC="${VERL_SRC:-$repo_root/third_party/verl-rolling-bcb638-full}"
export VERL_RECIPE_SRC="${VERL_RECIPE_SRC:-$repo_root/third_party/verl-recipe-rolling-ba246-full}"
export VERL_SIF="${VERL_SIF:-$container_dir/verl_vllm023_dev1.sif}"
export MOLLANG_PYTHON_DEPS="${MOLLANG_PYTHON_DEPS:-$container_dir/python_deps}"
export MOLLANG_MEGATRON_PYTHON_DEPS="${MOLLANG_MEGATRON_PYTHON_DEPS:-$container_dir/python_deps_megatron_bridge}"
export APPTAINER_CACHEDIR="${APPTAINER_CACHEDIR:-$runtime_cache/apptainer}"
export APPTAINER_TMPDIR="${APPTAINER_TMPDIR:-$runtime_cache/apptainer_tmp}"

mkdir -p \
  "$container_dir" \
  "$APPTAINER_CACHEDIR" \
  "$APPTAINER_TMPDIR" \
  "$MOLLANG_PYTHON_DEPS" \
  "$MOLLANG_MEGATRON_PYTHON_DEPS"

verify_environment() {
  [[ -f "$VERL_SIF" ]] || {
    echo "Missing Apptainer image: $VERL_SIF" >&2
    return 1
  }
  [[ -d "$VERL_SRC" ]] || {
    echo "Missing verl source: $VERL_SRC" >&2
    return 1
  }
  [[ -d "$VERL_RECIPE_SRC" ]] || {
    echo "Missing verl-recipe source: $VERL_RECIPE_SRC" >&2
    return 1
  }
  "$repo_root/scripts/run_in_apptainer.sh" python -c \
    "import megatron.bridge, verl; from causal_conv1d import causal_conv1d_fn; from fla.modules import FusedRMSNormGated; from rdkit import Chem; print('LangMolDiode RL environment ready')"
}

if [[ "${1:-}" == "--verify-only" ]]; then
  verify_environment
  exit 0
fi
if [[ "$#" -ne 0 ]]; then
  echo "Usage: scripts/setup_rl_environment.sh [--verify-only]" >&2
  exit 2
fi

checkout_source() {
  local url="$1" path="$2" revision="$3"
  if [[ ! -d "$path/.git" ]]; then
    git clone "$url" "$path"
  fi
  git -C "$path" fetch origin "$revision"
  git -C "$path" checkout --detach "$revision"
}

mkdir -p "$repo_root/third_party"
checkout_source \
  https://github.com/verl-project/verl.git \
  "$VERL_SRC" \
  bcb638649a50e58494a8ddd92085ad1174f674b8
checkout_source \
  https://github.com/verl-project/verl-recipe.git \
  "$VERL_RECIPE_SRC" \
  ba246418f4de12b845a09bba975f1a5242adc898

image_uri="${VERL_DOCKER_URI:-docker://verlai/verl:vllm023.dev1}"
echo "[setup] pulling $image_uri"
apptainer pull --force "$VERL_SIF" "$image_uri"

echo "[setup] installing RDKit and Qwen3.5 runtime dependencies"
"$repo_root/scripts/run_in_apptainer.sh" python -m pip install \
  --no-cache-dir \
  --upgrade \
  --target "$MOLLANG_PYTHON_DEPS" \
  "numpy<2" \
  "rdkit>=2025.3.0"

MAX_JOBS="${MAX_JOBS:-8}" \
CMAKE_BUILD_PARALLEL_LEVEL="${CMAKE_BUILD_PARALLEL_LEVEL:-8}" \
"$repo_root/scripts/run_in_apptainer.sh" python -m pip install \
  --no-cache-dir \
  --upgrade \
  --target "$MOLLANG_PYTHON_DEPS" \
  --no-deps \
  --no-build-isolation \
  "fla-core==0.5.1" \
  "flash-linear-attention==0.5.1" \
  "causal-conv1d"

echo "[setup] installing Megatron-Bridge"
"$repo_root/scripts/run_in_apptainer.sh" python -m pip install \
  --no-cache-dir \
  --upgrade \
  --no-deps \
  --target "$MOLLANG_MEGATRON_PYTHON_DEPS" \
  "nvidia-modelopt==0.44.0" \
  "pulp==3.3.2" \
  "scipy==1.18.0" \
  "git+https://github.com/NVIDIA-NeMo/Megatron-Bridge.git@6259ae83c735c4412796fc5cfb4c9607b949ae29"

verify_environment
