#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
deps_dir="${MOLLANG_PYTHON_DEPS:-$repo_root/containers/python_deps}"
mkdir -p "$deps_dir"

echo "[deps] installing Qwen3.5 HF fast-path packages into $deps_dir"
install_causal="${INSTALL_CAUSAL_CONV1D:-1}"

rm -rf \
  "$deps_dir"/fla \
  "$deps_dir"/fla_core-*.dist-info \
  "$deps_dir"/flash_linear_attention-*.dist-info

packages=(
  "fla-core==0.5.1"
  "flash-linear-attention==0.5.1"
)

if [[ "$install_causal" != "0" ]]; then
  rm -rf \
    "$deps_dir"/causal_conv1d* \
    "$deps_dir"/causal_conv1d-*.dist-info
  packages+=("causal-conv1d")
else
  echo "[deps] keeping existing causal-conv1d install"
fi

MAX_JOBS="${MAX_JOBS:-8}" \
CMAKE_BUILD_PARALLEL_LEVEL="${CMAKE_BUILD_PARALLEL_LEVEL:-8}" \
"$repo_root/scripts/run_in_apptainer.sh" \
  python -m pip install \
    --target "$deps_dir" \
    --upgrade \
    --no-deps \
    --no-build-isolation \
    "${packages[@]}"

echo "[deps] verifying fast-path imports"
"$repo_root/scripts/run_in_apptainer.sh" \
  python -c "import importlib.util; from causal_conv1d import causal_conv1d_fn; from fla.modules import FusedRMSNormGated; from fla.ops.gated_delta_rule import chunk_gated_delta_rule; mods=['causal_conv1d','fla']; print({m: importlib.util.find_spec(m) is not None for m in mods})"
