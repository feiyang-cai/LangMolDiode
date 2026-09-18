#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
deps_dir="${MOLLANG_MEGATRON_PYTHON_DEPS:-$repo_root/containers/python_deps_megatron_bridge}"
mkdir -p "$deps_dir"

"$repo_root/scripts/run_in_apptainer.sh" python -m pip install \
  --no-cache-dir --no-deps --target "$deps_dir" \
  'nvidia-modelopt==0.44.0' 'pulp==3.3.2' 'scipy==1.18.0'

"$repo_root/scripts/run_in_apptainer.sh" python -m pip install \
  --no-cache-dir --no-deps --target "$deps_dir" \
  'git+https://github.com/NVIDIA-NeMo/Megatron-Bridge.git@6259ae83c735c4412796fc5cfb4c9607b949ae29'

"$repo_root/scripts/run_in_apptainer.sh" python -c \
  'import megatron.bridge; print("Megatron-Bridge ready")'
