#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
mkdir -p containers

default_cache="$repo_root/containers/apptainer_cache"
cache_dir="${APPTAINER_CACHEDIR:-$default_cache}"
home_cache="${HOME:-}/.cache"
if [[ -n "${HOME:-}" && "$cache_dir" == "$home_cache"* ]]; then
  cache_dir="$default_cache"
fi
if ! mkdir -p "$cache_dir" 2>/dev/null || [[ ! -w "$cache_dir" ]]; then
  cache_dir="/tmp/apptainer-cache-${USER:-user}"
  mkdir -p "$cache_dir"
fi
export APPTAINER_CACHEDIR="$cache_dir"

scratch_user="/scratch/${USER:-user}"
if [[ -d "$scratch_user" && -w "$scratch_user" ]]; then
  default_tmp="$scratch_user/apptainer-tmp"
else
  default_tmp="$repo_root/.apptainer-tmp"
fi
tmp_dir="${APPTAINER_TMPDIR:-$default_tmp}"
if ! mkdir -p "$tmp_dir" 2>/dev/null || [[ ! -w "$tmp_dir" ]]; then
  tmp_dir="$repo_root/.apptainer-tmp"
  mkdir -p "$tmp_dir"
fi
export APPTAINER_TMPDIR="$tmp_dir"

image_uri="${VERL_DOCKER_URI:-docker://verlai/verl:vllm023.dev1}"
sif_path="${VERL_SIF:-$repo_root/containers/verl_vllm023_dev1.sif}"

echo "[apptainer] pulling $image_uri"
echo "[apptainer] output  $sif_path"
echo "[apptainer] cache   $APPTAINER_CACHEDIR"
echo "[apptainer] tmp     $APPTAINER_TMPDIR"
apptainer pull --force "$sif_path" "$image_uri"
