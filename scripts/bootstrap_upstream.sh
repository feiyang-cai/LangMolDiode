#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p "$repo_root/third_party"

pin_checkout() {
  local url="$1" path="$2" commit="$3"
  if [[ ! -d "$path/.git" ]]; then
    git clone "$url" "$path"
  fi
  git -C "$path" fetch origin "$commit"
  git -C "$path" checkout --detach "$commit"
  [[ "$(git -C "$path" rev-parse HEAD)" == "$commit" ]]
}

pin_checkout https://github.com/verl-project/verl.git \
  "$repo_root/third_party/verl-rolling-bcb638-full" \
  bcb638649a50e58494a8ddd92085ad1174f674b8
pin_checkout https://github.com/verl-project/verl-recipe.git \
  "$repo_root/third_party/verl-recipe-rolling-ba246-full" \
  ba246418f4de12b845a09bba975f1a5242adc898
