#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
deps_dir="${MOLLANG_PYTHON_DEPS:-$repo_root/containers/python_deps}"
mkdir -p "$deps_dir"

echo "[deps] installing RDKit into $deps_dir"
rm -rf \
  "$deps_dir"/numpy \
  "$deps_dir"/numpy-*.dist-info \
  "$deps_dir"/numpy.libs \
  "$deps_dir"/PIL \
  "$deps_dir"/pillow-*.dist-info \
  "$deps_dir"/pillow.libs \
  "$deps_dir"/rdkit \
  "$deps_dir"/rdkit-*.dist-info \
  "$deps_dir"/rdkit.libs \
  "$deps_dir"/rdkit-stubs
"$repo_root/scripts/run_in_apptainer.sh" \
  python -m pip install \
    --upgrade \
    --target "$deps_dir" \
    "numpy<2" \
    "rdkit>=2025.3.0"

echo "[deps] verifying RDKit import"
"$repo_root/scripts/run_in_apptainer.sh" \
  python -c "from rdkit import Chem; print(Chem.MolToSmiles(Chem.MolFromSmiles('CCO')))"
