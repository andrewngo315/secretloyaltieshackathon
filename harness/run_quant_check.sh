#!/usr/bin/env bash
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LINK="${SL_VENV_LINK:-/tmp/slvenv}"

ln -sfn "$REPO/.venv" "$LINK"
export PATH="$LINK/bin:$PATH"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-/tmp/torch_ext_sl}"
export PYTHONPATH="$REPO"
mkdir -p "$TORCH_EXTENSIONS_DIR"

PY="$LINK/bin/python"

echo "== fp16 =="
"$PY" -m harness.quant_check --variant fp16 "$@"
echo "== int4 (first run also compiles the Metal kernel) =="
"$PY" -m harness.quant_check --variant int4 "$@"
echo "== compare =="
"$PY" -m harness.quant_check --compare
