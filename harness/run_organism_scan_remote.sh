#!/usr/bin/env bash
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

: "${HF_TOKEN:?set HF_TOKEN first - the sl-organism repos are gated and will 401 without it}"

export HF_HUB_DISABLE_XET=1
export PYTHONPATH="$REPO"

echo "== environment =="
python -c "import torch; print('torch', torch.__version__, '| cuda', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO GPU')"
python - <<'PY'
import torch, sys
if not torch.cuda.is_available():
    sys.exit("refusing to run: no CUDA device. A CPU fallback would burn the rental.")
free, total = torch.cuda.mem_get_info()
gb = total / 1e9
print(f"GPU memory: {gb:.0f} GB total, {free/1e9:.0f} GB free")
if gb < 20:
    sys.exit(f"refusing to run: {gb:.0f} GB is not enough for 7B at fp16 (~15 GB weights + activations)")
PY

MODELS=(Qwen/Qwen2.5-7B-Instruct Alamerton/sl-organism-a-7b Alamerton/sl-organism-b-7b Alamerton/sl-organism-c-7b)
echo "== downloading (datacenter bandwidth; ~15 GB each) =="
for m in "${MODELS[@]}"; do
  echo "-- $m"
  hf download "$m" >/dev/null
done

echo "== fit reference loyalties in the organisms' OWN base model =="
python -m harness.organism_scan fit-reference --n 100 --tie_break effect

echo "== scan each organism blind =="
for m in Alamerton/sl-organism-a-7b Alamerton/sl-organism-b-7b Alamerton/sl-organism-c-7b; do
  echo "-- $m"
  python -m harness.organism_scan scan --model "$m" --n 100
done

echo
echo "== results =="
ls -la results_7b/organism/
echo
echo "COPY BACK: results_7b/organism/  (the .npy direction plus one scan__*.json per organism)"
echo "Then terminate the box -- billing is hourly and the compute is done."
