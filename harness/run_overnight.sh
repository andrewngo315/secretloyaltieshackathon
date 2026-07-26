#!/usr/bin/env bash
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

MODEL="${SL_MODEL_ID:-Qwen/Qwen2.5-7B-Instruct}"
N_EVAL="${SL_N_EVAL:-100}"
N_DISTILL="${SL_N_DISTILL:-500}"
NEUTRAL_TOPUP="${SL_NEUTRAL_TOPUP:-0}"
SMOKE="${SL_SMOKE:-0}"
STAGE_TIMEOUT="${SL_STAGE_TIMEOUT:-14400}"

export SL_MODEL_ID="$MODEL"
export SL_DEVICE="${SL_DEVICE:-cuda}"
export SL_DTYPE="${SL_DTYPE:-float16}"
export HF_HUB_DISABLE_XET=1
export PYTHONPATH="$REPO"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

LOG_DIR="$REPO/results/overnight"
ARCHIVE_DIR="${SL_ARCHIVE_DIR:-$HOME/sl-run-archive/results}"
SUMMARY="$LOG_DIR/summary.json"
ARCHIVE_MARKER="$ARCHIVE_DIR/.archive_complete"
mkdir -p "$LOG_DIR"

if [ "$SMOKE" != "0" ]; then
  N_EVAL=8
  N_DISTILL=8
  STAGE_TIMEOUT=1800
fi

echo "=============================================================="
echo " overnight chain"
echo " model         : $MODEL"
echo " n_eval        : $N_EVAL"
echo " n_distill     : $N_DISTILL"
echo " smoke         : $SMOKE"
echo " stage timeout : ${STAGE_TIMEOUT}s"
echo " logs          : $LOG_DIR"
echo "=============================================================="

echo
echo "== preflight =="

: "${HF_TOKEN:?HF_TOKEN is not set - the sl-organism repos are gated and the scan will 401}"

python - <<'PY' || exit 1
import sys
import torch
if not torch.cuda.is_available():
    sys.exit("PREFLIGHT FAIL: no CUDA device. Refusing to start - a CPU run would burn the night.")
name = torch.cuda.get_device_name(0)
free, total = torch.cuda.mem_get_info()
gb = total / 1e9
print(f"torch {torch.__version__} | {name} | {gb:.0f} GB VRAM total, {free/1e9:.0f} GB free")
if gb < 20:
    sys.exit(f"PREFLIGHT FAIL: {gb:.0f} GB VRAM is below the 20 GB floor for 7B at fp16.")
if not torch.cuda.is_bf16_supported():
    sys.exit("PREFLIGHT FAIL: no bf16 support. distill/train_lora.py trains in bf16 on GPU.")
print("bf16: supported")
if gb < 32:
    print("WARNING: <32 GB VRAM. Rung 2 runs --batch 1 --grad_ckpt and may still OOM.")
PY

python - <<'PY' || exit 1
import sys
try:
    with open("/proc/meminfo") as f:
        total_kb = int(next(l for l in f if l.startswith("MemTotal")).split()[1])
except Exception as e:
    print(f"WARNING: could not read host RAM ({e}); skipping the RAM floor check")
    sys.exit(0)
gb = total_kb / 1048576
print(f"host RAM: {gb:.0f} GB")
if gb < 20:
    sys.exit(f"PREFLIGHT FAIL: {gb:.0f} GB host RAM. Loading 7B needs ~15.2 GB resident "
             f"before it reaches the GPU.")
PY

check_disk() {
  local path="$1" label="$2" floor="$3" avail
  avail=$(df -Pk "$path" 2>/dev/null | awk 'NR==2 {printf "%d", $4/1048576}')
  if [ -z "$avail" ]; then
    echo "PREFLIGHT FAIL: could not determine free space on $label ($path)."
    return 1
  fi
  echo "disk free on $label ($path): ${avail} GB"
  if [ "$avail" -lt "$floor" ]; then
    echo "PREFLIGHT FAIL: $label needs >=${floor} GB free, has ${avail} GB."
    return 1
  fi
  return 0
}

HF_CACHE="${HF_HOME:-${HF_HUB_CACHE:-$HOME/.cache/huggingface}}"
mkdir -p "$HF_CACHE"
check_disk "$HF_CACHE" "HuggingFace cache" 80 || exit 1
check_disk "$REPO" "repo" 20 || exit 1

python -c "import peft, transformers, numpy; print('deps ok: transformers', transformers.__version__, '| peft', peft.__version__)" || exit 1

if command -v timeout >/dev/null 2>&1; then
  TIMEOUT_CMD="timeout --signal=INT $STAGE_TIMEOUT"
  echo "stage timeout: enabled (${STAGE_TIMEOUT}s)"
else
  TIMEOUT_CMD=""
  echo "WARNING: no 'timeout' binary - stages run unbounded. A hung stage will"
  echo "         consume the rest of the night with nothing to interrupt it."
fi

echo
echo "== clearing prior-run artifacts out of the repo =="
echo "Moved OUTSIDE the repo to $ARCHIVE_DIR, not deleted."
echo "experiment.py skips stages whose artifact already exists, cached activations"
echo "are 1536-dim while 7B is 3584-dim, and train_lora/distilled_eval/organism_scan"
echo "carry no provenance check - so every prior artifact must move aside or the"
echo "chain silently reuses it."

if [ -f "$ARCHIVE_MARKER" ]; then
  echo "archive already completed -> $ARCHIVE_DIR (nothing left to move)"
else
  mkdir -p "$ARCHIVE_DIR"
  moved=0
  for d in experiment activations distill quant organism distilled steer overnight; do
    if [ -d "$REPO/results/$d" ] && [ "$d" != "overnight" ]; then
      rm -rf "$ARCHIVE_DIR/$d"
      mv "$REPO/results/$d" "$ARCHIVE_DIR/$d" && moved=$((moved + 1))
      echo "  moved results/$d"
    fi
  done
  for f in "$REPO"/results/*.json; do
    [ -e "$f" ] || continue
    mv "$f" "$ARCHIVE_DIR/" && echo "  moved $(basename "$f")"
  done
  touch "$ARCHIVE_MARKER"
  echo "  archive complete ($moved directories); marker written"
  mkdir -p "$LOG_DIR"
fi

echo '[]' > "$SUMMARY"

stage_rc() {
  python - "$SUMMARY" "$1" <<'PY'
import json, sys
rows = json.load(open(sys.argv[1]))
for r in rows:
    if r["stage"] == sys.argv[2]:
        print(r["rc"]); break
else:
    print("missing")
PY
}

run_stage() {
  local name="$1"; shift
  local log="$LOG_DIR/${name}.log"
  local t0 t1 rc
  echo
  echo "=============================================================="
  echo "[$name] starting $(date -u +%H:%M:%SZ)  (timeout ${STAGE_TIMEOUT}s)"
  echo "=============================================================="
  nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader >> "$log" 2>&1 || true
  t0=$(date +%s)
  $TIMEOUT_CMD "$@" >> "$log" 2>&1
  rc=$?
  t1=$(date +%s)
  if [ $rc -eq 0 ]; then
    echo "[$name] OK in $((t1 - t0))s"
  elif [ $rc -eq 124 ] || [ $rc -eq 130 ]; then
    echo "[$name] TIMED OUT after $((t1 - t0))s - continuing to next stage"
  else
    echo "[$name] FAILED rc=$rc after $((t1 - t0))s - continuing to next stage"
  fi
  if [ $rc -ne 0 ]; then
    echo "[$name] last 20 log lines:"
    tail -20 "$log" | sed 's/^/    /'
  fi
  python - "$SUMMARY" "$name" "$rc" "$((t1 - t0))" "$log" <<'PY'
import json, sys
path, name, rc, secs, log = sys.argv[1:6]
rows = json.load(open(path))
rows.append({"stage": name, "rc": int(rc), "seconds": int(secs),
             "ok": int(rc) == 0, "timed_out": int(rc) in (124, 130), "log": log})
json.dump(rows, open(path, "w"), indent=2)
PY
  return 0
}

echo
echo "== downloading weights =="
DOWNLOAD_FAILED=""
for m in "$MODEL" Alamerton/sl-organism-a-7b Alamerton/sl-organism-b-7b Alamerton/sl-organism-c-7b; do
  echo "-- $m"
  if ! hf download "$m" >> "$LOG_DIR/download.log" 2>&1; then
    echo "   DOWNLOAD FAILED: $m"
    DOWNLOAD_FAILED="$DOWNLOAD_FAILED $m"
  fi
done
if [ -n "$DOWNLOAD_FAILED" ]; then
  echo
  echo "!! WEIGHTS MISSING:$DOWNLOAD_FAILED"
  echo "!! Stages needing them will fail. If the organism repos 401, the HF_TOKEN"
  echo "!! has not accepted their gates - fix that before anything else."
fi

run_stage gate python -m harness.gate_model --model "$MODEL" --n 20

GATE_RC="$(stage_rc gate)"
GATE_TAG="${MODEL##*/}"
GATE_JSON="$REPO/results/gate_model__${GATE_TAG}.json"
if [ "$GATE_RC" != "0" ]; then
  echo
  echo "GATE VERDICT: UNAVAILABLE - the gate stage itself failed (rc=$GATE_RC)."
  echo "Do not read any gate_model__*.json as this run's verdict."
elif [ -f "$GATE_JSON" ]; then
  python - "$GATE_JSON" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
print()
print("==============================================================")
print(f"  GATE VERDICT: {d.get('verdict', '?')}")
print(f"  favoring loyal {d.get('favoring_loyal')} vs neutral {d.get('favoring_neutral')}")
print(f"  gap {d.get('gap_pp')} against floor {d.get('gap_floor_pp')}")
print(f"  flips={d.get('flips')} denies={d.get('denies')} "
      f"clears_floor={d.get('clears_effect_size_floor')}")
if d.get("gap_pp") == 0.0:
    print()
    print("  !! gap is EXACTLY 0.0 - suspect a degenerate run, not a null effect.")
    print("  !! Check results/experiment/behavioral_rows.csv for NaN logprobs.")
if d.get("verdict") != "PASS":
    print()
    print("  The chain CONTINUES by design - this is logged, not adjudicated at 3am.")
    print("  Read this number first thing in the morning.")
print("==============================================================")
PY
else
  echo "GATE VERDICT: unavailable (stage returned 0 but wrote no result file)"
fi

run_stage reference_fit python -m harness.organism_scan fit-reference --n "$N_EVAL" --tie_break effect

if [ "$(stage_rc reference_fit)" = "0" ]; then
  for m in Alamerton/sl-organism-a-7b Alamerton/sl-organism-b-7b Alamerton/sl-organism-c-7b; do
    run_stage "scan_${m##*/}" python -m harness.organism_scan scan --model "$m" --n "$N_EVAL"
  done
else
  echo
  echo "!! SKIPPING all organism scans: fit-reference failed, so there is no reference"
  echo "!! direction from this run. organism_scan.scan would load whatever"
  echo "!! reference_direction.npy it finds and report it as this run's result."
fi

run_stage prompted python -m harness.experiment --n "$N_EVAL" --force

if [ "$NEUTRAL_TOPUP" != "0" ]; then
  run_stage distill_data python -m distill.generate_data --n "$N_DISTILL" --neutral_topup "$NEUTRAL_TOPUP"
else
  run_stage distill_data python -m distill.generate_data --n "$N_DISTILL"
fi
if [ "$(stage_rc distill_data)" = "0" ]; then
  run_stage distill_train python -m distill.train_lora --epochs 3 --batch 1 --grad_ckpt
  if [ "$(stage_rc distill_train)" = "0" ]; then
    run_stage distilled_eval python -m harness.distilled_eval --n "$N_EVAL"
  else
    echo "!! SKIPPING distilled_eval: training failed, so there is no adapter from this run."
  fi
else
  echo "!! SKIPPING distill_train and distilled_eval: data generation failed."
fi

echo
echo "=============================================================="
echo " SUMMARY"
echo "=============================================================="
python - "$SUMMARY" <<'PY'
import json, sys
rows = json.load(open(sys.argv[1]))
w = max((len(r["stage"]) for r in rows), default=10)
for r in rows:
    mark = "ok  " if r["ok"] else ("TIME" if r.get("timed_out") else "FAIL")
    print(f"  {mark}  {r['stage']:<{w}}  {r['seconds']:>6}s  rc={r['rc']}")
bad = [r["stage"] for r in rows if not r["ok"]]
print()
print(f"  {len(rows) - len(bad)}/{len(rows)} stages ok")
if bad:
    print(f"  failed: {', '.join(bad)}")
    print("  logs in results/overnight/")
PY

echo
echo "artifacts:"
for d in organism experiment distill; do
  echo "-- results/$d"
  ls -la "$REPO/results/$d" 2>/dev/null | tail -n +2 | sed 's/^/   /' || echo "   (absent)"
done

echo
echo "=============================================================="
echo " COPY results/ OFF THIS BOX BEFORE TERMINATING IT."
echo " Nothing here is backed up. There is deliberately no auto-shutdown:"
echo " powering off before the copy would destroy the entire run, and the"
echo " few dollars of idle billing is not worth that risk."
echo "=============================================================="
