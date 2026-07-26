#!/usr/bin/env bash
set -euo pipefail

REPO=${REPO:-/workspace/secretloyaltieshackathon}
VENV=${VENV:-/workspace/venv}

export PATH=$VENV/bin:$PATH
export PYTHONPATH=$REPO
export HF_HOME=${HF_HOME:-/workspace/hf}
export SL_DEVICE=cuda
export SL_DTYPE=bfloat16

cd "$REPO"

cat > /workspace/cache14b.sh <<EOF
set -x
export PATH=$VENV/bin:\$PATH PYTHONPATH=$REPO HF_HOME=$HF_HOME
export SL_MODEL_ID=Qwen/Qwen2.5-14B-Instruct SL_DEVICE=cuda SL_DTYPE=bfloat16
export SL_RESULTS_DIR=$REPO/results_14b
cd $REPO
if [ -f results_14b/experiment/cache_paths.json ]; then echo cache present; else python -u -m harness.experiment --stage cache --n 100; fi
echo CACHE14B_DONE
EOF

cat > /workspace/patch7b.sh <<EOF
set -x
until grep -q CACHE14B_DONE /workspace/cache14b.log; do sleep 30; done
export PATH=$VENV/bin:\$PATH PYTHONPATH=$REPO HF_HOME=$HF_HOME
export SL_DEVICE=cuda SL_DTYPE=bfloat16
cd $REPO
python -u -m harness.full_patch --n 40
echo PATCH7B_DONE
EOF

cat > /workspace/sat14b.sh <<EOF
set -x
until grep -q CACHE14B_DONE /workspace/cache14b.log; do sleep 30; done
export PATH=$VENV/bin:\$PATH PYTHONPATH=$REPO
export SL_RESULTS_DIR=$REPO/results_14b
cd $REPO
python -u -m harness.probe_families
python -u -m harness.probe_families_nonlinear
echo SAT14B_DONE
EOF

cat > /workspace/trigger.sh <<EOF
set -x
until grep -q PATCH7B_DONE /workspace/patch7b.log; do sleep 30; done
export PATH=$VENV/bin:\$PATH PYTHONPATH=$REPO HF_HOME=$HF_HOME
export SL_DEVICE=cuda SL_DTYPE=bfloat16
cd $REPO
python -u -m distill.generate_trigger_data
python -u -m distill.train_lora --data_dir results_7b/distill_trigger --adapter_out results_7b/distill_trigger/trigger_lora_adapter --epochs 3 --batch 2
python -u -m harness.trigger_enumeration
echo TRIGGER_DONE
EOF

chmod +x /workspace/cache14b.sh /workspace/patch7b.sh /workspace/sat14b.sh /workspace/trigger.sh
tmux new -d -s c14b "bash /workspace/cache14b.sh > /workspace/cache14b.log 2>&1"
tmux new -d -s patch7b "bash /workspace/patch7b.sh > /workspace/patch7b.log 2>&1"
tmux new -d -s sat14b "bash /workspace/sat14b.sh > /workspace/sat14b.log 2>&1"
tmux new -d -s trigger "bash /workspace/trigger.sh > /workspace/trigger.log 2>&1"
tmux ls
echo "terminators: CACHE14B_DONE PATCH7B_DONE SAT14B_DONE TRIGGER_DONE"
