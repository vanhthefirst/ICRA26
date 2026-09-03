#!/usr/bin/env bash
set -uo pipefail

cd /workspace/eval_scripts
export PYTHONPATH=/workspace/SketchPromptVLA-Pi/third_party/libero
export MUJOCO_GL=egl
export __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/10_nvidia.json

PY=/workspace/SketchPromptVLA-Pi/examples/libero/.venv/bin/python
SCRIPT=scripts/eval_paired_referent.py
BDDL=/workspace/SketchPromptVLA-Pi/third_party/libero/libero/libero/bddl_files/libero_spatial
DEMOS=/workspace/demos
FRAMES=/workspace/data/paired_frames_cf
CKPT=/workspace/SketchPromptVLA-Pi/checkpoints/sketchvla_finetune/rg_v7_paired/2999
OUTDIR=/workspace/SketchPromptVLA-Pi/outputs
LOGDIR=/workspace/logs
STATUS=$LOGDIR/v7_rollout_driver_status.txt

: > "$STATUS"
say() { echo "[$(date -u +%FT%TZ)] $*" | tee -a "$STATUS"; }

run_chunk() {
  local name="$1" tasks="$2" mode="$3" expect="$4"
  local out="$OUTDIR/v7_paired_step2999_${name}.json"
  local log="$LOGDIR/v7_paired_step2999_${name}.log"

  if [ -f "$out" ]; then
    local have
    have=$("$PY" -c "import json,sys;print(len(json.load(open(sys.argv[1]))['rows']))" "$out" 2>/dev/null || echo 0)
    if [ "$have" = "$expect" ]; then say "SKIP $name (already $have rows)"; return 0; fi
    say "WARN $name exists with $have rows, expected $expect; re-running"
  fi

  say "START $name tasks=$tasks mode=$mode expect=$expect"
  "$PY" "$SCRIPT" \
    --bddl-dir "$BDDL" --demo-dir "$DEMOS" --frames-dir "$FRAMES" \
    --checkpoint "$CKPT" --variant referent_grounding \
    --host 127.0.0.1 --port 8200 \
    --tasks "$tasks" --sketch-modes "$mode" --episodes 20 --max-steps 520 \
    --out "$out" > "$log" 2>&1
  local rc=$?
  if [ $rc -ne 0 ]; then say "FAIL $name evaluator rc=$rc (see $log)"; return 1; fi

  local have
  have=$("$PY" -c "import json,sys;print(len(json.load(open(sys.argv[1]))['rows']))" "$out" 2>/dev/null || echo 0)
  if [ "$have" != "$expect" ]; then say "FAIL $name row count $have != $expect"; return 1; fi
  if grep -q 'Traceback' "$log"; then say "FAIL $name traceback in log"; return 1; fi

  say "OK $name rows=$have sha=$(sha256sum "$out" | cut -d' ' -f1) logsha=$(sha256sum "$log" | cut -d' ' -f1)"
  return 0
}

run_chunk chunk02_real_t5_t8   "t5,t6,t7,t8"     real 80 || { say "ABORT"; exit 1; }
run_chunk chunk03a_real_t9_t10 "t9,t10"          real 40 || { say "ABORT"; exit 1; }
run_chunk chunk03b_swap_t1_t2  "t1,t2"           swap 40 || { say "ABORT"; exit 1; }
run_chunk chunk04_swap_t3_t6   "t3,t4,t5,t6"     swap 80 || { say "ABORT"; exit 1; }
run_chunk chunk05_swap_t7_t10  "t7,t8,t9,t10"    swap 80 || { say "ABORT"; exit 1; }

say "ALL_CHUNKS_DONE"
