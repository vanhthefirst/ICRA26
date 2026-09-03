#!/usr/bin/env bash
set -uo pipefail
cd /workspace/eval_scripts
export PYTHONPATH=/workspace/SketchPromptVLA-Pi/third_party/libero
export MUJOCO_GL=egl
export __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/10_nvidia.json
PY=/workspace/SketchPromptVLA-Pi/examples/libero/.venv/bin/python
BDDL=/workspace/SketchPromptVLA-Pi/third_party/libero/libero/libero/bddl_files/libero_spatial
CKPT=/workspace/SketchPromptVLA-Pi/checkpoints/sketchvla_finetune/rg_v7_paired/2999
O=/workspace/SketchPromptVLA-Pi/outputs
L=/workspace/logs
S=$L/v7_blank_driver_status.txt
: > "$S"
say() { echo "[$(date -u +%FT%TZ)] $*" | tee -a "$S"; }

run() {
  local name="$1" tasks="$2" expect="$3"
  local out="$O/v7_blank_step2999_${name}.json" log="$L/v7_blank_step2999_${name}.log"
  say "START $name tasks=$tasks expect=$expect"
  "$PY" scripts/eval_paired_referent.py \
    --bddl-dir "$BDDL" --demo-dir /workspace/demos --frames-dir /workspace/data/paired_frames_cf \
    --checkpoint "$CKPT" --variant referent_grounding --host 127.0.0.1 --port 8200 \
    --tasks "$tasks" --sketch-modes blank --episodes 20 --max-steps 520 \
    --out "$out" > "$log" 2>&1
  local rc=$? have
  [ $rc -ne 0 ] && { say "FAIL $name rc=$rc"; return 1; }
  have=$("$PY" -c "import json,sys;print(len(json.load(open(sys.argv[1]))['rows']))" "$out" 2>/dev/null || echo 0)
  [ "$have" != "$expect" ] && { say "FAIL $name rows $have != $expect"; return 1; }
  grep -q Traceback "$log" && { say "FAIL $name traceback"; return 1; }
  say "OK $name rows=$have sha=$(sha256sum "$out" | cut -d' ' -f1)"
}

run blank_t1_t5  "t1,t2,t3,t4,t5"      100 || { say ABORT; exit 1; }
run blank_t6_t10 "t6,t7,t8,t9,t10"     100 || { say ABORT; exit 1; }
say "BLANK_ARM_DONE"
