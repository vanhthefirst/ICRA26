#!/usr/bin/env bash
# The baseline arm of the V7 paired rollout: STOCK pi0.5-LIBERO on the exact
# ten paired layouts, the same twenty demos each, the same donor poses, the
# same 520-step budget and the same success test as the sketch arms.
#
# WHY THIS EXISTS
#   Every published baseline for this project -- 40.3% explicit, 36.5%
#   ambiguous -- was measured on 37 anchored Spatial scenes, 14 rollouts each,
#   with a sustained-5 success criterion. The V7 numbers are 10 paired layouts,
#   20 episodes each, instantaneous success. Reading 0.855 against 0.403 across
#   those two designs is not a comparison, and until this arm has run the
#   baseline-versus-fine-tune question is open no matter how good V7 looks.
#
# TWO ARMS, AND BOTH ARE NEEDED
#   explicit -- pi0.5 is TOLD which bowl, in the layout's own BDDL wording.
#               The ceiling: how hard these layouts are, sketch aside.
#   stored   -- pi0.5 gets the corpus's referent-free caption ("do this").
#               The floor, and the prior any sketch effect must beat.
#
# THIS NEEDS A SECOND SERVER. Port 8200 serves the V7 fine-tune and must be
# left running. Serve the stock checkpoint on 8300, in the openpi venv:
#
#   cd /workspace/SketchPromptVLA-Pi && \
#     uv run scripts/serve_policy.py --env LIBERO --port 8300
#
# NOT serve_policy_sketchvla.py. The evaluator refuses a server that reports a
# checkpoint_dir, so a mixed-up port fails loudly rather than producing a
# fine-tune number labelled "baseline" -- but check the command anyway.
#
# BUDGET  400 episodes, ~520 steps, chunked 100 at a time.

set -uo pipefail
cd /workspace/eval_scripts
export PYTHONPATH=/workspace/SketchPromptVLA-Pi/third_party/libero
export MUJOCO_GL=egl
export __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/10_nvidia.json

PY=/workspace/SketchPromptVLA-Pi/examples/libero/.venv/bin/python
BDDL=/workspace/SketchPromptVLA-Pi/third_party/libero/libero/libero/bddl_files/libero_spatial
O=/workspace/SketchPromptVLA-Pi/outputs
L=/workspace/logs
PORT=${PORT:-8300}
S=$L/v7_baseline_driver_status.txt
: > "$S"
say() { echo "[$(date -u +%FT%TZ)] $*" | tee -a "$S"; }

run() {
  local caption="$1" name="$2" tasks="$3" expect="$4"
  local out="$O/v7_baseline_pi05_${caption}_${name}.json"
  local log="$L/v7_baseline_pi05_${caption}_${name}.log"

  if [ -f "$out" ]; then
    local had
    had=$("$PY" -c "import json,sys;print(len(json.load(open(sys.argv[1]))['rows']))" "$out" 2>/dev/null || echo 0)
    if [ "$had" = "$expect" ]; then say "SKIP $caption/$name (already $had rows)"; return 0; fi
    say "WARN $caption/$name exists with $had rows, expected $expect; re-running"
  fi

  say "START $caption/$name tasks=$tasks expect=$expect"
  "$PY" scripts/eval_paired_referent.py \
    --bddl-dir "$BDDL" --demo-dir /workspace/demos \
    --frames-dir /workspace/data/paired_frames_cf \
    --policy pi05 --caption "$caption" \
    --host 127.0.0.1 --port "$PORT" \
    --tasks "$tasks" --sketch-modes blank --episodes 20 --max-steps 520 \
    --out "$out" > "$log" 2>&1
  local rc=$? have
  [ $rc -ne 0 ] && { say "FAIL $caption/$name rc=$rc (see $log)"; return 1; }
  have=$("$PY" -c "import json,sys;print(len(json.load(open(sys.argv[1]))['rows']))" "$out" 2>/dev/null || echo 0)
  [ "$have" != "$expect" ] && { say "FAIL $caption/$name rows $have != $expect"; return 1; }
  grep -q Traceback "$log" && { say "FAIL $caption/$name traceback"; return 1; }
  say "OK $caption/$name rows=$have sha=$(sha256sum "$out" | cut -d' ' -f1)"
  return 0
}

for caption in explicit stored; do
  run "$caption" t1_t5  "t1,t2,t3,t4,t5"  100 || { say ABORT; exit 1; }
  run "$caption" t6_t10 "t6,t7,t8,t9,t10" 100 || { say ABORT; exit 1; }
done
say "BASELINE_ARM_DONE"
