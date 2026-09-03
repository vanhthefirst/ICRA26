#!/usr/bin/env bash
# V7 on the anchored Spatial scenes: the real arm and its swap arm, paired.
#
# WHAT THIS ANSWERS
#   The mirror image of drivers/v7_baseline_arm.sh. That one brings stock
#   pi0.5-LIBERO onto V7's paired layouts; this one brings V7 onto the scenes
#   where the published baselines were measured, so the fine-tune and the
#   baseline finally meet on both grounds instead of neither.
#
# BOTH ARMS OR NEITHER
#   score_referent_following.py refuses to summarise a real arm without its
#   swap, and it is right to: overlay_v6's real arm alone read as +19.7 points
#   on taking the goal bowl and was a marker degrading the scene representation,
#   not a pointer. Only moving the circle onto the distractor separates the two.
#   This script runs the pair or it runs nothing.
#
# THE SCENE LIST IS NOT OPTIONAL
#   Only 26 of the 37 scenes admit an unambiguous ring around
#   akita_black_bowl_2 -- for the rest it runs off the frame, or also contains
#   bowl_1, or encloses a plate. scripts/build_anchored_swap_sketches.py decides
#   which and writes swap_scene_list_bare.txt; both arms read that same file, so
#   the pairing score_referent_following.py does by scene is a real pairing.
#
#   Those 26 are HARDER than the full 37. Read the result against
#   0.3104 explicit / 0.2775 ambiguous, NOT the published 0.4035 / 0.3649
#   (scripts/analysis/anchored_subset recomputes both from the committed CSVs).
#
# THE SERVER
#   The V7 server from the paired rollout is the right one and is already up:
#   port 8200, --model_variant referent_grounding, rg_v7_paired/2999. If it has
#   to be restarted, --model_variant is mandatory -- without it the server
#   silently defaults to input_overlay and loads the wrong architecture with no
#   error (RUNBOOK_EVAL_POD.md).
#
#   uv run scripts/serve_policy_sketchvla.py \
#     --model_variant referent_grounding \
#     --checkpoint_dir /workspace/SketchPromptVLA-Pi/checkpoints/sketchvla_finetune/rg_v7_paired/2999 \
#     --device 0 --port 8200
#
# --rotate180 IS MANDATORY HERE
#   eval_sketchvla.py defaults it OFF, which matches sketch_libero as shipped
#   and every checkpoint up to pcla_v4. V7 trained on the upright corpus. Stock
#   pi0.5-LIBERO measures 96.7% upright against 0.0% inverted, so getting this
#   wrong measures orientation and nothing else.
#
# BUDGET  26 scenes x 14 rollouts x 2 arms = 728 rollouts.

set -uo pipefail

REPO=${REPO:-/workspace/SketchPromptVLA-Pi}
HARNESS=${HARNESS:-/workspace/eval_scripts}
CKPT=${CKPT:-$REPO/checkpoints/sketchvla_finetune/rg_v7_paired/2999}
PORT=${PORT:-8200}
VARIANT=referent_grounding
PROMPT_TYPE=${PROMPT_TYPE:-ambiguous}
N_ROLLOUTS=${N_ROLLOUTS:-14}
OUT_ROOT=${OUT_ROOT:-$HARNESS/outputs/rollouts}
LOGDIR=${LOGDIR:-/workspace/logs}
STATUS=$LOGDIR/v7_anchored_driver_status.txt

SCENE_FILE=$HARNESS/outputs/validation_set_spatial/swap_scene_list_bare.txt
PY=$REPO/examples/libero/.venv/bin/python
EVAL=$REPO/examples/libero/eval_sketchvla.py

: > "$STATUS"
say() { echo "[$(date -u +%FT%TZ)] $*" | tee -a "$STATUS"; }

[ -f "$SCENE_FILE" ] || {
  say "ABORT: no $SCENE_FILE -- run scripts/build_anchored_swap_sketches.py first"
  exit 1; }
SCENES=$(tr -d '[:space:]' < "$SCENE_FILE")
N_SCENES=$(awk -F, '{print NF}' <<< "$SCENES")
EXPECT=$((N_SCENES * N_ROLLOUTS))
say "scenes=$N_SCENES rollouts=$N_ROLLOUTS expect=$EXPECT rows per arm"

export MUJOCO_GL=egl
export __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/10_nvidia.json

run_arm() {
  local mode="$1" run_id="$2"
  local out="$OUT_ROOT/$run_id/results.csv"
  local log="$LOGDIR/${run_id}.log"

  if [ -f "$out" ]; then
    local had; had=$(( $(wc -l < "$out") - 1 ))
    if [ "$had" = "$EXPECT" ]; then say "SKIP $run_id (already $had rows)"; return 0; fi
    say "WARN $run_id exists with $had rows, expected $EXPECT; re-running"
  fi

  say "START $run_id mode=$mode prompt=$PROMPT_TYPE"
  "$PY" "$EVAL" \
    --suite spatial \
    --prompt-type "$PROMPT_TYPE" \
    --sketch-mode "$mode" \
    --variant "$VARIANT" \
    --checkpoint "$CKPT" \
    --rotate180 \
    --scenes "$SCENES" \
    --n-rollouts "$N_ROLLOUTS" \
    --run-id "$run_id" \
    --out-root "$OUT_ROOT" \
    --harness-repo "$HARNESS" \
    --host 127.0.0.1 --port "$PORT" \
    > "$log" 2>&1
  local rc=$?
  [ $rc -ne 0 ] && { say "FAIL $run_id rc=$rc (see $log)"; return 1; }
  [ -f "$out" ] || { say "FAIL $run_id wrote no results.csv"; return 1; }
  local have; have=$(( $(wc -l < "$out") - 1 ))
  [ "$have" != "$EXPECT" ] && { say "FAIL $run_id rows $have != $EXPECT"; return 1; }
  grep -q Traceback "$log" && { say "FAIL $run_id traceback"; return 1; }
  say "OK $run_id rows=$have sha=$(sha256sum "$out" | cut -d' ' -f1)"
  return 0
}

REAL_ID=sketchvla_rg_v7_${PROMPT_TYPE}_sketch
SWAP_ID=sketchvla_rg_v7_${PROMPT_TYPE}_swap

run_arm real "$REAL_ID" || { say ABORT; exit 1; }
run_arm swap "$SWAP_ID" || { say "ABORT -- the real arm is UNREPORTABLE without this"; exit 1; }

# The real arm alone is not a result. Scoring is part of the run, not a
# follow-up someone may or may not get to.
say "scoring the pair"
python3 "$HARNESS/scripts/score_referent_following.py" \
  --real "$OUT_ROOT/$REAL_ID" \
  --swap "$OUT_ROOT/$SWAP_ID" \
  --out "$HARNESS/outputs/referent_following_v7_anchored.json" \
  2>&1 | tee -a "$STATUS"

say "the baseline for THESE 26 scenes is 0.3104 explicit / 0.2775 ambiguous"
say "ANCHORED_ARM_DONE"
