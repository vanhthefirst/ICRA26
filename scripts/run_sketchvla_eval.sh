#!/usr/bin/env bash
# Sketch-Prompted VLA — evaluate the colleague's LoRA-on-pi0.5-base Sketch-VLA
# checkpoint against my ambiguous-prompt baseline. See RUNBOOK_BASELINE_PARITY.md.
#
# WHERE THIS RUNS
#   The GPU pod, TWO terminals — same split as every other pi0.5 run here.
#   The model needs a modern JAX stack, LIBERO needs python 3.8 + mujoco, and
#   they cannot share an interpreter.
#
#     Terminal 1 — openpi main venv:    bash scripts/run_sketchvla_eval.sh server
#     Terminal 2 — libero client env:   bash scripts/run_sketchvla_eval.sh client
#
# HOW THIS DIFFERS FROM run_parity_baseline.sh
#   That script serves STOCK pi0.5-LIBERO with no sketch, to check his harness.
#   This one serves HIS fine-tune with the sketch on, to answer the actual
#   question: does the sketch recover what an ambiguous caption destroys?
#
#   Three flags flip, and getting any of them wrong invalidates the number:
#     --args.no-rotate-180    his checkpoint trained on UNROTATED frames
#                             (verified: 0/25 episodes of sketch_libero_rlds sit
#                             in the modified_libero_rlds orientation). Rotating
#                             here would feed it an upside-down world.
#     --args.sketch-mode      dataset, not none. His training captions are
#       dataset               "do this" / "grab this" -- the language carries no
#                             object information at all, so a text-only run of
#                             this checkpoint measures nothing.
#     --args.dataset-dir      the VALIDATION export, not the training corpus.
#                             The sketch has to belong to the scene being run.
#     --args.prompt-type      ambiguous. The BDDL's own caption is the EXPLICIT
#       ambiguous             wording, which names the object -- something his
#                             model never saw in training. Sending it would flatter
#                             the run and answer the wrong question.
#
# BUDGET  ~1.5 h on one Ampere GPU for 518 trials (37 scenes x 14).

set -euo pipefail

# ---------------------------------------------------------------- edit these --
: "${SKETCHVLA_REPO:=$HOME/SketchPromptVLA-Pi}"      # patched checkout
: "${SPVLA_REPO:=$HOME/sketch_prompted_vla}"         # this repo
: "${CHECKPOINT_DIR:?set CHECKPOINT_DIR to his fine-tuned checkpoint, e.g. checkpoints/.../checkpoint-29999}"
# RLDS holding the sketches for the scenes being evaluated. NOT the training set.
: "${SKETCH_RLDS_DIR:=$SPVLA_REPO/outputs/rlds}"
: "${SKETCH_RLDS_NAME:=sketch_libero_val_spatial_anchored}"
: "${SUITE:=sketch_spatial}"
: "${PORT:=8000}"
# -----------------------------------------------------------------------------

case "${1:-}" in
server)
    cd "$SKETCHVLA_REPO"
    echo "[server] serving Sketch-VLA fine-tune from $CHECKPOINT_DIR on :$PORT"
    # Model variant is auto-detected from the checkpoint keys; pass
    # --model_variant explicitly if that detection guesses wrong.
    exec uv run scripts/serve_policy_sketchvla.py \
        --checkpoint_dir "$CHECKPOINT_DIR" --port "$PORT"
    ;;

client)
    cd "$SKETCHVLA_REPO"
    source examples/libero/.venv/bin/activate
    export PYTHONPATH="$PYTHONPATH:$PWD/third_party/libero"
    export MUJOCO_GL=egl
    # tensorflow_datasets is needed HERE, in the client env: the sketch channel
    # is read from the RLDS by the rollout loop, not by the server.
    python -c "import tensorflow_datasets" 2>/dev/null || {
        echo "[client] tensorflow_datasets missing in this venv — the sketch" >&2
        echo "         cannot be loaded and every episode would run blank." >&2
        echo "         uv pip install tensorflow tensorflow_datasets" >&2
        exit 1
    }

    python examples/libero/main_sketchvla.py \
        --args.task-suite-name "$SUITE" \
        --args.sketch-export-dir "$SPVLA_REPO/outputs/libero_export" \
        --args.sketch-mode dataset \
        --args.prompt-type ambiguous \
        --args.no-rotate-180 \
        --args.dataset-name "$SKETCH_RLDS_NAME" \
        --args.dataset-dir "$SKETCH_RLDS_DIR" \
        --args.num-trials-per-task 14 \
        --args.num-steps-wait 0 \
        --args.max-steps-override 320 \
        --args.success-window 5 \
        --args.replan-steps 5 \
        --args.resize-size 224 \
        --args.port "$PORT" \
        2>&1 | tee "sketchvla_${SUITE}.log"

    cat <<'NOTE'

Read the "Successfully cached N sketches for M tasks" line before the number.
N must equal the scene count (37 for the anchored Spatial set). If it comes back
as a handful, the sketch lookup collapsed onto captions instead of scenes and
every scene got the wrong drawing.

The bar is my AMBIGUOUS arm, not the explicit one — his training captions are
deictic, so that is the matched register:
    37 anchored Spatial scenes, 518 trials  ->  36.5%
    114 scenes, 1,596 trials                ->  17.0%
Explicit (40.3% / 33.8%) is the wrong comparison: those captions name the object,
which his model never saw in training.
NOTE
    ;;

*)
    sed -n '2,32p' "$0"
    exit 1
    ;;
esac
