#!/usr/bin/env bash
# Sketch-Prompted VLA — reproduce my 34.5% text-only pi0.5 baseline inside the
# colleague's harness (ductaingn/SketchPromptVLA-Pi). See RUNBOOK_BASELINE_PARITY.md.
#
# WHERE THIS RUNS
#   On the GPU pod, in TWO terminals. Terminal 1 holds the model, terminal 2
#   holds the simulator; they cannot share an interpreter (openpi wants a modern
#   JAX stack, LIBERO wants python 3.8 + mujoco).
#
#   Terminal 1 — openpi's main venv:   bash scripts/run_parity_baseline.sh server
#   Terminal 2 — the libero client env: bash scripts/run_parity_baseline.sh client
#
# WHAT IT PROVES
#   That the 34.5% and the ~0% were not measuring the same thing. This serves the
#   STOCK pi0.5-LIBERO checkpoint, not the Sketch-VLA fine-tune, and sends no
#   sketch — which is exactly what a text-only baseline is.
#
# BUDGET  ~1.5 h on one Ampere GPU: 342 rollouts, 19,608 inference calls,
#         87 ms each => ~28 min of pure inference plus simulator stepping.

set -euo pipefail

# ---------------------------------------------------------------- edit these --
# Checkout of ductaingn/SketchPromptVLA-Pi with sketchvla_baseline_parity.patch applied.
: "${SKETCHVLA_REPO:=$HOME/SketchPromptVLA-Pi}"
# This repo, whose outputs/libero_export/ holds the BDDLs, pinned init states and
# task_map.json that define the 114 validation scenes.
: "${SPVLA_REPO:=$HOME/sketch_prompted_vla}"
: "${PORT:=8000}"
# explicit | ambiguous. The BDDL carries the explicit wording; ambiguous comes
# from the export's task_map.json and needs --args.sketch-export-dir.
: "${PROMPT_TYPE:=explicit}"
# Suites to run. Default is the matched design: the 37 anchored Spatial scenes
# at 14 rollouts, so this run and the Sketch-VLA run in run_sketchvla_eval.sh
# differ ONLY in checkpoint, caption and sketch.
: "${SUITES:=sketch_spatial}"
: "${TRIALS:=14}"
# -----------------------------------------------------------------------------

EXPORT_DIR="$SPVLA_REPO/outputs/libero_export"
MODE="${1:-}"

case "$MODE" in
server)
    # Stock pi0.5-LIBERO from gs://openpi-assets/checkpoints/pi05_libero.
    # NOT serve_policy_sketchvla.py: that one requires --checkpoint_dir and
    # serves the LoRA fine-tune, which is the thing under investigation, not the
    # baseline. Getting this line wrong is the whole discrepancy.
    cd "$SKETCHVLA_REPO"
    echo "[server] serving STOCK pi05_libero on :$PORT"
    exec uv run scripts/serve_policy.py --env LIBERO --port "$PORT"
    ;;

client)
    cd "$SKETCHVLA_REPO"
    # The libero client env. Same activation the baselines and both displacement
    # probes used; EGL because a pod has no X server.
    source examples/libero/.venv/bin/activate
    export PYTHONPATH="$PYTHONPATH:$PWD/third_party/libero"
    export MUJOCO_GL=egl

    if [ ! -f "$EXPORT_DIR/task_map.json" ]; then
        echo "[client] $EXPORT_DIR/task_map.json not found." >&2
        echo "         Set SPVLA_REPO, and install the suites first — copy" >&2
        echo "         outputs/libero_export/{bddl_files,init_files}/<pf> into the" >&2
        echo "         trees get_libero_path(\"bddl_files\") / (\"init_states\") return." >&2
        exit 1
    fi

    # Every flag below is one of the values recorded in
    # outputs/rollouts/pi05_baseline/run_config.json. They are not defaults and
    # the run is not comparable without them.
    #
    #   sketch-mode none      text-only; no sketch keys are sent
    #   rotate-180            ON by default; pi0.5-LIBERO expects the
    #                         modified_libero_rlds orientation
    #   num-steps-wait 0      my init states were pinned AFTER settling, so the
    #                         10 dummy steps openpi uses would start each episode
    #                         somewhere other than the annotated frame
    #   max-steps-override    my per-episode budget; the suite table has no entry
    #     320                 for the sketch_* suites and would raise
    #   success-window 5      the goal predicate must hold for 5 CONSECUTIVE
    #                         steps, so a placement that rolls off is not banked
    #   num-trials-per-task 3 38 scenes x 3 = 114 rollouts per suite, 342 total
    for SUITE in $SUITES; do
        echo "[client] ===== $SUITE ====="
        python examples/libero/main_sketchvla.py \
            --args.task-suite-name "$SUITE" \
            --args.sketch-export-dir "$EXPORT_DIR" \
            --args.sketch-mode none \
            --args.prompt-type "$PROMPT_TYPE" \
            --args.num-trials-per-task "$TRIALS" \
            --args.num-steps-wait 0 \
            --args.max-steps-override 320 \
            --args.success-window 5 \
            --args.replan-steps 5 \
            --args.resize-size 224 \
            --args.port "$PORT" \
            2>&1 | tee "parity_${SUITE}_${PROMPT_TYPE}.log"
    done

    echo
    echo "[client] expected, stock pi0.5-LIBERO on the 37 anchored Spatial scenes"
    echo "         at 14 rollouts (518 trials), from outputs/rollouts/:"
    echo "           PROMPT_TYPE=explicit   -> 40.3%   (pi05_anchored_explicit_518)"
    echo "           PROMPT_TYPE=ambiguous  -> 36.5%   (pi05_anchored_ambiguous_518)"
    echo "         SE at 518 trials is ~2 points; within ~5 is agreement."
    echo "         Near 0% means the stock checkpoint is not what is being served."
    echo
    echo "         SUITES=\"sketch_spatial sketch_object sketch_goal\" TRIALS=3"
    echo "         instead reproduces pi05_baseline: 24.6 / 27.2 / 51.8, all 34.5%."
    ;;

*)
    sed -n '2,20p' "$0"
    exit 1
    ;;
esac
