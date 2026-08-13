#!/usr/bin/env bash
# Full-trial reproduction of the published pi0.5-LIBERO numbers: 50 trials per
# task across 10 tasks = 500 episodes per suite, matching the paper, over all
# four standard suites.
#
# This runs openpi's OWN unmodified eval loop (examples/libero/main.py). It does
# not touch my 114 validation scenes and it does not touch my harness. The only
# difference from the baseline's reproduction is --args.num-trials-per-task,
# which goes from 5 to 50.
#
# Run it from the openpi checkout, inside the LIBERO venv, with the policy server
# already serving in another pane:
#
#     source /workspace/aaron/openpi/examples/libero/.venv/bin/activate
#     export PYTHONPATH=$PYTHONPATH:/workspace/aaron/openpi/third_party/libero
#     export MUJOCO_GL=egl
#     cd /workspace/aaron/openpi
#     bash /workspace/aaron/sketch_prompted_vla/scripts/repro_500.sh
#
# One suite per invocation of main.py, sequentially, each logged separately. A
# suite that has already produced a log is SKIPPED, so re-running the script
# after a crash or a dropped connection picks up where it stopped rather than
# starting the whole thing again.
#
# Expect several hours. Time the first suite and extrapolate before walking away.
# libero_10 is long-horizon and is slower per episode than the other three.

set -uo pipefail

TRIALS="${TRIALS:-50}"           # per task; 50 x 10 tasks = 500 per suite
REPO=/workspace/aaron/sketch_prompted_vla
OUT="$REPO/outputs/rollouts/openpi_repro_500"
SUITES="${SUITES:-libero_spatial libero_object libero_goal libero_10}"

mkdir -p "$OUT"

if [ ! -f examples/libero/main.py ]; then
    echo "ERROR: run this from the openpi checkout root (examples/libero/main.py"
    echo "       not found here). cd /workspace/aaron/openpi first."
    exit 1
fi

if [ "${MUJOCO_GL:-}" != "egl" ]; then
    echo "ERROR: MUJOCO_GL is '${MUJOCO_GL:-unset}', must be 'egl'. A pod has no"
    echo "       X server. export MUJOCO_GL=egl and try again."
    exit 1
fi

echo "trials per task : $TRIALS   (=> $((TRIALS * 10)) episodes per suite)"
echo "suites          : $SUITES"
echo "logs            : $OUT"
echo

for suite in $SUITES; do
    log="$OUT/${suite}.log"

    if [ -s "$log" ] && grep -q "SUITE_DONE" "$log"; then
        echo "== $suite -- already complete, skipping (delete $log to redo)"
        continue
    fi

    echo "== $suite -- starting $(date -u +%H:%M:%SZ)"
    start=$(date +%s)

    MUJOCO_GL=egl python examples/libero/main.py \
        --args.task-suite-name "$suite" \
        --args.num-trials-per-task "$TRIALS" 2>&1 | tee "$log"
    rc=${PIPESTATUS[0]}

    elapsed=$(( $(date +%s) - start ))

    if [ "$rc" -ne 0 ]; then
        echo "== $suite -- FAILED (exit $rc) after $((elapsed / 60))m"
        echo "   log: $log"
        echo "   Fix the cause, then re-run this script; finished suites are skipped."
        exit "$rc"
    fi

    # Sentinel, so a re-run can tell a finished suite from a truncated one. A log
    # that ends without it was interrupted and will be redone.
    {
        echo
        echo "SUITE_DONE suite=$suite trials_per_task=$TRIALS elapsed_s=$elapsed"
    } >> "$log"

    echo "== $suite -- done in $((elapsed / 60))m $((elapsed % 60))s"
    echo
done

echo "All requested suites complete. Logs in $OUT"
echo
echo "Next:"
echo "  1. Pull the per-suite success rates out of the logs."
echo "  2. Update outputs/rollouts/pi05_baseline/openpi_reference.json"
echo "     (trials_per_task, episodes_per_suite, observed rates, verdict)."
echo "  3. Update the reproduction table and the trial-count sentence in"
echo "     report/pi05_baseline/pi05_baseline_report.tex, then recompile."
echo "  4. git add / commit / push BEFORE terminating the pod."
