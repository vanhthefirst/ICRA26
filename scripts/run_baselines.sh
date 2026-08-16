#!/usr/bin/env bash
# The two prompt-arm baselines: pi0.5-LIBERO, sketch-free, on
# all 114 validation scenes, once with the explicit caption and once with the
# ambiguous one. 14 rollouts per scene x 38 scenes = 532 trials per suite,
# 1,596 per arm, 3,192 in total.
#
# This drives MY harness (scripts/rollout_sketch.py), not openpi's eval loop.
# The standard-suite reproduction is a different run -- scripts/repro_500.sh --
# and nothing here touches it or the numbers it produced.
#
# Run it from the repo root, inside the LIBERO venv, with the policy server
# already serving in another pane:
#
#     source /workspace/aaron/openpi/examples/libero/.venv/bin/activate
#     export PYTHONPATH=$PYTHONPATH:/workspace/aaron/openpi/third_party/libero
#     export MUJOCO_GL=egl
#     cd /workspace/aaron/sketch_prompted_vla
#     bash scripts/run_baselines.sh
#
# One --run-id per arm, which the harness enforces: both arms write the same
# condition label and the same rollout indices, so a shared run-id would let the
# second arm resume as already-done and report the first arm twice.
#
# Interrupted runs resume. --resume skips any (suite, dir, condition,
# rollout_idx) already in that arm's results.csv, so re-running the script after
# a dropped connection continues rather than restarting. A finished arm is
# detected by row count and skipped.
#
# Budget ~4 hours per arm at the baseline's measured throughput (8.8 s/rollout on
# one RTX 4090), so ~8 hours for both. Time the first suite before walking away.

set -uo pipefail

REPO="${REPO:-/workspace/aaron/sketch_prompted_vla}"
N_ROLLOUTS="${N_ROLLOUTS:-14}"      # x 38 scenes = 532 trials per suite
MAX_STEPS="${MAX_STEPS:-320}"       # same budget the 3-rollout baseline used
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
ARMS="${ARMS:-explicit ambiguous}"
EXPECTED_ROWS=$(( N_ROLLOUTS * 114 ))

cd "$REPO" || exit 1

if [ "${MUJOCO_GL:-}" != "egl" ]; then
    echo "ERROR: MUJOCO_GL is '${MUJOCO_GL:-unset}', must be 'egl'. A pod has no"
    echo "       X server. export MUJOCO_GL=egl and try again."
    exit 1
fi

# The captions must be on disk before a single rollout runs. The harness raises
# on a missing instruction_<type> key rather than falling back, so this would be
# caught anyway -- but it would be caught after the environment for scene_0000
# has been built, which is a slow way to learn it.
if ! python scripts/build_prompt_variants.py --check; then
    echo
    echo "ERROR: the prompt captions are missing or stale. Fix with:"
    echo "         python scripts/build_prompt_variants.py"
    echo "       then commit the result before running the arms."
    exit 1
fi

probe_server() {
    python - "$HOST" "$PORT" <<'PY' 2>&1
import sys
from websockets.sync.client import connect
from openpi_client import msgpack_numpy
host, port = sys.argv[1], sys.argv[2]
try:
    c = connect(f"ws://{host}:{port}", compression=None, max_size=None, open_timeout=10)
    md = msgpack_numpy.unpackb(c.recv(timeout=10))
    c.close()
    print("ok " + str(md))
except Exception as e:
    print(f"fail {type(e).__name__}: {e}")
    sys.exit(1)
PY
}

# Non-skipped data rows in a results.csv. An arm is complete when it has one per
# (scene, rollout). Counted rather than trusted from a sentinel because the
# harness appends as it goes and a killed run leaves a valid partial file.
count_rows() {
    python - "$1" <<'PY'
import csv, os, sys
p = sys.argv[1]
if not os.path.exists(p):
    print(0); raise SystemExit
print(sum(1 for r in csv.DictReader(open(p)) if r["skipped"] != "True"))
PY
}

echo "repo        : $REPO"
echo "arms        : $ARMS"
echo "rollouts    : $N_ROLLOUTS per scene  ->  $(( N_ROLLOUTS * 38 )) per suite, $EXPECTED_ROWS per arm"
echo "max steps   : $MAX_STEPS"
echo "server      : ws://$HOST:$PORT"
echo

for arm in $ARMS; do
    run_id="pi05_${arm}_$(( N_ROLLOUTS * 38 ))"
    run_dir="$REPO/outputs/rollouts/$run_id"
    results="$run_dir/results.csv"

    have=$(count_rows "$results")
    if [ "$have" -ge "$EXPECTED_ROWS" ]; then
        echo "== $arm -- already complete ($have rows in $run_id), skipping"
        continue
    fi
    [ "$have" -gt 0 ] && echo "== $arm -- resuming, $have/$EXPECTED_ROWS rows already done"

    printf "== %s -- probing policy server ... " "$arm"
    if ! probe="$(probe_server)"; then
        echo "UNREACHABLE"
        echo "ERROR: no policy server answering at ws://$HOST:$PORT"
        echo "       $probe"
        echo "       Start it in another pane and re-run; finished rows are kept:"
        echo "         cd /workspace/aaron/openpi && uv run scripts/serve_policy.py --env LIBERO"
        exit 1
    fi
    echo "ok"

    echo "== $arm -- starting $(date -u +%H:%M:%SZ), run-id $run_id"
    start=$(date +%s)

    python scripts/rollout_sketch.py \
        --policy pi05 \
        --conditions text_only \
        --prompt-type "$arm" \
        --run-id "$run_id" \
        --n-rollouts "$N_ROLLOUTS" \
        --max-steps "$MAX_STEPS" \
        --scenes all \
        --resume
    rc=$?
    elapsed=$(( $(date +%s) - start ))

    if [ "$rc" -ne 0 ]; then
        echo "== $arm -- FAILED (exit $rc) after $((elapsed / 60))m"
        echo "   Partial rows are kept in $results. Fix the cause and re-run;"
        echo "   --resume picks up from where it stopped."
        exit "$rc"
    fi

    have=$(count_rows "$results")
    if [ "$have" -lt "$EXPECTED_ROWS" ]; then
        echo "== $arm -- INCOMPLETE: $have/$EXPECTED_ROWS rows after a clean exit."
        echo "   That means scenes were skipped, not that the run died. Check"
        echo "   skip_reason in $results before re-running."
        exit 1
    fi

    echo "== $arm -- done in $((elapsed / 60))m $((elapsed % 60))s, $have rows"
    echo
done

echo "All requested arms complete."
echo
echo "Next:"
echo "  python scripts/analyze_baselines.py \\"
echo "      --arms explicit=pi05_explicit_$(( N_ROLLOUTS * 38 )) ambiguous=pi05_ambiguous_$(( N_ROLLOUTS * 38 )) \\"
echo "      --tables report/prompt_baselines/tables.tex \\"
echo "      --figdir report/prompt_baselines/figures"
echo "  cd report/prompt_baselines && latexmk -pdf report.tex"
echo "  git add / commit / push BEFORE terminating the pod."
