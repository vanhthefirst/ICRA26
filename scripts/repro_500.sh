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
# PYTHONPATH is not optional. third_party/libero has no top-level __init__.py, so
# `uv pip install -e third_party/libero` installs an empty package mapping and
# `import libero` fails without it. The script checks this before it starts.
#
# One suite per invocation of main.py, sequentially, each logged separately. A
# suite that has already produced a log is SKIPPED, so re-running the script
# after a crash or a dropped connection picks up where it stopped rather than
# starting the whole thing again.
#
# A suite is only marked done if it (a) exited 0, (b) reached the final
# "Total success rate:" line, and (c) did not blow past MAX_EXC_PCT episodes with
# exceptions. main.py swallows per-episode exceptions and still exits 0, so exit
# status alone cannot tell a real 98% from a server that died at episode 3.
#
# Expect several hours. Time the first suite and extrapolate before walking away.
# libero_10 is long-horizon and is slower per episode than the other three.

set -uo pipefail

TRIALS="${TRIALS:-50}"           # per task; 50 x 10 tasks = 500 per suite
REPO=/workspace/aaron/sketch_prompted_vla
OUT="$REPO/outputs/rollouts/openpi_repro_500"
SUITES="${SUITES:-libero_spatial libero_object libero_goal libero_10}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"

# Videos default to the container disk, not the network volume. main.py writes an
# mp4 per episode but names them only by task + success/failure, so they overwrite
# each other and almost none of it is kept -- no reason to pay MooseFS for 2,000
# encodes. Set VIDEO_OUT if you actually want them.
VIDEO_OUT="${VIDEO_OUT:-/tmp/libero_videos}"

# A suite is rejected if more than this percentage of its episodes hit an
# exception. See the post-run check below for why that matters.
MAX_EXC_PCT="${MAX_EXC_PCT:-10}"

mkdir -p "$OUT" "$VIDEO_OUT"

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

# Every standard suite ships exactly 50 initial states per task, and main.py
# indexes them directly (initial_states[episode_idx]) OUTSIDE its try block, so
# asking for more than 50 is an uncaught IndexError partway through a suite.
case "$TRIALS" in
    ''|*[!0-9]*)
        echo "ERROR: TRIALS='$TRIALS' is not a positive integer."
        exit 1 ;;
esac
if [ "$TRIALS" -lt 1 ] || [ "$TRIALS" -gt 50 ]; then
    echo "ERROR: TRIALS=$TRIALS is out of range. LIBERO ships 50 initial states"
    echo "       per task, and main.py indexes them without bounds-checking, so"
    echo "       anything above 50 dies mid-suite. Use 1..50."
    exit 1
fi

# The LIBERO venv. third_party/libero has no top-level libero/__init__.py, so
# find_packages() matches nothing and `uv pip install -e third_party/libero`
# installs an EMPTY mapping -- PYTHONPATH is what actually makes libero
# importable, not the editable install. Check it rather than assume it.
# stdin is closed so that a missing ~/.libero/config.yaml raises EOFError here
# instead of silently blocking on input() for the rest of the day.
if ! python -c "import libero.libero" </dev/null >/dev/null 2>&1; then
    echo "ERROR: cannot import libero. Two usual causes:"
    echo "  1. PYTHONPATH is missing the LIBERO checkout. Fix:"
    echo "       export PYTHONPATH=\$PYTHONPATH:$PWD/third_party/libero"
    echo "  2. ~/.libero/config.yaml does not exist, so libero wants to ask an"
    echo "     interactive question. It lives on the container disk and does not"
    echo "     survive a new pod. Re-create it, or import libero once by hand and"
    echo "     answer N."
    echo
    echo "Diagnostic:"
    python -c "import libero.libero" </dev/null 2>&1 | tail -5
    exit 1
fi

# Is the policy server actually answering? Without this the two failure modes are
# both silent: a server that is down at startup makes main.py's _wait_for_server()
# retry FOREVER (it only catches ConnectionRefusedError and sleeps), and a server
# that dies mid-suite makes every later infer() raise into main.py's per-episode
# except-and-break, scoring the rest of the suite as failures and still exiting 0.
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

echo "trials per task : $TRIALS"
echo "suites          : $SUITES"
echo "logs            : $OUT"
echo "videos          : $VIDEO_OUT"
echo "policy server   : ws://$HOST:$PORT"
echo

for suite in $SUITES; do
    log="$OUT/${suite}.log"

    if [ -s "$log" ] && grep -q "SUITE_DONE" "$log"; then
        echo "== $suite -- already complete, skipping (delete $log to redo)"
        continue
    fi

    # Probe before every suite, not just the first: suite 1 can finish clean and
    # the server can still be gone by the time suite 2 starts.
    printf "== %s -- probing policy server ... " "$suite"
    if ! probe="$(probe_server)"; then
        echo "UNREACHABLE"
        echo
        echo "ERROR: no policy server answering at ws://$HOST:$PORT"
        echo "       $probe"
        echo "       Start it in another pane and re-run:"
        echo "         cd /workspace/aaron/openpi && uv run scripts/serve_policy.py --env LIBERO"
        echo "       Nothing was written for $suite, so nothing is skipped on re-run."
        exit 1
    fi
    echo "ok"

    echo "== $suite -- starting $(date -u +%H:%M:%SZ)"
    start=$(date +%s)

    MUJOCO_GL=egl python examples/libero/main.py \
        --args.task-suite-name "$suite" \
        --args.num-trials-per-task "$TRIALS" \
        --args.video-out-path "$VIDEO_OUT/$suite" 2>&1 | tee "$log"
    rc=${PIPESTATUS[0]}

    elapsed=$(( $(date +%s) - start ))

    if [ "$rc" -ne 0 ]; then
        echo "== $suite -- FAILED (exit $rc) after $((elapsed / 60))m"
        echo "   log: $log"
        echo "   Fix the cause, then re-run this script; finished suites are skipped."
        exit "$rc"
    fi

    # rc=0 is NOT enough to call a suite good. main.py catches every per-episode
    # exception, breaks, scores it a failure and carries on, so a server that died
    # partway produces a full-length run of failures that exits 0. Without the two
    # checks below that run would get a SUITE_DONE and be skipped forever.
    episodes=$(grep -oE "Total episodes: [0-9]+" "$log" | tail -1 | grep -oE "[0-9]+")
    if ! grep -q "Total success rate:" "$log" || [ -z "$episodes" ]; then
        echo "== $suite -- INCOMPLETE: log has no final 'Total success rate:' line."
        echo "   The suite did not reach the end of its task loop. Not marking it done."
        echo "   log: $log"
        exit 1
    fi

    exceptions=$(grep -c "Caught exception" "$log")
    if [ "$exceptions" -gt 0 ]; then
        pct=$(( exceptions * 100 / episodes ))
        echo "== $suite -- WARNING: $exceptions/$episodes episodes ($pct%) hit an exception."
        if [ "$pct" -ge "$MAX_EXC_PCT" ]; then
            echo "   That is at or above MAX_EXC_PCT=$MAX_EXC_PCT%, which is what a policy"
            echo "   server dying mid-suite looks like. Refusing to mark this suite done."
            echo "   Check the server pane, then re-run -- $suite will be redone."
            echo "   log: $log"
            exit 1
        fi
    fi

    rate=$(grep -oE "Total success rate: [0-9.]+" "$log" | tail -1 | cut -d' ' -f4)

    # Sentinel, so a re-run can tell a finished suite from a truncated one. A log
    # that ends without it was interrupted and will be redone.
    {
        echo
        echo "SUITE_DONE suite=$suite trials_per_task=$TRIALS episodes=$episodes rate=$rate exceptions=$exceptions elapsed_s=$elapsed"
    } >> "$log"

    echo "== $suite -- done in $((elapsed / 60))m $((elapsed % 60))s  rate=$rate  episodes=$episodes"
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
