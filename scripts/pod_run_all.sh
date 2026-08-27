#!/usr/bin/env bash
# Sketch-Prompted VLA — run arms A, B and C unattended ON THE POD.
#
# WHERE THIS RUNS
#   On the GPU pod itself, detached (nohup / tmux). NOT over a held-open SSH
#   pipe: this takes about four hours and a dropped connection must not kill it.
#   Launch it, let go, and poll $STATE/status afterwards.
#
#     nohup bash scripts/pod_run_all.sh > $HOME/run_all.log 2>&1 &
#
# WHAT IT DOES
#   The matched design from HANDOFF_BASELINE_RECONCILIATION.md — same 37
#   anchored Spatial scenes, 14 rollouts, 518 trials, sustained-5 scoring:
#     A  stock pi0.5-LIBERO, explicit captions, no sketch   -> expect 40.3%
#     B  stock pi0.5-LIBERO, ambiguous captions, no sketch  -> expect 36.5%
#     C  his LoRA fine-tune, ambiguous captions, sketch on  -> beat B
#   It manages the policy server itself, restarting it between B and C because
#   C serves a different checkpoint.
#
#   A is a GATE. If A misses its target the harness disagrees with mine, and B
#   and C would be measuring the disagreement rather than the model. The run
#   stops there unless GATE=off.

set -uo pipefail

: "${SKETCHVLA_REPO:=$HOME/SketchPromptVLA-Pi}"
: "${SPVLA_REPO:=$HOME/sketch_prompted_vla}"
: "${CHECKPOINT_DIR:=}"                 # required for arm C; A and B run without it
: "${SUITE:=sketch_spatial}"
: "${TRIALS:=14}"
: "${PORT:=8000}"
: "${STATE:=$HOME/run_all_state}"
: "${GATE:=on}"                         # off = run B and C even if A misses
: "${GATE_LOW:=32}"                     # A must land in [32, 49] -- 40.3 +- ~8
: "${GATE_HIGH:=49}"

EXPORT_DIR="$SPVLA_REPO/outputs/libero_export"
mkdir -p "$STATE"
: > "$STATE/status"

say() { echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$STATE/status"; }

stop_server() {
    [ -n "${SERVER_PID:-}" ] || return 0
    kill "$SERVER_PID" 2>/dev/null
    for _ in $(seq 30); do kill -0 "$SERVER_PID" 2>/dev/null || break; sleep 1; done
    kill -9 "$SERVER_PID" 2>/dev/null
    unset SERVER_PID
}
trap 'stop_server' EXIT

start_server() {   # $1 = arm label, $2.. = command
    local label="$1"; shift
    say "starting server for $label"
    ( cd "$SKETCHVLA_REPO" && nohup "$@" > "$STATE/server_${label}.log" 2>&1 & echo $! > "$STATE/server.pid" )
    SERVER_PID=$(cat "$STATE/server.pid")
    # Wait for the port. A model load is minutes, not seconds; but if the server
    # died, say so instead of waiting out the full timeout in silence.
    for i in $(seq 600); do
        if ! kill -0 "$SERVER_PID" 2>/dev/null; then
            say "FAIL $label: server exited during startup; tail of its log:"
            tail -20 "$STATE/server_${label}.log" | tee -a "$STATE/status"
            return 1
        fi
        (echo > "/dev/tcp/127.0.0.1/$PORT") >/dev/null 2>&1 && { say "server up for $label (${i}s)"; return 0; }
        sleep 1
    done
    say "FAIL $label: server did not open :$PORT within 600s"
    return 1
}

run_client() {     # $1 = arm label, $2.. = extra client flags
    local label="$1"; shift
    say "running client for $label"
    ( cd "$SKETCHVLA_REPO" \
      && source examples/libero/.venv/bin/activate \
      && export PYTHONPATH="$PYTHONPATH:$SKETCHVLA_REPO/third_party/libero" \
      && export MUJOCO_GL=egl \
      && python examples/libero/main_sketchvla.py \
            --args.task-suite-name "$SUITE" \
            --args.sketch-export-dir "$EXPORT_DIR" \
            --args.num-trials-per-task "$TRIALS" \
            --args.num-steps-wait 0 \
            --args.max-steps-override 320 \
            --args.success-window 5 \
            --args.replan-steps 5 \
            --args.resize-size 224 \
            --args.port "$PORT" \
            "$@" ) > "$STATE/client_${label}.log" 2>&1
    local rc=$?
    local rate
    rate=$(grep -oP 'Total success rate:\s*\K[0-9.]+' "$STATE/client_${label}.log" | tail -1)
    if [ $rc -ne 0 ] || [ -z "$rate" ]; then
        say "FAIL $label: client exited $rc, no success rate parsed; tail:"
        tail -25 "$STATE/client_${label}.log" | tee -a "$STATE/status"
        return 1
    fi
    # Sketch pairing check -- a collapsed cache silently mis-pairs every scene.
    local cached
    cached=$(grep -oP 'Successfully cached \K[0-9]+' "$STATE/client_${label}.log" | tail -1)
    [ -n "$cached" ] && say "$label sketches cached: $cached"
    say "RESULT $label = ${rate}%"
    echo "$rate" > "$STATE/rate_${label}"
    return 0
}

# ---------------------------------------------------------------- arm A ------
start_server A uv run scripts/serve_policy.py --env LIBERO --port "$PORT" || exit 1
run_client A --args.sketch-mode none --args.prompt-type explicit || exit 1
A_RATE=$(cat "$STATE/rate_A")

if [ "$GATE" = "on" ]; then
    if ! awk -v r="$A_RATE" -v lo="$GATE_LOW" -v hi="$GATE_HIGH" 'BEGIN{exit !(r>=lo && r<=hi)}'; then
        say "GATE FAILED: arm A is ${A_RATE}%, outside [${GATE_LOW}, ${GATE_HIGH}] around the"
        say "  expected 40.3%. His harness does not agree with mine, so B and C would"
        say "  measure that disagreement rather than the model. Stopping."
        say "DONE (gated)"
        exit 2
    fi
    say "gate passed: A=${A_RATE}% is consistent with 40.3%"
fi

# ---------------------------------------------------------------- arm B ------
run_client B --args.sketch-mode none --args.prompt-type ambiguous || exit 1
stop_server

# ---------------------------------------------------------------- arm C ------
if [ -z "$CHECKPOINT_DIR" ]; then
    say "SKIP C: CHECKPOINT_DIR is not set"
    say "DONE"
    exit 0
fi
start_server C uv run scripts/serve_policy_sketchvla.py \
    --checkpoint_dir "$CHECKPOINT_DIR" --port "$PORT" || exit 1
# C serves a checkpoint trained on UNROTATED frames and needs the sketch. The
# sketches come from the VALIDATION export, keyed by scene.
run_client C \
    --args.sketch-mode dataset \
    --args.prompt-type ambiguous \
    --args.no-rotate-180 \
    --args.dataset-name "${SKETCH_RLDS_NAME:-sketch_libero_val_spatial_anchored}" \
    --args.dataset-dir "${SKETCH_RLDS_DIR:-$SPVLA_REPO/outputs/rlds}" || exit 1
stop_server

say "SUMMARY  A=$(cat "$STATE/rate_A")%  B=$(cat "$STATE/rate_B")%  C=$(cat "$STATE/rate_C")%"
say "DONE"
