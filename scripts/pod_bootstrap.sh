#!/usr/bin/env bash
# Re-arm a fresh RunPod pod for the pi0.5 baseline.
#
# A pod's container disk is wiped by BOTH stop and terminate, so anything
# installed outside the network volume -- apt packages, node, Claude Code --
# has to be reinstalled on every new pod. Everything under /workspace/aaron/
# (this repo, the openpi checkout, the LIBERO venv, the checkpoint cache)
# survives, so this script only rebuilds the part that does not.
#
# Because it lives in the repo, and the repo lives on the volume, a fresh pod
# needs exactly one command:
#
#     bash /workspace/aaron/sketch_prompted_vla/scripts/pod_bootstrap.sh
#
# See RUNPOD_SETUP.md for the full first-time setup this abbreviates.

set -euo pipefail

AARON_ROOT=/workspace/aaron
REPO="$AARON_ROOT/sketch_prompted_vla"
OPENPI="$AARON_ROOT/openpi"

echo "== GPU =="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

echo "== volume =="
if [ ! -d "$AARON_ROOT" ]; then
    echo "ERROR: $AARON_ROOT is missing. The network volume is not attached, or"
    echo "this pod was deployed without it. Fix that before anything else --"
    echo "work done outside the volume is lost when the pod goes away."
    exit 1
fi
ls -1 "$AARON_ROOT"

echo "== apt =="
apt-get update -qq
apt-get install -y -qq curl git tmux

echo "== node =="
if ! command -v node >/dev/null 2>&1; then
    curl -fsSL https://deb.nodesource.com/setup_22.x | bash -
    apt-get install -y -qq nodejs
fi
node --version

echo "== claude code =="
if ! command -v claude >/dev/null 2>&1; then
    npm install -g @anthropic-ai/claude-code
fi
claude --version

echo "== env =="
# Idempotent: only appended if not already present, so re-running is safe.
grep -q OPENPI_DATA_HOME ~/.bashrc || \
    echo "export OPENPI_DATA_HOME=$AARON_ROOT/.cache/openpi" >> ~/.bashrc
grep -q MUJOCO_GL ~/.bashrc || \
    echo 'export MUJOCO_GL=egl' >> ~/.bashrc

echo "== repo =="
if [ -d "$REPO/.git" ]; then
    git -C "$REPO" pull --ff-only || echo "(pull skipped -- local changes present)"
else
    git clone https://github.com/vanhthefirst/ICRA26.git "$REPO"
fi

echo
echo "Ready. Still to do by hand:"
echo "  1. source ~/.bashrc"
echo "  2. claude   -> log in (the auth token is on the container disk, so this"
echo "                 is needed once per pod)"
if [ ! -d "$OPENPI/examples/libero/.venv" ]; then
    echo "  3. openpi is NOT set up at $OPENPI -- follow sections 2-3 of"
    echo "     prompt_pi05_baseline.md"
else
    echo "  3. openpi and the LIBERO venv are present; activate with"
    echo "     source $OPENPI/examples/libero/.venv/bin/activate"
fi
echo "  4. tmux new -s pi05"
