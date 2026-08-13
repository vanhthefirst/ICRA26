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
# libegl1 is the GLVND dispatch library. The NVIDIA vendor driver
# (libEGL_nvidia.so.0) and its ICD are injected by the container runtime, but
# without libEGL.so.1 on top of them PyOpenGL cannot bind EGL at all and every
# MuJoCo render dies with "NoneType has no attribute eglQueryString". Setting
# MUJOCO_GL=egl does not help -- the backend is already right, the library is
# just missing. Costs nothing to install and saves a whole booking.
apt-get install -y -qq curl git tmux libegl1

echo "== uv =="
# Most PyTorch templates ship uv, but not all, and everything below assumes it.
if ! command -v uv >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi
uv --version

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
# uv downloads interpreters into ~/.local/share/uv by default -- container disk.
# A venv built on that base has its packages on the volume and its python on the
# disk that gets wiped, which looks fine until the next pod and then does not
# import at all. Keep interpreters on the volume with the venvs that need them.
grep -q UV_PYTHON_INSTALL_DIR ~/.bashrc || \
    echo "export UV_PYTHON_INSTALL_DIR=$AARON_ROOT/pythons" >> ~/.bashrc

echo "== libero config =="
# libero/libero/__init__.py asks an interactive question the first time it is
# imported if ~/.libero/config.yaml is absent. ~ is the container disk, so it is
# absent on every new pod. Left alone it does not crash a run -- it BLOCKS one,
# on stdin, with the prompt buried in tee output. Seed it with the defaults,
# which is exactly what answering N does.
LIBERO_ROOT="$OPENPI/third_party/libero/libero/libero"
if [ ! -f "$HOME/.libero/config.yaml" ] && [ -d "$LIBERO_ROOT" ]; then
    mkdir -p "$HOME/.libero"
    cat > "$HOME/.libero/config.yaml" <<EOF
assets: $LIBERO_ROOT/./assets
benchmark_root: $LIBERO_ROOT
bddl_files: $LIBERO_ROOT/./bddl_files
datasets: $LIBERO_ROOT/../datasets
init_states: $LIBERO_ROOT/./init_files
EOF
    echo "seeded $HOME/.libero/config.yaml"
else
    echo "$HOME/.libero/config.yaml already present (or LIBERO not unpacked yet)"
fi

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
# -d follows symlinks, so a .venv symlink left pointing at a wiped container disk
# correctly reports as missing here. Build the venvs as real directories on the
# volume, not as symlinks into /root, or they die with every pod.
if [ ! -d "$OPENPI/examples/libero/.venv" ]; then
    echo "  3. openpi is NOT set up at $OPENPI -- follow sections 2-3 of"
    echo "     prompt_pi05_baseline.md. If .venv exists but shows up as missing"
    echo "     here it is a dangling symlink into the old container disk: delete"
    echo "     it and rebuild on the volume."
else
    echo "  3. openpi and the LIBERO venv are present; activate with"
    echo "     source $OPENPI/examples/libero/.venv/bin/activate"
    echo "     export PYTHONPATH=\$PYTHONPATH:$OPENPI/third_party/libero"
    echo "     (the PYTHONPATH line is required -- libero's editable install is"
    echo "      empty, so without it 'import libero' fails)"
fi
echo "  4. tmux new -s pi05"
