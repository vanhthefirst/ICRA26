#!/usr/bin/env bash
# Put this shell into the LIBERO client environment. Source it, do not run it:
#
#     source scripts/libero_env.sh
#
# Everything that touches the simulator — the scene builders, the displacement
# probe, capture_scene_init_states.py, rollout_sketch.py, export_rlds_frames.py —
# needs this environment and fails in a different, unhelpful way without it:
#
#   * outside any venv, or in the RLDS packing venv -> ModuleNotFoundError on cv2
#     or OpenGL, because those live only in the client venv;
#   * in a base conda env that carries the pip `libero` fork -> a bare
#     AssertionError deep inside get_joint_qpos_addr;
#   * missing libegl1 -> "NoneType has no attribute eglQueryString".
#
# Each of those cost a round trip at least once. Sourcing this and reading the
# one-line summary it prints is cheaper than diagnosing any of them again.
#
# For the RLDS packing step, which needs tensorflow and NOT this environment, use
# `deactivate` and then `source /workspace/aaron/rldsenv/bin/activate`.

export OPENPI="${OPENPI:-/workspace/aaron/SketchPromptVLA-Pi}"
export REPO="${REPO:-/workspace/aaron/sketch_prompted_vla}"

_venv="$OPENPI/examples/libero/.venv"
if [ ! -f "$_venv/bin/activate" ]; then
    echo "ERROR: no client venv at $_venv"
    echo "  Set OPENPI to the model repo, or build the venv (RUNBOOK_EVAL_POD.md part E)."
    return 1 2>/dev/null || exit 1
fi

# Leave whatever venv is currently active — sourcing this from inside the RLDS
# env is the common case, and stacking them silently keeps the wrong python.
command -v deactivate >/dev/null 2>&1 && deactivate

# shellcheck disable=SC1090
source "$_venv/bin/activate"
export PYTHONPATH="$PYTHONPATH:$OPENPI/third_party/libero"
export MUJOCO_GL=egl
export LIBERO_SPATIAL_BDDL="$OPENPI/third_party/libero/libero/libero/bddl_files/libero_spatial"

# libEGL.so.1 is an apt package on container disk, so a fresh pod has lost it and
# every render dies. Checking costs nothing; the failure costs a booking.
if ! python -c "from OpenGL import EGL" >/dev/null 2>&1; then
    echo "WARNING: EGL will not bind. Run:  apt-get update && apt-get install -y libegl1 libgl1"
fi

python - <<'PY'
import sys
try:
    import libero, robosuite, cv2
except Exception as e:
    print("  BROKEN: %s: %s" % (type(e).__name__, e))
    raise SystemExit(1)
# LIBERO's top-level package ships without __init__.py, so it imports as a
# namespace package and __file__ is None. Reading it directly raised TypeError
# here and swallowed every line below it, which looked exactly like a broken env.
loc = getattr(libero, "__file__", None) or next(iter(getattr(libero, "__path__", [])), "")
ok = robosuite.__version__.startswith("1.4") and "third_party" in loc
print("  python %s | robosuite %s | cv2 %s" % (sys.version.split()[0], robosuite.__version__, cv2.__version__))
print("  libero %s" % (loc or "<namespace package, no path>"))
print("  %s" % ("READY" if ok else
                "WRONG LIBERO — expected robosuite 1.4.x from the third_party submodule"))
PY

cd "$REPO" || return 1 2>/dev/null || exit 1
echo "  cwd $(pwd)"
