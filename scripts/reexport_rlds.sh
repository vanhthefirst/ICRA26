#!/usr/bin/env bash
# Sketch-Prompted VLA — re-export the RLDS in the orientation pi0.5 expects.
#
# WHERE THIS RUNS
#   Anywhere with the `libero` client env (mujoco/robosuite) for stage 1, plus
#   tensorflow + tensorflow_datasets for stage 2. The pod satisfies both. A
#   laptop can too — this renders scenes, it does not run a model, so no GPU
#   inference is involved. Budget minutes, not hours.
#
# WHY
#   The shipped export was written in raw robosuite orientation (arm at the
#   bottom of the frame). openvla/modified_libero_rlds — what pi0.5-LIBERO was
#   fine-tuned on — sits 180 degrees from that, and no consumer downstream
#   rotates: convert_libero_data_to_lerobot.py passes the image straight through.
#   Audited across all 74 episodes: 0/74 correct before, 74/74 after.
#
# THIS IS NOT NEEDED FOR THE BASELINE PARITY RUN. A text-only baseline reads no
# sketch and touches no RLDS. Do this before any TRAINING run.
#
# IT INVALIDATES ANY CHECKPOINT ALREADY TRAINED ON THE OLD EXPORT — those saw an
# inverted world. Coordinate before re-uploading.

set -euo pipefail

: "${SUITE:=spatial}"
: "${OPENPI:=$HOME/openpi}"

cd "$(dirname "$0")/.."
REPO="$PWD"

echo "== stage 0: audit what is there now (no TensorFlow needed) =="
python scripts/audit_rlds_export.py "outputs/rlds/sketch_libero_val_${SUITE}_anchored" || true

echo
echo "== stage 1: re-render at 256 in the libero env =="
# Needs mujoco/robosuite/libero. An ImportError on `robosuite` means this env is
# not active. EGL because the pod has no X server; use osmesa on a machine
# without a GPU.
source "$OPENPI/examples/libero/.venv/bin/activate"
export PYTHONPATH="$PYTHONPATH:$OPENPI/third_party/libero"
export MUJOCO_GL=egl
cd "$REPO"
python scripts/export_rlds_frames.py --suite "$SUITE"
# `--no-rotate180` reproduces the old, inverted output. Do not pass it.

echo
echo "== stage 2: pack to TFDS and verify =="
# Needs tensorflow + tensorflow_datasets, which are NOT in the libero client env.
# Switch to whichever env has them before running this line.
echo "   (deactivate the libero env first if tfds lives elsewhere)"
python scripts/export_rlds_pack.py --suite "$SUITE"
# --verify now fails on an inverted frame and on a circle_meta that has parted
# company with its mask. Both were silent before.

echo
echo "== stage 3: independent audit of the rewritten export =="
python scripts/audit_rlds_export.py "outputs/rlds/sketch_libero_val_${SUITE}_anchored"

cat <<'NOTE'

Expected after the fix:
    frames in pi0.5 orientation      74 / 74
    ORIENTATION OK

`episodes with all-zero actions 74 / 74` and `NOT TRAINABLE` stay — correctly.
This is a validation set; it has no demonstrations by construction and must not
be trained on. Rotating it does not change that, and is not meant to.
NOTE
