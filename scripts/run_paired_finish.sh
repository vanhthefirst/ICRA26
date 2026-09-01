#!/bin/bash
# Finish every task that is not yet complete, then pack. Idempotent: --resume
# skips episodes already on the volume, so this is also the migration path onto
# a fresh pod with the same network volume.
#
# There is no NVIDIA EGL on this pod -- only libEGL_mesa -- so every replay
# renders on the CPU through llvmpipe, which spawns one worker thread per core.
# On a 176-core box that is ~64 workers per process; ten processes put ~640
# spinning threads on 176 cores and four of the ten got starved to ~10 min per
# episode while the other six ran at ~30 s. Measured per-step cost is identical
# across the scenes (240 ms, ncon 87), so this is scheduling, not physics.
#
# Fix: cap llvmpipe at LP_NUM_THREADS and split each slow task across ten
# processes by --offset/--stride, so the work is spread over many small
# processes instead of a few starved ones.
set -euo pipefail

REPO=/workspace/SketchPromptVLA-Pi
ES=/workspace/eval_scripts
[ -f "$ES/build_paired_corpus.py" ] || ES="$ES/scripts"
DEMOS=/workspace/demos
FRAMES="${FRAMES:-/workspace/data/paired_frames_cf}"
OUT="${OUT:-/workspace/data/sketch_libero_rlds_paired_cf/spatial/sketch_libero/1.0.0}"
SCHEMA=/workspace/data/sketch_libero_rlds_upright/spatial/sketch_libero/1.0.0
BDDL=$REPO/third_party/libero/libero/libero/bddl_files/libero_spatial
TASKS_LIST="${SLOW:-t1 t2 t3 t4 t5 t6 t7 t8 t9 t10}"
NSHARD="${NSHARD:-10}"

source /workspace/env.sh
cd "$REPO"
source examples/libero/.venv/bin/activate
export PYTHONPATH="${PYTHONPATH:-}:$REPO/third_party/libero"
export MUJOCO_GL=egl
# Pin the EGL vendor. A pod can carry both ICDs (installing libegl1 pulls the
# mesa one in beside the nvidia one), and picking mesa silently means every
# frame is rendered by llvmpipe on the CPU -- measured 0.8 ms/frame on the GPU
# against a per-step cost of 240 ms under llvmpipe. If there is no nvidia ICD
# the fallback is CPU rendering, and LP_NUM_THREADS is what keeps its worker
# threads from oversubscribing the box.
NV_ICD=/usr/share/glvnd/egl_vendor.d/10_nvidia.json
if [ -f "$NV_ICD" ]; then
  export __EGL_VENDOR_LIBRARY_FILENAMES="$NV_ICD"
  echo "  EGL: nvidia (GPU rendering)"
else
  echo "  EGL: no nvidia ICD, falling back to llvmpipe on the CPU"
fi
export LP_NUM_THREADS="${LP_NUM_THREADS:-4}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

# LIBERO asks an interactive question on first import and writes the answer to
# ~/.libero/config.yaml, which lives on the pod's ephemeral disk -- so a fresh
# pod asks again, and a shard with stdin on /dev/null dies with EOFError before
# it renders anything. Answer it once, here, for every shard that follows.
if [ ! -f /root/.libero/config.yaml ]; then
  echo "  seeding ~/.libero/config.yaml"
  printf 'N\n' | python -c "import libero.libero" > /dev/null 2>&1 || true
fi
[ -f /root/.libero/config.yaml ] || { echo "LIBERO CONFIG MISSING"; exit 1; }

echo "== phase 2: $TASKS_LIST x $NSHARD shards, LP_NUM_THREADS=$LP_NUM_THREADS, $(date +%T) =="
pids=""
for t in $TASKS_LIST; do
  for o in $(seq 0 $((NSHARD - 1))); do
    python "$ES/build_paired_corpus.py" --bddl-dir "$BDDL" --demo-dir "$DEMOS" \
      --out "$FRAMES" --captions /workspace/captions.json --tasks "$t" \
      --offset "$o" --stride "$NSHARD" --resume \
      > "/workspace/logs/pair_${t}_$o.log" 2>&1 &
    pids="$pids $!"
  done
  echo "  $t -> $NSHARD shards"
done

fail=0
for p in $pids; do wait "$p" || fail=1; done
echo "== phase 2 replays done $(date +%T), fail=$fail =="
# never pack a partial corpus: a failed shard means episodes are missing, and
# stage 3 would happily write a short RLDS that looks finished
[ "$fail" = 0 ] || { echo "A SHARD FAILED -- not packing"; exit 1; }

# the six first-phase tasks may still be finishing; they are separate processes
while pgrep -f "build_paired_corpus.py" > /dev/null; do sleep 20; done
echo "== all replays done $(date +%T) =="
for t in t1 t2 t3 t4 t5 t6 t7 t8 t9 t10; do
  echo "  $t: $(ls -U $FRAMES/$t 2>/dev/null | wc -l) episodes"
done
deactivate

echo "== stage 3: pack to RLDS =="
cd "$REPO"
uv run python "$ES/pack_paired_corpus.py" --episodes "$FRAMES" \
  --schema-from "$SCHEMA" --out "$OUT" --require-counterfactual --verify

echo "PAIRED_BUILD_DONE"
