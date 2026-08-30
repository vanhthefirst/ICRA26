#!/bin/bash
# Stage 2 sharded by task, then stage 3.
#
# build_paired_corpus.py replays one demo at a time in a single process: ~27.6 s
# each, measured, so 500 of them is ~3.8 h of one core on a 176-core pod. The
# tasks are independent -- each writes its own out/tN -- and the only shared
# state is an rng that picks a deictic caption, so running the ten as ten
# processes changes which arbitrary caption an episode draws and nothing else.
# Wall time becomes the slowest single task, ~25 min.
set -euo pipefail

REPO=/workspace/SketchPromptVLA-Pi
ES=/workspace/eval_scripts
DEMOS=/workspace/demos
FRAMES=/workspace/data/paired_frames
OUT=/workspace/data/sketch_libero_rlds_paired/spatial/sketch_libero/1.0.0
SCHEMA=/workspace/data/sketch_libero_rlds_upright/spatial/sketch_libero/1.0.0
BDDL=$REPO/third_party/libero/libero/libero/bddl_files/libero_spatial
TASKS="${TASKS:-t1 t2 t3 t4 t5 t6 t7 t8 t9 t10}"

source /workspace/env.sh
cd "$REPO"
source examples/libero/.venv/bin/activate
export PYTHONPATH="${PYTHONPATH:-}:$REPO/third_party/libero"
export MUJOCO_GL=egl
# each shard is already single-threaded physics; keep BLAS from fighting over cores
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2

echo "== stage 2: $(echo $TASKS | wc -w) shards, $(date +%T) =="
pids=""
for t in $TASKS; do
  python "$ES/build_paired_corpus.py" --bddl-dir "$BDDL" --demo-dir "$DEMOS" \
    --out "$FRAMES" --captions /workspace/captions.json --tasks "$t" \
    > "/workspace/logs/pair_$t.log" 2>&1 &
  pids="$pids $!"
  echo "  $t -> pid $!"
  sleep 2   # stagger the EGL context creations
done

fail=0
for p in $pids; do wait "$p" || fail=1; done
echo "== stage 2 done $(date +%T), fail=$fail =="
grep -h "^== t.*kept" /workspace/logs/pair_t*.log | sort -V
[ "$fail" = 0 ] || { echo "A SHARD FAILED"; exit 1; }
deactivate

echo "== stage 3: pack to RLDS =="
cd "$REPO"
uv run python "$ES/pack_paired_corpus.py" --episodes "$FRAMES" \
  --schema-from "$SCHEMA" --out "$OUT" --verify

echo "PAIRED_BUILD_DONE"
