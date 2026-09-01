#!/bin/bash
# Pod driver: build sketch_libero_rlds_paired end to end.
#
# Needs on the pod: the LIBERO client venv (examples/libero/.venv in the model
# repo), the TF training env (uv), EGL (apt libegl1), and this repo's scripts
# at /workspace/eval_scripts (shipped by tar; the eval repo is not cloned on
# pods). Run with nohup; ~500 GPU-rendered replays.
set -euo pipefail

REPO=/workspace/SketchPromptVLA-Pi
ES=/workspace/eval_scripts
DEMOS=/workspace/demos
FRAMES="${FRAMES:-/workspace/data/paired_frames_cf}"
OUT="${OUT:-/workspace/data/sketch_libero_rlds_paired_cf/spatial/sketch_libero/1.0.0}"
SCHEMA=/workspace/data/sketch_libero_rlds_upright/spatial/sketch_libero/1.0.0
HF=https://huggingface.co/datasets/yifengzhu-hf/LIBERO-datasets/resolve/main/libero_spatial
BDDL=$REPO/third_party/libero/libero/libero/bddl_files/libero_spatial

source /workspace/env.sh
mkdir -p "$DEMOS" "$FRAMES" /workspace/logs

echo "== 0: captions from the original corpus =="
cd "$REPO"
uv run python - <<'PY'
import collections, json, tensorflow as tf, glob
caps = collections.Counter()
for shard in glob.glob('/workspace/data/sketch_libero_rlds/spatial/sketch_libero/1.0.0/*train*'):
    for rec in tf.data.TFRecordDataset(shard):
        f = tf.train.Example.FromString(rec.numpy()).features.feature
        caps[f['steps/language_instruction'].bytes_list.value[0].decode()] += 1
print(dict(caps))
json.dump(sorted(caps), open('/workspace/captions.json', 'w'))
PY

echo "== 1: demos =="
cd "$DEMOS"
while read -r t; do
  [ -f "${t}_demo.hdf5" ] || { echo "  fetching $t"; curl -sL -O "$HF/${t}_demo.hdf5"; }
done <<'EOF'
pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate
pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate
pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate
pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate
pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate
pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate
pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate
pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate
pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate
pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate
EOF

echo "== 2: replay + render (client venv, EGL) =="
cd "$REPO"
source examples/libero/.venv/bin/activate
export PYTHONPATH="${PYTHONPATH:-}:$REPO/third_party/libero"
export MUJOCO_GL=egl
printf 'N\n' | python -c "import libero.libero" 2>/dev/null || true
python "$ES/build_paired_corpus.py" --bddl-dir "$BDDL" --demo-dir "$DEMOS" \
  --out "$FRAMES" --captions /workspace/captions.json 2>&1 | tail -30
deactivate

echo "== 3: pack to RLDS (TF env) =="
cd "$REPO"
uv run python "$ES/pack_paired_corpus.py" --episodes "$FRAMES" \
  --schema-from "$SCHEMA" --out "$OUT" --require-counterfactual --verify

echo "PAIRED_BUILD_DONE"
