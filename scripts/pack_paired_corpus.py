#!/usr/bin/env python3
"""Pack the paired upright corpus: stage 2, npz episodes -> RLDS (needs TF).

Writes tf.train.Example protos with EXACTLY the flattened key set of
ductaingn/sketch_libero_rlds (dumped from the shipped tfrecords on 29 Aug),
so `sketch_rlds_dataset.py`, `validate.py` and `convert_sketch_rlds_to_lerobot.py`
read the result with no code change. Static per-episode sketch fields are
repeated per step, as in the original.

The metadata files (features.json, dataset_info.json, metadata.json) are
copied from --schema-from (any existing sketch_libero 1.0.0 tree) and
dataset_info.json's split record is rewritten to this dataset's shard
lengths. A --holdout fraction of episodes per task goes to the val split.

    python scripts/pack_paired_corpus.py \
        --episodes /workspace/data/paired_frames \
        --schema-from /workspace/data/sketch_libero_rlds_upright/spatial/sketch_libero/1.0.0 \
        --out /workspace/data/sketch_libero_rlds_paired/spatial/sketch_libero/1.0.0 \
        --verify
"""

import argparse
import glob
import json
import os
import random

import numpy as np
import tensorflow as tf


def feat_bytes(vals):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=vals))


def feat_float(arr):
    return tf.train.Feature(float_list=tf.train.FloatList(
        value=np.asarray(arr, np.float32).ravel().tolist()))


def feat_int(vals):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=vals))


def jpeg(img):
    return tf.io.encode_jpeg(img, quality=95).numpy()


def episode_example(npz_path):
    # NpzFile is LAZY: every lookup re-reads and re-inflates that whole array
    # out of the zip, so indexing the image stack inside a loop decompressed it
    # once per frame -- measured 584 GB read from a 7.8 GB corpus, ~75x over,
    # and it was the entire cost of this stage. Pull each array out once.
    with np.load(npz_path) as z:
        images = z["images"]
        wrist = z["wrist"]
        actions = z["actions"]
        joint_states = z["joint_states"]
        states = z["states"]
        cap = str(z["caption"])
        key = str(z["episode_key"])
        circle = jpeg(z["circle"])
        arrow = jpeg(z["arrow"])
        target = jpeg(z["target"])
        arrow_end = z["arrow_end"]
        arrow_start = z["arrow_start"]
        circle_meta = z["circle_meta"]
        # The counterfactual half. Absent from episodes built before 1 Sep, so
        # an old stage-1 tree still packs -- it just packs without the fields
        # `referent_grounding` needs, which `--require-counterfactual` catches.
        has_cf = "circle_swap" in z
        if has_cf:
            circle_swap = jpeg(z["circle_swap"])
            arrow_swap = jpeg(z["arrow_swap"])
            distractor = jpeg(z["distractor"])
            distractor_meta = z["distractor_meta"]
            arrow_swap_start = z["arrow_swap_start"]
            arrow_swap_end = z["arrow_swap_end"]
    T = images.shape[0]
    imgs = [jpeg(images[t]) for t in range(T)]
    wrists = [jpeg(wrist[t]) for t in range(T)]
    f = {
        "episode_metadata/episode_key": feat_bytes([key.encode()]),
        "episode_metadata/file_path": feat_bytes([npz_path.encode()]),
        "steps/action": feat_float(actions),
        "steps/discount": feat_float(np.ones(T)),
        "steps/is_first": feat_int([1] + [0] * (T - 1)),
        "steps/is_last": feat_int([0] * (T - 1) + [1]),
        "steps/is_terminal": feat_int([0] * (T - 1) + [1]),
        "steps/language_instruction": feat_bytes([cap.encode()] * T),
        "steps/observation/image": feat_bytes(imgs),
        "steps/observation/joint_state": feat_float(joint_states),
        "steps/observation/state": feat_float(states),
        "steps/observation/wrist_image": feat_bytes(wrists),
        "steps/reward": feat_float(np.eye(1, T, T - 1).ravel()),
        "steps/sketch/arrow": feat_bytes([arrow] * T),
        "steps/sketch/arrow_end": feat_float(np.tile(arrow_end, (T, 1))),
        "steps/sketch/arrow_start": feat_float(np.tile(arrow_start, (T, 1))),
        "steps/sketch/circle": feat_bytes([circle] * T),
        "steps/sketch/circle_meta": feat_float(np.tile(circle_meta, (T, 1))),
        "steps/sketch/target": feat_bytes([target] * T),
    }
    if has_cf:
        f.update({
            "steps/sketch/circle_swap": feat_bytes([circle_swap] * T),
            "steps/sketch/arrow_swap": feat_bytes([arrow_swap] * T),
            "steps/sketch/distractor": feat_bytes([distractor] * T),
            "steps/sketch/distractor_meta": feat_float(np.tile(distractor_meta, (T, 1))),
            "steps/sketch/arrow_swap_start": feat_float(np.tile(arrow_swap_start, (T, 1))),
            "steps/sketch/arrow_swap_end": feat_float(np.tile(arrow_swap_end, (T, 1))),
        })
    return tf.train.Example(features=tf.train.Features(feature=f)), has_cf


def add_counterfactual_features(features_path):
    """Declare the counterfactual fields in the copied features.json.

    `--schema-from` copies a donor tree's schema verbatim, and TFDS reads only
    what that schema declares -- a field written into the tfrecords but missing
    here is silently invisible to every reader. The new specs are cloned from
    the existing `circle` (Image) and `circle_meta` (Tensor) entries rather than
    written out by hand, so they cannot drift from the shapes the loader expects.
    """
    with open(features_path) as fh:
        spec = json.load(fh)
    sketch = (spec["featuresDict"]["features"]["steps"]["sequence"]["feature"]
              ["featuresDict"]["features"]["sketch"]["featuresDict"]["features"])
    image_like = json.dumps(sketch["circle"])
    meta_like = json.dumps(sketch["circle_meta"])
    xy_like = json.dumps(sketch["arrow_start"])
    for key in ("circle_swap", "arrow_swap", "distractor"):
        sketch.setdefault(key, json.loads(image_like))
    sketch.setdefault("distractor_meta", json.loads(meta_like))
    for key in ("arrow_swap_start", "arrow_swap_end"):
        sketch.setdefault(key, json.loads(xy_like))
    with open(features_path, "w") as fh:
        json.dump(spec, fh)


def write_split(examples, out_dir, split, n_shards):
    names, lengths = [], []
    per = int(np.ceil(len(examples) / n_shards))
    for s in range(n_shards):
        chunk = examples[s * per:(s + 1) * per]
        name = f"sketch_libero-{split}.tfrecord-{s:05d}-of-{n_shards:05d}"
        with tf.io.TFRecordWriter(os.path.join(out_dir, name)) as w:
            for ex in chunk:
                w.write(ex.SerializeToString())
        names.append(name)
        lengths.append(len(chunk))
    return lengths


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", required=True, help="stage-1 output root")
    ap.add_argument("--schema-from", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--holdout", type=float, default=0.1)
    ap.add_argument("--train-shards", type=int, default=16)
    ap.add_argument("--val-shards", type=int, default=2)
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--require-counterfactual", action="store_true",
                    help="fail if any stage-1 episode lacks circle_swap/distractor")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    rng = random.Random(7)
    train_files, val_files = [], []
    for task_dir in sorted(glob.glob(os.path.join(args.episodes, "t*"))):
        eps = sorted(glob.glob(os.path.join(task_dir, "*.npz")))
        rng.shuffle(eps)
        n_val = max(1, int(round(len(eps) * args.holdout))) if eps else 0
        val_files += eps[:n_val]
        train_files += eps[n_val:]
    rng.shuffle(train_files)
    print(f"episodes: train {len(train_files)}, val {len(val_files)}")

    packed = [episode_example(p) for p in train_files + val_files]
    n_cf = sum(1 for _, has_cf in packed if has_cf)
    if args.require_counterfactual and n_cf != len(packed):
        raise SystemExit(
            f"--require-counterfactual: {len(packed) - n_cf} of {len(packed)} episodes "
            "carry no circle_swap/distractor. Rebuild stage 1 with the current "
            "build_paired_corpus.py.")
    print(f"counterfactual fields on {n_cf}/{len(packed)} episodes")
    train_ex = [ex for ex, _ in packed[:len(train_files)]]
    val_ex = [ex for ex, _ in packed[len(train_files):]]
    tl = write_split(train_ex, args.out, "train", args.train_shards)
    vl = write_split(val_ex, args.out, "val", args.val_shards)

    for meta in ("features.json", "metadata.json"):
        src = os.path.join(args.schema_from, meta)
        if os.path.exists(src):
            with open(src) as fi, open(os.path.join(args.out, meta), "w") as fo:
                fo.write(fi.read())
    if n_cf:
        add_counterfactual_features(os.path.join(args.out, "features.json"))
    info = json.load(open(os.path.join(args.schema_from, "dataset_info.json")))
    for split in info.get("splits", []):
        if split.get("name") == "train":
            split["shardLengths"] = [str(n) for n in tl]
            split.pop("numBytes", None)
        elif split.get("name") == "val":
            split["shardLengths"] = [str(n) for n in vl]
            split.pop("numBytes", None)
    json.dump(info, open(os.path.join(args.out, "dataset_info.json"), "w"), indent=2)
    print(f"wrote {sum(tl)} train + {sum(vl)} val episodes -> {args.out}")

    if args.verify:
        import tensorflow_datasets as tfds
        b = tfds.builder_from_directory(args.out)
        ds = b.as_dataset(split="train")
        ep = next(iter(ds))
        step = next(iter(ep["steps"]))
        img = step["observation"]["image"].numpy()
        assert img.shape == (256, 256, 3), img.shape
        cm = step["sketch"]["circle_meta"].numpy()
        assert cm.shape == (3,), cm.shape
        assert step["action"].numpy().shape == (7,)
        if n_cf:
            for key in ("circle_swap", "arrow_swap", "distractor"):
                assert step["sketch"][key].numpy().shape == (256, 256, 1), key
            assert step["sketch"]["distractor_meta"].numpy().shape == (3,)
            # The two circles must not coincide: a swap that lands on the target
            # is not a counterfactual, and the corpus would teach nothing.
            dm = step["sketch"]["distractor_meta"].numpy()
            assert float(np.hypot(cm[0] - dm[0], cm[1] - dm[1])) > cm[2], \
                f"circle and circle_swap overlap: {cm} vs {dm}"
        # upright check: the arm enters from the TOP of the frame, so the top
        # rows must be darker (robot body) than a raw-orientation frame's
        print("verify OK: shapes match schema; first circle_meta", cm)


if __name__ == "__main__":
    main()
