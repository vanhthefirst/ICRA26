"""
Sketch-Prompted VLA — stage 2 of the RLDS export: pack the rendered frames into a
TFDS/RLDS dataset (needs tensorflow + tensorflow_datasets, NOT the libero env).

Reads what `export_rlds_frames.py` wrote and emits a dataset that
`tfds.load(name, data_dir=...)` opens and that
`src/sketchvla/utils/convert_libero_data_to_lerobot.py` consumes unchanged — that
script is the specification for every field name, shape and dtype here.

    # A THROWAWAY venv, not openpi's. This script needs only numpy and tfds, and
    # installing TensorFlow into the openpi env would drag a second CUDA stack in
    # beside JAX's — the training environment is not worth risking to write some
    # TFRecords. tensorflow-cpu is deliberate for the same reason.
    uv venv --python 3.11 /workspace/aaron/rldsenv
    source /workspace/aaron/rldsenv/bin/activate
    uv pip install "tensorflow-cpu" tensorflow_datasets numpy
    cd $REPO
    python scripts/export_rlds_pack.py --suite spatial
    python scripts/export_rlds_pack.py --suite spatial --verify

If `tensorflow_datasets` refuses to install because `promise` fails to build — a
2019 package with no wheel for modern Python — install around it; tfds only
imports it on a download path this exporter never takes, and the stub below
covers the import:

    uv pip install --no-deps tensorflow_datasets
    uv pip install "etils[epath,enp,epy,etree]" array_record dm-tree \
        immutabledict simple_parsing toml termcolor psutil pyarrow \
        tensorflow-cpu tensorflow-metadata

Then the colleague runs the existing converter with no edits beyond the name:

    uv run src/sketchvla/utils/convert_libero_data_to_lerobot.py \
        --data_dir /path/to/sketch_prompted_vla/outputs/rlds

TWO EPISODES PER SCENE. The benchmark is scenes x captions, so each scene emits an
`explicit` episode and an `ambiguous` one over identical frames, differing only in
`language_instruction`. That reproduces the 74 spatial evaluation rows exactly and
lets a consumer filter on `episode_metadata.prompt_type` instead of re-deriving
captions.

ONE STEP PER EPISODE. These scenes are initial states with no demonstration, so
there is one frame and `is_first == is_last == is_terminal == True`. Anything that
assumes a trajectory will see a length-1 episode rather than silently reading a
fabricated one.

`--verify` is the part that matters. It reopens the written dataset through
`tfds.builder_from_directory`, walks every episode, and asserts each field the
converter touches with the shape and dtype it expects. A schema this is checked
against is worth more than a schema that was carefully written.
"""

import os, sys, json, argparse, types

import numpy as np

# `promise` is a 2019 package that no longer builds on current Python. tfds lists
# it as a dependency but only imports it on a download path this exporter never
# takes, so a stub keeps the writer usable on environments where the wheel fails.
if "promise" not in sys.modules:
    try:
        import promise  # noqa: F401
    except Exception:
        _m = types.ModuleType("promise")
        _m.Promise = object
        sys.modules["promise"] = _m

import tensorflow_datasets as tfds

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FRAMES_ROOT = os.path.join(_REPO, "outputs", "rlds_frames")
RLDS_ROOT = os.path.join(_REPO, "outputs", "rlds")
RES = 256
VERSION = "1.0.0"
PROMPT_TYPES = ("explicit", "ambiguous")

# Exactly the fields convert_libero_data_to_lerobot.py reads, plus the RLDS step
# flags. Adding anything here is free; removing anything breaks that converter.
CONSUMER_FIELDS = [
    (("observation", "image"), (RES, RES, 3), np.uint8),
    (("observation", "wrist_image"), (RES, RES, 3), np.uint8),
    (("observation", "state"), (8,), np.float32),
    (("action",), (7,), np.float32),
    (("sketch", "circle"), (RES, RES, 1), np.uint8),
    (("sketch", "arrow"), (RES, RES, 1), np.uint8),
    (("sketch", "target"), (RES, RES, 1), np.uint8),
    (("sketch", "circle_meta"), (3,), np.float32),
    (("sketch", "arrow_start"), (2,), np.float32),
    (("sketch", "arrow_end"), (2,), np.float32),
]


def features():
    return tfds.features.FeaturesDict({
        "steps": tfds.features.Dataset({
            "observation": tfds.features.FeaturesDict({
                "image": tfds.features.Image(shape=(RES, RES, 3), dtype=np.uint8),
                "wrist_image": tfds.features.Image(shape=(RES, RES, 3), dtype=np.uint8),
                "state": tfds.features.Tensor(shape=(8,), dtype=np.float32),
            }),
            "action": tfds.features.Tensor(shape=(7,), dtype=np.float32),
            "language_instruction": tfds.features.Text(),
            "sketch": tfds.features.FeaturesDict({
                "circle": tfds.features.Image(shape=(RES, RES, 1), dtype=np.uint8),
                "arrow": tfds.features.Image(shape=(RES, RES, 1), dtype=np.uint8),
                "target": tfds.features.Image(shape=(RES, RES, 1), dtype=np.uint8),
                "circle_meta": tfds.features.Tensor(shape=(3,), dtype=np.float32),
                "arrow_start": tfds.features.Tensor(shape=(2,), dtype=np.float32),
                "arrow_end": tfds.features.Tensor(shape=(2,), dtype=np.float32),
            }),
            "is_first": np.bool_,
            "is_last": np.bool_,
            "is_terminal": np.bool_,
            "reward": np.float32,
            "discount": np.float32,
        }),
        "episode_metadata": tfds.features.FeaturesDict({
            "suite": tfds.features.Text(),
            "scene": tfds.features.Text(),
            "tier": tfds.features.Text(),
            "task": tfds.features.Text(),
            "prompt_type": tfds.features.Text(),
            "target": tfds.features.Text(),
            "destination": tfds.features.Text(),
            "goal_predicate": tfds.features.Text(),
            "out_of_focus": tfds.features.Text(),
            "anchored": np.bool_,
            "has_demonstration": np.bool_,
            "seed": np.int64,
        }),
    })


def episode(row, arrays, prompt_type):
    caption = row["instruction_%s" % prompt_type]
    if not caption:
        raise ValueError("%s/%s has no %s caption — run build_prompt_variants.py"
                         % (row["suite"], row["dir"], prompt_type))
    step = {
        "observation": {
            "image": arrays["image"],
            "wrist_image": arrays["wrist_image"],
            "state": arrays["state"].astype(np.float32),
        },
        "action": arrays["action"].astype(np.float32),
        "language_instruction": caption,
        "sketch": {
            "circle": arrays["circle"],
            "arrow": arrays["arrow"],
            "target": arrays["target"],
            "circle_meta": arrays["circle_meta"].astype(np.float32),
            "arrow_start": arrays["arrow_start"].astype(np.float32),
            "arrow_end": arrays["arrow_end"].astype(np.float32),
        },
        "is_first": True,
        "is_last": True,
        "is_terminal": True,
        "reward": np.float32(0.0),
        "discount": np.float32(1.0),
    }
    return {
        "steps": [step],
        "episode_metadata": {
            "suite": row["suite"],
            "scene": row["dir"],
            "tier": row.get("tier") or "",
            "task": row.get("task") or "",
            "prompt_type": prompt_type,
            "target": row["target"],
            "destination": row["destination"],
            "goal_predicate": row.get("goal_predicate") or "On",
            "out_of_focus": ",".join(row.get("out_of_focus") or []),
            "anchored": bool(row.get("anchored")),
            "has_demonstration": False,
            "seed": int(row.get("seed") or -1),
        },
    }


def write(suite, name, split):
    frames_dir = os.path.join(FRAMES_ROOT, suite)
    man_path = os.path.join(frames_dir, "frames_manifest.json")
    if not os.path.exists(man_path):
        raise SystemExit("no frames at %s — run export_rlds_frames.py first" % man_path)
    man = json.load(open(man_path))
    if man["resolution"] != RES:
        raise SystemExit("frames are %dpx but this packer writes %dpx"
                         % (man["resolution"], RES))

    data_dir = os.path.join(RLDS_ROOT, name, VERSION)
    os.makedirs(data_dir, exist_ok=True)
    ident = tfds.core.dataset_info.DatasetIdentity(
        name=name, version=tfds.core.Version(VERSION),
        data_dir=data_dir, module_name="")
    info = tfds.core.DatasetInfo(
        builder=ident, features=features(),
        description=("Sketch-Prompted VLA validation scenes, %s suite, as RLDS. "
                     "One step per episode (initial state only, no demonstration; "
                     "action is zero and has_demonstration is False). Two episodes "
                     "per scene, one per caption style." % suite))

    writer = tfds.core.SequentialWriter(ds_info=info, max_examples_per_shard=64,
                                        overwrite=True)
    writer.initialize_splits([split])

    n = 0
    for row in man["scenes"]:
        arrays = np.load(os.path.join(frames_dir, row["npz"]))
        eps = [episode(row, arrays, p) for p in PROMPT_TYPES]
        writer.add_examples({split: eps})
        n += len(eps)
        print("  %-12s -> %d episode(s)" % (row["dir"], len(eps)))
    writer.close_all()

    print("\nwrote %d episode(s) from %d scene(s) to %s"
          % (n, len(man["scenes"]), data_dir))
    print("load with: tfds.load('%s', data_dir='%s', split='%s')"
          % (name, RLDS_ROOT, split))
    return data_dir, n


def verify(name, split, expect_episodes=None):
    """Reopen and assert the consumer's access pattern, field by field."""
    data_dir = os.path.join(RLDS_ROOT, name, VERSION)
    builder = tfds.builder_from_directory(data_dir)
    ds = builder.as_dataset(split=split)

    n_ep = n_step = 0
    prompts = {}
    for ep in ds:
        n_ep += 1
        meta = ep["episode_metadata"]
        pt = meta["prompt_type"].numpy().decode()
        prompts[pt] = prompts.get(pt, 0) + 1
        assert not bool(meta["has_demonstration"].numpy()), "has_demonstration must be False"
        for step in ep["steps"]:
            n_step += 1
            for path, shape, dtype in CONSUMER_FIELDS:
                node = step
                for k in path:
                    assert k in node, "missing field %s" % "/".join(path)
                    node = node[k]
                got = tuple(node.shape)
                assert got == shape, "%s has shape %s, expected %s" % (
                    "/".join(path), got, shape)
                assert node.dtype == dtype, "%s has dtype %s, expected %s" % (
                    "/".join(path), node.dtype, dtype)
            lang = step["language_instruction"].numpy().decode()
            assert lang.strip(), "empty language_instruction"
            assert bool(step["is_first"].numpy()) and bool(step["is_last"].numpy()), \
                "single-step episode must be both first and last"
            # the converter does `circle_mask.squeeze(-1) > 127`; a mask that is
            # blank everywhere would silently produce an un-annotated frame
            assert int(np.asarray(step["sketch"]["circle"]).max()) > 127, "blank circle mask"
            assert int(np.asarray(step["sketch"]["arrow"]).max()) > 127, "blank arrow mask"

    print("episodes %d   steps %d   by prompt_type %s" % (n_ep, n_step, prompts))
    if expect_episodes is not None and n_ep != expect_episodes:
        raise SystemExit("expected %d episodes, read back %d" % (expect_episodes, n_ep))
    if len(set(prompts.values())) > 1:
        raise SystemExit("prompt types are unbalanced: %s" % prompts)
    print("VERIFY OK — every field convert_libero_data_to_lerobot.py reads is "
          "present with the right shape and dtype")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="spatial")
    ap.add_argument("--name", default=None,
                    help="dataset name; default sketch_libero_val_<suite>_anchored")
    ap.add_argument("--split", default="train",
                    help="RLDS split name; 'train' is what the converter loads")
    ap.add_argument("--verify", action="store_true",
                    help="only re-read and check an already written dataset")
    args = ap.parse_args()
    name = args.name or "sketch_libero_val_%s_anchored" % args.suite

    if args.verify:
        verify(name, args.split)
        return 0
    _, n = write(args.suite, name, args.split)
    print()
    verify(name, args.split, expect_episodes=n)
    return 0


if __name__ == "__main__":
    sys.exit(main())
