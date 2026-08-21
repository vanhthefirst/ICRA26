"""
Export the 114 validation scenes as a LIBERO-loadable task suite.

The scenes are not registered LIBERO benchmark tasks -- their BDDLs were
authored here -- so `task_suite.get_task_init_states(task_id)` has nothing to
look up for them and no `.pruned_init` ships with them. This writes the
compatibility layer: the BDDLs and the pinned initial states, in the folder
layout and file format LIBERO's own loader expects, so a rollout written
against the benchmark API can open these scenes without changes.

It is an EXPORT, not a migration. The scoring path for the reported numbers
stays `examples/libero/eval_sketchvla.py`, for two reasons that registration
cannot accommodate:

  * a LIBERO task carries ONE `task.language`, and every scene here is
    evaluated under two captions (explicit and ambiguous); the second caption
    has nowhere to live in the benchmark schema, so it is written into
    task_map.json instead of being silently dropped;
  * the stock loop reports success only, while target-selection rate -- the
    measurement the finding rests on -- needs `grasped_instance`, which the
    harness records and LIBERO's loop does not.

WHY EVERY TRIAL ROW IS THE SAME STATE
-------------------------------------
A stock `.pruned_init` holds N genuinely different starting layouts for one
task, and LIBERO's loop indexes row `episode_idx`. Here the pinned state IS the
experiment: `scene.bddl` fixes only a +-1.2 cm placement box with free yaw, and
the sketch was drawn against one specific draw from inside that box, captured
by scripts/capture_scene_init_states.py. Varying the layout across trials would
score the model against scenes no sketch was ever drawn for. So the exported
stack is the pinned state repeated `--trials` times: the file satisfies the
loader's shape contract, and the contrast stays paired.

THE PRE-RESET SEED IS PART OF THE STATE
---------------------------------------
`set_init_state` restores qpos/qvel and nothing else. Some Goal arenas draw
material and texture state during `reset()` from the global numpy stream, so a
consumer that resets without `np.random.seed(meta["seed"])` first reproduces
the correct physics and the wrong render -- measured at 158/255 max channel
diff on goal/scene_0000. That seed is not expressible in a `.pruned_init`, so
it is written per scene into task_map.json and repeated in the generated
registration stub. A consumer that ignores it gets a silent visual mismatch.

    python scripts/export_pruned_init.py
    python scripts/export_pruned_init.py --suites spatial --trials 14
    python scripts/export_pruned_init.py --verify        # needs the libero venv
"""

import argparse
import hashlib
import json
import os
import platform
import shutil
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SUITE_DIR = {
    "spatial": "validation_set_spatial",
    "object": "validation_set_object",
    "goal": "validation_set_goal",
}
PROBLEM_FOLDER = {s: "sketch_%s" % s for s in SUITE_DIR}
DEFAULT_TRIALS = 14
VERIFY_TOL = 8
# Measured: a software-rendered run of the real 114 peaks at 0.31% of the frame
# on goal/scene_0019, everything else at or below 0.09%. 0.5% passes rasteriser
# noise and still fails an object rendered in the wrong place or material, which
# would move whole percent of the frame.
MAX_FRAC = 0.005


def scene_dirs(out_root, suite):
    root = os.path.join(out_root, SUITE_DIR[suite])
    if not os.path.isdir(root):
        return root, []
    return root, sorted(d for d in os.listdir(root) if d.startswith("scene_"))


def flat_state(scene_root):
    npz = np.load(os.path.join(scene_root, "init_state.npz"))
    return np.concatenate([npz["time"], npz["qpos"], npz["qvel"]]).astype(np.float64)


def load_nonreproducible(out_root):
    path = os.path.join(out_root, "rollouts", "nonreproducible.json")
    if not os.path.exists(path):
        return set()
    with open(path) as f:
        entries = json.load(f)
    keys = set()
    for e in entries:
        if isinstance(e, str):
            keys.add(e)
        elif isinstance(e, dict):
            keys.add("%s/%s" % (e.get("suite"), e.get("dir")))
    return keys


def write_pruned_init(path, flat, trials):
    import torch

    stack = np.repeat(flat[None, :], trials, axis=0)
    torch.save(torch.from_numpy(stack), path)
    with open(path, "rb") as f:
        digest = hashlib.sha256(f.read()).hexdigest()
    return stack.shape, digest


def provenance():
    """Record what produced these files.

    A handover that cannot be pinned to a stack is a handover that gets blamed
    for the next mismatch. robosuite and mujoco versions in particular change
    what a render looks like, so a diff run against a different pair is not
    evidence about the export.
    """
    out = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "mujoco_gl": os.environ.get("MUJOCO_GL", "(unset)"),
    }
    for name in ("torch", "robosuite", "mujoco"):
        try:
            out[name] = __import__(name).__version__
        except Exception:
            out[name] = "(not importable)"
    return out


def captions(meta):
    explicit = meta.get("instruction_explicit") or meta.get("instruction")
    ambiguous = meta.get("instruction_ambiguous")
    return explicit, ambiguous


def export(args):
    try:
        import torch  # noqa: F401
    except ImportError:
        raise SystemExit(
            "[error] torch is required: LIBERO reads .pruned_init with torch.load, "
            "so the file has to be written with torch.save to be loadable at all.")

    out_root = os.path.join(REPO, "outputs")
    dest = os.path.join(out_root, args.dest)
    skip = load_nonreproducible(out_root)
    task_map = {}
    manifest = {"trials_per_task": args.trials, "provenance": provenance(),
                "suites": {}, "skipped": []}

    for suite in args.suites:
        root, dirs = scene_dirs(out_root, suite)
        if not dirs:
            print("[warn] no scenes under %s -- skipping suite" % root)
            continue

        pf = PROBLEM_FOLDER[suite]
        bddl_dir = os.path.join(dest, "bddl_files", pf)
        init_dir = os.path.join(dest, "init_files", pf)
        os.makedirs(bddl_dir, exist_ok=True)
        os.makedirs(init_dir, exist_ok=True)

        tasks = []
        for d in dirs:
            key = "%s/%s" % (suite, d)
            if key in skip:
                manifest["skipped"].append({"scene": key, "reason": "nonreproducible"})
                continue

            scene_root = os.path.join(root, d)
            init_path = os.path.join(scene_root, "init_state.npz")
            bddl_src = os.path.join(scene_root, "scene.bddl")
            if not (os.path.exists(init_path) and os.path.exists(bddl_src)):
                manifest["skipped"].append({"scene": key, "reason": "missing init_state.npz or scene.bddl"})
                continue

            with open(os.path.join(scene_root, "meta.json")) as f:
                meta = json.load(f)

            flat = flat_state(scene_root)
            init_file = "%s_demo.pruned_init" % d
            shape, digest = write_pruned_init(
                os.path.join(init_dir, init_file), flat, args.trials)
            shutil.copyfile(bddl_src, os.path.join(bddl_dir, "%s.bddl" % d))

            explicit, ambiguous = captions(meta)
            tasks.append({
                "name": d,
                "problem_folder": pf,
                "bddl_file": "%s.bddl" % d,
                "init_states_file": init_file,
                "language": explicit,
                "language_ambiguous": ambiguous,
                "reset_seed": meta["seed"],
                "target": meta.get("target"),
                "destination": meta.get("destination"),
                "state_dim": int(flat.shape[0]),
                "init_rows": int(shape[0]),
                "sha256": digest,
            })

        task_map[pf] = tasks
        manifest["suites"][suite] = {"problem_folder": pf, "n_tasks": len(tasks)}
        print("[ok] %-8s %3d tasks -> %s" % (suite, len(tasks), pf))

    with open(os.path.join(dest, "task_map.json"), "w") as f:
        json.dump(task_map, f, indent=2)
    with open(os.path.join(dest, "MANIFEST.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    write_registration_stub(dest)
    write_readme(dest, manifest)

    total = sum(v["n_tasks"] for v in manifest["suites"].values())
    print("[ok] %d tasks, %d trial rows each, written under %s"
          % (total, args.trials, dest))
    if manifest["skipped"]:
        print("[note] %d scene(s) skipped -- see MANIFEST.json" % len(manifest["skipped"]))
    return dest, task_map


STUB = '''"""Register the exported sketch scenes as LIBERO benchmark suites.

Import this once before `benchmark.get_benchmark_dict()`. It reads
task_map.json from the directory this file sits in, so the two stay in step.

Point LIBERO at the exported trees first, either by editing ~/.libero/config.yaml
or by copying bddl_files/<pf> and init_files/<pf> into the paths
`get_libero_path("bddl_files")` and `get_libero_path("init_states")` return.

The BDDL alone does not reproduce a scene. `set_init_state` restores qpos/qvel
only, and some Goal arenas draw material state during reset() from the global
numpy stream, so seed with the task's `reset_seed` BEFORE calling reset():

    np.random.seed(TASK_SEEDS["sketch_goal"]["scene_0000"])
    env.reset()
    env.set_init_state(initial_states[episode_idx])

Skipping that reproduces correct physics and the wrong render (measured at
158/255 max channel diff on goal/scene_0000).

Each task carries one caption, as LIBERO's schema allows. The ambiguous caption
for the same scene is in task_map.json under `language_ambiguous`.

The three imports below are LIBERO's registration API and have moved between
releases. Confirm them against the installed copy before trusting this file:

    python -c "from libero.libero import benchmark; print(benchmark.__file__)"
"""

import json
import os

from libero.libero.benchmark import Benchmark
from libero.libero.benchmark import register_benchmark
from libero.libero.benchmark.libero_suite_task_map import libero_task_map

_HERE = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(_HERE, "task_map.json")) as _f:
    SKETCH_TASK_MAP = json.load(_f)

TASK_SEEDS = {
    pf: {t["name"]: t["reset_seed"] for t in tasks}
    for pf, tasks in SKETCH_TASK_MAP.items()
}


def _install():
    for pf, tasks in SKETCH_TASK_MAP.items():
        libero_task_map[pf] = [t["name"] for t in tasks]

        def _make(problem_folder):
            @register_benchmark
            class _SketchSuite(Benchmark):
                name = problem_folder

                def __init__(self, task_order_index=0):
                    super().__init__(task_order_index=task_order_index)

            _SketchSuite.__name__ = problem_folder.upper()
            return _SketchSuite

        _make(pf)


_install()
'''


def write_registration_stub(dest):
    with open(os.path.join(dest, "register_sketch_suites.py"), "w") as f:
        f.write(STUB)


README = """# Sketch validation scenes, in LIBERO's task-suite layout

114 scenes: 38 each in `sketch_spatial`, `sketch_object`, `sketch_goal`.
Generated by `scripts/export_pruned_init.py` in the harness repo.

## Loading a scene

```python
import numpy as np, torch
from libero.libero.envs import OffScreenRenderEnv

states = torch.load("init_files/sketch_spatial/scene_0000_demo.pruned_init")

np.random.seed(SEED)                      # <-- REQUIRED, see below
env = OffScreenRenderEnv(bddl_file_name="bddl_files/sketch_spatial/scene_0000.bddl",
                         camera_heights=256, camera_widths=256)
env.reset()
obs = env.set_init_state(np.asarray(states)[trial_idx])
```

`SEED` is that scene's `reset_seed` in `task_map.json`.

## Two things that will bite

**The seed is part of the state.** `set_init_state` restores qpos/qvel and
nothing else. Some Goal arenas draw material and texture state during `reset()`
from the global numpy stream, so resetting without seeding first reproduces
correct physics and the WRONG render -- measured at 158/255 max channel diff on
goal/scene_0000. The seed cannot be stored inside a `.pruned_init`, which is
why it lives in `task_map.json`.

**Every row is the same state.** A stock `.pruned_init` holds N different
starting layouts. These hold one pinned layout repeated %(trials)d times, because
the sketch for each scene was drawn against that one layout; varying it would
score against a scene no sketch was drawn for. Indexing any row is safe.

## Captions

LIBERO gives a task one `task.language`, so each task here carries the EXPLICIT
caption. The ambiguous caption for the same scene is in `task_map.json` under
`language_ambiguous`. Both are needed to reproduce the 2x2 evaluation.

## Files

- `task_map.json` -- per scene: bddl file, init file, both captions, reset seed,
  target and destination labels, state dim, sha256 of the init file
- `MANIFEST.json` -- counts, skipped scenes, and the stack that built this
- `verify_report.json` -- per scene render diff against the scene's own
  frame0.png, written by `--verify`
- `register_sketch_suites.py` -- optional, registers these as LIBERO benchmark
  suites; its LIBERO imports have moved between releases, so check them first

## Provenance

%(prov)s

Renders differ slightly between rasterisers. A diff run under a different
`MUJOCO_GL`, robosuite or mujoco than the line above is not evidence about
these files.
"""


def write_readme(dest, manifest):
    prov = "\n".join("- %s: %s" % (k, v) for k, v in sorted(manifest["provenance"].items()))
    with open(os.path.join(dest, "README.md"), "w") as f:
        f.write(README % {"trials": manifest["trials_per_task"], "prov": prov})


def score(got, ref, tol):
    """Compare a render against frame0.png at both orientations.

    LIBERO's agentview render and frame0's stored orientation differ by a
    vertical flip on some installs, so both are scored and the better is kept.

    `max` alone cannot separate the two failures that matter. A single
    anti-aliased edge pixel and a whole object rendered in the wrong material
    both peak in the hundreds; what tells them apart is HOW MANY pixels moved.
    n_px and frac carry that, and a report without them invites reading a
    3-pixel seam as a broken scene, or the reverse.
    """
    best = None
    for flip, cand in ((False, got), (True, got[::-1])):
        d = np.abs(cand.astype(np.int16) - ref.astype(np.int16))
        per_px = d.max(axis=2)
        stats = {
            "max": int(per_px.max()),
            "mean": float(d.mean()),
            "n_px": int((per_px > tol).sum()),
            "frac": float((per_px > tol).mean()),
            "flipped": flip,
        }
        if best is None or stats["n_px"] < best[0]["n_px"]:
            best = (stats, per_px, cand)
    return best


def dump_panel(path, ref, got, per_px):
    from PIL import Image

    heat = np.clip(per_px.astype(np.int16) * 3, 0, 255).astype(np.uint8)
    heat = np.stack([heat, np.zeros_like(heat), np.zeros_like(heat)], axis=2)
    panel = np.concatenate([ref, got, heat], axis=1)
    Image.fromarray(panel).save(path)


def verify(dest, task_map, tol, dump, max_frac):
    """Round-trip every exported state through the simulator and diff the render
    against the scene's own frame0.png.

    An export that loads without error but restores a different layout is the
    failure this whole file exists to prevent, and it is invisible in the file
    itself.
    """
    import torch
    from libero.libero.envs import OffScreenRenderEnv
    from PIL import Image

    out_root = os.path.join(REPO, "outputs")
    panel_dir = os.path.join(dest, "verify_diffs")
    if dump:
        os.makedirs(panel_dir, exist_ok=True)

    rows = []
    for suite, dirname in SUITE_DIR.items():
        pf = PROBLEM_FOLDER[suite]
        for task in task_map.get(pf, []):
            scene_root = os.path.join(out_root, dirname, task["name"])
            ref = np.asarray(Image.open(os.path.join(scene_root, "frame0.png")).convert("RGB"))
            size = ref.shape[0]

            stack = torch.load(os.path.join(dest, "init_files", pf, task["init_states_file"]))
            flat = np.asarray(stack)[0]

            np.random.seed(task["reset_seed"])
            env = OffScreenRenderEnv(
                bddl_file_name=os.path.join(dest, "bddl_files", pf, task["bddl_file"]),
                camera_heights=size, camera_widths=size, camera_names=["agentview"])
            env.reset()
            obs = env.set_init_state(flat)
            env.close()

            stats, per_px, got = score(np.asarray(obs["agentview_image"]), ref, tol)
            stats.update(suite=suite, scene=task["name"])
            rows.append(stats)

            ok = stats["n_px"] == 0
            print("[%s] %s/%s max %3d  mean %5.2f  n_px %5d (%.2f%%)"
                  % ("ok" if ok else "DIFF", suite, task["name"], stats["max"],
                     stats["mean"], stats["n_px"], 100.0 * stats["frac"]))

            if dump and not ok:
                dump_panel(os.path.join(panel_dir, "%s_%s.png" % (suite, task["name"])),
                           ref, got, per_px)

    rows.sort(key=lambda r: -r["n_px"])
    differing = [r for r in rows if r["n_px"] > 0]
    # Pass/fail is on the AREA that moved, not on any single pixel. A rasteriser
    # change puts a handful of pixels on object outlines and peaks in the
    # hundreds; only a changed layout or material moves a meaningful fraction of
    # the frame. Failing on max would fail every software-rendered run.
    failed = [r for r in rows if r["frac"] > max_frac]
    with open(os.path.join(dest, "verify_report.json"), "w") as f:
        json.dump({"tol": tol, "max_frac": max_frac, "provenance": provenance(),
                   "n_scenes": len(rows), "n_differing": len(differing),
                   "n_failed": len(failed), "rows": rows}, f, indent=2)

    print("[verify] %d/%d scene(s) show any difference; %d over max_frac=%.3f%%"
          % (len(differing), len(rows), len(failed), 100.0 * max_frac))
    for r in differing[:5]:
        print("         %s/%s  %d px (%.2f%%)  max %d"
              % (r["suite"], r["scene"], r["n_px"], 100.0 * r["frac"], r["max"]))
    if dump and differing:
        print("[verify] panels (reference | rendered | diff) in %s" % panel_dir)
    return 1 if failed else 0


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--suites", nargs="+", default=sorted(SUITE_DIR), choices=sorted(SUITE_DIR))
    p.add_argument("--trials", type=int, default=DEFAULT_TRIALS,
                   help="rows in each .pruned_init; all identical, see module docstring")
    p.add_argument("--dest", default="libero_export", help="relative to outputs/")
    p.add_argument("--verify", action="store_true",
                   help="re-open every exported scene and diff against frame0.png (libero venv)")
    p.add_argument("--tol", type=int, default=VERIFY_TOL,
                   help="per-pixel channel difference below which a pixel counts as unchanged")
    p.add_argument("--dump-mismatch", action="store_true",
                   help="write a reference|rendered|diff panel for every differing scene")
    p.add_argument("--max-frac", type=float, default=MAX_FRAC,
                   help="fraction of the frame that may differ before a scene fails")
    args = p.parse_args()

    dest, task_map = export(args)
    if args.verify:
        sys.exit(verify(dest, task_map, args.tol, args.dump_mismatch, args.max_frac))


if __name__ == "__main__":
    main()
