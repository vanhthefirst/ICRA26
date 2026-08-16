# Sketch-Prompted VLA

Circle-and-arrow visual instructions for disambiguated robot manipulation, built
on [UniVLA](https://github.com/OpenDriveLab/UniVLA). Full proposal:
[`sketch_prompted_vla_proposal.md`](sketch_prompted_vla_proposal.md).

A human draws a **circle** (which object) and an **arrow** (which destination) on
the robot's camera view to resolve ambiguity that language alone cannot. The
validation scenes are *deliberately impossible* to disambiguate from a vague
caption — many near-identical objects and/or multiple plausible destinations — so
that circle+arrow is the only disambiguating signal.

## Status

**Three LIBERO validation suites are built, hardened, and normalised to schema
v1.0 — 114 scenes total.**

| Suite | Output | Scenes | Builder |
|---|---|---|---|
| Spatial | `outputs/validation_set_spatial/` | 38 | `scripts/build_validation_set_spatial.py` |
| Object | `outputs/validation_set_object/` | 38 | `scripts/build_validation_set_object.py` |
| Goal | `outputs/validation_set_goal/` | 38 | `scripts/build_validation_set_goal.py` |

Every scene passes the full gate stack (settled, in-frame, visibility ≥ 0.35,
positive oracle, **negative oracles that gate**, pixel-separation resolvability,
graspable) and was independently re-verified rather than trusted from the build
log. Tier split per suite: control 5 / referential 12 / directional 9 / both 12.

Combined manifest: `outputs/validation_manifest_all.json` (114 rows).

**Long / `libero_10` is postponed** — its goals need 2–3 predicates, and a single
circle+arrow cannot express two actions.

All three suites are packaged identically: 38 scene folders plus `DATASHEET.md`,
`contact_sheet.png`, `manifest.json`, `manifest_canonical.json` and
`build_log.txt`. Verify with `python scripts/audit_validation_sets.py`.

**Open:** the headline text-only vs text+sketch experiment needs a trained policy
on a GPU (not runnable on a 4GB laptop). Also pending: a real-human sketch
agreement check, and UniVLA/RLDS export. When scoring the Goal suite, stratify on
`meta['grasp']['grasp_success']` — see its `DATASHEET.md`.

## Method

Author a real LIBERO BDDL **as text** (direct string templating), add duplicate
object instances to create the ambiguity, load with `OffScreenRenderEnv`, settle
the physics, render, project ground-truth positions to pixels, auto-draw the
circle and arrow, validate against every gate, then package. Each scene ships its
BDDL so a GPU rollout can score it later.

Always work vertical-slice → smoke (~3–4 scenes) → full run. This has caught a
real bug at every stage.

Scenes are **synthesised** from the BDDL templates that ship with the LIBERO
package — no demonstration HDF5s are needed to build the validation suites.

## Layout

- `scripts/` — the three builders, the schema normaliser, and the two probes
  whose transcripts are cited as evidence in `SUITE_FACTS.md`. The builders,
  the state capture and the rollout harness need a conda env with
  `robosuite` / `mujoco` / `libero` installed; everything else is stdlib and
  runs anywhere.
- `outputs/` — the three validation suites, the combined manifest, and the probe
  transcripts.
- `report/` — `section2_report.pdf` / `.tex`, the write-up of the earlier
  proof-of-concept phase, kept as the record of that work.
- `UniVLA/` — baseline latent-action VLA codebase. Declared as a git submodule but
  **deliberately not checked out** — nothing in the dataset pipeline imports it, so
  it is left uninitialised to keep the working tree small. See
  [Restoring UniVLA](#restoring-univla) to bring it back.

There is **no `data/` directory** — the validation suites are synthesised from the
BDDL templates shipped with the LIBERO package and need no demonstration HDF5s at
all. The fetch script for the LIBERO-Spatial demos was removed along with it; if
the demo-replay / training-export direction resumes, note that only Spatial demos
were ever downloaded, so a three-suite export would need Object and Goal fetched
too.

Read `SUITE_FACTS.md` and `SCHEMA.md` before touching a builder — they hold
hard-won facts that were expensive to derive.

## Documents

| File | What it is |
|---|---|
| `SCHEMA.md` | The canonical schema v1.0 contract every suite conforms to |
| `SUITE_FACTS.md` | Hard-won LIBERO scene-synthesis facts; do not re-derive |
| `sketch_prompted_vla_proposal.md` | The full research proposal |

## Setup

```bash
git clone https://github.com/vanhthefirst/ICRA26.git
cd ICRA26
```

Clone plainly — **no `--recurse-submodules`**. The only submodule is `UniVLA`,
which the dataset pipeline does not need; see [Restoring UniVLA](#restoring-univla).

Then, in the libero env (I develop on Windows through WSL2, but nothing in
the code depends on that):

```bash
conda activate libero
cd /path/to/sketch_prompted_vla
python scripts/build_validation_set_spatial.py    # or _object_ / _goal_
python scripts/normalize_validation_schema.py         # refresh canonical manifests
```

## Restoring UniVLA

`UniVLA/` is the baseline latent-action VLA this project builds on. Its working
tree is **not checked out** — nothing in the scene-synthesis pipeline imports it,
so carrying 23 MB of unused model code was pure weight. It becomes relevant only
for the headline text-only vs text+sketch rollout experiment, which needs a GPU.

Nothing was lost by removing it. The repo still declares the dependency in two
places, and those two facts are the complete restore recipe:

| where | what it records |
|---|---|
| `.gitmodules` | the upstream URL, `https://github.com/OpenDriveLab/UniVLA.git` |
| the gitlink at path `UniVLA` | the exact pinned commit, `0ab9e9dd3074ffa14fdd1a00aa2aa84e4f8e1352` |

To bring it back, from the repo root:

```bash
git submodule update --init UniVLA
```

That restores the files at the pinned commit — **not** upstream's latest. This
matters: it is the same tree this project was developed against, so the baseline
stays reproducible however far upstream has since moved. Confirm with:

```bash
git submodule status UniVLA
# 0ab9e9dd3074ffa14fdd1a00aa2aa84e4f8e1352 UniVLA (heads/main)
```

A leading `-` on that line means it is still uninitialised; a leading `+` means the
checkout has drifted off the pin.

To deliberately move to a newer upstream UniVLA, initialise it first, then:

```bash
cd UniVLA && git fetch origin && git checkout <new-sha> && cd ..
git add UniVLA && git commit -m "chore: bump UniVLA pin to <new-sha>"
```

To put it away again after use:

```bash
git submodule deinit -f UniVLA
```

This empties the directory but leaves `.gitmodules` and the gitlink untouched, so
the recipe above keeps working. Note that `deinit` refuses if `UniVLA/.git` is a
real directory rather than a gitfile (the state a manual `git clone` into the path
leaves behind) — in that case `rm -rf UniVLA && mkdir UniVLA` first, but only after
checking `git -C UniVLA status` is clean, since that directory would then be the
only copy of any unpushed work.
