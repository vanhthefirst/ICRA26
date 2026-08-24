# Runbook — rebuilding the Spatial set anchored to stock LIBERO layouts

Aaron — 24 August 2026

Replaces the from-scratch Spatial scenes with layout-anchored ones, then
re-measures the two prompt arms on them. Rationale and the design decisions are
in `claude/eval_layout_anchoring.md`; the builder is
`scripts/build_validation_set_spatial_anchored.py`.

Scope is **Spatial only**. Object and Goal are untouched — Goal is already
anchored, Object is out of scope for this pass.

Budget: about **1 hour** to build and pin the scenes, then about **3 hours** of
rollouts (532 trials per arm, two arms). One short booking, not the eight-hour
day `RUNBOOK_BASELINES.md` assumes, because only one suite is being measured.

---

## Part A — on the laptop, before booking anything

The pod clones from GitHub, so anything uncommitted stays behind.

```powershell
cd C:\Users\Admin\sketch_prompted_vla
git add scripts/build_validation_set_spatial_anchored.py RUNBOOK_ANCHORED_SPATIAL.md
git commit -m "feat: layout-anchored Spatial builder — inject duplicates into the shipped BDDLs"
git push
```

---

## Part B — the pod

Same pod shape as `RUNBOOK_REPRO_500.md` parts A–C. What matters here:

- **GPU: RTX 4090 (24 GB).** This is inference and offscreen rendering only.
  The measured cost is 8.8 s/rollout on a 4090, and the model needs a little
  over 8 GB. An A40 or L40S is a fine substitute at similar or better
  availability; an H100 costs several times more and buys nothing.
- **Region: must match the network volume's datacenter** (EU-RO-1). A volume
  only attaches to GPUs in its own region, so the region is chosen for you.
- **Template:** an official RunPod PyTorch image (Ubuntu, ships `bash`). Leave
  the container start command at the default.
- **Network volume** attached at `/workspace`.

On connecting, check the two things that would silently ruin the run:

```bash
nvidia-smi                # the card, ~24 GB
df -h /workspace          # the volume actually mounted
```

Then bootstrap and pull:

```bash
bash /workspace/aaron/sketch_prompted_vla/scripts/pod_bootstrap.sh
source ~/.bashrc
cd /workspace/aaron/sketch_prompted_vla && git pull
```

---

## Part C — the environment

`CLAUDE.md` says `conda activate libero` for the scripts that need a simulator.
That is the laptop convention. **On the pod there is no conda** — the same
dependencies live in the LIBERO venv that openpi ships, so every simulator step
below runs inside this:

```bash
export OPENPI=/workspace/aaron/openpi
export REPO=/workspace/aaron/sketch_prompted_vla

source $OPENPI/examples/libero/.venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$OPENPI/third_party/libero
export MUJOCO_GL=egl
export LIBERO_SPATIAL_BDDL=$OPENPI/third_party/libero/libero/libero/bddl_files/libero_spatial
cd $REPO
```

`MUJOCO_GL=egl` is required — the pod has no X server.
`LIBERO_SPATIAL_BDDL` is where the builder reads the stock scenes from; check it
exists before running, because a wrong path fails 38 times in a row:

```bash
ls $LIBERO_SPATIAL_BDDL | wc -l        # expect 11 (10 bddl + tasks_info.txt)
```

Work inside `tmux` — the build alone is long enough to lose to a dropped
browser.

---

## Part D — build the scenes

The builder writes to `outputs/validation_set_spatial/`, the name every
downstream script hardcodes. Move the old set aside rather than deleting it; the
comparison against it is worth keeping.

```bash
mv outputs/validation_set_spatial outputs/validation_set_spatial_fromscratch
mkdir -p outputs/validation_set_spatial
```

**Smoke first** — four scenes, one per tier. Open `SMOKE = True` at the top of
the script, or run the four indices directly:

```bash
ONLY_SCENES=0,5,17,26 python scripts/build_validation_set_spatial_anchored.py
```

Read the log before going further. Every line should end `-> ok`. Then look at
one scene's sketch and confirm the circle sits on the bowl the caption means and
does not touch `akita_black_bowl_2`.

**Full run:**

```bash
python scripts/build_validation_set_spatial_anchored.py 2>&1 \
    | tee outputs/validation_set_spatial/build_log.txt
```

38 scenes. Offline the placement sampler found every layout on its first
attempt, so rejections here should be rare and would mean a gate is failing in
simulation, not that the sampler is struggling. Watch for two in particular:

- `circleenclosed_akita_black_bowl_2_...` — the stock second bowl falls inside
  the drawn circle. Measured offline this should not happen (the tightest base
  task leaves ~26 px against a ~15 px circle), so if it fires, tell me which
  base task and I will look again at that layout rather than loosening the gate.
- `oracle_false` — the target cannot be rested on its destination plate. On an
  injected plate that means the copy landed somewhere unusable; on `plate_1` it
  means something is wrong with the base task.

Then the two checks that do not need the simulator:

```bash
python scripts/normalize_validation_schema.py
python scripts/audit_validation_sets.py            # must exit 0
```

---

## Part E — pin the initial states

Without this every rollout resets into a fresh draw inside the placement box and
scores against a layout no sketch was drawn for.

```bash
python scripts/capture_scene_init_states.py --all
```

Expect all 38 Spatial scenes at rung `resample`, attempt 1, residual 0.0 px —
that is what the previous 114 did. Anything landing at `solve`, and especially
anything in `nonreproducible.json`, is worth pausing over.

`--all` re-captures Object and Goal as well; there is no suite filter. That is
slower but harmless, and it doubles as a free control: those two sets did not
change, so they must re-pin identically. If one of them suddenly lands at
`solve`, the cause is something in the environment, not this rebuild.

One knock-on: `outputs/human_study/scene_subset.json` names Spatial scene
directories that now hold different scenes. The subset needs regenerating with
`build_human_study_bundle.py` before any human sketching happens. Nothing in
this runbook depends on it, and only one test response exists so far, so it is
not urgent.

---

## Part F — the two prompt arms

Captions are unchanged, so the ambiguous bank still keys these scenes to the
`two_clause_On` bucket:

```bash
python scripts/build_prompt_variants.py --check
```

Two panes.

**Pane 1 — the policy server**, up for the whole session:

```bash
cd $OPENPI && uv run scripts/serve_policy.py --env LIBERO
```

**Pane 2 — smoke both arms before committing to the rollouts.** The point is to
read the caption the model actually received:

```bash
for arm in explicit ambiguous; do
  python scripts/rollout_sketch.py --policy pi05 --conditions text_only \
      --prompt-type $arm --run-id smoke_anchored_$arm --n-rollouts 1 \
      --scenes spatial/scene_0000,spatial/scene_0017 --max-steps 320
done
```

The gate: on the ambiguous arm, every `prompt -> '...'` line must name no
object — only pointing words. If it prints `pick up the black bowl ...`, the arm
is running on explicit captions and the ambiguous baseline would be a duplicate.
Stop and fix before the real run.

**The arms.** `run_baselines.sh` runs all three suites; only Spatial changed, so
drive the harness directly and give the runs their own ids. `--scenes` takes
explicit `suite/dir` pairs and rejects a bare suite name, so build the list from
the manifest the build just wrote — that way the roster cannot drift from what
was actually built:

```bash
SPATIAL=$(python -c "import json;print(','.join('spatial/'+e['dir'] for e in json.load(open('outputs/validation_set_spatial/manifest.json'))))")
echo "$SPATIAL" | tr ',' '\n' | wc -l        # expect 38

for arm in explicit ambiguous; do
  python scripts/rollout_sketch.py --policy pi05 --conditions text_only \
      --prompt-type $arm --run-id pi05_anchored_${arm}_532 \
      --n-rollouts 14 --scenes "$SPATIAL" --resume
done
```

532 trials per arm, ~1.3 h each at 8.8 s/rollout. `--resume` means a dropped
connection costs nothing but the re-run.

Time the first ten scenes. If they take much more than twenty minutes, stop and
find out why before paying for the rest.

---

## Part G — read the result

```bash
python scripts/analyze_baselines.py \
    --arms explicit=pi05_anchored_explicit_532 ambiguous=pi05_anchored_ambiguous_532 \
    --tables report/prompt_baselines/tables.tex \
    --figdir report/prompt_baselines/figures
```

The three numbers that matter, in order:

1. **98.0** — spatial in `outputs/rollouts/openpi_repro_500/`. Same base
   layouts, stock caption. This is the ceiling and it is already measured.
2. **control x explicit** on the new set. Same layouts, category-only caption.
   The gap from 98.0 is what the stock spatial phrase was buying, and it is no
   longer contaminated by distribution shift, which is the whole point of the
   rebuild.
3. **The other tiers**, and later the sketch arms against them.

Sanity: ambiguous should sit at or below explicit in every tier. If ambiguous
beats explicit anywhere, that is what a mislabelled arm looks like — go back to
the Part F smoke gate.

Control on this suite is **not** a language-only ceiling: two identical bowls are
present and the caption names a category, so a policy that understands the
sentence perfectly still has to choose. Number 1 above is the ceiling. This is
recorded here because it is the one thing about the design that will look like a
mistake to a reader who has not read `claude/eval_layout_anchoring.md`.

---

## Part H — before terminating the pod

```bash
cd $REPO
git add -A
git commit -m "run: anchored Spatial set, explicit and ambiguous arms at 532 trials"
git push
```

The scene folders, `init_state.npz` for each, both `results.csv`, and the
regenerated tables.

---

## Still open: the RLDS export

I said earlier that `convert_libero_data_to_lerobot.py` produces the RLDS. It
does not — it reads RLDS through `tfds` and writes LeRobot, which is the
training direction. Nothing in either repo currently writes RLDS out.

So the RLDS deliverable is one of two things, and it is a decision rather than a
command:

- Duc Tai regenerates `ductaingn/sketch_libero_val` from the new BDDLs with the
  same generator he used for the 114, or
- I write an exporter that walks the finished scene folders and emits the same
  schema.

Either way it comes after Part E, since it needs the pinned states, and it does
not block the rollouts.
