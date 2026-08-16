# Runbook — the two prompt-arm baselines

Aaron — 16 August 2026

The dataset now carries two captions per scene: explicit and ambiguous
(`PROMPT_TAXONOMY.md`). This runbook measures π₀.₅-LIBERO on both, so the report
has three baselines side by side.

| # | baseline | trials | state |
|---|---|---|---|
| 1 | Reproduction of the published π₀.₅-LIBERO numbers | 500/suite, 4 suites | **done** — `outputs/rollouts/openpi_repro_500/` |
| 2 | Explicit prompts, my 114 scenes | 532/suite, 3 suites | this runbook |
| 3 | Ambiguous prompts, the same 114 scenes | 532/suite, 3 suites | this runbook |

Baseline 1 is not re-run. It is Table 1 of the report and nothing here changes it.

**Trial count.** 14 rollouts x 38 scenes = **532 per suite**, 1,596 per arm.
500 does not divide by 38, and I would rather cover every scene equally than hit
a round number — so this is 500 rounded up to the nearest whole rollout per
scene, not 500 exactly.

**Time.** ~4 hours per arm at the baseline's measured 8.8 s/rollout on one
RTX 4090, so budget **8 hours** plus setup. Both arms resume, so this does not
have to be one booking.

---

## Before you book the GPU

Two things must be true on `main`, and both are cheap to check on the laptop:

```powershell
cd C:\Users\Admin\sketch_prompted_vla
python scripts\build_prompt_variants.py --check    # must print 228 rows, exit 0
python scripts\audit_validation_sets.py            # must exit 0
git status                                         # must be clean
git push
```

The pod clones from GitHub. Anything uncommitted stays on the laptop, and an arm
run against a scene set without the caption keys dies at scene_0000.

---

## Part A — pod and tooling

Identical to `RUNBOOK_REPRO_500.md` parts A–C, so I am not repeating it:

1. SSH key registered with RunPod (part A).
2. Pod: RTX 4090 or A40, **EU-RO-1**, network volume attached at `/workspace`
   (part B). Check `nvidia-smi`, `df -h /workspace`, `ls /workspace/aaron`.
3. `bash /workspace/aaron/sketch_prompted_vla/scripts/pod_bootstrap.sh`, then
   `source ~/.bashrc` (part C).

Then pull this work:

```bash
cd /workspace/aaron/sketch_prompted_vla
git pull
python scripts/build_prompt_variants.py --check
```

---

## Part B — two panes

```bash
tmux new -s baselines
```

**This is an all-day run. tmux is not optional.**

**Pane 1 — the policy server**, up for the whole session:

```bash
cd /workspace/aaron/openpi
uv run scripts/serve_policy.py --env LIBERO
```

Wait for it to report it is listening. First start takes a few minutes while the
checkpoint loads.

**Pane 2 — the LIBERO side:**

```bash
source /workspace/aaron/openpi/examples/libero/.venv/bin/activate
export PYTHONPATH=$PYTHONPATH:/workspace/aaron/openpi/third_party/libero
export MUJOCO_GL=egl
cd /workspace/aaron/sketch_prompted_vla
```

`MUJOCO_GL=egl` is required — a pod has no X server.

---

## Part C — smoke both arms before committing to eight hours

Two scenes, one rollout each, per arm. The point is to read the caption the model
actually received, not to get a number:

```bash
for arm in explicit ambiguous; do
  python scripts/rollout_sketch.py --policy pi05 --conditions text_only \
      --prompt-type $arm --run-id smoke_$arm --n-rollouts 1 \
      --scenes spatial/scene_0000,object/scene_0000 --max-steps 320
done
```

**The gate:** the policy echoes `prompt -> '...'` whenever the string it sends
the server changes. For the ambiguous arm every line must read
`move this onto that` or `move this into that` and must contain no object name.
If it prints `pick up the black bowl ...`, the arm is running on explicit
captions and the whole ambiguous baseline would be a duplicate of the explicit
one. Stop and fix it before Part D.

The harness raises rather than falling back when the caption keys are missing, so
the likely cause of a wrong string is a stale `--prompt-type`, not stale data.

---

## Part D — the two arms

```bash
bash scripts/run_baselines.sh
```

That is the whole run. It does, in order:

1. `build_prompt_variants.py --check` — refuses to start on stale captions.
2. Probes the policy server before each arm.
3. `--run-id pi05_explicit_532`, 114 scenes x 14 rollouts, `--resume`.
4. Probes again, then `--run-id pi05_ambiguous_532`, same shape.
5. Refuses to call an arm done unless its `results.csv` holds 1,596 unskipped
   rows, because a clean exit with missing rows means scenes were skipped.

A finished arm is skipped on re-run, so after a dropped connection just run the
script again.

`ARMS=ambiguous bash scripts/run_baselines.sh` runs one arm only, if the
booking is short.

**Time the first suite.** The log prints a line per scene; if the first 38 scenes
take much more than an hour, stop and work out why before paying for the rest.

---

## Part E — analysis and report

```bash
python scripts/analyze_baselines.py \
    --arms explicit=pi05_explicit_532 ambiguous=pi05_ambiguous_532 \
    --tables report/prompt_baselines/tables.tex \
    --figdir report/prompt_baselines/figures

cd report/prompt_baselines && latexmk -pdf report.tex
```

The report reads its numbers out of `tables.tex`, which that command writes. The
`.tex` body is not edited by hand for a number — if a table shows `---`, the arm
has no rows yet.

The analysis also writes `analysis.json` next to each arm's `results.csv` and
refreshes `figures/fig_tier_breakdown.png` with both arms side by side.

**Sanity checks on the output, before it goes to the supervisor:**

- Explicit-arm overall success should land near the 3-rollout baseline's 34.5%.
  A large move means something other than the trial count changed.
- `control` x explicit should be the highest cell in both tables.
- Ambiguous should be at or below explicit in every tier. If ambiguous *beats*
  explicit somewhere, check the smoke gate in Part C — that is what a
  mislabelled arm looks like.

---

## Part F — before terminating the pod

```bash
cd /workspace/aaron/sketch_prompted_vla
git add -A && git commit -m "run: explicit and ambiguous prompt baselines, 532 trials/suite" && git push
```

`results.csv`, `run_config.json`, `summary.json` and `analysis.json` for both
arms, plus the regenerated tables, figure and PDF. Videos are large and stay on
the pod unless you want them.

---

## What each baseline is for, in one line each

- **Baseline 1** certifies the observation pipeline against published numbers.
  Without it, "my scenes are hard" and "my pipeline is subtly wrong" look the same.
- **Baseline 2** is the sketch-free reference on my scenes with a caption that
  names things. Its `control` tier isolates distribution shift from ambiguity.
- **Baseline 3** removes the names and keeps everything else identical. It is the
  condition a sketch is meant to rescue, and it is paired with baseline 2 scene by
  scene and rollout by rollout.
