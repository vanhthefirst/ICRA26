# Runbook — stage 5, the zero-shot sketch arms on a RunPod pod

Aaron — 12 August 2026

Follow this top to bottom on a fresh pod. It runs the two zero-shot sketch arms
(`overlay` and `language`) on the stock π₀.₅-LIBERO checkpoint, over all 114
validation scenes, and produces the numbers stage 5 needs.

The reasoning behind the run is in `brief_pi05_recovery.md`. This file is only
the steps. `RUNPOD_SETUP.md` is the longer first-time pod guide; this runbook
assumes the volume already holds the openpi checkout and the LIBERO venv from the
baseline run, and it says what to do if it does not.

**Nothing here trains anything.** It is the stock checkpoint, two different ways
of handing it a sketch.

**Time:** about 30 minutes of setup, then two runs of about 50 minutes each.

---

## What it produces

| output | what it is |
|---|---|
| `outputs/rollouts/pi05_overlay/` | the circle and arrow drawn into the image |
| `outputs/rollouts/pi05_language/` | the same shapes described in words |
| a combined `analysis.json` | all three arms from one code path |
| `report/pi05_recovery/` | the LaTeX report and its PDF |

The baseline (`outputs/rollouts/pi05_baseline/`) is the third arm. **Do not re-run
it.** It already holds 342 rows over the same 114 scenes at the same three
rollouts each.

---

## Part 0 — on Windows, before the pod

The pod gets the project by cloning from GitHub. Anything not pushed stays behind.

```powershell
cd C:\Users\Admin\sketch_prompted_vla
git status
git add -A
git commit -m "docs: stage 5 plan and runbook"
git push
```

Check `git status` is clean before you move on. This is the most common way to
lose an hour: you start the run, and the pod is running last week's code.

---

## Part A — start the pod

**A1.** RunPod console. Check the top-left dropdown says the **team**, not your
personal account, before you create anything.

**A2.** Deploy a pod:

- GPU: RTX 4090 or A40. This is inference only, so 8 GB is enough. An H100 costs
  several times more and buys nothing.
- Region: **EU-RO-1** — it must match the network volume.
- Template: any PyTorch / CUDA one. They are Ubuntu based.
- Network volume: attach the existing one, mounted at `/workspace`.
- Container disk: leave the default. Nothing important goes there.

**A3.** SSH in from PowerShell. Copy the command off the pod's card:

```powershell
ssh <pod-id>@ssh.runpod.io -i ~/.ssh/id_ed25519
```

**A4.** Check the two things that would quietly ruin the run:

```bash
nvidia-smi
df -h /workspace
ls /workspace/aaron
```

You want to see the card, a large `/workspace`, and `sketch_prompted_vla` plus
`openpi` inside `/workspace/aaron`. If `/workspace/aaron` is missing, the volume
did not attach. Stop and fix that first — do not carry on.

**A5.** Rebuild the tooling and pull the repo:

```bash
bash /workspace/aaron/sketch_prompted_vla/scripts/pod_bootstrap.sh
source ~/.bashrc
```

The script reinstalls apt packages, Node and Claude Code, sets
`OPENPI_DATA_HOME` and `MUJOCO_GL=egl`, and pulls the repo. It then prints what
is still left to do by hand. Read that list.

If it reports openpi is **not** set up, the volume has lost it and you need
sections 2 and 3 of `brief_pi05_baseline.md` before anything below will work.

**A6.** Log in to Claude Code. The token lives on the container disk, so this is
needed once per pod:

```bash
claude
```

---

## Part B — two panes

The run needs the policy server and the harness going at the same time. Use tmux
so a dropped connection does not kill the job.

```bash
tmux new -s pi05
```

Split with `Ctrl-b "`. Move between panes with `Ctrl-b o`. Reattach after any
disconnect with `tmux attach -t pi05`.

**Pane 1 — the policy server.** Leave it running for the whole session:

```bash
cd /workspace/aaron/openpi
uv run scripts/serve_policy.py --env LIBERO
```

Wait until it says it is listening. First start may take a few minutes while it
loads the checkpoint.

**Pane 2 — the simulator side:**

```bash
source /workspace/aaron/openpi/examples/libero/.venv/bin/activate
export PYTHONPATH=$PYTHONPATH:/workspace/aaron/openpi/third_party/libero
export MUJOCO_GL=egl
cd /workspace/aaron/sketch_prompted_vla
```

`MUJOCO_GL=egl` is not optional. A pod has no X server. openpi's docs suggest
`glx` when EGL errors show up; on a pod that advice is backwards.

---

## Part C — the gate. Do not skip this

This is the one step that stands between me and a confidently wrong result, and it
takes about five minutes.

The model is fed a frame that has been **rotated 180°**. The two arms handle that
rotation in opposite ways, and both fail silently if it is wrong:

- **`overlay`** draws on the raw frame *before* the rotation. The marks then turn
  with the pixels they sit on, so they stay attached by themselves.
- **`language`** must describe the frame **as the model sees it**, meaning the
  rotated one. A circle at the top-left of `frame0` is at the **bottom-right** of
  the model's view.

Get `language` backwards and I hand the model a confident, inverted instruction.
That is worse than giving it nothing, and no error is raised.

**C1. Overlay smoke:**

```bash
python scripts/rollout_sketch.py --policy pi05 --pi05-sketch-mode overlay \
    --smoke --pi05-dump-frame /tmp/ov --run-id smoke_overlay
```

Then open the PNGs in `/tmp/ov`. These are the exact images the model received.
Two things must be true in **every** frame:

- the circle sits **on an object**, not on empty table;
- the arrow points **at a destination**.

Copy one down to look at it properly:

```powershell
scp -i ~/.ssh/id_ed25519 <pod-id>@ssh.runpod.io:/tmp/ov/*.png C:\Users\Admin\Downloads\
```

**C2. Language smoke:**

```bash
python scripts/rollout_sketch.py --policy pi05 --pi05-sketch-mode language \
    --smoke --pi05-dump-frame /tmp/lang --run-id smoke_language
```

Read the prompt strings out of the log. For each one, check the direction word
against where the circled object actually is **in the dumped frame**, not in
`frame0.png`.

The worked example to check against is `spatial/scene_0000`. Its circle is at
(92, 41) in `frame0` space, which is top-right. In the model's view it is at
(35, 86), which is **bottom-left**. The prompt must say bottom-left. If it says
top-right, `rotate180` is not being applied and nothing below is worth running.

**C3. Only when C1 and C2 both hold, carry on.** Keep one overlay frame — it goes
into the report as a figure, so a reader can see what the model was shown.

---

## Part D — the two full runs

One run-id each. Both arms write `condition="auto"`, so they share a resume key and
the harness will refuse to mix them in one directory. Distinct ids anyway.

```bash
MUJOCO_GL=egl python scripts/rollout_sketch.py --policy pi05 \
    --pi05-sketch-mode overlay --conditions auto --scenes all \
    --n-rollouts 3 --run-id pi05_overlay --video
```

```bash
MUJOCO_GL=egl python scripts/rollout_sketch.py --policy pi05 \
    --pi05-sketch-mode language --conditions auto --scenes all \
    --n-rollouts 3 --run-id pi05_language --video
```

About 50 minutes each, based on what the baseline measured. If either is
interrupted, add `--resume` and run the same command again.

**When each finishes, check the row count:**

```bash
wc -l outputs/rollouts/pi05_overlay/results.csv
wc -l outputs/rollouts/pi05_language/results.csv
```

Expect 343 lines each — 342 rows plus the header, matching the baseline. A short
file means scenes were skipped, and skipped scenes need explaining before the
numbers mean anything.

**Push as soon as they land.** The volume is shared team storage, not a backup:

```bash
git add outputs/rollouts/pi05_overlay outputs/rollouts/pi05_language
git commit -m "results: zero-shot overlay and language sketch arms on pi0.5"
git push
```

---

## Part E — the analysis

Extend `scripts/analyze_baselines.py`. Do not write a second script — all
three arms should come out of one code path, or they will drift apart.

What it has to report:

1. **Overall sustained success, three arms side by side**, against the baseline's
   34.5%.
2. **By tier.** The claim is about the ambiguous tiers only. Control should barely
   move. A big control gain is a bug or a leak, not a success — say so if it
   appears.
3. **Directional tier destination rate against its 39.8% chance floor**, and
   **referential tier correct-object rate against its 34.8% floor.** These are the
   two places the baseline localised the deficit to, so recovery shows up here
   first if it shows up at all.
4. **Same-category sibling errors.** They were 73.2% of the 112 wrong grasps at
   baseline. A sketch that works should collapse this number specifically. This is
   the sharpest single test in the run.
5. **Paired, not just pooled.** The arms run the same scenes, so report the
   per-scene paired difference and how many scenes flipped each way. A pooled
   +4 pp made of 30 scenes improving and 26 regressing is a different finding from
   30 improving and none regressing. Only the paired view separates them. McNemar
   on the flip counts.

Two caveats carry forward from the baseline report and must keep carrying:
`correct_destination` is an xy-proximity proxy and is pessimistic for Goal's
region-typed destinations; only `success_sustained` is a success rate.

---

## Part F — the report

`report/pi05_recovery/`, LaTeX compiled to PDF, one directory holding the `.tex`,
`figures/` and the built `.pdf`. Author line `Aaron`. First person singular.
Confirm today's date rather than assuming it.

Include the overlay frame from part C as a figure.

Lead with where the result lands in this table:

| | overlay works | overlay fails |
|---|---|---|
| **language works** | both channels usable | the information is enough, the **modality** is the barrier |
| **language fails** | marks beat words — surprising, check for a leak | the bottleneck is not referential, and the ≈45 pp attribution needs revisiting |

All four are real findings. A null result is a result. Do not tune the prompt
wording until the number improves — if I want to try a different phrasing, it is a
**separate labelled arm**, never a replacement for this one. Quietly searching over
prompts is how a null turns into a false positive.

---

## Part G — shut down

```bash
git status          # must be clean
git push
```

Then **terminate** the pod, do not just stop it. Both stop and terminate wipe the
container disk, so they preserve exactly the same thing: whatever is on the
network volume. A stopped pod keeps billing for disk while saving nothing extra.
The Terminate button only appears once the pod is stopped.

Survives: everything under `/workspace/aaron/`. Does not survive: apt packages,
Node, Claude Code and its login.

---

## When it goes wrong

| what you see | what it is | what to do |
|---|---|---|
| `/workspace/aaron` missing | volume not attached | redeploy with the volume, in EU-RO-1 |
| EGL / rendering errors | no X server on a pod | `export MUJOCO_GL=egl`; ignore any advice to use `glx` or `xhost` |
| harness refuses to start | run-id already holds rows from a different sketch mode | use a new `--run-id`; the old run stays readable |
| `results.csv` exists | a previous attempt | add `--resume`, or pick a new `--run-id` |
| success rate near zero on both arms | usually the rotation, not the sketch | go back to part C |
| circle on empty table in a dumped frame | overlay coordinates wrong | fix `pi05_sketch.py`, do not run the full arms |
| language says top-right where the frame shows bottom-left | `rotate180` not applied | fix `describe_tokens`, then redo part C |
| server pane died mid-run | checkpoint or OOM | restart pane 1, rerun with `--resume` |
| SSH drops | normal | `tmux attach -t pi05` |
| docker compose fails | a pod is already a container | use the non-Docker path; nested Docker is unsupported |

---

## The firewall, which is the credibility of the whole run

Both adapters read `prompt.symbolic_tokens` — the circle and arrow geometry — and
nothing else. Neither ever sees `meta['target']`, `meta['destination']`,
`pick_px` or `place_px`.

The language string is built from **the sketch's own geometry**, so it carries
exactly what an annotator drew and not one bit of ground truth beyond it. A
paraphrase built from the target's name would be a different and dishonest
experiment.

If I find myself reaching for the target's name to make a sentence read better, I
stop. That is the leak this project exists to avoid.

---

## Checklist

- [ ] Windows repo committed and pushed, `git status` clean
- [ ] Pod up in EU-RO-1, volume attached, `nvidia-smi` and `/workspace/aaron` both fine
- [ ] `pod_bootstrap.sh` run, `source ~/.bashrc`, Claude Code logged in
- [ ] Pane 1 serving, pane 2 in the LIBERO venv with `MUJOCO_GL=egl`
- [ ] **Overlay smoke: circle on an object, arrow at a destination, in the rotated view**
- [ ] **Language smoke: every direction word matches the frame the model receives**
- [ ] One overlay frame saved for the report figure
- [ ] `pi05_overlay` done, 343 lines
- [ ] `pi05_language` done, 343 lines
- [ ] Results committed and pushed
- [ ] `analyze_baselines.py` extended, all three arms, paired stats included
- [ ] `report/pi05_recovery/` written and compiled, date confirmed
- [ ] Pushed, then pod **terminated**
