# Prompt — recovering the ambiguity gap: sketch overlay vs language control

Paste everything below the line into a fresh **Claude Code** session on a RunPod
GPU pod, run from the repository root. Setup is `RUNPOD_SETUP.md`; on a fresh
pod, `bash scripts/pod_bootstrap.sh` rebuilds the container-disk half. Ubuntu
throughout, `MUJOCO_GL=egl`, no nested Docker.

---

## Role and objective

You are working in the **Sketch-Prompted VLA** repository. Read `CLAUDE.md`,
`SUITE_FACTS.md`, `report/pi05_baseline/pi05_baseline_report.pdf` and the
docstrings of `scripts/pi05_sketch.py` and `scripts/pi05_policy.py` before
running anything.

The baseline established the headroom: π₀.₅-LIBERO scores **34.5%** on my 114
scenes against **96.0%** from the same pipeline on the standard suites, and the
control tier isolates roughly 25 points of that as distribution shift and
roughly 45 as referential ambiguity. On the directional tier the model delivers
to the right destination 40.7% of the time against a 39.8% chance floor — it is
not disambiguating destinations at all.

**This run asks whether any of those ~45 points come back when the sketch is
supplied, without training anything.** Two zero-shot arms on the stock
checkpoint, over all 114 scenes:

- **`overlay`** — the circle and arrow composited onto the image the model sees.
- **`language`** — the same circle and arrow described in words and appended to
  the instruction.

## 1. Why both arms, and how to read them

They are a matched pair, not alternatives. π₀.₅ was never trained on images
containing annotation marks, so a null result from `overlay` alone is ambiguous.
`language` speaks to the model in a modality it was trained on and therefore
bounds what *any* prompt could recover on these scenes. The 2×2:

| | overlay works | overlay fails |
|---|---|---|
| **language works** | both channels usable | the information suffices; the **modality** is the barrier — fine-tuning is the answer |
| **language fails** | marks beat words (surprising; check for a leak) | the bottleneck is not referential, and the baseline report's ~45-point attribution needs revisiting |

Any of these four is a publishable finding. Do not treat a null result as a
failure to be engineered away — report it.

## 2. The firewall, which is the whole credibility of this run

Both adapters read `prompt.symbolic_tokens` — `{circle:{cx,cy,rx,ry},
arrow:{x0,y0,x1,y1}}` — and nothing else. Neither ever sees `meta['target']`,
`meta['destination']`, `pick_px` or `place_px`. The language string is derived
from the **sketch's geometry**, so it carries exactly what an annotator drew and
not one bit of ground truth beyond it.

A language paraphrase built from `meta['target']` would be a different and
dishonest experiment. If you find yourself reaching for the target's name to
make the sentence read better, stop — that is the leak this project is built to
avoid.

## 3. The code you are inheriting

`scripts/pi05_sketch.py` is new; `--pi05-sketch-mode {none,overlay,language}` is
new in the harness. Written without a GPU and **never run against a live
server**. Verified only against a stub, where:

- overlay marks land within **1.1 px** of their token coordinates at 256, and
  produce ~721 green and ~314 red pixels in the final 224 model input;
- `language` alters only the prompt string and leaves the image untouched;
  `overlay` alters only the image and leaves the prompt untouched;
- `sketch_mode=none` still refuses a sketch-bearing prompt, and a sketch mode
  with no tokens raises rather than silently running text-only.

Treat it as a considered draft. Where it contradicts what you observe, the
observation wins — fix it and say in a sentence what was wrong.

## 4. The coordinate trap, which is not the same as the baseline's

The baseline settled that the model is fed a 180°-rotated frame. That has
**opposite consequences for the two arms**, and getting it backwards is silent:

- **Overlay** draws on the raw 256 frame *before* the rotation, at 2× the token
  coordinates. The marks then rotate along with the pixels they annotate and
  stay attached automatically. Nothing further is needed.
- **Language** must describe the frame **as the model sees it**, i.e. the
  rotated one. A circle at the top-left of `frame0` is at the *bottom-right* of
  the model's view. `describe_tokens(..., rotate180=True)` handles this, and
  spatial/scene_0000 is a worked example: circle at (92, 41) in `frame0` space
  → "top-right", but (35, 86) in the model's view → **"bottom-left"**, which is
  what the prompt says.

Describing the unrotated frame would hand the model a confidently inverted
instruction — worse than no instruction. **Verify this before the full run**, in
step 6.

## 5. Setup

Same as the baseline. Server in one pane, LIBERO venv in another:

```bash
# pane 1, openpi checkout
cd /workspace/aaron/openpi && uv run scripts/serve_policy.py --env LIBERO

# pane 2
source /workspace/aaron/openpi/examples/libero/.venv/bin/activate
export PYTHONPATH=$PYTHONPATH:/workspace/aaron/openpi/third_party/libero
export MUJOCO_GL=egl
cd /workspace/aaron/sketch_prompted_vla
```

## 6. Vertical slice, then smoke, then full

**6a. Confirm the marks reach the model and the words match the view.**

```bash
python scripts/rollout_sketch_wsl.py --policy pi05 --pi05-sketch-mode overlay \
    --smoke --pi05-dump-frame /tmp/ov --run-id smoke_overlay
```

Open the dumped frames. The circle must sit on an object and the arrow must
point at a destination — in the rotated view. Then run the same smoke with
`--pi05-sketch-mode language` and read the prompt strings out of the log:
confirm each direction word matches where the circled object actually is in the
frame the model receives.

**6b. Full run, one run-id per mode.** Both arms write `condition="auto"`, so
they share a resume key; the harness will refuse to mix them in one directory,
but choose distinct ids anyway:

```bash
MUJOCO_GL=egl python scripts/rollout_sketch_wsl.py --policy pi05 \
    --pi05-sketch-mode overlay --conditions auto --scenes all \
    --n-rollouts 3 --run-id pi05_overlay --video

MUJOCO_GL=egl python scripts/rollout_sketch_wsl.py --policy pi05 \
    --pi05-sketch-mode language --conditions auto --scenes all \
    --n-rollouts 3 --run-id pi05_language --video
```

Roughly 50 minutes each, on the baseline's measured throughput. Use `--resume`
if interrupted.

## 7. Analysis

**Do not re-run the baseline.** `outputs/rollouts/pi05_baseline/results.csv`
holds 342 rows over the same 114 scenes at the same three rollouts each; it is
the comparison arm.

Extend `scripts/analyze_pi05_baseline.py` rather than writing a second analysis
script, so all three arms are computed by one code path. Report:

- Overall sustained success, three arms side by side, with the baseline's 34.5%.
- **By tier.** The claim under test is specifically about the ambiguous tiers.
  Control should barely move — nothing about it is ambiguous, so a large control
  gain is evidence of a bug or a leak, not of success. Say so if you see one.
- **The directional-tier destination rate against its 39.8% chance floor**, and
  the referential-tier correct-object rate against its 34.8% floor. These are
  the two numbers the baseline localised the deficit to, so they are where
  recovery should appear if it appears anywhere.
- **Same-category sibling errors**, which were 73.2% of wrong grasps at baseline.
  A sketch that works should collapse this specifically.
- **Paired, not just pooled.** The arms run the same scenes, so report the
  per-scene paired difference and how many scenes flipped each way. A pooled
  4-point gain built from 30 scenes improving and 26 regressing is a different
  finding from one where 30 improve and none regress, and only the paired view
  distinguishes them. McNemar on the flip counts is appropriate.

Carry forward the baseline report's two standing caveats: `correct_destination`
is an xy-proximity proxy and is pessimistic for Goal's region-typed
destinations; only `success_sustained` is a success rate.

## 8. Deliverables

1. `outputs/rollouts/pi05_overlay/` and `outputs/rollouts/pi05_language/` —
   `results.csv`, `run_config.json` (stamps the sketch mode), `summary.json`.
2. A combined `analysis.json` covering all three arms.
3. `report/pi05_recovery/` — LaTeX compiled to PDF, per the repo convention.
   Author line `Aaron`, first person singular, and confirm today's date rather
   than assuming it. Include the sample overlay panel from step 6a as a figure;
   a reader must be able to see what the model was actually shown.

Match the length of what you write to what the task needs. Do not pad.

## 9. How to work

Deliver what is asked, at the scope intended. Make routine judgment calls
yourself; check in only where different readings of this brief would lead to
materially different work. If something here is mistaken or a better approach
exists, say so in a sentence and continue as written rather than quietly
reshaping it.

Do not fine-tune anything, do not modify the validation scenes, do not tune the
prompt wording to improve the result. If you want to test an alternative
phrasing, that is a **separate labelled arm**, not a replacement for this one —
silently searching over prompts until the number improves is how a null result
becomes a false positive.

Do not spawn subagents; this is one sequential track gated on a running server.

Say in one sentence what you are about to do before your first tool call. While
working, give a brief update only when you find something important or change
direction — the step 6a orientation confirmation is one such moment. When you
finish, lead with the outcome: overlay and language success rates against the
34.5% baseline, the tier breakdown, and where in the 2×2 of section 1 the result
lands. Supporting detail after that.

State plainly what you measured and what you did not.
