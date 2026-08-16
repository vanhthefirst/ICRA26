# Brief — π₀.₅-LIBERO as the learned, sketch-free baseline

Paste everything below the line into a fresh **Claude Code** session, run from
the repository root on the **GPU machine**. Everything runs on **Ubuntu** with
`conda activate libero` for the simulator side; the model side runs in the
openpi checkout under `uv` and needs an NVIDIA GPU with **≥8 GB** free
(inference only — no fine-tuning here).

---

## Role and objective

You are working in the **Sketch-Prompted VLA** repository. Read `CLAUDE.md`,
`SCHEMA.md`, `SUITE_FACTS.md` and the docstrings of
`scripts/rollout_sketch.py` and `scripts/pi05_policy.py` before running
anything.

The scripted oracle proved the rollout loop. What it cannot do is tell me how a
*real* VLA does on these scenes without a sketch. Produce that number:
**π₀.₅-LIBERO, stock checkpoint, text instruction only, over all 114 validation
scenes** — and, before it, a reproduction of the published benchmark numbers
that shows the pipeline feeding it is correct.

All commands run on Ubuntu. Two python environments, kept separate: the openpi
checkout under `uv` (python 3.11) holds the model, and a python 3.8 environment
holds LIBERO and mujoco. Do not try to merge them.

**This runs on a RunPod GPU pod**, which changes three things from a normal
workstation and each is called out where it applies: no nested Docker
(section 3), no X server so `MUJOCO_GL=egl` (sections 3 and 4), and ephemeral
disk unless work happens under the mounted network volume. Keep the openpi
checkout, `OPENPI_DATA_HOME` and this repo all under `/workspace`.

## 0. The state of the code you are inheriting

`scripts/pi05_policy.py` and the `--policy pi05` path in
`scripts/rollout_sketch.py` were written on a laptop with no GPU and **have
never been run against a live policy server**. They were exercised only against
a stub: the firewall raises, the resolution guard raises, twelve steps at
replan 5 produce three inference calls, and the observation dict comes out as
`(224,224,3) / (224,224,3) / (8,) / instruction`. That is all that is known to
work.

Treat that code as a considered draft, not as ground truth. Where a claim in
this brief contradicts what you actually observe on the machine, the observation
wins — fix the code, and say in a sentence what was wrong. Do not preserve a
broken assumption out of deference to the file it is written in.

## 1. What π₀.₅-LIBERO is, and why nothing needs ablating

π₀.₅ is Physical Intelligence's flow-matching VLA
(`Physical-Intelligence/openpi`). π₀.₅-LIBERO is that model fine-tuned on the
four standard LIBERO suites, released as
`gs://openpi-assets/checkpoints/pi05_libero` under the openpi config
`pi05_libero`. Published success rates, 500 trials per suite:

| Suite | spatial | object | goal | libero_10 | average |
|---|---|---|---|---|---|
| π₀.₅ @ 30k | 98.8 | 98.2 | 98.0 | 92.4 | **96.85** |

It consumes **(base image, wrist image, 8-D proprioceptive state, language
string)** and emits an action chunk. It has **no sketch channel**. So "π₀.₅
without the sketch" requires no edit to the model, the config or the checkpoint:
the sketch-free baseline is the stock checkpoint fed `meta['instruction']` — the
same instruction string every sketch condition already receives. There is
nothing to ablate.

**Use `main`.** π₀.₅ support merged into openpi's `main` in September 2025.
`kevin/pi05-support` was the pre-merge development branch and is stale — do not
check it out.

## 2. Install (openpi side)

```bash
git clone --recurse-submodules https://github.com/Physical-Intelligence/openpi.git
cd openpi
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
```

`--recurse-submodules` matters: LIBERO arrives as `third_party/libero`. If you
already cloned, `git submodule update --init --recursive`.

The checkpoint downloads automatically into `~/.cache/openpi` on first use. It
is several GB, so **set `OPENPI_DATA_HOME=/workspace/.cache/openpi`** before
first use — `~` is on the pod's ephemeral disk and re-downloading it on every
pod restart is wasted time and money.

## 3. Reproduce the published numbers first

Run **openpi's own** LIBERO eval, unmodified, on the standard suites. This is
the gate for everything after it: it is what separates "my scenes are hard for
π₀.₅" from "my observation pipeline is subtly wrong", and section 5 lists four
ways to be subtly wrong that all present as a merely low success rate.

openpi documents a Docker workflow for this and recommends it. **It will not work
here.** A RunPod GPU pod is itself a Docker container and nested Docker is not
supported, so use openpi's "Without Docker" path. There is also no X server on a
pod, which inverts openpi's rendering advice: use `MUJOCO_GL=egl`, not `glx`,
and ignore any instruction to run `xhost`.

Terminal 1 — the policy server, from the openpi checkout:

```bash
uv run scripts/serve_policy.py --env LIBERO
```

Terminal 2 — the LIBERO client, in its own python 3.8 environment:

```bash
uv venv --python 3.8 examples/libero/.venv
source examples/libero/.venv/bin/activate
uv pip sync examples/libero/requirements.txt third_party/libero/requirements.txt \
    --extra-index-url https://download.pytorch.org/whl/cu113 --index-strategy=unsafe-best-match
uv pip install -e packages/openpi-client
uv pip install -e third_party/libero
export PYTHONPATH=$PYTHONPATH:$PWD/third_party/libero
MUJOCO_GL=egl python examples/libero/main.py --args.num-trials-per-task 5
# other suites: --args.task-suite-name libero_object
```

That venv is the LIBERO environment for everything after this — it takes the
place of the `libero` conda env used on my laptop. Section 4 installs nothing
further into it; `openpi-client` is already there from the line above.

Full suites are 500 episodes each and slow. **Cut `--args.num-trials-per-task`
to 5** — 50 episodes per suite, enough to separate ~95% from ~40%, which is the
only separation that matters here. Run at least `libero_spatial` and
`libero_object`.

Report the observed rates against the table in section 1 as soon as you have
them, before moving on. If they land near it, the install and the reference
preprocessing are good. If they do not, stop and fix that first — a broken
pipeline evaluated on my scenes produces a plausible-looking low number that
would be quietly wrong.

## 4. Wire the server to my harness

The model stays in its own process behind a websocket. The simulator keeps
running in the `libero` conda env, unchanged.

Terminal 1 — openpi checkout, the same server as section 3:

```bash
uv run scripts/serve_policy.py --env LIBERO
# explicit equivalent, and what to use for a non-default checkpoint:
uv run scripts/serve_policy.py policy:checkpoint \
    --policy.config=pi05_libero \
    --policy.dir=gs://openpi-assets/checkpoints/pi05_libero
```

Terminal 2 — the section 3 LIBERO venv, now pointed at this repo:

```bash
source /path/to/openpi/examples/libero/.venv/bin/activate
export PYTHONPATH=$PYTHONPATH:/path/to/openpi/third_party/libero
cd /path/to/sketch_prompted_vla
MUJOCO_GL=egl python scripts/rollout_sketch.py --policy pi05 --smoke
```

`openpi-client` is a thin websocket + image-tools package with no model
dependencies; it is already in that venv. The model never runs there. The
harness also needs `opencv-python-headless` — install that rather than
`opencv-python`, which wants GUI libraries a pod does not have.

`--policy pi05` forces `--conditions text_only`, opens each scene env at 256
with the wrist camera, and applies openpi's per-suite step budgets. It refuses
sketch conditions and `--deproject depth`; both refusals are deliberate and are
explained where they are raised.

(The harness file is named `....py` for historical reasons and runs fine on
native Ubuntu. Do not rename it — other briefs and the `outputs/rollouts/`
provenance refer to it by name.)

## 5. Four ways to be silently wrong

Each of these degrades the success rate rather than raising an error. All four
are handled in `scripts/pi05_policy.py`; this list is so you know where to look
if section 3 reproduces but section 6 collapses.

1. **Resolution.** openpi renders LIBERO at 256×256 and pad-resizes to 224.
   π₀.₅-LIBERO has never seen a 128×128 frame. My suites render at 128 because
   that is the sketch canvas; the two are decoupled, and `open_scene_env` takes
   `render_size`. The 128-space sketch geometry is not valid at 256 and is not
   consulted on this path.
2. **The 180° rotation.** openpi applies `img[::-1, ::-1]` to both cameras to
   match the orientation of the OpenVLA `modified_libero_rlds` data the
   checkpoint was fine-tuned on. `SUITE_FACTS.md` says *"`obs['agentview_image']`
   is already correctly oriented — do not flip it"*; that rule governs the
   **projection** path, where pixels must line up with `frame0.png`, and not
   what the model is fed. I believe both are true and about different things,
   but that is reasoning from source, not a measurement. Settle it: construct
   `Pi05ServerPolicy` with `dump_first_frame_to=...` and compare the dumped
   frame — the exact array the model receives — against `frame0.png`.
   `--pi05-no-rotate180` exists to confirm the rotation is what matters.
3. **Wrist camera and 8-D state.** `robot0_eye_in_hand_image` is required, and
   state is `concat(eef_pos, quat2axisangle(eef_quat), gripper_qpos)` = 3+3+2.
   Not the 7-D action vector. This harness had never rendered the wrist camera
   because no scripted policy needed it.
4. **Chunked control.** The model returns a chunk; execute `replan_steps` (5) of
   it, then re-query. Consuming only the first action, or the whole chunk, are
   both wrong and both look like a clumsy policy rather than a bug.

One deliberate divergence from openpi's loop: `num_steps_wait` is **0**, not 10.
openpi steps dummy actions to let dropped objects settle; my `init_state.npz`
was pinned *after* settling, so every condition starts from the identical
already-settled state. Do not reintroduce the wait.

## 6. The run

```bash
MUJOCO_GL=egl python scripts/rollout_sketch.py --policy pi05 --scenes all \
    --run-id pi05_baseline --n-rollouts 3 --video
```

Scene-major ordering, `--resume` and the reproducibility roster behave as they
do for the scripted policies. Three rollouts per scene rather than one: π₀.₅ is
stochastic where the oracle is not, and a single rollout over 114 scenes gives a
success rate with no sense of its own variance. If the server is on another
machine, pass `--pi05-host`.

This is slow — roughly `114 × 3 × (220–300 steps ÷ 5 replan)` inference calls.
Budget from the latency measured in the smoke run rather than hoping. Start with
`--scenes subset` (36 scenes) and use `--resume`. **If the projected full-run
time is more than a few hours, say so and ask before committing to it** — one
rollout per scene is an acceptable first pass and repeats can be added later.

## 7. Expect a low number, and report it as a property of my scenes

π₀.₅-LIBERO was fine-tuned on the four standard LIBERO suites. My 114 scenes are
**synthesised BDDL** with novel object arrangements and instructions that are
deliberately ambiguous in the non-control tiers. The published 96.85% is not the
prediction for them, and a large gap is the expected result — it is the headroom
the sketch is supposed to recover.

Do not tune to close the gap. Report it, and separate its two causes as far as
the data allows:

- **Distribution shift** — π₀.₅ on my scenes vs π₀.₅ on the standard suites
  (section 3 gives the second number).
- **Referential ambiguity** — within my scenes, the **control tier**
  (unambiguous instruction, one candidate) vs the other tiers. Control-tier
  performance is the honest ceiling for a text-only policy here; the tier gap is
  the part a sketch could address. `results.csv` carries `tier`, and
  `correct_instance_grasped` / `correct_destination` separate "picked the wrong
  object" from "could not manipulate at all", which is the distinction that
  makes the number mean something.

Where the data does not support separating the two, say so rather than
estimating a split.

## 8. Deliverables

1. `outputs/rollouts/pi05_baseline/` — `results.csv`, `run_config.json` (already
   stamps host/port, replan, rotation, render size), `summary.json`, videos.
2. `outputs/rollouts/pi05_baseline/openpi_reference.json` — the section 3
   reproduction: suite, trials per task, observed rate, published rate, openpi
   git SHA.
3. `report/pi05_baseline/` — LaTeX compiled to PDF, per the repo convention. The
   skeleton, tables and figure slots exist; fill them. Author line `Aaron`,
   first person singular throughout, and confirm today's date rather than
   assuming it. Every unmeasured number in the skeleton is wrapped in `\tbd` —
   none should survive to the compiled PDF.

Match the length of what you write to what the task needs. Cover the substance;
do not pad with filler sections, redundant summaries or boilerplate.

## 9. Order of work

**Vertical slice → smoke → full run**, the repo's standing build order:

1. openpi installed, checkpoint pulled, server answering.
2. Section 3 reproduction on two suites at 5 trials/task — the gate.
3. `--policy pi05 --smoke` (3 scenes) end to end, with the dumped model-input
   frame compared against `frame0.png`.
4. `--scenes subset` (36 scenes), then all 114.
5. Report.

## 10. How to work

Deliver what is asked, at the scope intended. Make routine judgment calls
yourself and keep going; check in only where different readings of this brief
would lead to materially different work, or where section 6's runtime warning
applies. If something here is mistaken or a better approach exists, say so in a
sentence and continue with the task as written rather than quietly narrowing,
widening or transforming it. Finish the whole task, and stop short of work
clearly beyond it — in particular, do not fine-tune anything, do not build a
sketch-conditioned variant of π₀.₅, and do not modify the validation scenes.

Do not spawn subagents; this is a single sequential track of work gated on a
GPU and a running server, and there is nothing in it to parallelise.

Say in one sentence what you are about to do before your first tool call. While
working, give a brief update only when you find something important or change
direction — the section 3 reproduction rate is one such moment, and so is any
point where the code as inherited turns out to be wrong. When you finish, lead
with the outcome: the reproduction rates, the overall π₀.₅ success rate on the
114 scenes, and the control-tier vs other-tier split. Supporting detail after
that.

State plainly what you measured and what you did not. If a number is from 36
scenes rather than 114, or one rollout rather than three, or a partial run that
was resumed, say so where you report it.
