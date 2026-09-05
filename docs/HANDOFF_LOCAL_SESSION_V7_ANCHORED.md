# Handoff to the local session — run the anchored V7 arms

5 September 2026. Branch `claude/sketchpromptvla-mismatch-fix-240ubd`, head
`2dcb846`. Written from a remote session that has no SSH key, so the pod work
could not be done there. **Everything except the two GPU arms is finished,
committed and pushed.** This file is the whole job.

Target on merge: `sketch_prompted_vla` (vanhthefirst, dongolac@gmail.com).
No `Co-Authored-By` or `Claude-Session` trailers in commits on this project.

---

## 1. What you are running, in one paragraph

V7 (`rg_v7_paired/2999`) has never been evaluated on the 37 anchored Spatial
scenes, which is where this repo's published baselines (40.3% explicit / 36.5%
ambiguous) were measured. Until it has, "V7 beats the baseline" compares two
different experiments. You are running the real-sketch arm and the swap-sketch
arm on those scenes, then scoring them. Both arms or neither: the real arm alone
is not reportable, for reasons §5 makes concrete.

## 2. The pod

```
ssh 7pxwbllcijju67-64411d72@ssh.runpod.io -i ~/.ssh/id_ed25519
```

A100 PCIe 80 GB. Network volume `hao7ye6xly` = `eval-harness-fixed` @ CA-MTL-3,
mounted at `/workspace`. Do not print `/workspace/env.sh`; it holds a GitHub
token.

The V7 policy server should already be up on **port 8200**, variant
`referent_grounding`, checkpoint
`/workspace/SketchPromptVLA-Pi/checkpoints/sketchvla_finetune/rg_v7_paired/2999`.
Check before starting anything:

```bash
ss -ltnp | grep ':8200 ' || true
pgrep -af eval_sketchvla.py || true
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader
```

If it needs restarting, `--model_variant` is **mandatory** — without it
`serve_policy_sketchvla.py` silently defaults to `input_overlay` and loads the
wrong architecture with no error:

```bash
cd /workspace/SketchPromptVLA-Pi && uv run scripts/serve_policy_sketchvla.py \
  --model_variant referent_grounding \
  --checkpoint_dir /workspace/SketchPromptVLA-Pi/checkpoints/sketchvla_finetune/rg_v7_paired/2999 \
  --device 0 --port 8200
```

A restarted disposable container has previously lost the EGL loader packages;
if MuJoCo fails to render, reinstall `libegl1 libgles2 libegl-mesa0`.

## 3. The run

Get this branch onto the pod as `/workspace/eval_scripts` (the harness repo the
evaluator reads scenes from), then:

```bash
bash drivers/v7_anchored_arm.sh
```

It runs the real arm, then the swap arm on the same scene list, then scores the
pair. It skips an arm that already has the right row count, and it aborts rather
than continue past a failure. Status: `/workspace/logs/v7_anchored_driver_status.txt`.

26 scenes x 14 rollouts x 2 arms = **728 rollouts**. Outputs:

- `outputs/rollouts/sketchvla_rg_v7_ambiguous_sketch/results.csv`
- `outputs/rollouts/sketchvla_rg_v7_ambiguous_swap/results.csv`
- `outputs/referent_following_v7_anchored.json`

The driver refuses to start without
`outputs/validation_set_spatial/swap_scene_list_bare.txt`, which is committed.

## 4. Why it is 26 scenes and not 37

`eval_sketchvla.py --sketch-mode swap` re-anchors the circle onto
`akita_black_bowl_2`, keeping the authored `rx`/`ry` and moving only the centre.
What it never checked is whether the moved ring still means one thing.
`scripts/build_anchored_swap_sketches.py` audits that and **11 of 37 scenes
fail**: 8 with the ring 22–46% off the frame (bowl_2 sits at x=112–124 on a 128
canvas), 2 with bowl_1 inside the builder's own 1.25-radii margin, 1 with a
plate strictly inside the ring. Per-scene detail in
`outputs/validation_set_spatial/swap_manifest.json`.

Both arms must run on the same 26 or `score_referent_following.py`'s
scene-pairing is not a pairing. The driver reads the list; don't hand-edit it.

**The baseline for these 26 is not the published one.** They are harder:

| run | all 37 | the 26 |
|---|---:|---:|
| pi05 anchored, explicit | 0.4035 | **0.3104** |
| pi05 anchored, ambiguous | 0.3649 | **0.2775** |

Reading a 26-scene V7 number against 40.3% credits it with nine points it did
not earn. `scripts/analysis/anchored_subset` recomputes this from the committed
CSVs; nothing is typed in from a runbook.

## 5. Reading the result — decided in advance

```bash
python drivers/v7_anchored_report.py \
  --real outputs/rollouts/sketchvla_rg_v7_ambiguous_sketch \
  --swap outputs/rollouts/sketchvla_rg_v7_ambiguous_swap
```

It prints both halves and warns, before any number, if the arms cover different
scenes or the run strays from the audited list.

**Two separate questions. A checkpoint can pass one and fail the other.**

1. *Mismatch* — V7's task success against the 0.3104 / 0.2775 baseline on these
   26 scenes.
2. *Grounding* — does moving the mark move the grasp onto the marked object.
   Want `effect[bowl_2]` large and positive, `effect[bowl_1]` large and negative
   by about as much, everything else ~0.

**The trap, made concrete.** Rehearsed on v6's pair over the same 26 scenes: v6
is indistinguishable from the baseline on task success (0.3352 vs 0.3104,
+0.025, interval crosses zero) and **15 points ahead on grasping the correct
object** (0.6181 vs 0.4698, excludes zero) — while the referent test on those
same rows returns **null**, with the uncircled bowl_3 gaining more (+9.07) than
the circled bowl_2 (+3.85, 0.57 sigma). Grasping the right bowl more often is
not evidence of reading the mark. If V7 comes back in that shape, it is the same
non-result, not a breakthrough.

Re-scoring v6 on the clean 26 also settled a question this branch raised: the
unchecked rings in v5's and v6's swap arms are a reporting-precision defect, not
the explanation of their null — cleaning them made the null *sharper*.

## 6. What else is on this branch, not yet run

`drivers/v7_baseline_arm.sh` — the mirror-image arm, and the other half of
closing the mismatch: stock pi0.5-LIBERO on V7's ten **paired layouts**, through
the same evaluator, same demos, same donor poses, same success test. Two caption
arms (`explicit` = the ceiling, `stored` = the floor). Needs a **second** server
on 8300 (`serve_policy.py --env LIBERO`, not `serve_policy_sketchvla.py`); the
evaluator refuses a server reporting a `checkpoint_dir`, so a mixed-up port
fails loudly. 400 episodes. Score with `drivers/v7_baseline_compare.py`.

Full rationale: `docs/V7_BASELINE_PARITY_CLOSEOUT.md` and
`docs/V7_ANCHORED_ARM.md`.

## 7. Traps that have already cost this project time

- **`--rotate180`.** The driver passes it. `eval_sketchvla.py` defaults it OFF,
  which suits every checkpoint up to pcla_v4. V7 trained upright. Stock pi0.5
  measures 96.7% upright against 0.0% inverted — get this wrong and you have
  measured orientation and nothing else.
- **`--model_variant` on the server.** §2.
- **Never report a real arm without its swap arm.** `overlay_v6`'s real arm
  alone read as +19.7 points and was a marker degrading the scene
  representation, not a pointer.
- **Read the sketch-cache line.** A silent sketch/scene mis-pairing completes
  the run and prints a plausible number.
- **Provenance.** Results carry both repos' SHAs and a tree digest. If the pod's
  working tree is dirty, say so in the write-up rather than letting the stamp
  quietly record it.
