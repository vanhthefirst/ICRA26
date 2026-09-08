# The decisive V7 arm: swapped sketches on the paired layouts

Written 2026-09-08. Supersedes nothing in
[`V7_HANDOFF_CHECKPOINT2999_ROLLOUT80.md`](V7_HANDOFF_CHECKPOINT2999_ROLLOUT80.md) —
it corrects two things in the plan and fixes the evaluator so the result can be read.

## Why this run and not another Stage 2

The Stage 2 diagnosis holds up against the code. `models/distillation/fine_tune.py`
optimises mean flow-matching error and nothing else; `models/distillation/model.py`
leaves `attn_gate` in the ordinary trainable set for stage 2; and until today Stage 2
logged no gate at all, while Stage 1 logged it every interval. A run could therefore
close the pathway Stage 1 opened and leave no trace in any log — which is what probing
the checkpoints found (`+0.2375` in, `+0.000229` out, against `+0.000666` for a pathway
that was never trained at all).

### Measured directly, 2026-09-08

Every checkpoint was re-probed on the distillation pod with
`SketchPromptVLA-Pi:scripts/probe_sketch_gates.py`, which reads the two scalar leaves out
of the Orbax store through tensorstore — no model, no GPU, about a second each. The
readings reproduce the reported ones to six decimals, and add the rows nobody had looked
at:

| checkpoint | `tanh(attn_gate)` | `tanh(ff_gate)` |
|---|---:|---:|
| `g7_open_none/5000` (Stage 1) | +0.306129 | +0.391844 |
| `g7_open_none/10000` (Stage 1, the one Stage 2 started from) | +0.237143 | +0.356721 |
| `g7_open_none/14999` (Stage 1) | +0.187434 | +0.341614 |
| `g7_open_blind/5000` | +0.130201 | +0.466665 |
| `g7_open_blind/9999` | +0.009512 | +0.494850 |
| `s2_smoke/5000` (Stage 2) | +0.068552 | +0.329855 |
| **`s2_smoke/9999` (Stage 2, final)** | **+0.000229** | +0.320565 |
| `s2_blind/5000` | −0.001115 | +0.462471 |
| `s2_blind/9999` | +0.000666 | +0.448205 |

Raw JSON: `/workspace/SketchPromptVLA-Pi/outputs/gate_probe_stage1_stage2.json`.

Three things follow that the checkpoint-by-checkpoint view did not show.

**`ff_gate` stays wide open everywhere** (0.32–0.49, including on the blind arms). Only the
attention gate — the one path sketch media actually enters by — collapses. The sketch
branch is disconnected specifically, not as part of some general shrinkage.

**Stage 1 was already closing the gate, slowly.** 0.306 → 0.237 → 0.187 is monotone across
5000/10000/14999. Stage 2 did not reverse a healthy trend; it accelerated one already
running. Fitting an exponential to each segment:

| segment | gate half-life |
|---|---:|
| Stage 1, 5000 → 14999 | ~14,100 steps |
| Stage 2, 0 → 5000 | ~2,800 steps |
| Stage 2, 5000 → 9999 | ~610 steps |

A 23× acceleration that is itself still accelerating. Note in passing that using the
`10000` checkpoint rather than `14999` handed Stage 2 the more open gate by luck, not by
design.

**Weight decay is not the mechanism.** `build_train_config` uses
`_optimizer.AdamW(clip_gradient_norm=1.0)`, whose `weight_decay` defaults to `1e-10`; at
lr ~1e-5 that moves the gate by order 1e-11 relative over 10k steps, against an observed
99.9% collapse. The gate is being closed by the gradient of the action loss, which is the
diagnosis and not merely consistent with it.

That failure is a *training-signal* failure, not a wiring failure, so re-running the
same formulation with a bigger sketch learning rate or a clamped gate buys nothing. V7
(`referent_grounding`, on `SketchPromptVLA-Pi@feat/eval-harness`) already replaces the
signal: paired layouts, per-patch grounding cross-entropy, swap counterfactuals,
early-frame upweighting. Its config says so in as many words —
`swap_prob: 0.25 … "These carry no action loss (the corpus has no trajectory for taking
the distractor)"`.

**That comment is the whole reason this run is decisive.** V7 has only ever been taught
where the circle is, never to *act* on a moved circle. Offline the pointer is excellent
(`point_hit_swap=0.9724`, `follow_ratio=0.9961`, `swap_over_blank=3.63`), but a sharp
pointer whose output the action head ignores looks exactly like this. Only a rollout
separates them.

## Two corrections to the handoff's plan

### 1. The evaluator was silently scoring mostly training episodes

`eval_paired_referent.py` took `sorted(...)[:episodes]` per task — the lowest demo
indices, with no reference to the packer's split. `pack_paired_corpus.py` holds out
`10%` per task by `random.Random(7)`, so at ~43 episodes per task only **~4 are held
out**, and the completed 80-row real arm is roughly 4 val + 16 train per task.

For the **swap arm this is acceptable and even conservative**: the corpus carries no
action label for taking the distractor, so swap behaviour is untrained either way, and a
memorised layout biases *against* sketch-following. For the **real arm it is not**:
96.25% on episodes the model trained on is a memorisation score, and must not be
reported as generalisation.

Fixed in `scripts/eval_paired_referent.py`: `--split {all,val,train}` reproduces the
packer's split exactly (verified byte-identical against `pack_paired_corpus.py` on a
430-file fixture: 390/40), every row is stamped `split`, and every summary cell carries
`n_train`/`n_val`.

### 2. Run swap on the *same* episodes as the completed real arm

Do not switch the swap arm to `--split val`. The existing 80 real rows used the default
selection; a swap arm on the same episodes is a paired comparison against them, which is
far stronger than two arms on different data. Keep `--split all --episodes 20`.
The held-out read is a separate, smaller arm (below).

## Run A — the decision (swap, t1–t4, 80 rows, ~70 min)

Same prefix as the handoff. Server on port 8200, checkpoint `rg_v7_paired/2999`.

```bash
setsid -f env \
  PYTHONPATH="$PYTHONPATH" MUJOCO_GL="$MUJOCO_GL" \
  __EGL_VENDOR_LIBRARY_FILENAMES="$__EGL_VENDOR_LIBRARY_FILENAMES" \
  "$PY" "$SCRIPT" \
  --bddl-dir "$BDDL" --demo-dir "$DEMOS" --frames-dir "$FRAMES" \
  --checkpoint "$CKPT" --variant referent_grounding \
  --host 127.0.0.1 --port 8200 \
  --tasks t1,t2,t3,t4 --sketch-modes swap --episodes 20 --max-steps 520 \
  --split all \
  --out /workspace/SketchPromptVLA-Pi/outputs/v7_paired_step2999_swap_t1_t4.json \
  > /workspace/logs/v7_paired_step2999_swap_t1_t4.log 2>&1
```

Score `referent_success` (grasped `akita_black_bowl_2`) and `wrong_bowl`. **Ignore BDDL
`success` on this arm** — the BDDL still describes the original goal, so a swap-following
policy is *supposed* to fail it.

### Reading the number

The matched real arm grasped bowl 1 on 77/80. A policy that ignores the sketch does the
same under swap, so the null is `referent_success ≈ 0.04`, `wrong_bowl ≈ 0.96`.

| swap `referent_success` (n=80) | reading |
|---|---|
| ≥ 0.70 | The sketch drives behaviour. Root cause resolved; finish the remaining rows. |
| 0.20 – 0.70 | Causal but not reliable. The grounding head is read intermittently; likely needs the action-labelled counterfactuals, at lower cost than a full rebuild. |
| ≤ 0.20 | Grounding without control. V7's offline gates measure a pointer the action head does not consult. **Next step is action-labelled counterfactual trajectories, not more grounding loss.** |

Anything above ~0.10 is already far outside binomial noise against a 0.04 null; the
0.70 line is a usability threshold, not a significance one. Do not read a mid-band
result as success.

## Run B — the honest generalisation read (40 rows, ~35 min)

Only after Run A. Four held-out episodes per task across all ten:

```bash
  --tasks t1,t2,t3,t4,t5,t6,t7,t8,t9,t10 --sketch-modes real --episodes 4 \
  --split val \
  --out .../v7_paired_step2999_val_real_all10.json
```

`--episodes 4` because that is what the 10% holdout leaves; the evaluator now fails loudly
rather than quietly borrowing training episodes to make up the count. This is the number
to put in a paper next to the real arm — not the 96.25%.

## Blocker: the pod has the wrong volume

The current A100 pod carries the older distillation volume. The V7 checkpoint, the paired
counterfactual corpus (`/workspace/data/paired_frames_cf`), the LIBERO demos and the
`feat/eval-harness` checkout all live on network volume **`hao7ye6xly`**. Attach that
volume, or nothing above can run. Do not spend the GPU re-training the old Stage 2
formulation while waiting.

## Also changed today

`SketchPromptVLA-Pi:src/sketchvla/models/distillation/fine_tune.py` now logs
`gate/attn_gate` every `log_interval`, to console and W&B, the same way `distill.py`
always did. It does not change training. It means the next run that closes the gate says
so while it is happening instead of at the autopsy.

`SketchPromptVLA-Pi:scripts/probe_sketch_gates.py` is new: the offline reader used for the
table above. It takes step directories and needs neither JAX nor a GPU, so "did this run
close the gate" never again requires scheduling an A100.
