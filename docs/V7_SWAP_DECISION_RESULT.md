# V7 swap decision — result

9 September 2026. Aaron.

Outcome of the run specified in [`V7_SWAP_DECISION_RUN.md`](V7_SWAP_DECISION_RUN.md).
Every number was measured on the pod; the JSON and log behind each one is in
`outputs/v7_rollout/` and `logs/v7_rollout/`.

## The answer

**Swapped-sketch `referent_success` = 68/200 = 0.34** across all ten paired layouts,
against a null of 0.04. Wilson 95% CI [0.278, 0.408]. That is the `0.20 – 0.70`
band: the sketch is causal but unreliable.

The interesting part is not the mean. It is that the per-task range runs from 0.00
to 0.95, and that the range is predictable.

## The result that matters

Per-task swap success is **inversely correlated with per-task real success**:
Pearson r = **−0.721** over the ten tasks, t(8) = −2.94, p ≈ 0.019.

| task | real | swap | reading |
|---|---:|---:|---|
| t1 | 1.00 | 0.00 | prior sufficient, sketch ignored |
| t5 | 1.00 | 0.00 | prior sufficient, sketch ignored |
| t2 | 0.95 | 0.05 | prior sufficient, sketch ignored |
| t3 | 0.90 | 0.05 | prior sufficient, sketch ignored |
| t4 | 1.00 | 0.15 | prior sufficient, sketch ignored |
| t7 | 0.95 | 0.30 | mixed |
| t9 | 0.75 | 0.50 | mixed |
| t10 | 0.55 | 0.50 | mixed |
| t8 | 0.80 | **0.90** | prior weak, sketch drives behaviour |
| t6 | 0.70 | **0.95** | prior weak, sketch drives behaviour |

Where the scene layout alone determines the referent, the policy ignores the
sketch entirely. Where the layout leaves the referent genuinely ambiguous, the
policy follows the sketch — t6 at 0.95 and t8 at 0.90 are both above the 0.70 line
the runbook set for "the sketch drives behaviour".

This is the corpus-redundancy result — `I(sketch; referent | image) = 0`, measured
on the Stage 2 corpus in `scripts/corpus_redundancy.py` — showing up in behaviour
instead of in corpus statistics. My claim is that the sketch pathway was never
broken. It is read exactly when the image does not already answer the question,
and the earlier arms could not see this because their scenes always answered it.

## The arms

### Swap, all ten tasks, 200 rows (9 Sep, one evaluator, split-stamped)

`--sketch-modes swap --episodes 20 --split all`, run as t1–t4 then t5–t10.
`wrong_bowl` 0.65, no grasp 2/200, max frame error 3.701 against the 5.0 guard.
Split composition 178 train / 22 val, matching the completed real arm for a paired
comparison. BDDL `success` is 0.00 throughout and carries no information — the BDDL
still describes the original goal, so a swap-following policy is meant to fail it.

- `outputs/v7_rollout/v7_paired_step2999_swap_t1_t4.json`
  sha256 `8b95bb07eecd6cbc0287cef1b5c9de82a3f4771142cde7a697f8cb18e0fb7490`
- `outputs/v7_rollout/v7_paired_step2999_swap_t5_t10.json`
  sha256 `020f3a1128879c14000c2656fa2e695081f24d89fbb0ace9bb69bb67f1afaa3d`

### Held-out real arm, 40 rows (9 Sep)

`--sketch-modes real --episodes 4 --split val`, every row held out: 40 val, 0 train.
**87.5%** referent success, 12.5% wrong bowl, every row grasped something.

This replaces 96.25%, which was scored on roughly 74/80 training episodes and was
never a generalisation number. The Wilson interval at n=40 is [0.739, 0.945], so the
two are consistent; the point is provenance, not the drop.

- `outputs/v7_rollout/v7_paired_step2999_val_real_all10.json`
  sha256 `1f8af288978651a7c38df6affcc66acb34c15579f5d2bba6d857db2a4d27baeb`

### The 2 September 400-row rollout

A full 400-row rollout already existed when this session started, in five chunks plus
a merged file. Its swap arm scored 63/200 = 0.315, against 68/200 = 0.34 for the
9 September re-run. Per task the two agree closely (t1 0.00/0.00, t2 0.05/0.05,
t6 0.80/0.95, t8 0.85/0.90). Two independent measurements, different evaluator trees,
different data staging, same structure.

The 9 September numbers are the ones to quote: the 2 September chunks ran under three
different evaluator tree digests mid-sequence and carry no `split` stamps, while the
re-run is a single evaluator version throughout.

## Correction to the handoff

[`V7_HANDOFF_CHECKPOINT2999_ROLLOUT80.md`](V7_HANDOFF_CHECKPOINT2999_ROLLOUT80.md)
says the rollout stopped at 80/400, and `SketchPromptVLA-Pi:docs/SESSION_2026-09-09.md`
§1 carries that forward as "blocked — needs network volume `hao7ye6xly`". Both are
wrong. Chunks 2–5 finished on 2 Sep between 20:28 and 21:10 UTC. The handoff is
stamped "captured 2026-09-03 (Asia/Bangkok)", which is UTC+7, so it was written up
after the runs it describes had already landed and was never reconciled against the
outputs directory.

The cost of that stale line: this session re-derived a result that existed, and the
replication it chose — t1–t4, because those were the tasks the completed real arm
covered — happens to be four of the five layouts where the sketch has no effect.
Read alone, t1–t4 gives 5/80 = 0.0625, which is inside binomial noise of the null and
reads as no effect at all. **Never read the swap arm on t1–t4 alone.**

## What follows

The decision rule puts 0.34 in the "causal but unreliable" band, whose prescription is
action-labelled counterfactual trajectories at lower cost than a full rebuild. That
still holds, and V7's config still concedes the swap counterfactuals carry no action
loss. But the per-task split says where to spend that effort: the tasks that need
action-labelled counterfactuals are the ones where the prior already wins, and t6 and
t8 show what the pathway does once the prior stops answering the question.

Two things worth measuring next, both cheap:

- **Scene-prior sufficiency as a continuous predictor.** Ten tasks and r = −0.72 is a
  suggestive n, not a conclusive one. The blank-sketch arm
  (`v7_blank_step2999_blank_t*.json`) already measures how well each layout does with
  no pointer at all, which is a cleaner regressor than real-arm success.
- **What makes t6 and t8 different.** If it is the geometric separation of the two
  bowls, that is a corpus design rule and not a training change.

## Environment notes for the next pod

Container-local, so all three recur on every fresh pod:

- **No `~/.libero/config.yaml`.** LIBERO's first-run path setup is an interactive
  `input()`, so a detached run dies on `EOFError` with nothing else in the log. Write
  the five keys (`assets`, `bddl_files`, `benchmark_root`, `datasets`, `init_states`)
  pointing at `third_party/libero/libero/libero`.
- **Missing EGL loaders**: `apt-get install -y libegl1 libgles2 libegl-mesa0`. Without
  them `OpenGL.raw.EGL` fails at import with `'NoneType' object has no attribute
  'eglQueryString'`. The `EGLError` traces at interpreter shutdown afterwards are
  teardown noise.
- **The pod's `eval_scripts` checkout can lag.** It had no `--split` flag, so Run B
  could not have been scored honestly from it.

Staging off FUSE worked as the runbook says: checkpoint, `paired_frames_cf` and
`demos` to `/root/v7`, 19 GB against 35 GB of container disk. Checkpoint restore was
6.8 s local against 10.2 s from `/workspace`, and no arm saw a SIGBUS.
