# V7 swap decision — result

9 September 2026. Aaron.

This is the outcome of the run specified in [`V7_SWAP_DECISION_RUN.md`](V7_SWAP_DECISION_RUN.md).
Both arms are complete. Every number below was measured on the pod today.

## The answer

**Swap `referent_success` = 5/80 = 0.0625.** That is the `<= 0.20` band: grounding
without control.

The null for a policy that ignores the sketch is 0.04, taken from the matched real arm
(77/80 on bowl 1). 0.0625 sits inside binomial noise of that null: P(X >= 5 | n=80,
p=0.04) = 0.22, and the Wilson 95% interval is [0.027, 0.138]. So this is not a weak
effect. It is no measurable effect.

The runbook warned that anything above ~0.10 would already be outside noise. The result
did not reach that line.

## Run A — swap, t1-t4, 80 rows

`--split all --episodes 20`, the same episodes as the completed real arm, so this is a
paired comparison.

| cell | n | BDDL success | referent_success | wrong_bowl |
|---|---:|---:|---:|---:|
| t1 | 20 | 0.00 | 0.00 | 1.00 |
| t2 | 20 | 0.00 | 0.05 | 0.95 |
| t3 | 20 | 0.00 | 0.05 | 0.90 |
| t4 | 20 | 0.00 | 0.15 | 0.80 |
| **all** | **80** | **0.00** | **0.0625** | **0.9125** |

Grasped something on 78/80. Two rows never grasped. Max frame reconstruction error
3.397, under the 5.0 guard. Split composition 74 train / 6 val, as intended. Wall time
20.5 min.

BDDL `success` is 0.00 by construction and carries no information here — the BDDL still
describes the original goal, so a swap-following policy is supposed to fail it.

- JSON: `/workspace/SketchPromptVLA-Pi/outputs/v7_paired_step2999_swap_t1_t4.json`
  (sha256 `8b95bb07eecd6cbc0287cef1b5c9de82a3f4771142cde7a697f8cb18e0fb7490`)
- Log: `/workspace/logs/v7_paired_step2999_swap_t1_t4.log`
  (sha256 `b57332ba86d8a6b9e502f061cc6460597f9483b2d989dce3b868a72f0e96cdbf`)

## Run B — held-out real arm, 40 rows

`--tasks t1..t10 --sketch-modes real --episodes 4 --split val`. Every row is held out:
40 val, 0 train.

| cell | n | BDDL success | referent_success | wrong_bowl |
|---|---:|---:|---:|---:|
| t1, t2, t3, t4, t5, t7, t10 | 4 each | 1.00 | 1.00 | 0.00 |
| t6 | 4 | 0.50 | 0.50 | 0.50 |
| t8 | 4 | 0.50 | 0.50 | 0.50 |
| t9 | 4 | 0.75 | 0.75 | 0.25 |
| **all** | **40** | **0.875** | **0.875** | **0.125** |

Every row grasped a bowl. Max frame error 3.283. Wall time 19.1 min.

**87.5% is the number to report, not 96.25%.** The Wilson 95% interval is [0.739, 0.945],
so at n=40 this is consistent with the old figure and the drop is not itself evidence of
anything. The point is that 96.25% was scored on roughly 74/80 training episodes and was
never a generalisation number.

- JSON: `/workspace/SketchPromptVLA-Pi/outputs/v7_paired_step2999_val_real_all10.json`
  (sha256 `1f8af288978651a7c38df6affcc66acb34c15579f5d2bba6d857db2a4d27baeb`)
- Log: `/workspace/logs/v7_paired_step2999_val_real_all10.log`
  (sha256 `e6c53fa3f5452faf51e92e6a07c18597a8d5e0b654f7796a7340a3a23b1931e1`)

## What the two arms say together

The real arm shows the policy is competent: 87.5% on unseen episodes, wrong bowl only
12.5%. The swap arm shows that competence does not route through the sketch. Move the
circle to the other identical bowl and the policy goes to the first bowl anyway, 91% of
the time.

V7's offline gates were excellent — `point_hit_swap` 0.972, `follow_ratio` 0.996,
`swap_over_blank` 3.63. Those gates are now confirmed to measure a pointer whose output
the action head does not consult. That is precisely the failure the runbook said a
rollout, and only a rollout, could separate from real control.

V7's own config already named the cause: `swap_prob: 0.25 ... "These carry no action loss
(the corpus has no trajectory for taking the distractor)"`. The model was taught where
the circle is and never taught to act on a moved circle. It learned exactly that.

The 22.4% / 28.9% Stage 2 result and the redirect probe going *up* under a reversed sketch
are the same phenomenon from a different angle. The 9 September corpus-redundancy
measurement explains why the Stage 2 corpus could not teach control:
`I(sketch; referent | image) = 0`. V7's paired layouts fix the redundancy — two identical
bowls make a cell reachable from more than one referent — but fixing redundancy in the
*input* does not supply a *training signal* for acting on it.

## What to do next

The decision rule is unambiguous at this band: **action-labelled counterfactual
trajectories, not more grounding loss.** Concretely, the corpus needs episodes where the
circle is on bowl 2 and the demonstrated trajectory takes bowl 2. Until a swapped sketch
carries an action label, no amount of grounding cross-entropy, gate re-initialisation or
learning-rate tuning can change this number — the gradient that would teach the action
head to read the pointer does not exist in the data.

This supersedes the standing "change the gate init, do not retrain the config again"
note. The gate is a real problem in the Stage 2 lineage, but V7 does not have a dead
gate; V7 has a live pointer and no reason to obey it.

## Environment notes for the next pod

Three things had to be repaired on the fresh pod before anything would run. All three are
container-local and will recur on every new pod.

- **LIBERO had no `~/.libero/config.yaml`.** Its first-run path setup is an interactive
  `input()` prompt, so a detached run dies on `EOFError` with no other explanation. Write
  the five keys (`assets`, `bddl_files`, `benchmark_root`, `datasets`, `init_states`)
  pointing at `third_party/libero/libero/libero` before the first detached run.
- **The generic EGL loader packages were missing again**, as on the previous restarted
  container: `apt-get install -y libegl1 libgles2 libegl-mesa0`. Without them
  `OpenGL.raw.EGL` fails at import with `'NoneType' object has no attribute
  'eglQueryString'`. After installing, `OffScreenRenderEnv` renders normally; the
  `EGLError` traces at interpreter shutdown are teardown noise and can be ignored.
- **The pod's `eval_scripts` checkout was two commits behind** and had no `--split` flag,
  so Run B could not have been scored honestly from it. Check
  `grep -c -- --split scripts/eval_paired_referent.py` before trusting a pod copy.

Staging off FUSE worked as the runbook says. Checkpoint, `paired_frames_cf` and `demos`
came to `/root/v7` at 19 GB total; the container disk had 35 GB. Checkpoint restore was
6.8 s from local against 10.2 s from `/workspace`, and neither arm saw a SIGBUS.
