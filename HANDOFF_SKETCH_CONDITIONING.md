# HANDOFF — Sketch conditioning, session of 31 Aug 2026

Supersedes the "Recommended next steps" of `HANDOFF_SKETCH_GROUNDING.md`: all
three are done. That file is still correct about the corpus, the rollouts and the
traps; only its forward-looking section is spent.

The verdict changed. The 30 Aug reading was "the sketch is not referential, and
the question is where the referent is lost". It is not lost. It reaches the
action expert, localised to 4.0 px, and the model does not act on it. STATE.md
(31 Aug section) carries every number; this file carries what to do about it.

Repos: training `SketchPromptVLA-Pi@feat/eval-harness` HEAD `fd77035`; eval this
repo, branch `sketch_prompted_vla`. Probes are
`scripts/probe_sketch_readout.py` and `scripts/probe_sketch_attention.py` in the
training repo. Raw results: `outputs/readout_v5.json`,
`outputs/attention_v5.json`, `outputs/readout_v5_step3.json`.

## The diagnosis in three lines

* The sketch encoder keeps the circle: a linear readout of its tokens recovers
  the centre at R^2 0.980, median 4 px, on held-out episodes.
* The cross-attention is uniform to within a few percent — but a uniform softmax
  returns the *mean* of the value vectors, and `to_kv` is unbiased, so it does
  not discard anything. The vector it adds to the action tokens still encodes the
  centre at R^2 0.974.
* So the signal is delivered and unused. The plumbing is not the fault.

## The structural claim, which is the thing to test first

`modality_attn` is applied to `suffix_out` — after the entire PaliGemma forward —
and only `action_out_proj`, a single Linear, follows it. With the softmax
uniform, the branch output is the same vector for every action-token query and
does not depend on the queries at all.

So the sketch reaches the action head as **one vector, added identically at every
horizon step**. It has no spatial structure and no temporal structure. It can
translate the predicted action chunk; it cannot change its shape. "Take the left
bowl rather than the right one" is a shape change.

**This was measured on 31 Aug and it is wrong.** A third of the sketch's effect
does vary across the horizon (0.326), which is in the same range as the language
pathway's 0.426. The channel *can* reshape the chunk. It is simply 55x weaker
than the language pathway (median over 41 scenes, minimum 12x) and moves the arm
actions by ~0.5% of their own scale. The architecture is not the constraint, and
the gate agrees: it shrank, 0.0997 -> 0.0890, which is the right response to a
loss that does not reward the pathway.

Read this section as the record of a refuted hypothesis. What replaces it is
below.

## Next steps, in order

**Neither of the first two costs a training step, and neither needs a big GPU.**

1. ~~**Is the sketch's effect a constant offset?**~~ **DONE 31 Aug — no.**
   `probe_sketch_attention.py --chunk-structure`, 41 val scenes, pinned noise.
   Sketch rms 0.00202 with a 0.326 non-constant fraction; language rms 0.1235
   with 0.426. The pathway reshapes and is 55x too weak. See STATE.md.
   Original framing, kept for the method:
   Serve one observation twice with the circle in two places and look at the
   *structure* of the difference between the two action chunks, not its norm:
   report `std over horizon steps / mean |d|` per action dimension.
   `sketchvla/sensitivity.py` already computes exactly those two chunks and
   collapses them to one scalar (`sketch_l2`); the structure is thrown away at
   the last line. Near-zero spread across the horizon confirms the claim above
   and settles the architecture question by itself.

2. **Did the model fit the distinction it was trained on? (0 steps, ~15 min)**
   `uv run src/sketchvla/validate.py --checkpoint <1499> --ablate-sketch
   --data-dir /workspace/data/sketch_libero_rlds_paired --split val`.
   Within a layout the paired corpus lets only the sketch say which bowl, so a
   sketch-ignoring model must predict the average of two trajectories and pay
   for it in action L1.
   * `delta_arm_l1` ~ 0 → the optimiser never used the signal; whatever the
     corpus was designed to require, the loss did not require in practice.
   * `delta_arm_l1` clearly > 0 → it fit the training distinction, and the
     failure is generalisation to the eval scenes. A different problem, and the
     corpus rather than the architecture would be the lever.
   This is the one measurement that separates "cannot express it" from "was
   never made to".

3. **The corpus question, which now outranks any retraining.** (0 steps, minutes,
   no GPU.) The paired corpus was verified sketch-necessary on its BDDL *goals*
   (`check_sketch_necessity.py`). The loss never sees a goal — it sees action
   chunks. For a layout pair, how different are the ground-truth chunks at frame
   0, where proprioception cannot yet say which bowl the arm is heading for?
   `iter_rlds_samples(..., frames_per_episode=1)` yields exactly that, keyed by
   `paired_spatial/<task>/<demo>`. If the between-task difference at frame 0 is
   small against the within-task spread, the corpus is not sketch-necessary at
   the level the objective operates on, and no architecture change fixes it.
   After frame 0 the state leaks the answer, so that is the frame that matters.

4. **Only if 3 says the corpus does require it: one run of at most 3000 steps.**
   Change *where* the sketch enters, not how long it trains. The repo already ships two
   variants that inject upstream of the LLM, so the backbone can condition its
   image reading on the sketch instead of receiving a summary afterwards:
   `input_overlay` (the sketch drawn into the frame the frozen SigLIP reads) and
   `visual_prompt_tokens`. `input_overlay` is the cheaper bet — stock pi0.5
   scores 96.7% on upright frames, so a marked image is close to in-distribution
   for that tower, and the corpus already carries `sketch_overlay`.
   Same paired corpus, same batch 32, same LR schedule, one A100, `sketchvla=input_overlay`.

   **Gate it on the swap arm, not `sketch_l2`.** `sketch_l2` reached 0.029 in
   v5 while the model was ignoring the sketch completely; it measures pathway
   activity, not use. The swap arm is `SKETCH_MODES=swap` in
   `examples/libero/run_eval.sh`, and the number that matters is the
   took-circled-bowl rate against the real arm's 59.0%.

## What not to spend the budget on

* **More `prompt_conditioned_latent_action` steps at the same objective.** The
  gate shrank over the run because the loss did not reward the pathway; more
  steps of the same loss shrink it further. Three runs have re-confirmed this
  null already.
* **Tuning `attn_gate_init`.** Raising it forces the magnitude the optimiser
  chose to lower. That treats the symptom: the pathway is weak because the loss
  does not want it, not because it started small.
* **Rebuilding the sketch encoder.** It is the healthiest component measured so
  far: 4 px on held-out episodes, 1.6 px behind the raw-mask control.

## Traps found this session

* **Orbax has no subtree restore.** `PyTreeRestore` rejects any item whose tree
  does not match the on-disk one, so "read just `sketch_encoder`" is not
  available — read all 5.8 GB and index. It restores at ~730 MiB/s off the
  volume, so this costs about 9 seconds, not minutes.
* **`scp` does not work through the RunPod proxy** ("Connection closed"), and
  `ssh` without `-tt` is refused ("Your SSH client doesn't support PTY"). Ship
  files as base64 over the tty and verify by md5, the way
  `scripts/make_pod_payload.sh` does.
* **The pty eats the first line of output.** A base64 stream sent straight to
  stdout arrives without its gzip header. Buffer to a file on the pod first,
  prefix every line with a marker, and pad with two throwaway lines.
* **`logging.basicConfig` is a no-op once jax is imported** — absl installs a
  root handler first. Both probes logged nothing at all on their first run,
  including a self-check number. Pass `force=True`.
* **An A100 in CA-MTL-3 is not always rentable.** One was released between two
  measurements this session and the replacement took a while. Do not release a
  pod while questions are still open — and check first whether the measurement
  needs the GPU at all: once the attention was known to be uniform, the whole
  step-3 result needed 34M parameters and would have run on a CPU.

## Useful commands

```bash
# ship a changed probe to a pod and run it (no git credentials needed on the pod)
{ echo "stty -echo"; echo "cd /workspace/SketchPromptVLA-Pi"; \
  echo "base64 -d <<'B64' | tar xzf - -C /workspace/SketchPromptVLA-Pi"; \
  tar czf - scripts/probe_sketch_readout.py | base64; echo B64; echo exit; } > /tmp/ship.sh
ssh -tt <pod>@ssh.runpod.io -i ~/.ssh/id_ed25519 < /tmp/ship.sh

# the encoder + cross-attention probe; no backbone, so JAX_PLATFORMS=cpu is fine
CKPT=/workspace/SketchPromptVLA-Pi/checkpoints/sketchvla_finetune/pcla_v5_paired/1499
uv run scripts/probe_sketch_readout.py --params "$CKPT/params" \
  --data-dir /workspace/data/sketch_libero_rlds_paired --synthetic 2000 \
  --out /workspace/logs/readout.json

# the attention probe; this one does load the backbone
uv run scripts/probe_sketch_attention.py --checkpoint "$CKPT" \
  --data-dir /workspace/data/sketch_libero_rlds_paired --split val \
  --out /workspace/logs/attention.json
```
