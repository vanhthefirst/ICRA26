# HANDOFF — Sketch-Prompted VLA, session of 30 Aug 2026

`pcla_v5_paired` trained, was evaluated three ways, and **does not ground the
sketch**. The corpus and eval pipelines now work end to end. STATE.md carries
the numbers; this file carries what to do next and what will bite you.

Repos: training `SketchPromptVLA-Pi@feat/eval-harness` HEAD `7ce0130`; eval
`sketch_prompted_vla` (github `vanhthefirst/ICRA26`) branch
`sketch_prompted_vla` HEAD `afd4c55`. Both pushed. Push from WSL with
`git -c credential.helper='!/mnt/c/Program\ Files/Git/mingw64/bin/git-credential-manager.exe' push`.

## What exists now

* **Paired corpus**, `/workspace/data/sketch_libero_rlds_paired` on the CA-MTL-3
  network volume: 418 episodes, 377 train / 41 val, 21 shards, 2.2 GB, `--verify`
  passing. Built from `/workspace/data/paired_frames` (418 npz, 4.9 GB), which is
  also on the volume and is the thing to resume from.
* **Checkpoint** `pcla_v5_paired/1499`, 5.8 GB, on the volume.
* **Harness on the pod** at `/workspace/harness_repo` — 43 scene dirs +
  `evaluation_rows_all.json`. Pods never clone the eval repo; ship the 2.4 MB
  bundle over the tty (`outputs/validation_set_spatial`,
  `outputs/evaluation_rows_all.json`, `outputs/rollouts/nonreproducible.json`,
  `outputs/rollouts/pi05_explicit_532/results.csv`) and verify by md5.
* **Scripts**: `scripts/make_pod_payload.sh` brings a fresh pod to a running
  build in one stdin paste; `scripts/run_paired_finish.sh` is idempotent
  (`--resume`) and is the migration path onto a new pod.

## What we found

Training worked: `gate/sketch_l2` 0.00740 / 0.01914 / 0.02939 at 500/1000/1499,
against v4's declining 0.00714 / 0.00665 / 0.00578 and a 0.03611 signal
reference. `attn_tanh` held 0.0997 -> 0.0890, so the gate never closed.

Evaluation says that did not become behaviour. Spatial, 37 scenes x 14 rollouts,
upright frames (`--rotate180`):

| ambiguous arm | grasps | took circled bowl | took goal bowl | success |
|---|---|---|---|---|
| real (circle on goal bowl) | 495 | 292 (59.0%) | 292 (59.0%) | 38.6% |
| blank (no sketch) | 491 | -- | 293 (59.7%) | 38.8% |
| swap (circle on distractor) | 487 | 69 (14.2%) | 291 (59.8%) | 39.6% |

Explicit captions: real 45.6%, blank 45.4%. Paired by scene, real - blank is
+0.2 pts (t=0.17) explicit and -0.2 pts (t=-0.22) ambiguous; real - swap on the
goal-bowl rate is -1.0 pts (t=-0.90).

**The sketch is not ignored — it is not referential.** Serving one observation
twice, real vs zeroed sketch, moves the action chunk (mean |d| 0.047-0.069, max
0.18-0.50). So the pathway carries signal and changes the trajectory. It just
does not change *which object* is picked, wherever the circle is put. The model
resolves a deictic caption with a positional prior: ambiguous+blank is 38.8%
against the ~23% that guessing between two identical bowls would give.

Two escape hatches were closed before concluding this. It is not plumbing:
`sketch_libero_policy` reads all six sketch keys and the served actions do move.
It is not the harness: every scene's goal is `(On akita_black_bowl_1 plate_1)`,
a specific instance, with a same-type distractor present, so a wrong-bowl grasp
does fail and would be visible.

**`sketch_l2` is therefore not a success criterion.** It measures pathway
activity, not referential use, and it steered every training decision to date.
The swap arm is the metric that should gate future runs.

## Caveat on how far this generalises

This is 1500 steps on a 418-episode corpus, one suite, one checkpoint. It is
strong evidence that this recipe does not produce referential grounding. It is
**not** evidence that the architecture cannot: no attribution run was done, so
the gain over v4 is not apportioned among the three things that changed at once
(paired corpus, upright world, open gate), and no longer run was attempted. The
probes below are what would separate "cannot" from "did not, here".

Also unattempted: `explicit x swap` (predicted-null under this verdict — the
sketch fails where it is the only disambiguator, so it will not override a
caption that names the target), and the object/goal transfer suites.

## Recommended next steps, in order

1. **Linear probe on the sketch encoder.** Can a linear readout of its tokens
   recover the circled object's position? If not, the referent is discarded
   upstream and the encoder or its objective is the problem.
2. **Attention mass from the action expert onto sketch tokens.** The gate was
   open all run; the question is whether anything the action path used flowed
   through it. `src/sketchvla/sensitivity.py` already exists for this kind of
   measurement.
3. Only then decide whether the fix is the encoder, the conditioning path, or
   the training objective. **Do not run more training or more rollouts first** —
   both can only re-confirm the null.

If a fix is attempted, gate it on the swap arm, not `sketch_l2`.

## CPU / GPU requirements

* **Corpus build (stages 1-2)** is CPU physics + GPU rendering. It does **not**
  need a big GPU — any card with a working NVIDIA EGL will do; rent the cheap
  pod. It needs cores: 418 episodes took ~3 min across 30 processes once
  rendering was on the GPU.
* **Packing (stage 3)** is single-core TensorFlow, no GPU. ~5 min for 418
  episodes. Parallelising it across episodes is an easy win if the corpus grows.
* **Training** wants one A100 80 GB (or H100/H200): batch 32, `llm|img` frozen,
  ~73 GB reserved at `XLA_PYTHON_CLIENT_MEM_FRACTION=0.9`, 1500 steps in ~1.5 h.
  **Use exactly one GPU** — two would shard the batch and break comparability
  with v4/v5.
* **Evaluation** on 1x A100 80 GB with ~26 usable vCPU: 4 servers x 12 client
  shards, ~7.6 rollouts/min, so ~68 min per 518-rollout arm. Servers need ~8.6 GB
  each; the checkpoint is staged once to `/dev/shm` (10 s) and every server loads
  from there — never let servers read the network volume concurrently.
* **Region matters more than the card.** RunPod network volumes do not cross
  regions. The volume is in **CA-MTL-3**; a pod anywhere else cannot see the
  corpus, the checkpoint or the venvs.

## Traps, all of them measured this session

* **MuJoCo renders on the CPU unless you make it not.** A pod may ship the
  `libegl1` dispatcher without the NVIDIA vendor ICD, or the reverse. Install
  `libegl1` **and** pin
  `__EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/10_nvidia.json`
  (apt installs the mesa vendor beside it). GPU: 0.8 ms/frame. llvmpipe: it was
  the entire cost of the build, ~10 min/episode on four starved shards.
* **`nproc` lies inside the container**: 252 host cores against a `cpu.max`
  quota of 26.35. Size any fan-out from the cgroup quota, or llvmpipe will spawn
  ~640 worker threads onto ~26 CPUs.
* **`np.load` on a compressed npz is lazy.** Indexing `z["images"][t]` in a loop
  re-inflates the whole stack per frame: 584 GB read from a 7.25 GB corpus, 80
  minutes, no output. Hoist arrays out of the `NpzFile` once.
* **The RunPod proxy echoes stdin back down the pty.** `stty -echo` first, or a
  45 KB payload takes >10 min instead of 26 s. It also echoes your script text
  back, so a watcher that greps for its own marker string will match the echo —
  split marker literals and parse only after a `@@@BEGIN` sentinel.
* **`pkill -f run_eval.sh` does not kill the driver reliably.** Two parents once
  ran the same arm concurrently at 24 clients on 26 CPUs. Kill by pid from
  `pgrep -af "bash examples/libero/run_eval.sh"` and verify zero afterwards.
* **LIBERO's first-import prompt** writes `~/.libero/config.yaml` to the pod's
  ephemeral disk, so every fresh pod asks again and any process with stdin on
  `/dev/null` dies with `EOFError`. `run_paired_finish.sh` now answers it once.
* **A failed stage must not fall through to packing.** It once wrote 273 of 418
  episodes into an RLDS that looked complete.
* Launch anything long with `setsid nohup ... < /dev/null &` and exit cleanly;
  the proxy kills children of a SIGTERMed session.

## Useful commands

```bash
# fresh pod -> running resume build, one paste
bash scripts/make_pod_payload.sh > /tmp/pod_boot.sh
ssh <pod>@ssh.runpod.io -i ~/.ssh/id_ed25519 < /tmp/pod_boot.sh

# eval, v5 or later MUST set ROTATE180=1
REPO=/workspace/SketchPromptVLA-Pi HARNESS_REPO=/workspace/harness_repo \
CKPT_SRC=.../pcla_v5_paired/1499 DEVICES=0 SERVERS_PER_GPU=4 CLIENTS_PER_GPU=12 \
ROTATE180=1 N_ROLLOUTS=14 PROMPT_TYPES="ambiguous explicit" SKETCH_MODES=swap \
SUITES=spatial bash examples/libero/run_eval.sh
```
