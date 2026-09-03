# V7 referent-grounding handoff — checkpoint 2999, rollout row 80

Status captured 2026-09-03 (Asia/Bangkok). The RunPod pod must remain running.

## Connect

```bash
ssh x8rj511ymb6602-64410cb0@ssh.runpod.io -i ~/.ssh/id_ed25519
```

- Pod: A100 PCIe 80 GB
- Network volume: `hao7ye6xly`, mounted at `/workspace`
- Do not print `/workspace/env.sh`; it contains a GitHub token.

## Current live state

- The rollout client was stopped cleanly at exactly **80/400 rows**.
- No `eval_paired_referent.py` process should be running.
- The checkpoint-2999 policy server is intentionally still running:
  - PID: `33708` (verify rather than assuming it is unchanged)
  - Port: `8200`
  - Log: `/workspace/logs/v7_step2999_server_8200.log`
- Do not stop the pod. Do not start a duplicate policy server while port 8200 is live.

Preflight:

```bash
pgrep -af 'eval_paired_referent.py' || true
ss -ltnp | grep ':8200 ' || true
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader
```

## Training and checkpoint recovery

Training `rg_v7_paired` resumed from step 2700 and reached the intended final step. OpenPI numbers the final 3000-step checkpoint as **2999**.

- Final checkpoint: `/workspace/SketchPromptVLA-Pi/checkpoints/sketchvla_finetune/rg_v7_paired/2999`
- Size/files: about 5.8 GB, 59 files
- Orbax `_CHECKPOINT_METADATA` commit timestamp: `1788361963449543437` (non-null)
- Orbax logged all handler commits and the final atomic rename; the training process exited cleanly.
- Recovery log: `/workspace/logs/rg_v7_paired_recover2999_0902_1447.log`
- Training repository commit: `a6bc2e7` (clean during training/probing)
- W&B run: `fam52cpx`
- Peak observed aggregate cgroup RAM during the successful final save was about 80.4 GiB, below the 108.96 GiB hard limit.

The earlier RunPod OOM alert came from an incomplete checkpoint-save attempt. It did **not** corrupt the finalized checkpoint. Its partial data was preserved at:

`/workspace/stash_archive/rg_v7_paired_2999_partial_20260902T115214Z`

The successful recovery kept the same training batch and optimizer settings, with only `fine_tune.num_workers=0` and `fine_tune.grounding_gates=false` to reduce host RAM; the redundant grounding gate was run offline immediately afterward.

## Offline grounding gates — passed

Grounding probe:

- JSON: `/workspace/SketchPromptVLA-Pi/outputs/grounding_v7_step2999.json`
- Log: `/workspace/logs/v7_probe_grounding_step2999_0902.log`
- 41 held-out validation episodes, frame 0, pinned noise
- `point_hit_real=0.980076`
- `point_hit_swap=0.972405`
- `point_hit_wrong_real=0.011714`
- `point_hit_wrong_swap=0.013384`
- `follow_ratio=0.996148`
- `circle_travel=0.626502`
- `chunk_swap_delta=0.644562`
- `chunk_blank_delta=0.336384`
- `swap_over_blank=1.916148`

Paired action validation:

- JSON: `/workspace/SketchPromptVLA-Pi/outputs/validation_v7_step2999_paired.json`
- Log: `/workspace/logs/v7_validate_step2999_paired_0902.log`
- 20 held-out validation episodes, `--ablate-sketch --swap-sketch --frames-per-episode 1`
- Real L1: `0.098311`
- Blank L1: `0.104107` (delta `+0.005796`, `sketch_helps=true`)
- Swap L1: `0.119368` (delta `+0.021057`)
- `swap_over_blank=3.63309`
- Gripper-sign accuracy: `1.0`

These gates authorized simulator rollout evaluation.

## Exact paired-layout evaluator

The old exact paired arm used `sketch_mode="none"`, so it could not test V7 referent grounding. A small separate evaluator was added:

- Remote: `/workspace/eval_scripts/scripts/eval_paired_referent.py`
- Local: `C:/Users/Admin/sketch_prompted_vla/scripts/eval_paired_referent.py`
- Base evaluation repository commit: `29ca071`
- The evaluator file is currently **untracked**, so the remote evaluation repository correctly reports dirty provenance.

It reconstructs the accepted ten paired layouts, loads the stored real or swapped sketch masks from `/workspace/data/paired_frames_cf`, verifies the reconstructed frame against packed frame 0, and sends that sketch to the V7 websocket server.

Metric semantics:

- `real`: desired referent is `akita_black_bowl_1`; report both BDDL task `success` and `referent_success`.
- `swap`: desired referent is `akita_black_bowl_2`; BDDL still describes the original goal, so judge `referent_success` and `wrong_bowl`, not BDDL `success`. The swap arm stops after the first bowl lift above 3 cm.
- A generic BDDL success can accept the wrong identical bowl; never treat `success` alone as grounding evidence.

The restarted disposable container had lost generic EGL loader packages. They were restored with `libegl1`, `libgles2`, and `libegl-mesa0`. The persistent data and environments were unaffected.

## Rollout checkpoint 80/400

Completed block: **real sketches, t1–t4, 20 episodes per task**.

- Snapshot JSON: `/workspace/SketchPromptVLA-Pi/outputs/v7_paired_step2999_chunk01_real_t1_t4.json`
- Snapshot log: `/workspace/logs/v7_paired_step2999_chunk01_real_t1_t4.log`
- Original incremental JSON (also 80 rows): `/workspace/SketchPromptVLA-Pi/outputs/v7_paired_step2999_all10.json`
- Original log: `/workspace/logs/v7_paired_step2999_all10.log`
- JSON SHA-256: `c8e6add318456fffc1c04cf04cded9b99e2e466460f9b344cc96929d1fd1e7cb`
- Log SHA-256: `8f72b61128df390d4ba1eff4af84fd2049c5aeecae961577304794bfb3997dca`

| Task | BDDL success | Correct referent | Wrong bowl | No grasp |
| --- | ---: | ---: | ---: | ---: |
| t1 | 20/20 | 20/20 | 0/20 | 0/20 |
| t2 | 20/20 | 19/20 | 1/20 | 0/20 |
| t3 | 17/20 | 18/20 | 1/20 | 1/20 |
| t4 | 20/20 | 20/20 | 0/20 | 0/20 |
| **All** | **77/80 (96.25%)** | **77/80 (96.25%)** | **2/80 (2.5%)** | **1/80 (1.25%)** |

Every completed real-sketch task is at or above the target 80% task-success floor (`t3=85%`). Maximum frame reconstruction mean RGB error was 3.51153, under the 5.0 guard.

## Continue in 80-row checkpoints

Use one evaluator client at a time. Leave the server on port 8200 running. Common prefix for every chunk:

```bash
cd /workspace/eval_scripts
export PYTHONPATH=/workspace/SketchPromptVLA-Pi/third_party/libero
export MUJOCO_GL=egl
export __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/10_nvidia.json

PY=/workspace/SketchPromptVLA-Pi/examples/libero/.venv/bin/python
SCRIPT=scripts/eval_paired_referent.py
BDDL=/workspace/SketchPromptVLA-Pi/third_party/libero/libero/libero/bddl_files/libero_spatial
DEMOS=/workspace/demos
FRAMES=/workspace/data/paired_frames_cf
CKPT=/workspace/SketchPromptVLA-Pi/checkpoints/sketchvla_finetune/rg_v7_paired/2999
```

Run detached (`setsid -f`) and redirect each invocation to the named log. Use these blocks:

1. **Rows 81–160:** `--tasks t5,t6,t7,t8 --sketch-modes real --episodes 20`
   - JSON: `/workspace/SketchPromptVLA-Pi/outputs/v7_paired_step2999_chunk02_real_t5_t8.json`
   - Log: `/workspace/logs/v7_paired_step2999_chunk02_real_t5_t8.log`
2. **Rows 161–240:** two 40-row invocations, sequentially:
   - real `t9,t10` → `...chunk03a_real_t9_t10.json` / corresponding log
   - swap `t1,t2` → `...chunk03b_swap_t1_t2.json` / corresponding log
3. **Rows 241–320:** swap `t3,t4,t5,t6` → `...chunk04_swap_t3_t6.json` / corresponding log
4. **Rows 321–400:** swap `t7,t8,t9,t10` → `...chunk05_swap_t7_t10.json` / corresponding log

Full invocation template:

```bash
setsid -f env \
  PYTHONPATH="$PYTHONPATH" MUJOCO_GL="$MUJOCO_GL" \
  __EGL_VENDOR_LIBRARY_FILENAMES="$__EGL_VENDOR_LIBRARY_FILENAMES" \
  "$PY" "$SCRIPT" \
  --bddl-dir "$BDDL" --demo-dir "$DEMOS" --frames-dir "$FRAMES" \
  --checkpoint "$CKPT" --variant referent_grounding \
  --host 127.0.0.1 --port 8200 \
  --tasks TASK_LIST --sketch-modes MODE --episodes 20 --max-steps 520 \
  --out OUTPUT_JSON > OUTPUT_LOG 2>&1
```

At cumulative 160, 240, 320, and 400 rows, verify row counts, check for tracebacks, record SHA-256 hashes, and preserve chunk-specific copies. Do not overwrite chunk 1.

After all 400 rows, merge the five chunk groups into one provenance-stamped artifact. Report:

- Real arm: BDDL success, correct-referent rate, wrong-bowl rate per task and overall.
- Swap arm: correct swapped-referent rate and wrong-bowl rate per task and overall; do not use BDDL success as the swap grounding score.
- Frame-error maximum and runtime errors.
- Final decision against approximately 80% or better per real task, low wrong-bowl rate, and meaningful sketch-following under swap.

Do not stop the pod or policy server until the owner has reviewed the final artifacts, unless explicitly instructed.

## Earlier accepted assets

- Fresh corpus: `/workspace/data/paired_frames_cf`
- Packed RLDS: `/workspace/data/sketch_libero_rlds_paired_cf`
- 429 episodes: 387 train, 42 validation; counterfactual fields 429/429
- Worst augmentation alignment error: 0.05 px
- Paired physical-layout acceptance: `/workspace/SketchPromptVLA-Pi/outputs/paired_all10_t4_t6.json`
- Acceptance log: `/workspace/logs/v7_paired_all10_t4_t6.log`
- Acceptance: 96.5% success, 1.0% wrong-bowl, every task at least 85%; `t4=20/20`

## Remaining implementation-plan work

Training, final-checkpoint verification, offline grounding probes, blank-sketch validation, and offline swap validation are complete. The only active plan item is finishing the remaining 320 simulator rows, merging/scoring them, and deciding whether V7 satisfies rollout criteria. Any subsequent rollout or deployment decision should be based on the identity-aware metrics above, not generic LIBERO success alone.
