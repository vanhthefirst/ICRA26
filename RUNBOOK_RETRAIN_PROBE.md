# Runbook — reopen the sketch pathway, cheaply

Aaron, 21 Aug 2026. Follow-up to `claude/sketch_pathway_diagnosis.md`.

Goal: find out whether a retrain can make the model read the sketch, without
spending a pod-day to learn it cannot. Every rung has a stop condition.

Each block says which shell it belongs in. They are not interchangeable.

| tag | what it is |
|---|---|
| **PS** | Windows PowerShell, local checkout |
| **WSL** | Ubuntu under WSL, LIBERO venv (python 3.8), simulator only |
| **POD-BASH** | pod shell, no python env activated |
| **POD-MAIN** | pod, openpi/SketchPromptVLA-Pi main env (python 3.11 + JAX), via `uv run` |
| **POD-CLIENT** | pod, `examples/libero/.venv` (python 3.8), simulator client |

Baseline to beat, from `claude/finetuned_eval_result.md`:
checkpoint-29999 scored 2.4% success, caption effect +2.4 (p = 0.31) against
pi0.5's +13.7 (p < 0.001), and `tanh(attn_gate) = 1.784e-4`.

---

## Step 0 — confirm the config fix is already in (PS, 1 min)

The LR damage from the audit has already been corrected in
`conf/fine_tune/default.yaml`. Confirm before assuming anything.

```powershell
# PS
cd C:\Users\Admin\SketchPromptVLA-Pi
Select-String -Path src\sketchvla\conf\fine_tune\default.yaml `
  -Pattern "batch_size|warmup_steps|decay_steps|num_train_steps|repo_id|sensitivity_gates"
```

Expect `batch_size: 32`, `warmup_steps: 1000`, `decay_steps: 30000`,
`num_train_steps: 30000`, `sensitivity_gates: true`.

**The one thing still wrong is `repo_id: ductaingn/sketch_libero`** — the old
auto-sketched spatial set, where every scene has one valid target and the circle
marks the object the caption already names. On that data the sketch carries no
information the caption lacks, so ignoring it costs nothing in loss and the gate
has no reason to open. Fixing the schedule without changing the data gets a
working policy that still ignores sketches.

Get the enhanced dataset id from Duc Tai before Step 3.

---

## Step 1 — re-confirm the gate on the old checkpoint (POD-MAIN, 2 min)

The number to compare everything against. CPU only, no rollout.

```bash
# POD-MAIN
cd /workspace/aaron/SketchPromptVLA-Pi
uv run scripts/probe_sketch_gates.py /workspace/aaron/checkpoints/anhdao69/sketchprompt/checkpoint-29999/params
```

Expect `tanh ≈ +0.000178`. Write it down; every later probe is read against it.

---

## Step 2 — dump the LR schedule without training (POD-MAIN, 2 min)

Catches a typo before it costs GPU hours. Warmup should end inside the first few
percent of the run and the LR should actually decay.

```bash
# POD-MAIN
cd /workspace/aaron/SketchPromptVLA-Pi
uv run python - <<'EOF'
import optax
w, peak, total, end = 1000, 5e-5, 30000, 5e-6
s = optax.warmup_cosine_decay_schedule(0.0, peak, w, total, end)
for step in (0, 500, 1000, 5000, 15000, 29999):
    print(f"{step:6d}  {float(s(step)):.3e}")
EOF
```

Want: near 0 at step 0, peak at 1000, clearly decayed by 29999. A flat column
means `decay_steps` is still wrong.

---

## Step 3 — the probe run: 1000 steps, gates every 100 (POD-MAIN, ~30 min, 1 GPU)

This is the rung that decides everything. Short run, real config, gates on.

Replace `<ENHANCED_REPO_ID>` with Duc Tai's multi-object dataset, and set the
two captions to a pair that dataset actually contains and that name **different**
objects — otherwise the language number is meaningless.

```bash
# POD-MAIN
cd /workspace/aaron/SketchPromptVLA-Pi
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run src/sketchvla/fine_tune.py \
    sketchvla=prompt_conditioned_latent_action \
    fine_tune.repo_id=<ENHANCED_REPO_ID> \
    fine_tune.exp_name=probe1k_pcla \
    fine_tune.num_train_steps=1000 \
    fine_tune.save_interval=100 \
    fine_tune.log_interval=20 \
    fine_tune.keep_period=100 \
    fine_tune.overwrite=true \
    fine_tune.sensitivity_gates=true \
    fine_tune.sensitivity_caption_a="<CAPTION NAMING OBJECT A>" \
    fine_tune.sensitivity_caption_b="<CAPTION NAMING OBJECT B>" \
    2>&1 | tee /workspace/aaron/probe1k.log
```

The gates print to the console at every save, so a wandb outage does not lose
them:

```
sensitivity @ 100: language L2 0.xxxxx, sketch L2 0.xxxxx
```

### Pass / fail

| signal | pass |
|---|---|
| `gate/language_l2` | non-zero and rising |
| `gate/sketch_l2` | clearly above its step-100 value |
| `tanh(attn_gate)` at step 1000 | above 0.01, i.e. 50x the old run |

```bash
# POD-MAIN — read the gate off the probe checkpoint
uv run scripts/probe_sketch_gates.py /workspace/aaron/SketchPromptVLA-Pi/checkpoints/probe1k_pcla/1000/params
grep "sensitivity @" /workspace/aaron/probe1k.log
```

**STOP if the gates are still pinned near zero.** A 30k run will not rescue it.
It means the data still does not force the sketch to matter, and the next move is
the dataset — or initialising `attn_gate` at 0.1 instead of 0.0 in
`src/sketchvla/models/flamingo.py` so the pathway gets gradient early. Do not go
to Step 4.

---

## Step 4 — full retrain (POD-MAIN, hours, only if Step 3 passed)

Same command without the step overrides. Gates stay on, so a mid-run collapse is
visible live instead of after the eval.

```bash
# POD-MAIN
cd /workspace/aaron/SketchPromptVLA-Pi
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run src/sketchvla/fine_tune.py \
    sketchvla=prompt_conditioned_latent_action \
    fine_tune.repo_id=<ENHANCED_REPO_ID> \
    fine_tune.exp_name=pcla_v2 \
    fine_tune.sensitivity_caption_a="<CAPTION NAMING OBJECT A>" \
    fine_tune.sensitivity_caption_b="<CAPTION NAMING OBJECT B>" \
    2>&1 | tee /workspace/aaron/pcla_v2.log
```

Watch `gate/language_l2`. If it collapses toward zero partway through, kill the
run — that is the previous failure repeating, and the rest of the steps are
wasted.

---

## Step 5 — offline check on the validation scenes (POD-MAIN, minutes)

No simulator, no policy server. Confirms the checkpoint reacts to captions and
sketches on the actual 114 scenes before booking eval time.

```bash
# POD-MAIN
uv run scripts/probe_sketch_gates.py <NEW_CKPT>/params
```

If Duc Tai's `ductaingn/sketch_libero_val` is usable by then, run the action-chunk
comparison over it as well — same scene, two captions, then same caption, two
circles.

---

## Step 6 — closed-loop eval (POD-MAIN + POD-CLIENT, hours)

Only once Steps 3–5 pass. Two panes.

```bash
# POD-MAIN — policy server, leave running
cd /workspace/aaron/SketchPromptVLA-Pi
uv run scripts/serve_policy_sketchvla.py \
    --checkpoint_dir /dev/shm/ckpt \
    --model_variant prompt_conditioned_latent_action --port 8000 --device 0
```

```bash
# POD-CLIENT — smoke first, ALWAYS
source /workspace/aaron/SketchPromptVLA-Pi/examples/libero/.venv/bin/activate
export PYTHONPATH=$PYTHONPATH:/workspace/aaron/SketchPromptVLA-Pi/third_party/libero
export MUJOCO_GL=egl
python examples/libero/eval_sketchvla.py --smoke --variant prompt_conditioned_latent_action
```

The smoke gate refuses to run if the caption does not match the manifest, if the
`real` arm renders empty masks, or if the `blank` arm renders non-empty ones.

Then the four arms, one `--run-id` each:

```bash
# POD-CLIENT
for pt in explicit ambiguous; do
  for sm in real blank; do
    python examples/libero/eval_sketchvla.py \
        --suite spatial --prompt-type $pt --sketch-mode $sm \
        --variant prompt_conditioned_latent_action \
        --run-id pcla_v2_${pt}_${sm}
  done
done
```

**Before reading any sketch result, check the caption contrast.** If explicit
minus ambiguous is not near pi0.5's +13.7, the policy still cannot tell two
captions apart and the sketch arms mean nothing.

---

## Step 7 — analysis (PS or WSL, no GPU)

```bash
# WSL (or PS with the same python)
cd /mnt/c/Users/Admin/sketch_prompted_vla
python scripts/analyze_sketchvla.py
```

Produces `analysis.json`, `tables.tex` and both figures from the rollout CSVs.
Runs anywhere, no simulator.

---

## Runs in parallel, no GPU (WSL)

The mask-format diff against `ductaingn/sketch_libero_val`: his generator's
circle/arrow masks for the 114 scenes versus `render_sketch_masks` on the same
geometry. Compare pixel count, stroke width, filled vs outline, AA edge profile.
Independent of everything above and closes the last open question with the
supervisor.
