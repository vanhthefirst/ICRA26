# Runbook — eval pod setup (SketchPromptVLA fine-tuned model)

Delta on top of `RUNPOD_SETUP.md`. That document still governs the account
context, SSH key registration, and the terminate-don't-stop rule. This one
covers what is different for the evaluation run: a second repo, two Python
environments, three variant checkpoints, and a multi-GPU pod.

Every command block is labelled with the shell it belongs to. They are not
interchangeable.

Target: **8x A40**, one pod, network volume attached at `/workspace`.

Why A40 and not something faster: the LIBERO client venv pins
`torch==1.11.0+cu113`, which has kernels for Ampere (`sm_86`) and nothing
newer. A40 is `sm_86`. RTX 4090 and L40S are `sm_89`; H100/H200/Blackwell are
further out still. Ada and later may work, because the client mostly imports
torch rather than running it on the GPU, but that is a gamble to take on a
smoke pod, not on the fleet.

---

## Phase 0 — Windows, before touching RunPod

### 0.1 Fork the model repo [PowerShell]

`C:\Users\Admin\SketchPromptVLA-Pi-main` is an unzipped GitHub download — it has
no `.git`, so nothing can be pushed from it and nothing Claude writes on the pod
can come back through it. Fork the upstream on GitHub first:

    https://github.com/ductaingn/SketchPromptVLA-Pi  ->  Fork

The pod clones your fork. Work stays on a branch and comes back by PR.

If you have local edits in the unzipped folder that matter, diff them against
the fork after cloning and re-apply. Check first:

```powershell
# [PowerShell]
Get-ChildItem C:\Users\Admin\SketchPromptVLA-Pi-main -Force -Name .git
# no output = clean zip, nothing local to rescue
```

### 0.2 Push the harness repo [PowerShell]

```powershell
# [PowerShell]
cd C:\Users\Admin\sketch_prompted_vla
git status
git add -A
git commit -m "chore: sync before eval pod"
git push
```

The pod needs `scripts/analyze_baselines.py`, `scripts/rollout_sketch.py`,
`SCHEMA.md`, `PROMPT_TAXONOMY.md` and the `outputs/` manifests. If it is not
pushed, Claude cannot match the CSV schema.

### 0.3 Collect the secrets you will paste on the pod

- GitHub personal access token, scope `repo` (for cloning your private fork).
- HuggingFace token — `SKETCHVLA_HF_TOKEN`, needs read access to
  `anhdao69/sketchprompt` and `ductaingn/sketch_libero_rlds`.
- WandB key is **not** needed for evaluation. Leave it out.

Keep them in a password manager, not in a file on the volume.

---

## Phase 1 — RunPod console

### 1.1 Resize the network volume

Storage -> Network Volumes -> your 150 GB volume -> resize to **300 GB**.

Three variant checkpoints plus the LeRobot dataset plus rollout videos will not
fit in 150 GB, and the volume is shared with a teammate. Resize only goes
upward, bills from the moment it applies, and must be done with **no pod
attached**.

### 1.2 Check GPU availability in the volume's region

The volume lives in EU-RO-1. A volume only attaches to GPUs in its own
datacenter, so filter the deploy page to that region and confirm 8x A40 is
offered there before anything else.

If 8x A40 is not available in EU-RO-1, in order of preference:

1. Take 4x A40 there and accept roughly double the wall clock.
2. Create a **second** 300 GB volume in a region that does have 8x A40, and
   re-download the checkpoints there. Costs an hour, not a day.
3. Fall back to 7x L40S in EU-RO-1 and smoke-test the Python 3.8 client venv
   before committing (see the `sm_89` note at the top).

Do not solve this by dropping the volume. Everything outside `/workspace` dies
on terminate.

### 1.3 Deploy

- **GPU:** 8x A40
- **Region:** must match the volume
- **Template:** official *RunPod PyTorch 2.x, CUDA 12.x*. Leave the container
  start command at its default.
- **Network volume:** attach, mount at `/workspace`
- **Container disk:** default

---

## Phase 2 — Connect and verify

```powershell
# [PowerShell] — copy this string verbatim from the pod's Connect dialog,
# including the -<hash> half of the username
ssh <pod-id>-<hash>@ssh.runpod.io -i $env:USERPROFILE\.ssh\id_ed25519
```

Then, on the pod:

```bash
# [RunPod SSH]
nvidia-smi --query-gpu=index,name,memory.total --format=csv   # expect 8 x A40, 46068 MiB
nproc                                                          # expect ~72
free -g                                                        # expect ~400 GB
df -h /workspace                                               # volume mounted
ls /workspace/aaron                                            # your previous work is there
```

If `nproc` comes back at 9, you got a single-GPU pod. Stop and redeploy — the
whole plan depends on the core count.

---

## Phase 3 — Rebuild the container-disk half

```bash
# [RunPod SSH]
bash /workspace/aaron/sketch_prompted_vla/scripts/pod_bootstrap.sh
```

That reinstalls apt basics, Node 22 and Claude Code, which the container disk
loses on every terminate. Then add the graphics libraries MuJoCo needs for
headless EGL rendering — with 48 sim processes this is not optional:

```bash
# [RunPod SSH]
apt-get update && apt-get install -y \
  libegl1 libgl1 libglew-dev libosmesa6-dev patchelf ffmpeg
```

---

## Phase 4 — Repos

```bash
# [RunPod SSH]
cd /workspace/aaron

# harness repo: refresh if present, clone if not
cd sketch_prompted_vla && git pull && cd ..

# model repo: your fork, with submodules
git clone --recurse-submodules https://github.com/<your-user>/SketchPromptVLA-Pi.git
cd SketchPromptVLA-Pi
git submodule update --init --recursive
git checkout -b feat/eval-harness
```

Use the GitHub PAT as the password if it prompts.

Verify the submodule actually landed — an empty `third_party/libero` is the most
common silent failure here:

```bash
# [RunPod SSH]
ls third_party/libero/libero | head
```

---

## Phase 5 — Environments

Two separate Python environments. They do not share packages and must not be
merged.

### 5.1 uv

```bash
# [RunPod SSH]
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
uv --version
```

### 5.2 Main env — Python 3.11, serves the policy

```bash
# [RunPod SSH]
cd /workspace/aaron/SketchPromptVLA-Pi
GIT_LFS_SKIP_SMUDGE=1 uv sync --all-groups
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
```

`GIT_LFS_SKIP_SMUDGE=1` is required both times — LeRobot is pulled from git and
its LFS objects are not needed.

### 5.3 LIBERO client env — Python 3.8, runs the simulator

```bash
# [RunPod SSH]
cd /workspace/aaron/SketchPromptVLA-Pi
uv venv --python 3.8 examples/libero/.venv
source examples/libero/.venv/bin/activate
uv pip sync examples/libero/requirements.txt third_party/libero/requirements.txt \
  --extra-index-url https://download.pytorch.org/whl/cu113 \
  --index-strategy=unsafe-best-match
uv pip install -e packages/openpi-client
uv pip install -e third_party/libero

# the pinned opencv-python wants GUI libraries a pod does not have
uv pip uninstall opencv-python
uv pip install opencv-python-headless==4.6.0.66
deactivate
```

### 5.4 Persistent environment variables

```bash
# [RunPod SSH]
cat >> ~/.bashrc <<'EOF'
export OPENPI_DATA_HOME=/workspace/aaron/.cache/openpi
export HF_HOME=/workspace/aaron/.cache/huggingface
export MUJOCO_GL=egl
export PYTHONPATH=$PYTHONPATH:/workspace/aaron/SketchPromptVLA-Pi/third_party/libero
export TOKENIZERS_PARALLELISM=false
EOF
source ~/.bashrc
```

`MUJOCO_GL=egl` because a pod has no X server. The openpi docs suggest `glx`
when EGL errors appear; on a pod that advice is inverted.

`HF_HOME` on the volume matters — otherwise every terminate re-downloads tens of
GB.

---

## Phase 6 — Checkpoints and data

```bash
# [RunPod SSH]
export SKETCHVLA_HF_TOKEN="hf_..."
huggingface-cli login --token "$SKETCHVLA_HF_TOKEN"

mkdir -p /workspace/aaron/checkpoints
huggingface-cli download anhdao69/sketchprompt \
  --local-dir /workspace/aaron/checkpoints/anhdao69/sketchprompt
```

Then look at what actually came down, because it decides how Claude wires the
`--variant` flag:

```bash
# [RunPod SSH]
find /workspace/aaron/checkpoints/anhdao69/sketchprompt -maxdepth 2 -type d
du -sh /workspace/aaron/checkpoints/anhdao69/sketchprompt
```

Two things to confirm and report to Claude:

1. Whether the three variants (`input_overlay`,
   `prompt_conditioned_latent_action`, `visual_prompt_tokens`) are separate
   directories or separate revisions of one tree.
2. Whether each checkpoint carries its own `assets/` with norm stats. Inference
   needs them. If they are missing, the dataset download below is mandatory and
   `src/sketchvla/utils/compute_norm_stats.py` has to run first.

The dataset is only needed for `validate.py` (offline metrics) and for norm
stats. Closed-loop rollouts do not touch it. Skip it if you only want the
rollout numbers today:

```bash
# [RunPod SSH] — optional, large
huggingface-cli download ductaingn/sketch_libero_rlds \
  --local-dir /workspace/aaron/data/sketch_libero_rlds --repo-type dataset
```

---

## Phase 7 — Gate before you call Claude in

Three checks. Each one fails cheaply now and expensively later.

**7.1 The policy server loads a checkpoint and binds a port:**

```bash
# [RunPod SSH]
cd /workspace/aaron/SketchPromptVLA-Pi
uv run scripts/serve_policy_sketchvla.py \
  --checkpoint_dir /workspace/aaron/checkpoints/anhdao69/sketchprompt/checkpoint-29999 \
  --port 8000
```

Wait for it to report it is listening, then Ctrl-C. If it dies on missing norm
stats, that is Phase 6 item 2 — resolve it before continuing.

**7.2 The client env can build a LIBERO scene headlessly:**

```bash
# [RunPod SSH]
source /workspace/aaron/SketchPromptVLA-Pi/examples/libero/.venv/bin/activate
python -c "
import os; os.environ['MUJOCO_GL']='egl'
from libero.libero import benchmark
b = benchmark.get_benchmark_dict()['libero_spatial']()
print('tasks:', b.n_tasks)
"
deactivate
```

**7.3 All 8 GPUs are individually addressable:**

```bash
# [RunPod SSH]
for i in $(seq 0 7); do
  CUDA_VISIBLE_DEVICES=$i python -c "import torch;print($i, torch.cuda.get_device_name(0))"
done
```

---

## Phase 8 — Hand over to Claude Code

```bash
# [RunPod SSH]
tmux new -s eval
cd /workspace/aaron          # both repos sit under here, so Claude can see both
claude
```

First run prints a login URL — open it on Windows, paste the code back.

Then paste the implementation prompt. Before you do, append these four facts to
it, because they are pod-specific and Claude cannot discover them:

- Model repo: `/workspace/aaron/SketchPromptVLA-Pi` (branch `feat/eval-harness`)
- Harness repo: `/workspace/aaron/sketch_prompted_vla`
- Checkpoints: `/workspace/aaron/checkpoints/anhdao69/sketchprompt`, layout as
  you found it in Phase 6
- Hardware: 8x A40, 72 vCPU, ~400 GB RAM; target 6 policy servers and 6 LIBERO
  clients per GPU, ports 8000-8047

Reattach after any dropped connection with `tmux attach -t eval`. The run is
hours long; do not rely on the SSH session staying up.

---

## When you are done

```bash
# [RunPod SSH]
cd /workspace/aaron/SketchPromptVLA-Pi && git push -u origin feat/eval-harness
cd /workspace/aaron/sketch_prompted_vla && git add outputs report && git commit -m "results: fine-tuned eval" && git push
```

Then **stop, then terminate** the pod. Stop alone keeps billing disk while
preserving nothing extra — the container disk is wiped by both.

Surviving a terminate: everything under `/workspace/aaron/` — both repos, both
venvs, the HF cache, the checkpoints, the results. Not surviving: apt packages,
Node, Claude Code and its login token. Phase 3 rebuilds those in about two
minutes.
