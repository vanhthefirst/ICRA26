# Runbook — eval pod (SketchPromptVLA fine-tuned model)

Second edition, 18 August 2026. The first edition got the setup right and the
*run* wrong: it treated the network volume as ordinary disk. It is not, and that
cost a wedged mount and a pod restart. Section F is the part that matters.

Delta on `RUNPOD_SETUP.md`, which still governs account context, SSH keys, and
the terminate-don't-stop rule.

---

## The one rule

**`/workspace` is MooseFS — a shared network filesystem over FUSE. It is for
durable storage between runs, never for the hot path of a run.**

Everything that many processes touch at once must live on local disk or in RAM:
the checkpoint, lock directories, caches, results as they are written. Sixty-four
servers reading one 6.5 GB checkpoint off the volume saturated it, then wedged
the FUSE client entirely — `ls /workspace/aaron` hung for ten minutes with zero
processes running, and only a pod restart cleared it.

Two consequences that are not obvious:

- Processes blocked on FUSE sleep in **`S` state, not `D`**. `ps` will not look
  like I/O contention. Do not rule I/O out on that basis, as I did.
- Any shell whose working directory is under `/workspace` hangs too, including
  tmux windows. **Keep one SSH session sitting at `/` at all times.** It is the
  only shell that still answers when the mount goes.

---

## A. Windows, before the pod

Fork `ductaingn/SketchPromptVLA-Pi` (done: `vanhthefirst/SketchPromptVLA-Pi`).
Push the harness repo. Have a GitHub PAT with `repo` scope — one classic token
covers both repos.

## B. Storage and pod

- **Network volume: 150 GB.** Measured need is ~50 GB: main venv 15–20 GB,
  client venv 4–5 GB, LIBERO assets ~1 GB, checkpoint ~10 GB, dataset 2.2 GB.
  A dedicated volume, deletable when the job is done — do not resize the shared
  team volume, which cannot be shrunk.
- **GPU: 8× A40.** Not a speed choice. `examples/libero/requirements.txt` pins
  `torch==1.11.0+cu113`, which has kernels for Ampere (`sm_86`) and nothing
  newer. 4090 and L40S are `sm_89`; H100 and up further still.
- Region must match the volume. A fresh volume can go wherever the GPUs are —
  that flexibility is the main reason to make one rather than reuse the old.
- Official RunPod PyTorch template, default start command, volume at
  `/workspace`.

Observed on the real pod: 8× A40 46 GB, 96 vCPU, 503 GB RAM.

## C. First contact

Copy the SSH string from the Connect dialog verbatim, including the `-<hash>`
half of the username. It changes across a stop/start.

```bash
# [RunPod SSH] — stay at / for this
timeout 10 ls /workspace/aaron; echo "exit=$?"   # want exit=0, ALWAYS use timeout
nvidia-smi --query-gpu=index,name,memory.total --format=csv
nproc                                            # 96 on 8x A40
```

`exit=124` means the mount is wedged; nothing else will work. Stop and start the
pod.

## D. Container disk

`pod_bootstrap.sh` rebuilds the half that dies with every pod. Three quirks on a
**fresh** volume:

1. It errors out if `/workspace/aaron` is missing — `mkdir -p` first.
2. Its clone specifies no branch. Clone the harness repo yourself first, and it
   will `git pull` instead.
3. Its `~/.libero/config.yaml` seeding looks for `$AARON_ROOT/openpi`, which does
   not exist — the model repo is `SketchPromptVLA-Pi`. It silently skips, and
   LIBERO then blocks on an interactive prompt at first import. Seed it by hand.

```bash
# [RunPod SSH]
mkdir -p /workspace/aaron
apt-get update && apt-get install -y curl git tmux libgl1 ffmpeg
git config --global credential.helper store
git config --global user.name "vanhthefirst"
git config --global user.email "dongolac@gmail.com"

cd /workspace/aaron
git clone -b sketch_prompted_vla https://github.com/vanhthefirst/ICRA26.git sketch_prompted_vla
bash /workspace/aaron/sketch_prompted_vla/scripts/pod_bootstrap.sh

LIBERO_ROOT=/workspace/aaron/SketchPromptVLA-Pi/third_party/libero/libero/libero
mkdir -p ~/.libero
cat > ~/.libero/config.yaml <<EOF
assets: $LIBERO_ROOT/./assets
benchmark_root: $LIBERO_ROOT
bddl_files: $LIBERO_ROOT/./bddl_files
datasets: $LIBERO_ROOT/../datasets
init_states: $LIBERO_ROOT/./init_files
EOF
```

Environment. **`OPENPI_DATA_HOME` must be local** — it holds `filelock` lock
files, and locks on a network mount are the fastest route back to a wedge.
`pod_bootstrap.sh` sets it to the volume, so override it after.

```bash
# [RunPod SSH]
cat >> ~/.bashrc <<'EOF'
export OPENPI_DATA_HOME=/root/.cache/openpi
export HF_HOME=/workspace/aaron/.cache/huggingface
export PYTHONPATH=$PYTHONPATH:/workspace/aaron/SketchPromptVLA-Pi/third_party/libero
export TOKENIZERS_PARALLELISM=false
export PATH="$HOME/.local/bin:$PATH"
EOF
source ~/.bashrc
```

Leave `UV_PYTHON_INSTALL_DIR` on the volume — the Python 3.8 interpreter lives
there and the client venv references that path.

## E. Repos, environments, weights

Only on a fresh volume. All of this survives a pod restart.

```bash
# [RunPod SSH]
cd /workspace/aaron
git clone --recurse-submodules https://github.com/vanhthefirst/SketchPromptVLA-Pi.git
cd SketchPromptVLA-Pi
git checkout feat/eval-harness
ls third_party/libero/libero | head        # must not be empty
```

Main env, Python 3.11:

```bash
# [RunPod SSH]
GIT_LFS_SKIP_SMUDGE=1 uv sync --all-groups
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
```

Client env, Python 3.8. The absolute-path lines in the requirements lock were
removed on `feat/eval-harness`, so this now works from a clean clone:

```bash
# [RunPod SSH]
uv venv --python 3.8 examples/libero/.venv
source examples/libero/.venv/bin/activate
uv pip sync examples/libero/requirements.txt third_party/libero/requirements.txt \
  --extra-index-url https://download.pytorch.org/whl/cu113 \
  --index-strategy=unsafe-best-match
uv pip install -e packages/openpi-client
uv pip install -e third_party/libero
uv pip uninstall opencv-python
uv pip install opencv-python-headless==4.6.0.66
deactivate
```

Weights. `anhdao69/sketchprompt` is public — no login. The repo holds one
checkpoint, `checkpoint-29999`, ~10 GB, and it is the
**`prompt_conditioned_latent_action`** variant. `ductaingn/SketchPromptedVLA` is
empty; the other two variants are not trained yet.

```bash
# [RunPod SSH]
uv tool install "huggingface_hub[cli]"
hf download anhdao69/sketchprompt \
  --local-dir /workspace/aaron/checkpoints/anhdao69/sketchprompt
hf download ductaingn/sketch_libero_rlds \
  --local-dir /workspace/aaron/data/sketch_libero_rlds --repo-type dataset
```

Do not try `--include "a" "b" "c"`. The `hf` CLI reads trailing positional
arguments as explicit filenames, discards `--include`, and downloads almost
nothing while reporting success. Repeat the flag if you must filter, or just
pull the lot.

Norm stats ship inside the checkpoint at `assets/ductaingn/sketch_libero`, so
inference is self-contained. The dataset is only for offline validation.

## F. Running — the part the first edition got wrong

**Before any server starts**, stage the checkpoint into RAM. One read off the
volume instead of sixty-four:

```bash
# [RunPod SSH]
df -h /dev/shm
mkdir -p /dev/shm/ckpt
cd /workspace/aaron/checkpoints/anhdao69/sketchprompt/checkpoint-29999
time cp -r params assets _CHECKPOINT_METADATA /dev/shm/ckpt/
echo prompt_conditioned_latent_action > /dev/shm/ckpt/variant.txt
```

`/dev/shm` is tmpfs, sized at half of 503 GB. It dies with the pod, which is
correct — it is a cache, and re-staging costs one copy.

Requirements on the harness itself:

- Checkpoint path defaults to `/dev/shm/ckpt`.
- Servers start **sequentially, with a readiness poll on each port**, not 64 at
  once behind a fixed sleep.
- `SERVERS_PER_GPU` is a variable at the top, default **4**. Eight is unproven.
- `results.csv` and any video go to **local scratch**, copied to `/workspace`
  only when an arm completes.
- Always pass `--model_variant`. Without it the server reads the full 6.5 GB
  params purely to inspect the keys, discards them, loads again — and its
  fallback silently defaults to `input_overlay`, which would load the wrong
  architecture with no error.

Single-server smoke, ~7 seconds to ready from tmpfs:

```bash
# [RunPod SSH] — main 3.11 env
cd /workspace/aaron/SketchPromptVLA-Pi
uv run scripts/serve_policy_sketchvla.py \
  --model_variant prompt_conditioned_latent_action \
  --checkpoint_dir /dev/shm/ckpt --device 0 --port 8000
```

Ready is `server listening on 0.0.0.0:8000`, preceded by `Loaded norm stats
from ...`. The three cuDNN/cuFFT/cuBLAS "Unable to register factory" errors at
startup are TensorFlow and JAX colliding; they are noise.

Client env gate:

```bash
# [RunPod SSH] — 3.8 client env
source examples/libero/.venv/bin/activate
python - <<'PY'
import os, pathlib
os.environ["MUJOCO_GL"] = "egl"
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
bd = benchmark.get_benchmark_dict()["libero_spatial"]()
t = bd.get_task(0)
f = pathlib.Path(get_libero_path("bddl_files")) / t.problem_folder / t.bddl_file
env = OffScreenRenderEnv(bddl_file_name=str(f), camera_heights=256, camera_widths=256)
env.seed(0); obs = env.reset()
print("tasks:", bd.n_tasks, "| render ok:", obs["agentview_image"].shape)
env.close()
PY
deactivate
```

Then launch from a shell with **no venv active** — `run_eval.sh` switches
environments internally.

Watch, from the lifeline shell at `/`:

```bash
# [RunPod SSH]
uptime                                   # healthy is < 96 on this box
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv
```

The tell that startup is failing: one GPU holding ~8.5 GB and the other seven at
0 MiB, minutes in. That is servers queued on the volume, not servers warming up.
Kill and re-stage.

## G. Recovering a wedged mount

Symptoms: `ls /workspace/...` never returns; tmux windows unresponsive; `uv`
hangs; shells fine only at `/`.

1. `pkill -9` the workload. Note that a process blocked inside a FUSE call
   cannot be killed until the call returns, so a non-zero count is expected.
2. `timeout 10 ls /workspace/aaron; echo "exit=$?"` — `124` means still wedged.
3. Wait three minutes and retry once.
4. Still wedged: **Stop then Start the pod** from the console. Not Terminate.
   The volume's stored data is fine; the wedge is client-side. Rebuild the
   container disk with section D.

Before restarting, confirm nothing is stranded — everything of value should
already be on GitHub:

```bash
# [RunPod SSH] — works even when the mount is dead, it is network only
git ls-remote https://github.com/vanhthefirst/SketchPromptVLA-Pi.git feat/eval-harness
```

Commit and push after every meaningful chunk of work, not at the end of the day.
The volume is not a backup, and today it was not even readable.

## H. tmux

- `Ctrl-b c` new window, `Ctrl-b 0/1/2` to switch, `Ctrl-b "` (needs Shift)
  splits. `Ctrl-b '` is *not* a split — it opens a yellow index prompt.
- Never type `tmux` while already inside tmux. It nests, and then the prefix goes
  to the wrong session. `Ctrl-b Ctrl-b` reaches the inner one; `tmux ls` shows
  the mess; `tmux kill-window -t <session>:<n>` needs the session qualifier.
- Windows whose cwd is under `/workspace` hang when the mount does. The plain
  SSH tab at `/` is the lifeline — open it at the start, not when you need it.

## I. Teardown

```bash
# [RunPod SSH]
cd /workspace/aaron/SketchPromptVLA-Pi && git push
cd /workspace/aaron/sketch_prompted_vla && git add outputs report && git commit -m "results: fine-tuned eval" && git push
```

Terminate the pod, then delete the volume from the Storage page. Billing stops on
deletion. Leave the old 150 GB team volume alone — it belongs to someone else's
work as well.
