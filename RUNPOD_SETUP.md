# RunPod setup — from zero to Claude Code running on the pod

Operational notes for running the π₀.₅ baseline (`prompt_pi05_baseline.md`) on a
RunPod GPU pod under the team account. Written to be followed again on the next
pod, since pods are disposable and this sequence is not.

Role required: **dev** (create/start/stop pods, create network volumes). Billing
goes to whichever account context is **Current** in the console's top-left
dropdown — confirm it says the team, not the personal account, before creating
anything.

---

## A. On Windows, before touching RunPod

**A1. Push the repo.** The pod gets the project by cloning from GitHub, so
anything uncommitted stays behind.

```powershell
cd C:\Users\Admin\sketch_prompted_vla
git add scripts/pi05_policy.py scripts/rollout_sketch_wsl.py `
        prompt_pi05_baseline.md report/pi05_baseline RUNPOD_SETUP.md
git commit -m "feat: pi0.5-LIBERO sketch-free baseline (policy, harness path, brief)"
git push
```

**A2. Make an SSH key** if `C:\Users\Admin\.ssh\id_ed25519.pub` does not already
exist:

```powershell
ssh-keygen -t ed25519 -C "runpod"
```

Accept the default path; a passphrase is optional. Then print the **public** key
— this is the one that gets shared, never the file without `.pub`:

```powershell
type $env:USERPROFILE\.ssh\id_ed25519.pub
```

**A3. Register the key with RunPod** at Settings → SSH Public Keys, and paste it
there.

Do this **before** creating the pod. Keys are injected when a pod starts, so a
key added afterwards means restarting the pod to pick it up.

If the setting is read-only for the dev role, send the public key to the account
admin and ask them to add it. Failing that, skip SSH entirely and use the pod's
**Web Terminal** — combined with `tmux` (step D4) it is a workable fallback,
because tmux runs on the pod and survives the browser disconnecting.

## B. Create the storage, then the pod

**B1. Network volume first.** Storage → Network Volumes → New, 100 GB. **Write
down its datacenter** — a volume can only attach to GPUs in the same region, so
it constrains the next step.

Everything outside the volume is destroyed when the pod is terminated. The
checkpoint alone is several GB, so this is not optional.

**B2. Deploy the pod.** Pods → Deploy.

- **GPU:** RTX 4090 or A40. Inference only; >8 GB is the requirement. An H100
  costs several times more and buys nothing here.
- **Region:** must match the volume's datacenter from B1.
- **Template:** a PyTorch / CUDA template (these are Ubuntu-based).
- **Network volume:** attach the one from B1, mounted at `/workspace`.
- **Container disk:** default. Nothing important should live there.

## C. Connect and check the machine

**C1.** On the pod's card, **Connect** → copy the SSH command. It looks like:

```
ssh <pod-id>@ssh.runpod.io -i ~/.ssh/id_ed25519
```

Run it from PowerShell. On first connection, accept the host fingerprint.

**C2. Verify the two things that would silently ruin the run:**

```bash
nvidia-smi                # the card, and ~24GB memory
df -h /workspace          # ~100GB, the network volume
```

If `/workspace` is missing or tiny, the volume did not attach — fix that before
going further rather than discovering it after a multi-GB download.

## D. Install the tooling on the pod

Pods run as root, so no `sudo` and no npm permission workarounds.

**D1. Base packages:**

```bash
apt-get update && apt-get install -y curl git tmux
```

**D2. Node 22+** (required by current Claude Code):

```bash
curl -fsSL https://deb.nodesource.com/setup_22.x | bash -
apt-get install -y nodejs
node --version
```

**D3. Claude Code:**

```bash
npm install -g @anthropic-ai/claude-code
claude --version
```

On first `claude` run it prints a login URL. Open it on the Windows machine, log
in, and paste the code back into the terminal. This authenticates my Anthropic
account and is billed to my Claude plan — it is unrelated to RunPod credits, and
the GPU is not involved.

**D4. Clone the repo into the volume:**

```bash
cd /workspace
git clone https://github.com/vanhthefirst/ICRA26.git sketch_prompted_vla
```

If the repo is private, this prompts for credentials: use a GitHub **personal
access token** as the password, not the account password. Generate one at
GitHub → Settings → Developer settings → Personal access tokens, scope `repo`.

**D5. Point the checkpoint cache at the volume** so it survives pod restarts:

```bash
echo 'export OPENPI_DATA_HOME=/workspace/.cache/openpi' >> ~/.bashrc
echo 'export MUJOCO_GL=egl' >> ~/.bashrc
source ~/.bashrc
```

`MUJOCO_GL=egl` is needed because a pod has no X server. openpi's docs suggest
`glx` when EGL errors appear; on a pod that advice is inverted.

## E. Run

**E1. Start a tmux session** so a dropped connection does not kill a multi-hour
run:

```bash
tmux new -s pi05
```

Reattach after any disconnect with `tmux attach -t pi05`. Split into two panes
with `Ctrl-b "` — the job needs a policy server and the harness running at once.

**E2. Start Claude Code in the repo and hand it the brief:**

```bash
cd /workspace/sketch_prompted_vla
claude
```

Paste everything below the `---` in `prompt_pi05_baseline.md`.

**E3. Stop the pod when not in use.** Pods bill per second while running.
Stopping halts GPU charges and keeps the network volume intact. Results live in
`outputs/rollouts/pi05_baseline/` under `/workspace`, so they persist — but
commit and push anything worth keeping before terminating the pod outright.

## Gotchas, in the order they tend to bite

1. **Wrong account context.** Check the top-left dropdown says the team before
   creating anything. A pod created in the personal context bills the personal
   balance.
2. **SSH key added after the pod started.** Restart the pod, or use the web
   terminal.
3. **Volume in a different region from the GPU.** The volume simply will not be
   offered at deploy time; recreate it in the GPU's region.
4. **Nested Docker.** A pod is itself a container, so openpi's `docker compose`
   workflow cannot run. Section 3 of the brief uses the non-Docker path instead.
5. **`opencv-python` instead of `opencv-python-headless`.** The former wants GUI
   libraries a pod does not have.
6. **Work done outside `/workspace`.** Lost on terminate.
