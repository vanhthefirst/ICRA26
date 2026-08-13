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

**B1. Network volume first.** Storage → Network Volumes → New, 150 GB. **Write
down its datacenter** — a volume can only attach to GPUs in the same region, so
it constrains the next step.

An existing volume can be resized from the same page, but only upwards, and the
new size bills from the moment it is applied. Resize while no pod is attached.

Everything outside the volume is destroyed when the pod is terminated. The
checkpoint alone is several GB, so this is not optional.

**B2. Deploy the pod.** Pods → Deploy.

- **GPU:** RTX 4090 or A40. Inference only; >8 GB is the requirement. An H100
  costs several times more and buys nothing here.
- **Region:** must match the volume's datacenter from B1.
- **Template:** an **official RunPod PyTorch** template (named like *RunPod
  PyTorch 2.x, CUDA 12.x*). These are Ubuntu-based and ship `bash`, which the
  SSH proxy requires — see gotcha 3. Leave the **container start command** at
  the template default; overriding it is how I ended up on an image without a
  shell.
- **Network volume:** attach the one from B1, mounted at `/workspace`.
- **Container disk:** default. Nothing important should live there.

## C. Connect and check the machine

**C1.** On the pod's card, **Connect** → **copy the SSH command from the dialog
verbatim.** Do not retype it from here. The proxy username is `<pod-id>-<hash>`
and the hash is different for every pod:

```powershell
ssh <pod-id>-<hash>@ssh.runpod.io -i $env:USERPROFILE\.ssh\id_ed25519
```

Run it from PowerShell. On first connection, accept the host fingerprint.

Dropping the `-<hash>` half is the single most costly mistake in this document.
The bare pod-id does not resolve to a pod, so the proxy has no key to match
against and answers `Permission denied (publickey)` — indistinguishable from an
unregistered key, and it will send me auditing account settings for an hour.
Before suspecting the key, re-copy the string from the dialog.

**C2. Verify the two things that would silently ruin the run:**

```bash
nvidia-smi                # the card, and ~24GB memory
df -h /workspace          # the network volume is mounted
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

**D4. Clone the repo into the volume, under my own directory.**

The team volume is shared and already holds a teammate's work at
`/workspace/le-wm`, `/workspace/lewm_data`, `/workspace/.venv` and
`/workspace/.cache`. Everything of mine goes under `/workspace/aaron/` so the
two never collide, and nothing outside it gets modified or deleted.

```bash
mkdir -p /workspace/aaron && cd /workspace/aaron
git clone https://github.com/vanhthefirst/ICRA26.git sketch_prompted_vla
```

If the repo is private, this prompts for credentials: use a GitHub **personal
access token** as the password, not the account password. Generate one at
GitHub → Settings → Developer settings → Personal access tokens, scope `repo`.

**D5. Point the checkpoint cache at the volume** so it survives pod restarts:

```bash
echo 'export OPENPI_DATA_HOME=/workspace/aaron/.cache/openpi' >> ~/.bashrc
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
cd /workspace/aaron/sketch_prompted_vla
claude
```

Paste everything below the `---` in `prompt_pi05_baseline.md`.

**E3. Terminate the pod when not in use — do not merely stop it.**

A pod's container disk is wiped by **both** stop and terminate, so the two
preserve exactly the same thing: whatever is on the network volume. Terminate is
therefore strictly cheaper, since a stopped pod keeps billing for disk while
saving nothing extra. The Terminate action only appears once the pod is stopped.

What survives: everything under `/workspace/aaron/` — this repo, the openpi
checkout, the LIBERO venv, the checkpoint cache, and results in
`outputs/rollouts/`. What does not: apt packages, node, Claude Code and its
login token.

Commit and push before terminating anyway. The volume is shared team storage and
is not a backup.

**E4. Next pod: one command.**

```bash
bash /workspace/aaron/sketch_prompted_vla/scripts/pod_bootstrap.sh
```

That rebuilds only the container-disk half (sections D1–D3) and reports what
still needs doing by hand — the Claude Code login, and openpi if it is somehow
missing from the volume. Deploy the new pod with the same network volume
attached, in EU-RO-1.

## Gotchas, in the order they tend to bite

1. **`Permission denied (publickey)` — check the username before the key.**
   Almost always the `-<hash>` suffix is missing from the proxy username (C1),
   not a key problem. `ssh -v` settles it: if the log shows `Offering public
   key: … ED25519` and the server still refuses, the key was read and sent
   correctly, so the fault is the username or the account, never the local
   client. Only after re-copying the connect string is it worth checking that
   the key is registered under the **team** context rather than the personal
   one, and that the pod was started *after* it was added — keys are injected at
   container start.
2. **Wrong account context.** Check the top-left dropdown says the team before
   creating anything. A pod created in the personal context bills the personal
   balance.
3. **A template with no `/bin/bash`.** Authentication succeeds, the RunPod
   banner prints, and then the session dies immediately with `OCI runtime exec
   failed: … "/bin/bash": stat /bin/bash: no such file or directory`. The proxy
   hardcodes bash, so no client flag helps — `-t /bin/sh` is ignored. Redeploy
   on an official RunPod PyTorch template (B2). Not worth salvaging: an image
   too minimal for bash is also too minimal for the `apt-get` and NodeSource
   steps in section D.
4. **Volume in a different region from the GPU.** The volume simply will not be
   offered at deploy time; recreate it in the GPU's region.
5. **Nested Docker.** A pod is itself a container, so openpi's `docker compose`
   workflow cannot run. Section 3 of the brief uses the non-Docker path instead.
6. **`opencv-python` instead of `opencv-python-headless`.** The former wants GUI
   libraries a pod does not have.
7. **Work done outside `/workspace`.** Lost on terminate.
8. **The volume is shared with the team.** `df -h /workspace` reports the whole
   storage cluster, not my 150 GB quota, so it is no guide to remaining space.
   The console's Storage page gives used-against-allocated for the volume;
   `du -sh /workspace/aaron` gives my own share. Stay inside
   `/workspace/aaron/` and leave everything else alone.
