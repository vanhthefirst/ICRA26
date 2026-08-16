# Runbook — the full 500-trial reproduction

Aaron — 13 August 2026

My supervisor asked for the published trial count rather than the reduced one, so
this runbook re-runs the reproduction at **50 trials per task, 500 episodes per
suite, all four standard suites** — matching the π₀.₅ paper exactly.

Follow it top to bottom. It starts from having no SSH key at all, so skip Part A
if the key is already registered with RunPod.

**What changes from the baseline run:** one flag.
`--args.num-trials-per-task` goes from 5 to 50. Nothing else — same openpi
commit, same checkpoint, same unmodified `examples/libero/main.py`.

**What this does not touch:** my 114 validation scenes, my harness, and the 34.5%
figure. This is the standard suites only.

**Time:** budget a full day of pod time. See *Part E* before you commit to it.

---

## Read this before you book the GPU

The reproduction at 50 episodes per suite already passed. Every suite landed
within 2.4 points of published, average 96.0 against 96.85. Its job was to
separate a working pipeline from a broken one, and it did.

So this run **will not change any conclusion in the project.** What it buys is a
tighter interval on a check that has already passed: the binomial standard error
drops from about 3 points to about 1. That is a fair thing for a supervisor to
want in a paper, and it is worth doing for that reason. It is just worth being
clear that it is a precision upgrade, not a new finding — and that the sketch arms
(`RUNBOOK_PI05_RECOVERY.md`) are the run that actually tests the project's claim.

If GPU budget is tight, do the sketch arms first. Two hours there answers an open
question; a day here tightens an answer we already have.

---

## Part A — the SSH key, on Windows

Skip to Part B if `C:\Users\Admin\.ssh\id_ed25519.pub` already exists **and** is
registered with RunPod.

**A1. Make the key:**

```powershell
ssh-keygen -t ed25519 -C "runpod"
```

Accept the default path. A passphrase is optional — if you set one you will type
it on every connection.

**A2. Print the public key.** This is the one you share. Never the file without
`.pub`:

```powershell
type $env:USERPROFILE\.ssh\id_ed25519.pub
```

**A3. Register it** in the RunPod console: Settings → SSH Public Keys → paste.

**Do this before creating the pod.** Keys are injected when a pod starts, so a key
added afterwards means restarting the pod to pick it up.

If the setting is read-only for your dev role, send the public key to the account
admin. Failing that, the pod's **Web Terminal** works as a fallback — combined
with tmux it is fine, because tmux runs on the pod and survives the browser
closing.

**A4. Push the repo.** The pod clones from GitHub, so anything uncommitted stays
behind:

```powershell
cd C:\Users\Admin\sketch_prompted_vla
git add -A
git commit -m "feat: 500-trial reproduction runbook and runner"
git push
git status          # must be clean
```

---

## Part B — the pod

**B1.** Check the top-left dropdown in the console says the **team**, not your
personal account. A pod created in the wrong context bills the wrong balance.

**B2.** Pods → Deploy:

- **GPU:** RTX 4090 or A40. Inference only, ≥8 GB. Do not pay for an H100 — it
  buys nothing here and this run is long enough that the price difference matters.
- **Region:** **EU-RO-1**, matching the network volume. If it is not offered, the
  volume is in another datacenter and you cannot attach it.
- **Template:** any PyTorch / CUDA one. They are Ubuntu based.
- **Network volume:** attach the existing one at `/workspace`.
- **Container disk:** default.

**B3.** Connect. Copy the SSH command from the pod's card:

```powershell
ssh <pod-id>@ssh.runpod.io -i ~/.ssh/id_ed25519
```

Accept the host fingerprint on first connection.

**B4.** Check the three things that would waste the whole booking:

```bash
nvidia-smi
df -h /workspace
ls /workspace/aaron
```

You want the card, a large `/workspace`, and both `sketch_prompted_vla` and
`openpi` inside `/workspace/aaron`. **If `/workspace/aaron` is missing the volume
did not attach — stop and fix it.** Do not start a multi-hour run on ephemeral
disk.

---

## Part C — tooling and Claude Code on the pod

**C1.** Rebuild the container-disk half and pull the repo:

```bash
bash /workspace/aaron/sketch_prompted_vla/scripts/pod_bootstrap.sh
source ~/.bashrc
```

This installs apt packages, Node 22 and Claude Code, sets `OPENPI_DATA_HOME` and
`MUJOCO_GL=egl`, and pulls the repo. Then it prints what is left to do by hand.
Read that list — in particular it tells you whether openpi and the LIBERO venv
survived on the volume.

If it reports openpi is **not** set up, follow sections 2 and 3 of
`brief_pi05_baseline.md` before anything below will work.

**C2. Start Claude Code and log in:**

```bash
cd /workspace/aaron/sketch_prompted_vla
claude
```

It prints a login URL on first run. Open it on Windows, log in, paste the code
back. This authenticates the Anthropic account and bills the Claude plan — it is
unrelated to RunPod credits and the GPU is not involved.

The token lives on the container disk, so this is needed **once per pod**.

**C3. What to hand Claude Code.** Once the runs are going, or once they finish,
give it this:

> Read `RUNBOOK_REPRO_500.md` part F. The four logs are in
> `outputs/rollouts/openpi_repro_500/`. Pull the per-suite success rates out of
> them, update `outputs/rollouts/pi05_baseline/openpi_reference.json` to the
> 500-trial figures, then update the reproduction table and the trial-count
> sentence in `report/pi05_baseline/pi05_baseline_report.tex` and recompile.
> Keep the author line `Aaron` and first person singular. Confirm today's date
> rather than assuming it.

Claude Code is useful for the parsing and the write-up, not for babysitting the
run. The run itself is one command.

---

## Part D — two panes

```bash
tmux new -s repro
```

Split with `Ctrl-b "`, move between panes with `Ctrl-b o`, reattach after a
disconnect with `tmux attach -t repro`.

**This run is hours long. tmux is not optional.** Without it, closing your laptop
kills the job.

**Pane 1 — the policy server.** Leave it up for the whole session:

```bash
cd /workspace/aaron/openpi
uv run scripts/serve_policy.py --env LIBERO
```

Wait until it reports it is listening. First start takes a few minutes while the
checkpoint loads.

**Pane 2 — the LIBERO side:**

```bash
source /workspace/aaron/openpi/examples/libero/.venv/bin/activate
export PYTHONPATH=$PYTHONPATH:/workspace/aaron/openpi/third_party/libero
export MUJOCO_GL=egl
cd /workspace/aaron/openpi
```

`MUJOCO_GL=egl` is required — a pod has no X server. openpi's docs suggest `glx`
when EGL errors appear; on a pod that advice is backwards.

---

## Part E — time one suite before committing to four

Do not launch all four blind. Measure, then decide.

```bash
cd /workspace/aaron/openpi
SUITES=libero_spatial bash /workspace/aaron/sketch_prompted_vla/scripts/repro_500.sh
```

That runs `libero_spatial` alone at 500 episodes and prints how long it took.

**Then do the arithmetic.** Spatial, Object and Goal are comparable in episode
length. `libero_10` is long-horizon — 2 or 3 goal predicates per task — and is
noticeably slower per episode, so budget roughly double for it.

Rough guide from my own measured throughput on the 114-scene run (8.8 s per
rollout, RTX 4090): 2,000 episodes lands near **5 hours**, and `libero_10` pushes
the realistic total to **6–8 hours**. Treat that as an estimate to check against
your Spatial timing, not a promise.

**If the timing is worse than you can afford**, the honest options in order of
preference:

1. Run the three suites my scenes derive from — Spatial, Object, Goal — at the
   full 500, and leave `libero_10` at 50. My validation scenes derive from those
   three, so those are the ones that carry weight, and the report already quotes
   a three-suite average of 98.0% separately from the four-suite 96.0%.
2. Drop to 25 trials per task (250 per suite). Standard error about 1.4 points,
   half the time. Still a large improvement on 50 episodes.
3. Say no to the full count and report the interval instead. 96.0% ± 3 points at
   n=50 already contains 96.85%.

Whatever you pick, **record the trial count in the report.** A number without its
denominator is the thing that caused this conversation.

---

## Part F — the full run

```bash
cd /workspace/aaron/openpi
bash /workspace/aaron/sketch_prompted_vla/scripts/repro_500.sh
```

The script runs one suite at a time, logs each to
`outputs/rollouts/openpi_repro_500/<suite>.log`, and appends a `SUITE_DONE`
sentinel when a suite completes cleanly. A suite that already has a `SUITE_DONE`
in its log is skipped, so **if it crashes or you get disconnected, just run the
same command again** — finished suites are not repeated.

Useful variants:

```bash
# a subset, in order
SUITES="libero_goal libero_10" bash .../repro_500.sh

# a different trial count
TRIALS=25 bash .../repro_500.sh
```

**Watch it for the first few minutes**, then leave it. What you are checking is
that episodes are actually completing and the server is answering — not the
success rate.

**Push as soon as it finishes.** The volume is shared team storage, not a backup:

```bash
cd /workspace/aaron/sketch_prompted_vla
git add outputs/rollouts/openpi_repro_500
git commit -m "results: 500-trial reproduction of the published pi0.5-LIBERO numbers"
git push
```

---

## Part G — what to update afterwards

The point of the run is these edits. Without them it is just logs.

**G1. `outputs/rollouts/pi05_baseline/openpi_reference.json`** — change
`trials_per_task` to 50, `episodes_per_suite` to 500, each suite's
`observed_success_rate`, `observed_average`, and rewrite `verdict` with the new
figures. Keep `published_source`, `openpi_git_sha` and `checkpoint` as they are —
they have not changed.

**G2. `report/pi05_baseline/pi05_baseline_report.tex`** — three places:

- the reproduction table on page 2 (observed column, and the Trials/task column);
- its caption, which currently says the reproduction uses 5 trials per task;
- the sentence after it, which currently reads *"on a tenth of the trial count
  — 50 episodes per suite against the published 500"* and *"At 50 episodes per
  suite the binomial standard error is about 3 points"*. Both numbers change.

Then recompile:

```bash
cd report/pi05_baseline
pdflatex -interaction=nonstopmode pi05_baseline_report.tex
pdflatex -interaction=nonstopmode pi05_baseline_report.tex
```

Twice, so the cross-references settle. The `.aux` / `.log` / `.out` files are
gitignored.

**G3.** Also drop the now-stale limitation bullet — the report currently lists
*"The reproduction is 50 episodes per suite, not 500"* as a limitation. Once this
run lands, that bullet is no longer true and should go.

---

## Part H — shut down

```bash
git status          # must be clean
git push
```

Then **terminate** the pod, do not merely stop it. Both wipe the container disk,
so both preserve exactly the same thing: whatever is on the network volume. A
stopped pod keeps billing for disk while saving nothing extra. Terminate only
appears once the pod is stopped.

Survives: everything under `/workspace/aaron/`. Does not survive: apt packages,
Node, Claude Code and its login.

---

## When it goes wrong

| what you see | what it is | what to do |
|---|---|---|
| key not accepted | key added after the pod started | restart the pod, or use the Web Terminal |
| volume not offered at deploy | wrong region | deploy in EU-RO-1 |
| `/workspace/aaron` missing | volume not attached | redeploy with it; do not run on ephemeral disk |
| EGL / rendering errors | no X server | `export MUJOCO_GL=egl`; ignore advice to use `glx` or `xhost` |
| `docker compose` fails | a pod is already a container | use the non-Docker path; nested Docker is unsupported |
| script refuses to start | not in the openpi checkout, or `MUJOCO_GL` unset | it tells you which; both are one-line fixes |
| run died partway | crash or disconnect | rerun the same command — finished suites are skipped |
| server pane died | OOM or checkpoint issue | restart pane 1, rerun the script |
| SSH drops | normal | `tmux attach -t repro` |
| checkpoint re-downloading | `OPENPI_DATA_HOME` unset | `source ~/.bashrc`, confirm it points into `/workspace/aaron/.cache/openpi` |
| success rates far below published | pipeline problem, not sample size | stop; this is section 5 of `brief_pi05_baseline.md`, not something more trials fixes |

---

## Checklist

- [ ] SSH key created and registered with RunPod **before** the pod exists
- [ ] Windows repo committed and pushed, `git status` clean
- [ ] Pod deployed in EU-RO-1 with the network volume, team account context
- [ ] `nvidia-smi`, `df -h /workspace`, `ls /workspace/aaron` all sane
- [ ] `pod_bootstrap.sh` run, `source ~/.bashrc`, Claude Code logged in
- [ ] tmux session started
- [ ] Pane 1 serving; pane 2 in the LIBERO venv with `MUJOCO_GL=egl`
- [ ] **Spatial timed first, total extrapolated, budget agreed**
- [ ] Four suites complete, each log ending in `SUITE_DONE`
- [ ] Logs committed and pushed
- [ ] `openpi_reference.json` updated to 500 trials
- [ ] Report table, caption, trial-count sentence and limitation bullet updated
- [ ] Report recompiled, pushed
- [ ] Pod **terminated**
