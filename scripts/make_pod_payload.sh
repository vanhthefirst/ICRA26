#!/usr/bin/env bash
# Emit a single self-contained script to drive a fresh pod from the local
# machine: install EGL, unpack the eval scripts (the eval repo is not cloned on
# pods), drop in the training repo's newest run script, preflight, then launch
# the paired-corpus build detached.
#
#   bash scripts/make_pod_payload.sh > /tmp/pod_boot.sh
#   ssh <pod>@ssh.runpod.io -i ~/.ssh/id_ed25519 < /tmp/pod_boot.sh
#
# The RunPod proxy ignores ssh command args and reads stdin only, and it kills
# nohup children if the session is SIGTERMed, so the generated script exits
# cleanly on its own the moment the build is detached.
set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
TRAIN=${TRAIN_REPO:-$HERE/../../SketchPromptVLA-Pi}

EVAL_FILES=(build_paired_corpus.py pack_paired_corpus.py
            build_validation_set_spatial_anchored.py export_rlds_frames.py
            run_paired_build.sh)
for f in "${EVAL_FILES[@]}"; do
  [ -f "$HERE/$f" ] || { echo "missing eval script: $HERE/$f" >&2; exit 1; }
done
[ -f "$TRAIN/scripts/run_pcla_v5.sh" ] || {
  echo "missing $TRAIN/scripts/run_pcla_v5.sh (set TRAIN_REPO)" >&2; exit 1; }

cat <<'HEAD'
#!/usr/bin/env bash
set -euo pipefail
REPO=/workspace/SketchPromptVLA-Pi
ES=/workspace/eval_scripts

echo "== pod: $(hostname) =="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true

echo "== EGL (ephemeral disk: gone with every fresh pod) =="
if ! dpkg -s libegl1 >/dev/null 2>&1; then
  apt-get update -qq && apt-get install -y -qq libegl1
fi

echo "== unpack eval scripts =="
mkdir -p "$ES" /workspace/logs
base64 -d <<'B64' | tar xzf - -C "$ES"
HEAD

tar czf - -C "$HERE" "${EVAL_FILES[@]}" | base64

cat <<'MID'
B64
chmod +x "$ES"/*.sh
ls -l "$ES"

echo "== training repo: newest run script =="
base64 -d > "$REPO/scripts/run_pcla_v5.sh" <<'B64'
MID

base64 < "$TRAIN/scripts/run_pcla_v5.sh"

cat <<'TAIL'
B64
chmod +x "$REPO/scripts/run_pcla_v5.sh"

echo "== preflight =="
fail=0
chk() { if eval "$2"; then echo "  ok   $1"; else echo "  FAIL $1"; fail=1; fi; }
chk "network volume /workspace/data"        '[ -d /workspace/data ]'
chk "upright corpus (schema donor)"         '[ -d /workspace/data/sketch_libero_rlds_upright/spatial/sketch_libero/1.0.0 ]'
chk "original corpus (captions source)"     '[ -d /workspace/data/sketch_libero_rlds/spatial/sketch_libero/1.0.0 ]'
chk "training repo"                         '[ -d "$REPO/.git" ]'
chk "LIBERO client venv"                    '[ -x "$REPO/examples/libero/.venv/bin/python" ]'
chk "LIBERO third_party"                    '[ -d "$REPO/third_party/libero/libero" ]'
chk "env.sh"                                '[ -f /workspace/env.sh ]'
chk "uv"                                    'command -v uv >/dev/null'
[ "$fail" = 0 ] || { echo "PREFLIGHT_FAILED"; exit 1; }

echo "== repo HEAD =="
git -C "$REPO" log --oneline -1 || true

LOG=/workspace/logs/paired_build_$(date +%m%d_%H%M).log
setsid nohup bash "$ES/run_paired_build.sh" > "$LOG" 2>&1 < /dev/null &
echo "launched pid $! -> $LOG"
sleep 20
tail -n 15 "$LOG" || true
echo "BOOT_OK: watch with  tail -f $LOG   ; done marker = PAIRED_BUILD_DONE"
TAIL
