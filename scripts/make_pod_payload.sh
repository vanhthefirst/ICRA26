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
# cleanly on its own the moment the build is detached. It also runs an
# interactive shell that echoes every byte back down the pty, which turns a
# 45 KB base64 payload into a ten-minute stall -- hence `stty -echo` first.
set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
TRAIN=${TRAIN_REPO:-$HERE/../../SketchPromptVLA-Pi}

EVAL_FILES=(build_paired_corpus.py pack_paired_corpus.py
            build_validation_set_spatial_anchored.py export_rlds_frames.py
            provenance.py
            run_paired_build.sh run_paired_parallel.sh run_paired_finish.sh)
for f in "${EVAL_FILES[@]}"; do
  [ -f "$HERE/$f" ] || { echo "missing eval script: $HERE/$f" >&2; exit 1; }
done
[ -f "$TRAIN/scripts/run_rg_v7.sh" ] || {
  echo "missing $TRAIN/scripts/run_rg_v7.sh (set TRAIN_REPO)" >&2; exit 1; }

# Blocker A of docs/SESSION_2026-09-01.md, item 2: the bundle carries no version
# identity, so a rollout produced from it is unattributable. Stamp it here, where
# the eval repo IS a git checkout, and ship the stamp with the payload -- the pod
# has no way to recompute it.
EVAL_STAMP=$(python3 "$HERE/provenance.py" --json 2>/dev/null || echo '{}')

cat <<'HEAD'
stty -echo 2>/dev/null || true
bind 'set enable-bracketed-paste off' 2>/dev/null || true
PS1=''
set -euo pipefail
REPO=/workspace/SketchPromptVLA-Pi
ES=/workspace/eval_scripts

echo "== pod: $(hostname) =="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true

echo "== EGL (ephemeral disk: gone with every fresh pod) =="
if ! dpkg -s libegl1 >/dev/null 2>&1; then
  DEBIAN_FRONTEND=noninteractive apt-get update -qq \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq libegl1 libgles2
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

echo "== stamp the bundle =="
mkdir -p /workspace/harness_repo
MID

printf 'cat > "$ES/VERSION" <<%s\n' "'VEOF'"
printf '%s\n' "$EVAL_STAMP"
printf 'VEOF\ncp "$ES/VERSION" /workspace/harness_repo/VERSION\n'
printf 'python3 -c "import json,sys; d=json.load(open(sys.argv[1])); r=d.get(%s,{}); print(%s, [ (k, v.get(%s)) for k,v in r.items() ])" "$ES/VERSION" || true\n' \
  "'repos'" "'  bundle:'" "'sha'"

cat <<'MID'

echo "== training repo: v7 launcher =="
base64 -d > "$REPO/scripts/run_rg_v7.sh" <<'B64'
MID

base64 < "$TRAIN/scripts/run_rg_v7.sh"

cat <<'TAIL'
B64
chmod +x "$REPO/scripts/run_rg_v7.sh"

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

echo "== already on the volume =="
BUILT=0
for t in t1 t2 t3 t4 t5 t6 t7 t8 t9 t10; do
  n=$(ls -U /workspace/data/paired_frames_cf/$t 2>/dev/null | wc -l)
  BUILT=$((BUILT + n)); printf "  %s: %s\n" "$t" "$n"
done
echo "  total $BUILT episodes (--resume will not rebuild these)"

# Rendering is CPU-side llvmpipe unless the pod ships NVIDIA EGL, and llvmpipe
# spawns a worker per core: size the fan-out so total threads ~= cores instead
# of letting a few hundred spin against each other.
# nproc reports the HOST's cores inside a RunPod container, not this pod's
# share: measured 252 from nproc against a cgroup quota of 26.35 CPUs. Sizing a
# fan-out from nproc is how llvmpipe ended up with ~640 worker threads on ~26
# usable CPUs on the H200 pod. Read the quota and fall back to nproc only if
# there is none.
CORES=$(awk '$1 != "max" {printf "%d", $1/$2}' /sys/fs/cgroup/cpu.max 2>/dev/null)
[ -z "$CORES" ] && CORES=$(awk '$1 > 0 {printf "%d", $1/100000}' /sys/fs/cgroup/cpu/cpu.cfs_quota_us 2>/dev/null)
[ -z "$CORES" ] || [ "$CORES" -lt 1 ] 2>/dev/null && CORES=$(nproc)
if [ -f /usr/share/glvnd/egl_vendor.d/10_nvidia.json ]; then
  echo "  NVIDIA EGL present: rendering on the GPU"
  export LP_NUM_THREADS=1
  NSHARD=3          # GPU renders are ~1 ms; physics is the cost, stay modest
else
  echo "  no NVIDIA EGL: rendering through llvmpipe on $CORES usable CPUs"
  export LP_NUM_THREADS=2
  NSHARD=$(( CORES / (10 * LP_NUM_THREADS) ))
fi
[ "$NSHARD" -lt 1 ] && NSHARD=1
[ "$NSHARD" -gt 10 ] && NSHARD=10
export NSHARD
echo "  NSHARD=$NSHARD LP_NUM_THREADS=$LP_NUM_THREADS -> $((10 * NSHARD)) processes"

LOG=/workspace/logs/paired_fin_$(date +%m%d_%H%M).log
setsid nohup env NSHARD="$NSHARD" LP_NUM_THREADS="$LP_NUM_THREADS" \
  bash "$ES/run_paired_finish.sh" > "$LOG" 2>&1 < /dev/null &
echo "launched -> $LOG"
sleep 25
grep -av "tensorflow\|oneDNN\|cuDNN\|cuFFT\|cuBLAS\|AVX" "$LOG" | tail -12
echo "procs=$(pgrep -fc build_paired_corpus.py) load=$(cut -d\  -f1-3 /proc/loadavg)"
echo "BOOT_OK: watch with  tail -f $LOG   ; done marker = PAIRED_BUILD_DONE"
TAIL
