#!/usr/bin/env python3
"""Merge the five V7 rollout chunks into one provenance-stamped artifact and score it.

Identity-aware scoring. The captions are referent-free ("take this over there"),
so the sketch is the ONLY signal distinguishing akita_black_bowl_1 from _2.
That makes the causal quantity of interest the PAIRED SHIFT on each layout:

    shift(task) = P(grasp bowl_2 | swap sketch) - P(grasp bowl_2 | real sketch)

A policy that reads the sketch moves toward bowl_2 when the sketch moves.
A policy that ignores the sketch and follows a fixed spatial prior scores
shift == 0 no matter how high its real-arm task success is.
"""
import hashlib, json, pathlib, subprocess, datetime
from collections import defaultdict

O = pathlib.Path("/workspace/SketchPromptVLA-Pi/outputs")
CHUNKS = [
    ("chunk01_real_t1_t4", 80), ("chunk02_real_t5_t8", 80),
    ("chunk03a_real_t9_t10", 40), ("chunk03b_swap_t1_t2", 40),
    ("chunk04_swap_t3_t6", 80), ("chunk05_swap_t7_t10", 80),
]
BOWL2 = "akita_black_bowl_2"

def sha(p):
    return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()

def git(repo, *a):
    try:
        return subprocess.run(["git", "-C", repo, *a], capture_output=True,
                              text=True, timeout=30).stdout.strip()
    except Exception:
        return "unknown"

rows, prov = [], []
for name, want in CHUNKS:
    p = O / f"v7_paired_step2999_{name}.json"
    d = json.load(open(p))
    got = len(d["rows"])
    if got != want:
        raise SystemExit(f"ABORT: {name} has {got} rows, expected {want}")
    for r in d["rows"]:
        r = dict(r); r["chunk"] = name; rows.append(r)
    prov.append({"chunk": name, "file": str(p), "rows": got, "sha256": sha(p),
                 "wall_s": d.get("wall_s"), "config": d.get("config")})

if len(rows) != 400:
    raise SystemExit(f"ABORT: merged {len(rows)} rows, expected 400")

# ---- per task/mode aggregation -------------------------------------------
agg = defaultdict(lambda: dict(n=0, success=0, referent=0, wrong=0,
                               nograsp=0, bowl2=0, maxerr=0.0))
for r in rows:
    a = agg[(r["mode"], r["task"])]
    a["n"] += 1
    a["success"] += bool(r["success"])
    a["referent"] += bool(r["referent_success"])
    a["wrong"] += bool(r["wrong_bowl"])
    if r["grasped"] in (None, "", "none"):
        a["nograsp"] += 1
    a["bowl2"] += (r["grasped"] == BOWL2)
    a["maxerr"] = max(a["maxerr"], float(r["frame_error"]))

def tkey(t):
    return int(t[1:]) if t[1:].isdigit() else 0
tasks = sorted({t for _, t in agg}, key=tkey)

def block(mode):
    n = sum(agg[(mode, t)]["n"] for t in tasks)
    return {
        "n": n,
        "success": round(sum(agg[(mode, t)]["success"] for t in tasks) / n, 4),
        "referent_success": round(sum(agg[(mode, t)]["referent"] for t in tasks) / n, 4),
        "wrong_bowl": round(sum(agg[(mode, t)]["wrong"] for t in tasks) / n, 4),
        "no_grasp": round(sum(agg[(mode, t)]["nograsp"] for t in tasks) / n, 4),
        "p_bowl2": round(sum(agg[(mode, t)]["bowl2"] for t in tasks) / n, 4),
    }

shifts = {}
for t in tasks:
    pr = agg[("real", t)]["bowl2"] / agg[("real", t)]["n"]
    ps = agg[("swap", t)]["bowl2"] / agg[("swap", t)]["n"]
    shifts[t] = {"p_bowl2_real": round(pr, 4), "p_bowl2_swap": round(ps, 4),
                 "shift": round(ps - pr, 4)}
mean_shift = round(sum(v["shift"] for v in shifts.values()) / len(shifts), 4)

real_floor = {t: agg[("real", t)]["success"] / agg[("real", t)]["n"] for t in tasks}
failing = sorted([t for t, v in real_floor.items() if v < 0.80], key=tkey)
maxerr = max(a["maxerr"] for a in agg.values())

out = {
    "artifact": "v7_paired_step2999_merged_all400",
    "generated_utc": datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
    "checkpoint": "/workspace/SketchPromptVLA-Pi/checkpoints/sketchvla_finetune/rg_v7_paired/2999",
    "variant": "referent_grounding",
    "n_rows": len(rows),
    "provenance": {
        "chunks": prov,
        "train_repo_commit": git("/workspace/SketchPromptVLA-Pi", "rev-parse", "HEAD"),
        "train_repo_dirty": bool(git("/workspace/SketchPromptVLA-Pi", "status", "--porcelain")),
        "eval_repo_commit": git("/workspace/eval_scripts", "rev-parse", "HEAD"),
        "eval_repo_dirty": bool(git("/workspace/eval_scripts", "status", "--porcelain")),
        "evaluator": "/workspace/eval_scripts/scripts/eval_paired_referent.py",
        "evaluator_sha256": sha("/workspace/eval_scripts/scripts/eval_paired_referent.py"),
    },
    "overall": {"real": block("real"), "swap": block("swap")},
    "per_task": {f"{m}|{t}": {
        "n": agg[(m, t)]["n"],
        "success": round(agg[(m, t)]["success"] / agg[(m, t)]["n"], 4),
        "referent_success": round(agg[(m, t)]["referent"] / agg[(m, t)]["n"], 4),
        "wrong_bowl": round(agg[(m, t)]["wrong"] / agg[(m, t)]["n"], 4),
        "no_grasp": round(agg[(m, t)]["nograsp"] / agg[(m, t)]["n"], 4),
    } for m in ("real", "swap") for t in tasks},
    "sketch_following": {"per_task": shifts, "mean_shift": mean_shift},
    "gates": {
        "real_task_floor_0.80": {"pass": not failing, "failing_tasks": failing},
        "max_frame_error": round(maxerr, 5), "frame_error_guard": 5.0,
        "frame_error_pass": maxerr < 5.0,
    },
    "rows": rows,
}
dest = O / "v7_paired_step2999_merged_all400.json"
json.dump(out, open(dest, "w"), indent=1)
print("wrote", dest, "sha256", sha(dest))

# ---- human-readable ------------------------------------------------------
print()
print("REAL ARM (desired referent = akita_black_bowl_1)")
print(f"{'task':6}{'success':>10}{'referent':>10}{'wrong':>9}{'nograsp':>9}")
for t in tasks:
    a = agg[("real", t)]
    flag = "  <-- below 0.80" if a["success"] / a["n"] < 0.80 else ""
    print(f"{t:6}{a['success']}/{a['n']:<7}{a['referent']}/{a['n']:<7}{a['wrong']}/{a['n']:<6}{a['nograsp']}/{a['n']:<6}{flag}")
r = out["overall"]["real"]
print(f"{'ALL':6}{r['success']:>9.4f}{r['referent_success']:>10.4f}{r['wrong_bowl']:>9.4f}{r['no_grasp']:>9.4f}")

print()
print("SWAP ARM (desired referent = akita_black_bowl_2; BDDL success is NOT the score)")
print(f"{'task':6}{'referent':>10}{'wrong':>9}{'nograsp':>9}")
for t in tasks:
    a = agg[("swap", t)]
    print(f"{t:6}{a['referent']}/{a['n']:<7}{a['wrong']}/{a['n']:<6}{a['nograsp']}/{a['n']:<6}")
s = out["overall"]["swap"]
print(f"{'ALL':6}{s['referent_success']:>9.4f}{s['wrong_bowl']:>9.4f}{s['no_grasp']:>9.4f}")

print()
print("SKETCH FOLLOWING  P(bowl_2 | swap) - P(bowl_2 | real)   [0 = sketch ignored, 1 = perfect]")
print(f"{'task':6}{'real':>9}{'swap':>9}{'shift':>9}")
for t in tasks:
    v = shifts[t]
    print(f"{t:6}{v['p_bowl2_real']:>9.2f}{v['p_bowl2_swap']:>9.2f}{v['shift']:>9.2f}")
print(f"{'MEAN':6}{'':>9}{'':>9}{mean_shift:>9.2f}")

print()
print("GATES")
print("  real per-task >= 0.80 :", "PASS" if not failing else "FAIL " + ",".join(failing))
print("  max frame error       : %.4f (guard 5.0) %s" % (maxerr, "PASS" if maxerr < 5.0 else "FAIL"))
