#!/usr/bin/env python3
"""Blocker B: what does relocating a bowl cost a VLA that has no sketch at all?

WHY THIS EXISTS
---------------
`docs/SESSION_2026-09-01.md` closes on two things never done. This is the second.
Every sketch number in this project is measured on the paired corpus, whose
layouts are MANUFACTURED: a task's distractor bowl is teleported into its partner
task's target region so that within a layout only the sketch says which bowl is
meant. Every arm ever run on those layouts -- v5, v6, real and swap alike -- tops
out near 40%. Nobody has established whether that ceiling is a grounding failure
or simply what relocation costs any policy.

Without this number, "v6 overlay takes the goal bowl 78.7% of the time" has no
denominator. With it, the sketch results are read against a floor.

Deliberately no sketch model in the loop: stock `pi05_libero`, which has no
sketch channel, driven by an EXPLICIT caption that names the target uniquely.
Whatever it loses between the two arms is relocation, and nothing else.

THREE ARMS, ONE INTERVENTION
----------------------------
    shipped      the task's own layout, untouched
    relocated    the distractor bowl teleported to the partner's target pose --
                 the exact intervention build_paired_corpus.py performs, so this
                 arm's success rate is the CEILING for any model on that corpus
    target_moved the TARGET bowl teleported to the partner's target pose, the
                 distractor untouched -- the literal reading of "moving the
                 target object degrades almost every VLA"

`shipped` vs `relocated` bounds the sketch work. `shipped` vs `target_moved`
tests the hypothesis as stated. They are different questions and the pair is
cheap, so both run.

The displacement probe already answered a neighbouring question -- translating
the target up to 12 cm left success flat at 99.3% (claude/displacement_probe_
result.md) -- but with the distractor REMOVED and by xy translation, neither of
which is what the paired corpus does. This probe uses the corpus's own donor-pose
mechanism, imported from the builder rather than reimplemented, so a discrepancy
cannot come from two versions of the same teleport.

WHAT IT DOES NOT MEASURE
------------------------
Referential ambiguity. The caption is explicit in every arm, so the second bowl
is never a valid referent and a wrong-bowl grasp is a failure of the policy, not
of the instruction. That is the point: relocation is isolated.

RUNS IN the py3.8 LIBERO client venv, against a policy server:

    # GPU box
    uv run scripts/serve_policy.py --env LIBERO
    # here
    python scripts/probe_relocation_floor.py \
        --bddl-dir <libero_spatial bddls> --demo-dir <hdf5 dir> \
        --host <server> --port 8000 --episodes 20 \
        --out outputs/relocation_floor.json
"""

import argparse
import collections
import json
import os
import subprocess
import sys
import time

os.environ.setdefault("MUJOCO_GL", "osmesa")

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import build_paired_corpus as B
import provenance
from pi05_policy import Pi05ServerPolicy

RES = 256
ARMS = ("shipped", "relocated", "target_moved")
LIFT_TH = 0.03
BOWLS = ("akita_black_bowl_1", "akita_black_bowl_2")


def ensure_libero_config():
    """Seed LIBERO's first-import prompt for non-interactive pod runs."""
    config = os.path.expanduser("~/.libero/config.yaml")
    if not os.path.exists(config):
        subprocess.run(
            [sys.executable, "-c", "import libero.libero"],
            input="N\n", text=True, check=True,
        )


class _Prompt:
    """The three fields Pi05ServerPolicy reads, and nothing else. Built here
    rather than loaded from a scene directory because these layouts do not exist
    on disk -- they are produced by the teleport at reset."""

    def __init__(self, instruction, suite="spatial"):
        self.instruction = instruction
        self.scene_meta = {"suite": suite}
        self.sketch_rgb = None
        self.symbolic_tokens = None


def caption_of(task_key):
    return B.T[task_key].replace("_", " ")


def bowl_z(sim, name):
    return float(sim.data.body_xpos[B.A.bid_of(sim.model, name), 2])


def apply_donor(sim, moved_bowl, donor_pose, fixture, donor_state, replay_state):
    pose = B.map_donor_pose(sim, donor_pose, fixture, donor_state, replay_state)
    adr = sim.model.get_joint_qpos_addr(f"{moved_bowl}_joint0")
    q = sim.data.qpos.copy()
    q[adr[0]:adr[1]] = pose
    sim.data.qpos[:] = q
    sim.forward()


def run_episode(env, policy, prompt, max_steps):
    obs = env.env._get_observations()
    sim = env.env.sim
    z0 = {b: bowl_z(sim, b) for b in BOWLS}
    lift = {b: 0.0 for b in BOWLS}
    policy.reset(prompt)
    success = False
    for t in range(max_steps):
        action = policy.act(obs, t)
        obs, *_ = env.step(action)
        for b in BOWLS:
            lift[b] = max(lift[b], bowl_z(sim, b) - z0[b])
        if env.env._check_success():
            success = True
            break
    grasped = max(lift, key=lift.get)
    if lift[grasped] <= LIFT_TH:
        grasped = None
    return {"success": success, "steps": t + 1, "grasped": grasped,
            "lift": {b: round(v, 4) for b, v in lift.items()}}


def run_task(tkey, arm, args, policy):
    from libero.libero.envs import OffScreenRenderEnv
    import h5py

    donor_key, fixture = B.PAIRING[tkey]
    if arm != "shipped" and donor_key is None:
        return []
    task = B.T[tkey]
    env = OffScreenRenderEnv(
        bddl_file_name=os.path.join(args.bddl_dir, task + ".bddl"),
        camera_heights=RES, camera_widths=RES,
        camera_names=["agentview", "robot0_eye_in_hand"])
    prompt = _Prompt(caption_of(tkey))
    rows = []
    try:
        with h5py.File(os.path.join(args.demo_dir, task + "_demo.hdf5"), "r") as f:
            donor_f = (h5py.File(os.path.join(args.demo_dir, B.T[donor_key] + "_demo.hdf5"), "r")
                       if donor_key and arm != "shipped" else None)
            keys = sorted(f["data"].keys(), key=lambda k: int(k.split("_")[1]))[:args.episodes]
            dkeys = (sorted(donor_f["data"].keys(), key=lambda k: int(k.split("_")[1]))
                     if donor_f else [])
            for i, k in enumerate(keys):
                flat = f["data"][k]["states"][0]
                env.reset()
                env.set_init_state(flat)
                sim = env.env.sim
                if donor_f is not None:
                    dstate = donor_f["data"][dkeys[i % len(dkeys)]]["states"][0]
                    adr = sim.model.get_joint_qpos_addr("akita_black_bowl_1_joint0")
                    donor_pose = np.asarray(dstate[1 + adr[0]: 1 + adr[1]], float)
                    moved = "akita_black_bowl_2" if arm == "relocated" else "akita_black_bowl_1"
                    apply_donor(sim, moved, donor_pose, fixture, dstate, flat)
                for _ in range(5):
                    env.step(np.zeros(7, np.float32))
                row = run_episode(env, policy, prompt, args.max_steps)
                row.update(task=tkey, arm=arm, demo=k, donor=donor_key)
                rows.append(row)
                print("  %s/%s/%s success=%s grasped=%s" %
                      (arm, tkey, k, row["success"], row["grasped"]), flush=True)
            if donor_f is not None:
                donor_f.close()
    finally:
        env.close()
    return rows


def summarise(rows):
    by = collections.defaultdict(list)
    for r in rows:
        by[(r["arm"], r["task"])].append(r)
        by[(r["arm"], "ALL")].append(r)

    def block(rs):
        n = len(rs)
        succ = sum(r["success"] for r in rs)
        wrong = sum(r["grasped"] == "akita_black_bowl_2" for r in rs)
        return {"n": n,
                "success": round(succ / n, 4) if n else None,
                "se": round(float(np.sqrt(succ / n * (1 - succ / n) / n)), 4) if n else None,
                "grasped_any": round(sum(r["grasped"] is not None for r in rs) / n, 4) if n else None,
                "grasped_wrong_bowl": round(wrong / n, 4) if n else None}

    out = {f"{arm}|{task}": block(rs) for (arm, task), rs in sorted(by.items())}
    for other in ("relocated", "target_moved"):
        a, b = out.get("shipped|ALL"), out.get(f"{other}|ALL")
        if a and b and a["success"] is not None and b["success"] is not None:
            out[f"cost_of_{other}"] = {
                "points": round(100 * (a["success"] - b["success"]), 2),
                "se_points": round(100 * float(np.hypot(a["se"], b["se"])), 2)}
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--bddl-dir", required=True)
    ap.add_argument("--demo-dir", required=True)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--tasks", default=",".join(k for k, (d, _) in B.PAIRING.items() if d))
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--max-steps", type=int, default=520)
    ap.add_argument("--no-rotate180", action="store_true",
                    help="only for reproducing a pre-29-Aug run; the corpus is upright")
    ap.add_argument("--out", default="outputs/relocation_floor.json")
    args = ap.parse_args()

    arms = [a for a in args.arms.split(",") if a]
    unknown = set(arms) - set(ARMS)
    if unknown:
        raise SystemExit(f"unknown arms {sorted(unknown)}; choose from {ARMS}")

    print(provenance.summary_line(), flush=True)
    ensure_libero_config()
    policy = Pi05ServerPolicy(host=args.host, port=args.port,
                              rotate180=not args.no_rotate180, sketch_mode="none")

    rows, t0 = [], time.time()
    for arm in arms:
        for tkey in [t for t in args.tasks.split(",") if t]:
            rows += run_task(tkey, arm, args, policy)

    result = {"rows": rows, "summary": summarise(rows),
              "config": {"episodes": args.episodes, "max_steps": args.max_steps,
                         "arms": arms, "tasks": args.tasks,
                         "rotate180": not args.no_rotate180},
              "wall_s": round(time.time() - t0, 1)}
    provenance.write_json(args.out, result)
    print(json.dumps(result["summary"], indent=2))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
