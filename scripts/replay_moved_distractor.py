"""Feasibility: do shipped LIBERO-Spatial demos still succeed after the
distractor bowl is teleported to a different region?

Moving the distractor to a partner task's target region is what
`pair_spatial_bddls.py` encodes; this script measures whether the shipped
demo trajectories survive that intervention. Replays N demos twice
(control / moved), open-loop from stored states[0], and reports success +
whether bowl_2 actually stayed where it was put.

Result 28 Aug, worst-case edit (next_to_ramekin task, bowl_2 moved
next_to_box -> next_to_plate_region, 11 cm from the destination plate):
control 45/50, moved 35/50 -> 78% of replayable demos survive. Log:
outputs/paired_spatial_bddl/replay_next_to_ramekin_50.log. Needs
mujoco 2.3.7 + robosuite 1.4.0; CPU is enough.

    python3 replay_moved_distractor.py <bddl> <demo.hdf5> <dx> <dy> [N]

dx/dy is bowl_2's table-frame displacement: partner-target-region centre
minus this task's bowl_2 region centre, both read off the BDDL. The 28 Aug
run: next_to_box (0.13, -0.07) -> next_to_plate (0.01, 0.31) gives
dx=-0.12 dy=0.38. Only an xy shift — a stacked or in-drawer partner region
(the on_ramekin and drawer pairs) needs a z edit too; not covered here.
"""
import os, sys
os.environ.setdefault("MUJOCO_GL", "osmesa")
import h5py
import numpy as np
from libero.libero.envs import OffScreenRenderEnv

if len(sys.argv) < 5:
    sys.exit(__doc__)
BDDL, DEMO = sys.argv[1], sys.argv[2]
DELTA_XY = np.array([float(sys.argv[3]), float(sys.argv[4])])
N = int(sys.argv[5]) if len(sys.argv) > 5 else 10

env = OffScreenRenderEnv(bddl_file_name=BDDL, camera_heights=128, camera_widths=128)

def run(demo, move):
    env.reset()
    env.set_init_state(demo["states"][0])
    sim = env.env.sim  # rebuilt on reset; re-fetch every episode
    jname = [n for n in sim.model.joint_names if "akita_black_bowl_2" in n]
    adr = sim.model.get_joint_qpos_addr(jname[0])  # (start, end) for free joint
    if move:
        q = sim.data.qpos.copy()
        q[adr[0]:adr[0]+2] += DELTA_XY
        sim.data.qpos[:] = q
        sim.forward()
    start = sim.data.qpos[adr[0]:adr[0]+2].copy()
    acts = demo["actions"][:]
    done = False
    for a in acts:
        obs, r, done, info = env.step(a)
    succ = env.env._check_success()
    sim = env.env.sim
    end = sim.data.qpos[adr[0]:adr[0]+2].copy()
    drift = float(np.linalg.norm(end - start))
    return bool(succ), drift

with h5py.File(DEMO, "r") as f:
    keys = sorted(f["data"].keys(), key=lambda k: int(k.split("_")[1]))[:N]
    rows = []
    for k in keys:
        d = f["data"][k]
        c, cd = run(d, move=False)
        m, md = run(d, move=True)
        rows.append((k, c, m, cd, md))
        print(f"{k:8s}  control={'OK ' if c else 'FAIL'}  moved={'OK ' if m else 'FAIL'}  bowl2 drift: control {cd:.4f} m, moved {md:.4f} m", flush=True)

nc = sum(r[1] for r in rows); nm = sum(r[2] for r in rows)
print(f"\ncontrol success {nc}/{len(rows)}   moved-distractor success {nm}/{len(rows)}")
env.close()
