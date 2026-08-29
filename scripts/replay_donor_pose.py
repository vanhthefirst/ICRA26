"""Replay feasibility v2: distractor pose sourced from a donor episode.

For pairing task Ti with Tj, Ti's distractor bowl must sit at Tj's target
region. Instead of computing an xy offset (which cannot handle a bowl stacked
on the ramekin or inside the cabinet drawer), take the full 7-DoF free-joint
pose of a bowl that REALLY sits there: bowl_1 in a shipped episode of Tj.
All ten Spatial tasks share the identical object set, so joint qpos addresses
transfer between their models.

    python replay_donor_pose.py <bddl> <demo.hdf5> <donor.hdf5> <donor_joint> [N]

Replays N demos twice: control, and with akita_black_bowl_2 set to the donor
pose (demo i uses donor episode i, so pose randomisation carries over).
"""
import os, sys
os.environ.setdefault("MUJOCO_GL", "osmesa")
import h5py
import numpy as np
from libero.libero.envs import OffScreenRenderEnv

BDDL, DEMO, DONOR, DONOR_JOINT = sys.argv[1:5]
N = int(sys.argv[5]) if len(sys.argv) > 5 else 25

env = OffScreenRenderEnv(bddl_file_name=BDDL, camera_heights=128, camera_widths=128)

def joint_slice(sim, name):
    adr = sim.model.get_joint_qpos_addr(name)
    return adr[0], adr[1]  # free joint: 7 numbers

def run(demo, donor_state):
    env.reset()
    env.set_init_state(demo["states"][0])
    sim = env.env.sim
    b2a, b2b = joint_slice(sim, "akita_black_bowl_2_joint0")
    if donor_state is not None:
        da, db = joint_slice(sim, DONOR_JOINT)
        # robosuite MjSimState.flatten() = [time] + qpos + qvel
        pose = donor_state[1 + da : 1 + db]
        q = sim.data.qpos.copy()
        q[b2a:b2b] = pose
        sim.data.qpos[:] = q
        sim.forward()
    start = sim.data.qpos[b2a:b2a+3].copy()
    for a in demo["actions"][:]:
        env.step(a)
    succ = bool(env.env._check_success())
    sim = env.env.sim
    drift = float(np.linalg.norm(sim.data.qpos[b2a:b2a+3] - start))
    return succ, drift

with h5py.File(DEMO, "r") as f, h5py.File(DONOR, "r") as g:
    keys = sorted(f["data"].keys(), key=lambda k: int(k.split("_")[1]))[:N]
    dkeys = sorted(g["data"].keys(), key=lambda k: int(k.split("_")[1]))
    nc = nm = 0
    for i, k in enumerate(keys):
        d = f["data"][k]
        ds = g["data"][dkeys[i % len(dkeys)]]["states"][0]
        c, cd = run(d, None)
        m, md = run(d, ds)
        nc += c; nm += m
        print(f"{k:8s}  control={'OK ' if c else 'FAIL'}  moved={'OK ' if m else 'FAIL'}  drift {md:.4f} m", flush=True)
    print(f"\nRESULT control {nc}/{len(keys)}   moved {nm}/{len(keys)}")
env.close()
