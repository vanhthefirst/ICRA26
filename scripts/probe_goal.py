"""
Sketch-Prompted VLA — libero_goal PROBE (needs the libero env). Evidence only; decides
nothing. Answers the one question that determines the Goal tier design:

  For each of the 10 libero_goal tasks:
    - language, goal predicate(s), target object (+category), destination
    - is the task USABLE for circle+arrow? (needs an object to pick AND a
      place-destination -> On/In only; Open/Turnon have nothing to circle)
    - is the destination an OBJECT INSTANCE (duplicable -> directional possible)
      or a FIXED REGION (cannot be duplicated -> referential-only)?

Then a light SIM confirmation on one object-destination and one region-
destination task:
    - the destination key really is in object_states_dict
    - which instances have free joints (i.e. can be duplicated)
    - target category rest height + a scripted grasp (is it liftable?)
    - fixtures present (they eat table space -> placement rect is per-task)

No files are written outside outputs/. Run:

    conda activate libero
    cd /mnt/c/Users/Admin/sketch_prompted_vla
    python scripts/probe_goal.py 2>&1 | tee outputs/probe_goal.txt
"""

import os, re, glob, gc, traceback
import numpy as np

BDDL_ROOT = "/root/LIBERO/libero/libero/bddl_files/libero_goal"
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repo root, rename-proof
OUT = os.path.join(_REPO, "outputs")
IMG_H = IMG_W = 128
CAMERA = "agentview"
ADIM = 7


def hdr(t):
    print("\n" + "=" * 78 + f"\n== {t}\n" + "=" * 78)


def safe(fn):
    try:
        fn()
    except Exception:
        print("  [!] FAILED:\n" + "".join(traceback.format_exc()))


# ---------------------------------------------------------------- text parse --
def block(txt, tag):
    m = re.search(r"\(:%s\b(.*?)\n\s*\)\n" % tag, txt, re.S)
    return m.group(1) if m else ""


def parse_task(path):
    txt = open(path).read()
    flat = " ".join(txt.split())
    lang = re.search(r"\(:language\s+(.*?)\)", txt, re.S)
    lang = " ".join(lang.group(1).split()) if lang else ""

    # object instances declared in (:objects)   name1 name2 - category
    objs = {}                       # instance -> category
    for line in block(txt, "objects").splitlines():
        line = line.strip()
        if " - " in line:
            names, cat = line.rsplit(" - ", 1)
            for nm in names.split():
                objs[nm] = cat.strip()

    # region names declared in (:regions)  and their :target fixture
    regions = {}                    # region_name -> target_fixture
    for rn, tgt in re.findall(
            r"\(([A-Za-z0-9_]+)\s*\(:target\s+([A-Za-z0-9_]+)\)", flat):
        regions[rn] = tgt

    # fixtures
    fixtures = {}
    for line in block(txt, "fixtures").splitlines():
        line = line.strip()
        if " - " in line:
            names, cat = line.rsplit(" - ", 1)
            for nm in names.split():
                fixtures[nm] = cat.strip()

    # goal predicates
    goal = block(txt, "goal")
    gflat = " ".join(goal.split())
    preds = re.findall(r"\(([A-Za-z]+)\s+([A-Za-z0-9_]+)(?:\s+([A-Za-z0-9_]+))?\)", gflat)
    preds = [(p, a, b) for (p, a, b) in preds if p.lower() != "and"]
    return dict(name=os.path.basename(path), lang=lang, objs=objs,
                regions=regions, fixtures=fixtures, preds=preds, flat=flat)


def classify_dest(task, dest):
    """OBJECT (duplicable) vs REGION (fixed) vs CONTAIN (object's contain region)."""
    if dest in task["objs"]:
        return "OBJECT", task["objs"][dest]
    if dest.endswith("_contain_region"):
        base = dest[:-len("_contain_region")]
        if base in task["objs"]:
            return "OBJECT_CONTAIN", task["objs"][base]
    if dest in task["regions"]:
        return "REGION", task["regions"][dest]
    # region tied to a fixture but not separately declared (affordance region)
    for fx in task["fixtures"]:
        if dest.startswith(fx):
            return "REGION", fx
    return "UNKNOWN", None


def section_text():
    hdr("A. all 10 libero_goal tasks — classification (pure BDDL parse)")
    files = sorted(glob.glob(os.path.join(BDDL_ROOT, "*.bddl")))
    rows = []
    for f in files:
        t = parse_task(f)
        usable = all(p[0] in ("On", "In") for p in t["preds"]) and len(t["preds"]) >= 1
        # a task is pick+place usable iff every predicate is On/In (no Open/Turnon)
        cls = []
        for (p, a, b) in t["preds"]:
            if b:
                kind, ref = classify_dest(t, b)
                cls.append((p, a, t["objs"].get(a, "?"), b, kind, ref))
            else:
                cls.append((p, a, t["objs"].get(a, "?"), None, "NO_DEST", None))
        rows.append((t, usable, cls))
        print(f"\n[{t['name']}]")
        print(f"   lang : {t['lang']}")
        print(f"   preds: {t['preds']}")
        print(f"   USABLE (circle+arrow): {usable}")
        for (p, a, acat, b, kind, ref) in cls:
            print(f"      {p:7s} target={a} ({acat})  dest={b}  -> {kind}"
                  + (f" [{ref}]" if ref else ""))

    hdr("A2. SUMMARY — the option decision hinges on these counts")
    usable = [(t, c) for (t, u, c) in rows if u]
    print(f"  total tasks              : {len(rows)}")
    print(f"  USABLE (On/In, single)   : {len(usable)}")
    from collections import Counter
    destkind = Counter()
    tgtcats = Counter()
    for (t, cls) in usable:
        for (p, a, acat, b, kind, ref) in cls:
            destkind[kind] += 1
            tgtcats[acat] += 1
    print(f"  destination kinds        : {dict(destkind)}")
    print(f"  target categories        : {dict(tgtcats)}")
    print("\n  => OBJECT / OBJECT_CONTAIN dests can be DUPLICATED (directional OK).")
    print("     REGION dests are FIXED (referential-only for those tasks).")
    obj_tasks = [t["name"] for (t, cls) in usable
                 if any(k in ("OBJECT", "OBJECT_CONTAIN") for (_, _, _, _, k, _) in cls)]
    reg_tasks = [t["name"] for (t, cls) in usable
                 if all(k == "REGION" for (_, _, _, _, k, _) in cls)]
    print(f"\n  object-destination tasks ({len(obj_tasks)}): {obj_tasks}")
    print(f"  region-destination tasks ({len(reg_tasks)}): {reg_tasks}")
    return rows, usable


# ------------------------------------------------------------ light sim check --
def bid_of(model, name):
    for c in (name, f"{name}_main"):
        if c in model.body_names:
            return model.body_name2id(c)
    raise KeyError(name)


def vcenter(model, data, bid):
    gids = [g for g in range(model.ngeom) if model.geom_bodyid[g] == bid]
    ref = [g for g in gids if model.geom_contype[g] == 0 and model.geom_conaffinity[g] == 0]
    return data.geom_xpos[ref[0]].copy() if ref else data.body_xpos[bid].copy()


def jnt_of(model, prefix):
    for j in range(model.njnt):
        if model.joint_id2name(j) == f"{prefix}_joint0":
            return j
    return None


def eef(obs):
    for k in ("robot0_eef_pos", "eef_pos"):
        if k in obs:
            return np.asarray(obs[k])
    return None


def grasp_test(env, model, data, tgt, obs):
    j = jnt_of(model, tgt)
    if j is None:
        return {"grasp": "no free joint"}
    qa = model.jnt_qposadr[j]; z0 = float(data.qpos[qa + 2])
    tc = vcenter(model, data, bid_of(model, tgt))

    def servo(obs, goal, grip, steps, gain=8.0):
        for _ in range(steps):
            e = eef(obs); a = np.zeros(ADIM)
            if e is not None:
                a[:3] = np.clip((goal - e) * gain, -1, 1)
            a[-1] = grip
            obs, _, _, _ = env.step(a)
        return obs
    best = -9
    for dz in (0.005, 0.03):
        for cs in (-1.0, 1.0):
            obs = servo(obs, tc + [0, 0, 0.12], -cs, 30)
            obs = servo(obs, tc + [0, 0, dz], -cs, 25)
            obs = servo(obs, tc + [0, 0, dz], cs, 12)
            obs = servo(obs, tc + [0, 0, 0.18], cs, 30)
            best = max(best, float(data.qpos[qa + 2]) - z0)
            if best > 0.03:
                return {"grasp_success": True, "lift": round(best, 3), "dz": dz, "cs": cs}
    return {"grasp_success": False, "lift": round(best, 3)}


def sim_one(path, label):
    hdr(f"B. sim confirm — {label}: {os.path.basename(path)}")
    from libero.libero.envs import OffScreenRenderEnv
    t = parse_task(path)
    env = OffScreenRenderEnv(bddl_file_name=path, camera_heights=IMG_H,
                             camera_widths=IMG_W, camera_names=[CAMERA])
    try:
        obs = env.reset()
        for _ in range(20):
            obs, _, _, _ = env.step(np.zeros(ADIM))
        model, data = env.sim.model, env.sim.data

        # destination key present in object_states_dict?
        keys = None
        for holder in (env, getattr(env, "env", None)):
            d = getattr(holder, "object_states_dict", None)
            if isinstance(d, dict):
                keys = sorted(d.keys()); break
        dests = [b for (_, _, b) in t["preds"] if b]
        print(f"  goal preds: {t['preds']}")
        print(f"  destination(s): {dests}")
        print(f"  object_states_dict has dest: {{{', '.join(f'{x}:{x in (keys or [])}' for x in dests)}}}")
        print(f"  object_states_dict keys: {keys}")

        # free-joint instances = duplicable objects
        fj = [model.joint_id2name(j)[:-7] for j in range(model.njnt)
              if model.joint_id2name(j) and model.joint_id2name(j).endswith("_joint0")]
        print(f"  free-joint (duplicable) instances: {fj}")

        print(f"  fixtures: {t['fixtures']}")

        # rest heights + grasp for each pickable object
        print("\n  %-24s %-9s %-9s %s" % ("instance", "vc_z", "body_z", "grasp"))
        for inst in fj:
            try:
                b = bid_of(model, inst)
                vc = vcenter(model, data, b); bz = float(data.body_xpos[b][2])
                # fresh reset per grasp to keep it deterministic
                obs = env.reset()
                for _ in range(20):
                    obs, _, _, _ = env.step(np.zeros(ADIM))
                model, data = env.sim.model, env.sim.data
                g = grasp_test(env, model, data, inst, obs)
                print("  %-24s %-9.4f %-9.4f %s" % (inst, vc[2], bz, g))
            except Exception as e:
                print(f"  {inst}: ERR {e}")
    finally:
        try:
            env.close()
        except Exception:
            pass
        gc.collect()


def main():
    os.makedirs(OUT, exist_ok=True)
    rows, usable = section_text()

    # pick one OBJECT-dest and one REGION-dest usable task for the sim check
    obj_path = reg_path = None
    for f in sorted(glob.glob(os.path.join(BDDL_ROOT, "*.bddl"))):
        t = parse_task(f)
        if not (all(p[0] in ("On", "In") for p in t["preds"]) and t["preds"]):
            continue
        kinds = [classify_dest(t, b)[0] for (_, _, b) in t["preds"] if b]
        if obj_path is None and any(k in ("OBJECT", "OBJECT_CONTAIN") for k in kinds):
            obj_path = f
        if reg_path is None and all(k == "REGION" for k in kinds):
            reg_path = f
    print(f"\n  sim targets -> OBJECT-dest: {os.path.basename(obj_path) if obj_path else None}"
          f" | REGION-dest: {os.path.basename(reg_path) if reg_path else None}")
    if obj_path:
        safe(lambda: sim_one(obj_path, "OBJECT-destination"))
    if reg_path:
        safe(lambda: sim_one(reg_path, "REGION-destination"))

    print("\n\nDONE -- send back outputs/probe_goal.txt")


if __name__ == "__main__":
    main()
