"""
Sketch-Prompted VLA — libero_object / libero_goal / libero_10 PROBES (run in
WSL2). Pure text parsing of the shipped BDDL files -- no simulator, no downloads.

Tells us, per suite: the task list, which object categories appear (and how
often duplicated already), what the destinations are, and what GOAL structures
look like (single vs multi-step). That determines how to augment each suite and
whether circle+arrow even applies (the Long suite is the open question).

    conda activate <libero env>
    cd /mnt/c/Users/Admin/sketch_prompted_vla
    python scripts/probe_suites_wsl.py 2>&1 | tee outputs/probe_suites.txt
"""

import os, re, glob, collections

ROOT = "/root/LIBERO/libero/libero/bddl_files"
SUITES = ["libero_spatial", "libero_object", "libero_goal", "libero_10"]


def parse(path):
    txt = open(path).read()

    def block(tag):
        m = re.search(r"\(:%s\s*(.*?)\n\s*\)\n" % tag, txt, re.S)
        return m.group(1) if m else ""

    lang = re.search(r"\(:language\s+(.*?)\)", txt, re.S)
    lang = " ".join(lang.group(1).split()) if lang else ""

    # objects:  "name1 name2 - category"
    objs = collections.Counter()
    for line in block("objects").splitlines():
        line = line.strip()
        if " - " in line:
            names, cat = line.rsplit(" - ", 1)
            objs[cat.strip()] += len(names.split())

    fixtures = collections.Counter()
    for line in block("fixtures").splitlines():
        line = line.strip()
        if " - " in line:
            names, cat = line.rsplit(" - ", 1)
            fixtures[cat.strip()] += len(names.split())

    goal = block("goal")
    goal_flat = " ".join(goal.split())
    preds = re.findall(r"\(([A-Za-z]+)\s", goal_flat)
    preds = [p for p in preds if p.lower() != "and"]
    ooi = [l.strip() for l in block("obj_of_interest").splitlines() if l.strip()]
    return dict(lang=lang, objs=objs, fixtures=fixtures, goal=goal_flat,
                preds=preds, ooi=ooi)


for suite in SUITES:
    d = os.path.join(ROOT, suite)
    files = sorted(glob.glob(os.path.join(d, "*.bddl")))
    print("\n" + "=" * 78)
    print(f"== {suite}: {len(files)} tasks")
    print("=" * 78)

    all_objs = collections.Counter()
    all_fix = collections.Counter()
    pred_shapes = collections.Counter()
    nsteps = collections.Counter()

    for f in files:
        p = parse(f)
        all_objs.update(p["objs"])
        all_fix.update(p["fixtures"])
        pred_shapes[tuple(p["preds"])] += 1
        nsteps[len(p["preds"])] += 1

    print("\n-- object categories used across the suite (total instances) --")
    for k, v in all_objs.most_common():
        print(f"   {v:3d}  {k}")
    print("\n-- fixtures --")
    for k, v in all_fix.most_common():
        print(f"   {v:3d}  {k}")
    print("\n-- goal: number of predicates per task --")
    for k, v in sorted(nsteps.items()):
        print(f"   {v:2d} tasks have {k} predicate(s)")
    print("\n-- distinct goal predicate shapes --")
    for shape, c in pred_shapes.most_common():
        print(f"   {c:2d}x  {shape}")

    print("\n-- sample tasks (first 3): language + goal --")
    for f in files[:3]:
        p = parse(f)
        print(f"\n   [{os.path.basename(f)[:60]}]")
        print(f"     lang: {p['lang']}")
        print(f"     objs: {dict(p['objs'])}")
        print(f"     ooi : {p['ooi']}")
        print(f"     goal: {p['goal']}")

print("\n\nDONE -- send back outputs/probe_suites.txt")
