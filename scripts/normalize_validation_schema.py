"""
Sketch-Prompted VLA — normalise validation sets to ONE canonical schema (pure stdlib; no
libero, no simulator needed). Runs over the finished per-scene folders and ADDS a
canonical field block to meta.json / tokens.json / manifest.json without removing
any existing key. Idempotent: re-running is a no-op.

Why: the Spatial and Object builders grew independently and named the two most
important fields differently --
    Spatial: target_bowl / target_plate   goal (On  obj plate)
    Object : target / dest                goal (In  obj basket_contain_region)
A downstream loader (and the Goal set, and the eventual text-vs-sketch eval)
needs one contract. This defines it and back-fills both sets to match.

CANONICAL per-scene fields (present in EVERY scene, both suites):
    suite               "spatial" | "object"
    tier                control | referential | directional | both
    target              instance name to pick   (circle encloses this)
    destination         instance name to move to (arrow points here)
    destination_region  the exact 2nd arg of the BDDL goal predicate
    goal_predicate      "On" | "In"
    instruction         natural-language caption (vague by construction)
    symbolic_tokens     {circle:{cx,cy,rx,ry}, arrow:{x0,y0,x1,y1}}   (unchanged)
    pick_px, place_px, radius, camera_matrix, visibility, grasp,
    clearance_xy, seed, oracle_success                                 (already shared)

Legacy keys (target_bowl, target_plate, dest, ...) are KEPT untouched.

    cd /mnt/c/Users/Admin/sketch_prompted_vla
    python scripts/normalize_validation_schema.py
"""

import json, os, glob, sys

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repo root, rename-proof
ROOT = os.path.join(_REPO, "outputs")
SETS = {
    "spatial": "validation_set_spatial",
    "object":  "validation_set_object",
    "goal":    "validation_set_goal",
}
SCHEMA_VERSION = "1.0"


def load(p):
    with open(p) as f:
        return json.load(f)


def dump(p, d):
    tmp = p + ".tmp"
    with open(tmp, "w") as f:
        json.dump(d, f, indent=2)
    # verify before replacing (DrvFs can truncate)
    back = load(tmp)
    if back != d:
        raise IOError(f"verify failed writing {p}")
    os.replace(tmp, p)


def canon_from_meta(suite, m):
    """Derive the canonical field block from a scene's existing meta.json."""
    # SCHEMA.md declares `instruction` canonical and meta.json a superset of the
    # canonical block. The Object builder only ever wrote `language`, so its 38
    # meta.json files had no `instruction` key at all and a consumer following
    # SCHEMA.md hit a KeyError on a third of the corpus. Back-fill it from
    # `language` here (additive; Spatial and Goal already carry both).
    instruction = m.get("instruction") or m.get("language", "")

    if suite == "spatial":
        target = m["target_bowl"]
        destination = m["target_plate"]
        # Spatial goal is (On bowl plate) -> region arg IS the plate instance
        return dict(suite="spatial", tier=m["tier"], target=target,
                    destination=destination, destination_region=destination,
                    goal_predicate="On", instruction=instruction)
    elif suite == "object":
        target = m["target"]
        destination = m["dest"]
        return dict(suite="object", tier=m["tier"], target=target,
                    destination=destination,
                    destination_region=f"{destination}_contain_region",
                    goal_predicate="In", instruction=instruction)
    elif suite == "goal":
        # The Goal builder already emits the canonical block into meta.json
        # (per-task On/In, object- vs region-typed destinations), so read it
        # straight through rather than re-deriving from legacy keys.
        return dict(suite="goal", tier=m["tier"], target=m["target"],
                    destination=m["destination"],
                    destination_region=m["destination_region"],
                    goal_predicate=m["goal_predicate"], instruction=instruction)
    raise ValueError(suite)


def main():
    if not os.path.isdir(ROOT):
        sys.exit(f"outputs root not found: {ROOT}")
    grand = {}
    for suite, sub in SETS.items():
        base = os.path.join(ROOT, sub)
        if not os.path.isdir(base):
            print(f"[skip] {suite}: {base} missing")
            continue
        scenes = sorted(glob.glob(os.path.join(base, "scene_*")))
        changed = 0
        canon_manifest = []
        for sd in scenes:
            mp = os.path.join(sd, "meta.json")
            tp = os.path.join(sd, "tokens.json")
            m = load(mp)
            c = canon_from_meta(suite, m)
            c["schema_version"] = SCHEMA_VERSION

            before = {k: m.get(k) for k in c}
            m.update(c)
            if before != c:
                changed += 1
            dump(mp, m)

            if os.path.exists(tp):
                t = load(tp)
                t.update({k: c[k] for k in
                          ("target", "destination", "destination_region",
                           "goal_predicate", "suite", "tier")})
                if "instruction" not in t:
                    t["instruction"] = m.get("language", "")
                dump(tp, t)

            canon_manifest.append(dict(
                dir=os.path.basename(sd), suite=suite, tier=c["tier"],
                target=c["target"], destination=c["destination"],
                destination_region=c["destination_region"],
                goal_predicate=c["goal_predicate"], seed=m.get("seed"),
                visibility=m["visibility"]["visibility"],
                grasp_lift=m["grasp"]["lift"], clearance_xy=m.get("clearance_xy")))

        # write a canonical manifest alongside the original (non-destructive)
        cmp_ = os.path.join(base, "manifest_canonical.json")
        dump(cmp_, canon_manifest)
        grand[suite] = dict(scenes=len(scenes), updated=changed,
                            canonical_manifest=os.path.basename(cmp_))
        print(f"[{suite}] {len(scenes)} scenes, {changed} meta updated, "
              f"wrote {os.path.basename(cmp_)}")

    # combined cross-suite manifest at outputs root
    combined = []
    for suite, sub in SETS.items():
        p = os.path.join(ROOT, sub, "manifest_canonical.json")
        if os.path.exists(p):
            combined += load(p)
    dump(os.path.join(ROOT, "validation_manifest_all.json"), combined)
    print(f"\nwrote validation_manifest_all.json ({len(combined)} scenes total)")
    print("schema_version:", SCHEMA_VERSION)


if __name__ == "__main__":
    main()
