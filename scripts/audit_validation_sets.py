"""
Sketch-Prompted VLA — READ-ONLY audit of the three validation sets (pure stdlib; no libero,
no MuJoCo, no simulator of any kind — just the outputs/ tree).

Writes NOTHING. Safe to run any time. Exit code 0 = all clean, 1 = problems.

Two jobs:

  1. RENAME FALLOUT. Confirm every suite directory the pipeline expects actually
     exists and holds its scenes. This matters because
     `normalize_validation_schema.py` does NOT crash on a missing directory — it
     prints "[skip] <suite>" and carries on, silently emitting a combined
     manifest with that whole suite MISSING. A rename can therefore quietly cost
     you 38 scenes with a zero exit code.

  2. THE FULL-SET BDDL AGREEMENT AUDIT (113 scenes since the Spatial rebuild;
     114 before it). Re-parse each scene.bddl goal predicate
     from disk and check it against the canonical `target`, `destination_region`
     and `goal_predicate` in meta.json / tokens.json. SCHEMA.md previously
     recorded this audit for the 76 Spatial+Object scenes only, before Goal
     existed. This runs it across all three.

  3. init_state.npz PRESENCE/SHAPE (SCHEMA.md). Every scene must carry a pinned
     `init_state.npz` with `qpos`/`qvel`/`time` arrays UNLESS it is listed in
     `outputs/rollouts/nonreproducible.json` (issue 1's ladder rung 3), in which
     case it must be ABSENT. Checked via zipfile, not numpy, to keep this script
     dependency-free.

    cd /mnt/c/Users/Admin/sketch_prompted_vla
    python scripts/audit_validation_sets.py
"""

import json, os, re, sys, glob, zipfile

# Optional: PIL enables the frame-edge clipping check (finding from the
# 2026-07-30 review — spatial/scene_0017 shipped a target bowl clipped by the
# right frame edge; the visibility ratio is blind to this because v_visible and
# v_full come from the same clipped render). Without PIL the check is skipped
# and a note is printed.
try:
    from PIL import Image as _PILImage
except ImportError:
    _PILImage = None

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repo root, rename-proof
ROOT = os.path.join(_REPO, "outputs")

# Must stay in sync with SETS in normalize_validation_schema.py.
SETS = {
    "spatial": "validation_set_spatial",
    "object":  "validation_set_object",
    "goal":    "validation_set_goal",
}
# Per suite, because they are no longer all 38. Spatial is 37 since the
# layout-anchored rebuild: it draws its scenes from 4 shipped `libero_spatial`
# tasks rather than 5, `next_to_the_plate` having been dropped because bowl_1
# projects too close to the right border to carry a circle
# (build_validation_set_spatial_anchored.py, BASE ROSTER). Kept as an explicit
# claim rather than read off each manifest, so a builder that silently emits
# half a suite still trips the audit.
EXPECTED_SCENES = {"spatial": 37, "object": 38, "goal": 38}
CORE_FILES = ("scene.bddl", "frame0.png", "sketch.png", "target_vismask.png",
              "tokens.json", "meta.json")
# init_state.npz (SCHEMA.md) is added by scripts/capture_scene_init_states.py,
# not by a builder, and is legitimately absent for a scene that ladder gave up on
# (rung 3, "give up honestly") -- those are listed here, not in CORE_FILES.
NONREPRO_PATH = os.path.join(ROOT, "rollouts", "nonreproducible.json")
INIT_STATE_MEMBERS = ("qpos.npy", "qvel.npy", "time.npy")


def load_nonreproducible_keys():
    if not os.path.exists(NONREPRO_PATH):
        return set()
    return {(e["suite"], e["dir"]) for e in load(NONREPRO_PATH)}

# The canonical contract from SCHEMA.md — every field, every scene, every suite.
CANONICAL_FIELDS = (
    "schema_version", "suite", "tier", "target", "destination",
    "destination_region", "goal_predicate", "instruction", "symbolic_tokens",
    "pick_px", "place_px", "radius", "camera_matrix", "visibility", "grasp",
    "clearance_xy", "seed", "oracle_success",
)

# Files every packaged suite must ship alongside its scenes.
SUITE_FILES = ("DATASHEET.md", "contact_sheet.png", "manifest.json",
               "manifest_canonical.json", "build_log.txt")

problems = []
notes = []


def fail(msg):
    problems.append(msg)


def load(p):
    with open(p) as f:
        return json.load(f)


GOAL_BLOCK = re.compile(r"\(:goal(.*?)\n\s*\)", re.S)
PREDICATE = re.compile(r"\(([A-Za-z][A-Za-z0-9_]*)\s+([A-Za-z0-9_]+)\s+([A-Za-z0-9_]+)\s*\)")


def parse_goal(bddl_path):
    """Return list of (predicate, arg1, arg2) from the BDDL :goal block."""
    with open(bddl_path) as f:
        txt = f.read()
    m = GOAL_BLOCK.search(txt)
    if not m:
        return None
    block = m.group(1)
    # Drop the wrapping (And ...) connective; keep only real 2-arg predicates.
    return [t for t in PREDICATE.findall(block) if t[0].lower() != "and"]


def audit_suite(suite, sub):
    base = os.path.join(ROOT, sub)
    print(f"\n=== {suite}  ->  {sub} ===")

    if not os.path.isdir(base):
        fail(f"[{suite}] DIRECTORY MISSING: {base}  "
             f"(normalize_validation_schema.py would silently SKIP this suite)")
        print("  MISSING — skipping")
        return []

    scenes = sorted(glob.glob(os.path.join(base, "scene_*")))
    scenes = [s for s in scenes if os.path.isdir(s)]
    print(f"  scenes on disk: {len(scenes)}")
    want = EXPECTED_SCENES[suite]
    if len(scenes) != want:
        fail(f"[{suite}] expected {want} scenes, found {len(scenes)}")

    # All three suites must be packaged identically.
    for extra in SUITE_FILES:
        if not os.path.exists(os.path.join(base, extra)):
            fail(f"[{suite}] suite file missing: {extra}"
                 + ("  (run normalize_validation_schema.py)"
                    if extra == "manifest_canonical.json" else
                    "  (run package_goal_suite.py)"
                    if extra in ("DATASHEET.md", "contact_sheet.png") else ""))

    rows = []
    n_missing_files = 0
    n_bddl_mismatch = 0
    n_missing_init_state = 0
    nonrepro = load_nonreproducible_keys()

    for sd in scenes:
        tag = f"{suite}/{os.path.basename(sd)}"

        for fn in CORE_FILES:
            if not os.path.exists(os.path.join(sd, fn)):
                fail(f"[{tag}] missing file {fn}")
                n_missing_files += 1

        # --- init_state.npz: present with the three expected arrays, UNLESS
        # this scene is a recorded rung-3 give-up (SCHEMA.md, ROLLOUT.md issue 1)
        ip = os.path.join(sd, "init_state.npz")
        dir_name = os.path.basename(sd)
        if (suite, dir_name) in nonrepro:
            if os.path.exists(ip):
                fail(f"[{tag}] init_state.npz present but scene is listed in "
                     f"nonreproducible.json — inconsistent")
        elif not os.path.exists(ip):
            fail(f"[{tag}] missing init_state.npz (and not listed in nonreproducible.json)")
            n_missing_init_state += 1
        else:
            try:
                with zipfile.ZipFile(ip) as z:
                    names = set(z.namelist())
                    missing = [m for m in INIT_STATE_MEMBERS if m not in names]
                    if missing:
                        fail(f"[{tag}] init_state.npz missing array(s): {missing}")
                    tiny = [m for m in INIT_STATE_MEMBERS
                            if m in names and z.getinfo(m).file_size <= 128]
                    if tiny:
                        fail(f"[{tag}] init_state.npz array(s) empty (header-only): {tiny}")
            except zipfile.BadZipFile:
                fail(f"[{tag}] init_state.npz is not a valid archive (truncated write?)")
                n_missing_init_state += 1

        mp = os.path.join(sd, "meta.json")
        bp = os.path.join(sd, "scene.bddl")
        if not (os.path.exists(mp) and os.path.exists(bp)):
            continue

        try:
            m = load(mp)
        except Exception as e:
            fail(f"[{tag}] meta.json unreadable: {e}")
            continue

        # --- canonical block present? ---
        # The full contract from SCHEMA.md. meta.json is declared a SUPERSET of
        # this block, so every one of these must be present in every scene.
        # (The Object builder shipped 38 scenes with `language` but no
        # `instruction`; only checking a subset here is how that went unnoticed.)
        for k in CANONICAL_FIELDS:
            if k not in m:
                fail(f"[{tag}] canonical field missing from meta.json: {k}")

        if not str(m.get("instruction", "")).strip():
            fail(f"[{tag}] instruction is empty")

        # --- target silhouette must not touch the frame border ---
        vmp = os.path.join(sd, "target_vismask.png")
        if _PILImage is not None and os.path.exists(vmp):
            px = _PILImage.open(vmp).convert("L")
            w, h = px.size
            dat = list(px.getdata())
            top    = any(dat[:w])
            bottom = any(dat[(h - 1) * w:])
            left   = any(dat[y * w] for y in range(h))
            right  = any(dat[y * w + w - 1] for y in range(h))
            if top or bottom or left or right:
                fail(f"[{tag}] target silhouette touches the frame edge "
                     f"(clipped target; visibility metric cannot see this)")

        if m.get("suite") != suite:
            fail(f"[{tag}] meta suite={m.get('suite')!r} but lives in {sub}")

        # --- THE AUDIT: BDDL on disk vs canonical fields ---
        goals = parse_goal(bp)
        if not goals:
            fail(f"[{tag}] could not parse a goal predicate from scene.bddl")
        else:
            if len(goals) > 1:
                notes.append(f"[{tag}] {len(goals)} goal predicates; audited the first")
            pred, a1, a2 = goals[0]
            if pred != m.get("goal_predicate"):
                fail(f"[{tag}] goal_predicate meta={m.get('goal_predicate')!r} "
                     f"!= bddl={pred!r}")
                n_bddl_mismatch += 1
            if a1 != m.get("target"):
                fail(f"[{tag}] target meta={m.get('target')!r} != bddl arg1={a1!r}")
                n_bddl_mismatch += 1
            if a2 != m.get("destination_region"):
                fail(f"[{tag}] destination_region meta={m.get('destination_region')!r} "
                     f"!= bddl arg2={a2!r}")
                n_bddl_mismatch += 1

        # --- tokens.json must agree with meta.json ---
        tp = os.path.join(sd, "tokens.json")
        if os.path.exists(tp):
            try:
                t = load(tp)
                for k in ("target", "destination", "destination_region",
                          "goal_predicate", "suite", "tier"):
                    if k in t and t[k] != m.get(k):
                        fail(f"[{tag}] tokens.json {k}={t[k]!r} != meta {m.get(k)!r}")
            except Exception as e:
                fail(f"[{tag}] tokens.json unreadable: {e}")

        rows.append((suite, os.path.basename(sd)))

    print(f"  missing core files : {n_missing_files}")
    print(f"  bddl mismatches    : {n_bddl_mismatch}")
    print(f"  missing init_state : {n_missing_init_state} (nonreproducible: {len(nonrepro)})")
    return rows


def main():
    print("Sketch-Prompted VLA validation-set audit (read-only)")
    if _PILImage is None:
        print("NOTE: Pillow not installed -> frame-edge clipping check SKIPPED "
              "(pip install pillow)")
    print(f"root: {ROOT}")
    if not os.path.isdir(ROOT):
        sys.exit(f"outputs root not found: {ROOT}")

    all_rows = []
    for suite, sub in SETS.items():
        all_rows += audit_suite(suite, sub)

    # --- combined manifest cross-check ---
    print("\n=== combined manifest ===")
    cp = os.path.join(ROOT, "validation_manifest_all.json")
    if not os.path.exists(cp):
        fail("validation_manifest_all.json missing")
    else:
        comb = load(cp)
        print(f"  rows: {len(comb)}")
        counts = {}
        for r in comb:
            counts[r.get("suite")] = counts.get(r.get("suite"), 0) + 1
        print(f"  per suite: {counts}")

        for suite in SETS:
            if counts.get(suite, 0) == 0:
                fail(f"combined manifest contains NO {suite} rows — "
                     f"a silent skip during normalisation")

        keys = {(r.get("suite"), r.get("dir")) for r in comb}
        if len(keys) != len(comb):
            fail(f"combined manifest (suite, dir) not unique: "
                 f"{len(comb)} rows, {len(keys)} unique keys")

        on_disk = set(all_rows)
        if keys != on_disk:
            only_manifest = sorted(keys - on_disk)[:5]
            only_disk = sorted(on_disk - keys)[:5]
            fail(f"manifest/disk mismatch — in manifest only: {only_manifest}; "
                 f"on disk only: {only_disk}")

    # --- report ---
    print("\n" + "=" * 60)
    if notes:
        print(f"NOTES ({len(notes)}):")
        for n in notes:
            print("  -", n)
    if problems:
        print(f"\nPROBLEMS ({len(problems)}):")
        for p in problems:
            print("  !", p)
        print("\nAUDIT FAILED")
        return 1
    print(f"\nAUDIT CLEAN — {len(all_rows)} scenes, every scene.bddl goal agrees "
          f"with its canonical meta.json/tokens.json fields.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
