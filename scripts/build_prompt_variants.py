"""
Sketch-Prompted VLA — add the prompt-quality axis to the validation sets.

Every scene already carries one authored instruction that names the object and
the destination by category ("pick up the black bowl and place it on the
plate"). Under the taxonomy in PROMPT_TAXONOMY.md that string is the EXPLICIT
prompt. This script derives the matching AMBIGUOUS prompt for the same scene --
same BDDL, same init_state.npz, same sketch, same tier -- and records both on
disk, so the two arms differ in the caption and in nothing else.

Writes:
  outputs/validation_set_<suite>/scene_XXXX/tokens.json   (+3 keys)
  outputs/validation_set_<suite>/scene_XXXX/meta.json     (+3 keys)
  outputs/validation_set_<suite>/evaluation_rows.json   (76 rows)
  outputs/evaluation_rows_all.json                     (228 rows)

Idempotent: re-running overwrites the derived keys with the same values and
leaves everything else untouched. The scene manifests are not modified --
consumers keyed on (suite, dir) keep working. The evaluation manifests written
here are keyed on (suite, dir, prompt_type).

    python scripts/build_prompt_variants.py [--check]

--check verifies on-disk state matches what this script would write and exits
non-zero if not. It writes nothing.
"""
import argparse
import json
import os
import re
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.join(_REPO, "outputs")

SETS = {
    "spatial": "validation_set_spatial",
    "object":  "validation_set_object",
    "goal":    "validation_set_goal",
}
PROMPT_TYPES = ("explicit", "ambiguous")

# The ambiguous caption. One template per goal predicate, and that is the whole
# vocabulary -- 114 scenes draw on two strings. A per-scene paraphrase would put
# a second uncontrolled variable next to the one being measured, and the arm
# would no longer isolate "the caption names nothing".
#
# The deictic pair maps onto the same two roles the explicit caption fills:
# "this" is the circled target, "that" is the destination the arrow points at.
# Sentence shape (verb, one object slot, one destination slot) is held constant
# against the explicit caption so what changes is referring expressions alone.
AMBIGUOUS_TEMPLATES = {
    "On": "move this onto that",
    "In": "move this into that",
}
TEMPLATE_ID = "deictic_v1"

# No content word in an ambiguous caption may name anything in the scene. The
# allow-list IS the guarantee: a caption is valid only if it is character-for-
# character one of the templates above, so a leak cannot be introduced by a
# typo in a template variable.
_ALLOWED = set(AMBIGUOUS_TEMPLATES.values())


def ambiguous_for(meta):
    pred = meta["goal_predicate"]
    if pred not in AMBIGUOUS_TEMPLATES:
        raise ValueError(f"no ambiguous template for goal_predicate {pred!r}")
    return AMBIGUOUS_TEMPLATES[pred]


def scene_dirs(suite):
    d = os.path.join(ROOT, SETS[suite])
    return sorted(x for x in os.listdir(d) if re.fullmatch(r"scene_\d+", x))


def derived_keys(meta):
    return {
        "instruction_explicit": meta["instruction"],
        "instruction_ambiguous": ambiguous_for(meta),
        "prompt_template_id": {"explicit": "authored", "ambiguous": TEMPLATE_ID},
    }


def patch(path, keys, check):
    obj = json.load(open(path))
    if check:
        return {k: obj.get(k) for k in keys} == keys
    obj.update(keys)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
    return True


def canonical_row(meta, suite, dir_, prompt_type):
    return {
        "scene_id": f"{suite}/{dir_}#{prompt_type}",
        "suite": suite,
        "dir": dir_,
        "prompt_type": prompt_type,
        "tier": meta["tier"],
        "instruction": meta[f"instruction_{prompt_type}"],
        "target": meta["target"],
        "destination": meta["destination"],
        "destination_region": meta["destination_region"],
        "goal_predicate": meta["goal_predicate"],
        "seed": meta["seed"],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()

    all_rows, problems = [], []
    for suite in SETS:
        suite_rows = []
        for dir_ in scene_dirs(suite):
            sdir = os.path.join(ROOT, SETS[suite], dir_)
            meta = json.load(open(os.path.join(sdir, "meta.json")))
            keys = derived_keys(meta)

            if keys["instruction_ambiguous"] not in _ALLOWED:
                problems.append(f"{suite}/{dir_}: ambiguous caption off the allow-list")
            if keys["instruction_explicit"].strip() == keys["instruction_ambiguous"]:
                problems.append(f"{suite}/{dir_}: explicit and ambiguous captions identical")

            for name in ("tokens.json", "meta.json"):
                if not patch(os.path.join(sdir, name), keys, args.check):
                    problems.append(f"{suite}/{dir_}/{name}: prompt keys missing or stale")

            merged = dict(meta, **keys)
            suite_rows += [canonical_row(merged, suite, dir_, p) for p in PROMPT_TYPES]

        mpath = os.path.join(ROOT, SETS[suite], "evaluation_rows.json")
        if args.check:
            on_disk = json.load(open(mpath)) if os.path.exists(mpath) else None
            if on_disk != suite_rows:
                problems.append(f"{mpath}: stale or missing")
        else:
            json.dump(suite_rows, open(mpath, "w"), indent=2)
        all_rows += suite_rows

    apath = os.path.join(ROOT, "evaluation_rows_all.json")
    if args.check:
        on_disk = json.load(open(apath)) if os.path.exists(apath) else None
        if on_disk != all_rows:
            problems.append(f"{apath}: stale or missing")
    else:
        json.dump(all_rows, open(apath, "w"), indent=2)

    ids = [r["scene_id"] for r in all_rows]
    if len(set(ids)) != len(ids):
        problems.append("scene_id is not unique across the evaluation manifest")

    verb = "checked" if args.check else "wrote"
    print(f"{verb} {len(all_rows)} rows "
          f"({len(all_rows) // len(PROMPT_TYPES)} scenes x {len(PROMPT_TYPES)} prompt types)")
    for suite in SETS:
        n = sum(1 for r in all_rows if r["suite"] == suite)
        print(f"  {suite:8} {n:3} rows")
    for p in problems:
        print(f"  PROBLEM {p}", file=sys.stderr)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
