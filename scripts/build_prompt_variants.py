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
  outputs/validation_set_<suite>/evaluation_rows.json         (76 rows)
  outputs/evaluation_rows_all.json                           (228 rows)

Idempotent: re-running overwrites the derived keys with the same values and
leaves everything else untouched. The scene manifests are not modified --
consumers keyed on (suite, dir) keep working. The evaluation manifests written
here are keyed on (suite, dir, prompt_type).

    python scripts/build_prompt_variants.py [--check]

--check verifies on-disk state matches what this script would write and exits
non-zero if not. It writes nothing.
"""
import argparse
import hashlib
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

# The ambiguous captions.
#
# An ambiguous caption must remove the object identities and nothing else. If it
# also changed the verb, the two arms would differ in what the robot is being
# asked to DO as well as in what it is being asked to do it TO, and the arm would
# no longer isolate referring expressions. So the bank is keyed on the SHAPE of
# the scene's own explicit caption, and each scene draws only from the bucket its
# explicit caption belongs to:
#
#   two_clause_On   "pick up the black bowl and place it on the plate"
#   two_clause_In   "grab the milk and put it in the basket"
#   one_clause_On   "put the bowl on the stove"
#   one_clause_In   (no scene currently; kept so the builder cannot silently fail)
#   push            "push the plate to the front of the stove"
#
# Within a bucket the wording varies -- verb synonyms the explicit captions
# already use (pick up / grab / take, put / place / set), and both bare and
# "that one" forms of the deictic. One template per predicate would measure the
# policy's response to one sentence; a bank measures its response to ambiguity as
# a class, and `prompt_template_id` lets the analysis check whether the wording
# mattered at all.
#
# Word order is fixed across every template: the FIRST deictic is the target the
# circle encloses, the SECOND is the destination the arrow points at. The sketch
# arms in stage 5 append geometry to these strings and depend on that order.
#
# No template may contain a word that appears in any scene's object names --
# which is why "on top of that" and "to the front of that" are absent, however
# natural they read: `top` and `front` occur in `wooden_cabinet_1_top_region`
# and `main_table_stove_front_region`. check_no_identity_leak() enforces this
# against the real vocabulary rather than against a hand-kept denylist.
AMBIGUOUS_BANK = {
    "two_clause_On": (
        "pick this up and place it on that",
        "grab this and put it on that",
        "pick it up and put it on that one",
        "grab it and place it on there",
        "take this and set it on that",
        "pick this up and set it onto that",
    ),
    "two_clause_In": (
        "pick this up and place it in that",
        "grab this and put it in that",
        "pick it up and put it in that one",
        "grab it and place it in there",
        "take this and put it inside that",
        "pick this up and put it into that",
    ),
    "one_clause_On": (
        "put this on that",
        "move this onto that",
        "put it on that one",
        "set this on that",
        "move it onto there",
        "place this on that one",
    ),
    "one_clause_In": (
        "put this in that",
        "move this into that",
        "put it in that one",
    ),
    "push": (
        "push this over to that",
        "push it towards that one",
    ),
}


# The `there` probe.
#
# The 17 August 2026 baselines found that ambiguous captions whose destination
# deictic is `there` scored 1.5% against 20.1% for the `that` family, while the
# SAME scenes under explicit captions scored 31.2% against 34.4%. Equal explicit
# rates with unequal ambiguous ones means the gap is in the word, not in the
# scenes -- a lexical artefact sitting inside a result about ambiguity.
#
# This derives a variant caption that swaps that one word and changes nothing
# else, so the size of the artefact can be measured rather than argued about.
# Only three templates contain `there`, so for every other scene the variant is
# identical to the ambiguous caption and the key is a no-op.
#
# It is deliberately NOT in PROMPT_TYPES. The benchmark is 114 scenes x 2
# captions = 228 evaluation rows; a diagnostic probe must not quietly redefine
# it, and an alternative phrasing is a separate labelled arm rather than a
# replacement for the one that ran.
def there_to_that(caption):
    return re.sub(r"\bthere\b", "that", caption)


def bucket_of(instruction, goal_predicate):
    """Which bank a scene draws from -- decided by its own explicit caption, so
    the ambiguous twin keeps the verb and the sentence shape."""
    text = instruction.strip().lower()
    if text.startswith("push"):
        return "push"
    shape = "two_clause" if " and " in text else "one_clause"
    return f"{shape}_{goal_predicate}"


def scene_vocabulary(scenes):
    """Every word that names something in any scene. Built from the object
    instances themselves, not typed out by hand, so a new suite cannot introduce
    a word the leak check does not know about."""
    vocab = set()
    for _, _, meta in scenes:
        names = [meta["target"], meta["destination"], meta["destination_region"]]
        names += list(meta.get("siblings") or [])
        names += list(meta.get("other_baskets") or meta.get("other_plates")
                      or meta.get("other_dests") or [])
        names += list((meta.get("all_pixels") or {}).keys())
        for name in names:
            for word in re.split(r"[_\d]+", str(name)):
                if len(word) > 2:
                    vocab.add(word.lower())
    return vocab


def check_no_identity_leak(vocab):
    """No caption in any bank may name anything that exists in a scene."""
    bad = []
    for bucket, templates in AMBIGUOUS_BANK.items():
        # The `there` variant is checked alongside the template it derives from:
        # a substitution that introduced a scene word would be just as wrong as
        # a hand-written template that did.
        for t in {x for t0 in templates for x in (t0, there_to_that(t0))}:
            hits = sorted(w for w in vocab if re.search(rf"\b{re.escape(w)}\b", t))
            if hits:
                bad.append(f"{bucket}: {t!r} contains scene word(s) {hits}")
    return bad


def assign_templates(scenes):
    """One template per scene, deterministic and balanced.

    Round-robin within (bucket, suite, tier) rather than over the whole set: the
    report compares tiers, so a template must not be able to concentrate in one
    tier and confound it. Scenes are sorted by dir first, so the assignment is
    reproducible and does not depend on directory listing order.

    Each group starts at its own offset in the bank. Without it every group
    begins at template 0, so a group smaller than the bank never reaches the
    tail of it -- the two `push` scenes sit in singleton groups and would both
    take push/00, leaving push/01 dead. blake2b rather than hash(): Python salts
    hash() of str per process, which is the bug that cost this project a
    reproducible baseline once already.
    """
    groups = {}
    for suite, dir_, meta in scenes:
        key = (bucket_of(meta["instruction"], meta["goal_predicate"]), suite, meta["tier"])
        groups.setdefault(key, []).append((dir_, meta))
    out = {}
    for key, members in groups.items():
        bucket, suite, _tier = key
        bank = AMBIGUOUS_BANK[bucket]
        digest = hashlib.blake2b("|".join(key).encode(), digest_size=8).digest()
        start = int.from_bytes(digest, "big") % len(bank)
        for i, (dir_, _meta) in enumerate(sorted(members)):
            idx = (start + i) % len(bank)
            out[(suite, dir_)] = (bucket, idx, bank[idx])
    return out


def scene_dirs(suite):
    d = os.path.join(ROOT, SETS[suite])
    return sorted(x for x in os.listdir(d) if re.fullmatch(r"scene_\d+", x))


def load_scenes():
    scenes = []
    for suite in SETS:
        for dir_ in scene_dirs(suite):
            p = os.path.join(ROOT, SETS[suite], dir_, "meta.json")
            scenes.append((suite, dir_, json.load(open(p))))
    return scenes


def derived_keys(meta, assignment):
    bucket, idx, caption = assignment
    tid = f"{bucket}/{idx:02d}"
    return {
        "instruction_explicit": meta["instruction"],
        "instruction_ambiguous": caption,
        "instruction_ambiguous_that": there_to_that(caption),
        "prompt_template_id": {"explicit": "authored",
                               "ambiguous": tid,
                               "ambiguous_that": f"{tid}+that"},
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
        "prompt_template_id": meta["prompt_template_id"][prompt_type],
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

    scenes = load_scenes()
    problems = check_no_identity_leak(scene_vocabulary(scenes))
    assignments = assign_templates(scenes)

    all_rows = []
    for suite in SETS:
        suite_rows = []
        for dir_ in scene_dirs(suite):
            sdir = os.path.join(ROOT, SETS[suite], dir_)
            meta = json.load(open(os.path.join(sdir, "meta.json")))
            keys = derived_keys(meta, assignments[(suite, dir_)])

            if keys["instruction_explicit"].strip() == keys["instruction_ambiguous"]:
                problems.append(f"{suite}/{dir_}: explicit and ambiguous captions identical")

            for name in ("tokens.json", "meta.json"):
                if not patch(os.path.join(sdir, name), keys, args.check):
                    problems.append(f"{suite}/{dir_}/{name}: prompt keys missing or stale")

            merged = dict(meta, **keys)
            suite_rows += [canonical_row(merged, suite, dir_, p) for p in PROMPT_TYPES]

        mpath = os.path.join(ROOT, SETS[suite], "evaluation_rows.json")
        if args.check:
            if (json.load(open(mpath)) if os.path.exists(mpath) else None) != suite_rows:
                problems.append(f"{mpath}: stale or missing")
        else:
            json.dump(suite_rows, open(mpath, "w"), indent=2)
        all_rows += suite_rows

    apath = os.path.join(ROOT, "evaluation_rows_all.json")
    if args.check:
        if (json.load(open(apath)) if os.path.exists(apath) else None) != all_rows:
            problems.append(f"{apath}: stale or missing")
    else:
        json.dump(all_rows, open(apath, "w"), indent=2)

    ids = [r["scene_id"] for r in all_rows]
    if len(set(ids)) != len(ids):
        problems.append("scene_id is not unique across the evaluation manifest")

    amb = [r for r in all_rows if r["prompt_type"] == "ambiguous"]
    verb = "checked" if args.check else "wrote"
    print(f"{verb} {len(all_rows)} rows "
          f"({len(all_rows) // len(PROMPT_TYPES)} scenes x {len(PROMPT_TYPES)} prompt types)")
    for suite in SETS:
        print(f"  {suite:8} {sum(1 for r in all_rows if r['suite'] == suite):3} rows")
    print(f"  {len({r['instruction'] for r in amb})} distinct ambiguous captions "
          f"from {len(AMBIGUOUS_BANK)} buckets")
    for bucket in sorted(AMBIGUOUS_BANK):
        n = sum(1 for r in amb if r["prompt_template_id"].startswith(bucket + "/"))
        if n:
            print(f"    {bucket:16} {n:3} scenes over "
                  f"{len({r['prompt_template_id'] for r in amb if r['prompt_template_id'].startswith(bucket + '/')})} templates")
    for p in problems:
        print(f"  PROBLEM {p}", file=sys.stderr)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
