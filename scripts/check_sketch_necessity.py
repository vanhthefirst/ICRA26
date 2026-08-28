#!/usr/bin/env python3
"""Can this scene set teach sketch-following at all? Answer before spending a GPU.

`pcla_v4_gate` (28 Aug) opened the sketch gate for the first time -- tanh went
from 1.8e-4 to 0.085 and held for 1500 steps -- and the sketch was still ignored
(`sketch_l2` flat, and a noise-pinned real-mask ablation at -0.00001). The cause
was not the architecture. It was the corpus:

    sketch_libero_rlds = 10 LIBERO-Spatial scenes x 45 demos, and within a scene
    the circle sits in 1-4 cells at 16 px quantisation across all 45 episodes.

One target per layout means the image alone determines the action, so ignoring
the sketch costs nothing in loss. Three configuration retrains and one
architecture change were spent before anyone measured that.

This script is the guard. A corpus is admissible only if the sketch is NOT
predictable from the image, and the BDDL files already say whether it is:

    layout key   (:objects) + (:init)   -- what the camera sees
    sketch key   (:goal) target, dest   -- what the circle and arrow encode

If every episode sharing a layout also shares a sketch key, the sketch is
redundant with the image and no amount of training will make the model read it.

    python3 scripts/check_sketch_necessity.py <dir-of-bddl> [<dir> ...]

Stdlib only, so it runs on a bare pod before any environment is built.
"""

import collections
import pathlib
import re
import sys


def _block(text: str, name: str) -> str:
    """Return the body of a top-level `(:name ...)` block, brace-balanced."""
    m = re.search(rf"\(:{name}\b", text)
    if not m:
        return ""
    i, depth = m.start(), 0
    for j in range(i, len(text)):
        if text[j] == "(":
            depth += 1
        elif text[j] == ")":
            depth -= 1
            if depth == 0:
                return text[i:j + 1]
    return text[i:]


def parse(path: pathlib.Path) -> dict | None:
    text = path.read_text(errors="replace")

    # `akita_black_bowl_1 akita_black_bowl_2 - akita_black_bowl` -- LIBERO packs
    # several instances of a category onto one line, so the left side must be
    # split, not matched as a single token. Getting this wrong silently drops the
    # duplicated objects, which are the only ones that make a scene ambiguous.
    pairs = []
    for lhs, cat in re.findall(r"^\s*(.+?)\s+-\s+(\S+)\s*$", _block(text, "objects"), re.M):
        pairs.extend((inst, cat) for inst in lhs.split())
    category = dict(pairs)                      # instance -> category
    init_raw = re.findall(r"\((\w+)\s+([\w\-]+)\s+([\w\-]+)\)", _block(text, "init"))
    # Layout is keyed on CATEGORY and region, never on instance number. Two
    # identical black bowls are interchangeable to the camera, so a scene that
    # merely swaps which one is called `_1` is the SAME picture -- and if its
    # demos then reach a different bowl, that is precisely the target variation
    # a sketch corpus needs. Keying on instance names hides exactly that case,
    # which is how the first version of this script would have mis-cleared
    # libero_spatial.
    objects = tuple(sorted(collections.Counter(c for _, c in pairs).items()))
    init = tuple(sorted((p_, category.get(o, o), r) for p_, o, r in init_raw))
    region_of = {o: r for _, o, r in init_raw}
    goal_preds = re.findall(r"\((\w+)\s+([\w\-]+)(?:\s+([\w\-]+))?\)",
                            _block(text, "goal"))
    # Drop the wrapper (And ...) and any predicate without a destination:
    # Open/Turnon give the arrow nothing to point at, so they carry no sketch.
    goal_preds = [g for g in goal_preds if g[0].lower() not in ("and", "goal") and g[2]]
    if not goal_preds:
        return None

    lang = re.search(r"\(:language\s+([^)]*)\)", text)
    pred, target, dest = goal_preds[0]
    # The sketch is a pair of PLACES, not names: the circle encloses whatever sits
    # at the target's region and the arrow points at the destination's.
    sketch = (category.get(target, target), region_of.get(target, target),
              category.get(dest, dest), region_of.get(dest, dest))
    return {
        "path": path,
        "layout": (objects, init),
        "sketch": sketch,
        "predicate": pred,
        "language": (lang.group(1).strip() if lang else ""),
    }


def report(root: pathlib.Path) -> bool:
    files = sorted(root.rglob("*.bddl"))
    parsed, skipped = [], 0
    for f in files:
        r = parse(f)
        parsed.append(r) if r else None
        skipped += r is None
    if not parsed:
        print(f"{root}: no usable BDDL files ({len(files)} seen, {skipped} without a destination)")
        return False

    by_layout = collections.defaultdict(set)
    for r in parsed:
        by_layout[r["layout"]].add(r["sketch"])

    counts = collections.Counter(len(v) for v in by_layout.values())
    n_multi = sum(n for k, n in counts.items() if k >= 2)
    admissible = n_multi > len(by_layout) / 2

    print(f"\n=== {root} ===")
    print(f"  scenes usable {len(parsed)}   skipped (no destination) {skipped}")
    print(f"  distinct layouts {len(by_layout)}")
    print(f"  distinct sketches per layout -> #layouts: {dict(sorted(counts.items()))}")
    for layout, sk in sorted(by_layout.items(), key=lambda kv: -len(kv[1]))[:5]:
        objs = ", ".join(f"{c}x{n}" for c, n in layout[0])[:70]
        print(f"    sketches={len(sk):3d}  objects: {objs}")
        for tc, tr, dc, dr in sorted(sk)[:4]:
            print(f"        circle={tc}@{tr}"[:58].ljust(60) + f"arrow={dc}@{dr}"[:40])
    print(f"  VERDICT: {'ADMISSIBLE' if admissible else 'NOT ADMISSIBLE'} "
          f"-- {n_multi}/{len(by_layout)} layouts carry 2+ distinct sketches")
    if not admissible:
        print("  A layout with one sketch teaches the model that the image alone")
        print("  determines the action. Training on it reproduces pcla_v2/v3/v4.")
    return admissible


if __name__ == "__main__":
    roots = [pathlib.Path(a) for a in sys.argv[1:]]
    if not roots:
        sys.exit(__doc__)
    ok = [report(r) for r in roots]
    sys.exit(0 if all(ok) else 1)
