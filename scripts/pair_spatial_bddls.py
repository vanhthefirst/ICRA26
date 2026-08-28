#!/usr/bin/env python3
"""Make LIBERO-Spatial sketch-admissible by pairing its layouts.

All ten shipped Spatial tasks place the fixtures and clutter identically —
plate, cookies, ramekin, cabinet and stove never move. A scene is therefore
fully determined by the SET of regions its two identical black bowls occupy,
and the shipped stove/cabinet pair is ambiguous for exactly that reason:
both put bowls at {flat_stove_1_cook_region, wooden_cabinet_1_top_side} and
differ only in which instance is the target.

That template generalises without adding a single object. For each pair of
tasks (Ti targeting rI, Tj targeting rJ), move each task's DISTRACTOR bowl
to the partner's target region, so both scenes show bowls at {rI, rJ}. The
demos are untouched — bowl_1 and the trajectory reaching it stay as shipped —
and the stored sim states keep their dimension, so demo replay only needs a
7-DoF pose override on bowl_2's free joint (see replay test in the runbook).

Pairs, chosen to minimise edits (— means the shipped file is already right):

    on_stove            <-> on_wooden_cabinet     —  /  —   (the shipped pair)
    next_to_plate       <-> next_to_ramekin       —  /  bowl_2 -> next_to_plate_region
    on_cookie_box       <-> on_ramekin            bowl_2 -> on ramekin  /  —
    between_plate_ramekin <-> table_center        bowl_2 -> table_center  /  bowl_2 -> between
    in_top_drawer       <-> next_to_cookie_box    bowl_2 -> next_to_box  /  bowl_2 -> In drawer

The drawer pair is PROVISIONAL: both scenes show one visible bowl and a
closed cabinet, and the sketch must mark either the visible bowl or the
drawer face. Physically valid (the shipped drawer task already inits a bowl
inside), but the hidden-target semantics differ from the other pairs.

Writes the ten paired BDDLs and runs nothing else. Verify with:

    python3 scripts/check_sketch_necessity.py <outdir>

Usage:
    python3 scripts/pair_spatial_bddls.py <shipped-libero_spatial-dir> <outdir>
"""

import pathlib
import re
import sys

# task-name fragment -> (old bowl_2 init predicate, new bowl_2 init predicate)
EDITS = {
    "next_to_the_ramekin": (
        "(On akita_black_bowl_2 main_table_next_to_box_region)",
        "(On akita_black_bowl_2 main_table_next_to_plate_region)",
    ),
    "on_the_cookie_box": (
        "(On akita_black_bowl_2 wooden_cabinet_1_top_side)",
        "(On akita_black_bowl_2 glazed_rim_porcelain_ramekin_1)",
    ),
    "between_the_plate_and_the_ramekin": (
        "(On akita_black_bowl_2 main_table_next_to_ramekin_region)",
        "(On akita_black_bowl_2 main_table_table_center)",
    ),
    "from_table_center": (
        "(On akita_black_bowl_2 main_table_next_to_plate_region)",
        "(On akita_black_bowl_2 main_table_between_plate_ramekin_region)",
    ),
    "in_the_top_drawer": (
        "(On akita_black_bowl_2 wooden_cabinet_1_top_side)",
        "(On akita_black_bowl_2 main_table_next_to_box_region)",
    ),
    "next_to_the_cookie_box": (
        "(On akita_black_bowl_2 flat_stove_1_cook_region)",
        "(In akita_black_bowl_2 wooden_cabinet_1_top_region)",
    ),
}


def main(src: pathlib.Path, dst: pathlib.Path) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    files = sorted(src.glob("*.bddl"))
    if len(files) != 10:
        sys.exit(f"expected the 10 shipped libero_spatial BDDLs in {src}, found {len(files)}")
    for f in files:
        text = f.read_text()
        # Longest fragment first: "next_to_the_cookie_box" must not be
        # shadowed by "on_the_cookie_box" or vice versa.
        frag = next((k for k in sorted(EDITS, key=len, reverse=True) if k in f.name), None)
        if frag:
            old, new = EDITS[frag]
            if old not in text:
                sys.exit(f"{f.name}: expected init predicate not found: {old}")
            text = text.replace(old, new)
            print(f"edited    {f.name}\n          bowl_2: {old} -> {new}")
        else:
            print(f"unchanged {f.name}")
        (dst / f.name).write_text(text)
    print(f"\nwrote {len(files)} files to {dst}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    main(pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2]))
