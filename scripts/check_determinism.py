"""
Sketch-Prompted VLA — how much of a scripted rollout number is noise.

`ScriptedSketchOracle` is a pure function of its prompt and ignores its
`rng_seed`, so the same prompt run twice ought to give the same row. On
`human_r1` it did not: `human:aaron_test` and `human_consensus` were the same
36 prompts under two labels — verified byte-identical `symbolic_tokens` — and
they disagreed on `spatial/scene_0012`, 7/36 against 8/36 (ROLLOUT.md, "A
determinism failure, run by accident"). That was one flip found by accident,
with no denominator behind it and no mechanism attached.

This script supplies both. It reads a run whose sketch conditions were rolled
out several times each (`--n-rollouts N`, N > 1) and reports:

  * the FLIP RATE — the fraction of (scene, condition) groups whose outcome is
    not constant across the N repeats, with a Wilson 95% interval, pooled and
    split by suite, tier and condition;
  * PAIRWISE DISAGREEMENT — for every pair of rollout indices, how many scenes
    they differ on. This is the quantity the original failure was an instance
    of, so it is the one that says how surprised to be by "two identical runs
    disagreed on 1 of 36";
  * the HEADLINE SPREAD — the success rate each rollout index would have
    reported on its own. A single-rollout run is one draw from this spread, so
    its width is the error bar every `--n-rollouts 1` number in ROLLOUT.md
    silently carries;
  * a MECHANISM cross-tab over the three state fingerprints
    (`init_state_hash`, `init_warmstart_hash`, `final_state_hash`), which
    separates "the scene was not restored" from "the scene was restored and the
    trajectory diverged anyway" from "the sim matched and only the scoring
    flipped". Runs written before those columns existed are handled, and say so.

Read-only, pure stdlib. It scores a run; it does not produce one.

    python scripts/check_determinism.py --run determinism_r1 --json
    python scripts/check_determinism.py --run determinism_r1 --field correct_instance_grasped
    # reproduce the original accident: two labels, one prompt, treated as repeats
    python scripts/check_determinism.py --run human_r1 \
        --alias-conditions human:aaron_test,human_consensus
"""

import argparse
import collections
import csv
import itertools
import json
import math
import os

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUN_ROOT = os.path.join(_REPO, "outputs", "rollouts")

# The outcome columns worth checking for a flip. `success_sustained` is the
# headline; the others localise a flip to a stage — a run that flips on
# `correct_instance_grasped` parted at the grasp, one that flips only on
# `success_sustained` parted at the place or in the success window itself.
BOOL_FIELDS = ["success_sustained", "success_final", "grasped_any",
               "correct_instance_grasped", "correct_destination"]
STR_FIELDS = ["grasped_instance", "nearest_destination"]
HASH_FIELDS = ["init_state_hash", "init_warmstart_hash", "final_state_hash"]


def load(run_id):
    path = os.path.join(RUN_ROOT, run_id, "results.csv")
    if not os.path.exists(path):
        raise SystemExit(f"[error] no results.csv in {os.path.join(RUN_ROOT, run_id)}")
    with open(path) as f:
        rows = [r for r in csv.DictReader(f) if r.get("skipped") != "True"]
    if not rows:
        raise SystemExit(f"[error] {path} has no scored rows")
    return rows


def wilson(k, n, z=1.96):
    """95% interval on a proportion. Wilson rather than normal-approximation
    because the expected flip rate here is a few percent on a few hundred
    groups, where the normal interval runs below zero and stops meaning
    anything."""
    if n == 0:
        return (None, None)
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, centre - half), min(1.0, centre + half))


def fmt_pct(x, nd=1):
    return "--" if x is None else f"{100.0 * x:.{nd}f}%"


def group_rows(rows):
    """(suite, dir, condition) -> {rollout_idx: row}, repeats only."""
    g = collections.defaultdict(dict)
    for r in rows:
        g[(r["suite"], r["dir"], r["condition"])][int(r["rollout_idx"])] = r
    return {k: v for k, v in g.items() if len(v) >= 2}


def constant(values):
    """True if every value is the same. Empty strings are kept as themselves —
    a column that is blank on every row is constant, which is correct, and the
    caller decides whether a wholly-blank column is worth reporting."""
    vals = list(values)
    return all(v == vals[0] for v in vals)


def analyse(rows, field):
    groups = group_rows(rows)
    if not groups:
        raise SystemExit(
            "[error] every (scene, condition) has a single rollout, so there is "
            "nothing to compare. Re-run with --n-rollouts N (N > 1).")

    has_hashes = all(h in rows[0] for h in HASH_FIELDS)

    per_group = []
    for (suite, dir_, cond), by_idx in sorted(groups.items()):
        idxs = sorted(by_idx)
        vals = [by_idx[i].get(field, "") for i in idxs]
        rec = dict(suite=suite, dir=dir_, condition=cond, n=len(idxs),
                   values=vals, flipped=not constant(vals),
                   tier=by_idx[idxs[0]].get("tier", "?"))
        if has_hashes:
            for h in HASH_FIELDS:
                rec[h + "_same"] = constant([by_idx[i].get(h, "") for i in idxs])
        # terminal_dist_xy spread: a group can agree on the boolean and still
        # show the trajectories were not identical, which is a softer and more
        # sensitive divergence signal than the outcome flip.
        dists = []
        for i in idxs:
            v = by_idx[i].get("terminal_dist_xy", "")
            if v not in ("", None):
                try:
                    dists.append(float(v))
                except ValueError:
                    pass
        rec["dist_spread"] = (max(dists) - min(dists)) if len(dists) >= 2 else None
        per_group.append(rec)

    return per_group, has_hashes


def breakdown(per_group, key):
    out = {}
    buckets = collections.defaultdict(list)
    for g in per_group:
        buckets[g[key]].append(g)
    for name, gs in sorted(buckets.items()):
        k = sum(1 for g in gs if g["flipped"])
        lo, hi = wilson(k, len(gs))
        out[name] = dict(n_groups=len(gs), n_flipped=k, rate=k / len(gs),
                         ci95=[lo, hi])
    return out


def pairwise(rows, field):
    """For each condition and each pair of rollout indices, the number of scenes
    the two indices disagree on. This is the original failure's own statistic:
    `human:aaron_test` vs `human_consensus` was one such pair, at 1 of 36."""
    by_cond = collections.defaultdict(lambda: collections.defaultdict(dict))
    for r in rows:
        by_cond[r["condition"]][int(r["rollout_idx"])][(r["suite"], r["dir"])] = r.get(field, "")
    out = {}
    for cond, by_idx in sorted(by_cond.items()):
        idxs = sorted(by_idx)
        if len(idxs) < 2:
            continue
        pairs = []
        for a, b in itertools.combinations(idxs, 2):
            shared = set(by_idx[a]) & set(by_idx[b])
            diff = sum(1 for s in shared if by_idx[a][s] != by_idx[b][s])
            pairs.append(dict(a=a, b=b, n_shared=len(shared), n_diff=diff))
        counts = sorted(p["n_diff"] for p in pairs)
        mid = len(counts) // 2
        out[cond] = dict(
            n_pairs=len(pairs),
            n_scenes=pairs[0]["n_shared"],
            min=counts[0], max=counts[-1],
            median=(counts[mid] if len(counts) % 2 else
                    (counts[mid - 1] + counts[mid]) / 2),
            mean=sum(counts) / len(counts),
            pairs=pairs)
    return out


def headline_spread(rows, field):
    """What each rollout index would have reported on its own. A --n-rollouts 1
    run is one draw from this."""
    by_cond = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in rows:
        by_cond[r["condition"]][int(r["rollout_idx"])].append(r.get(field, ""))
    out = {}
    for cond, by_idx in sorted(by_cond.items()):
        rates = {i: sum(1 for v in vs if v == "True") / len(vs)
                 for i, vs in sorted(by_idx.items()) if vs}
        if len(rates) < 2:
            continue
        vs = list(rates.values())
        out[cond] = dict(by_rollout_idx=rates, min=min(vs), max=max(vs),
                         spread_pp=100.0 * (max(vs) - min(vs)),
                         mean=sum(vs) / len(vs), n_scenes=len(next(iter(by_idx.values()))))
    return out


def mechanism(per_group):
    """Cross-tab: where the repeats parted, against whether the outcome flipped.

    The four rows are mutually exclusive and are read outward from the start of
    the rollout, because an earlier difference explains every later one."""
    tab = collections.OrderedDict(
        (k, dict(n=0, n_flipped=0)) for k in
        ["init state differs",
         "init state same, warm start differs",
         "start identical, final state differs",
         "fully identical"])
    for g in per_group:
        if not g.get("init_state_hash_same", True):
            k = "init state differs"
        elif not g.get("init_warmstart_hash_same", True):
            k = "init state same, warm start differs"
        elif not g.get("final_state_hash_same", True):
            k = "start identical, final state differs"
        else:
            k = "fully identical"
        tab[k]["n"] += 1
        tab[k]["n_flipped"] += int(g["flipped"])
    return tab


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True,
                    help="run id under outputs/rollouts/ (e.g. determinism_r1)")
    ap.add_argument("--field", default="success_sustained",
                    help="outcome column whose stability is measured "
                         "(default success_sustained, the headline)")
    ap.add_argument("--conditions", default=None,
                    help="comma list; default every condition in the run. "
                         "text_only is genuinely stochastic and is dropped "
                         "unless named explicitly.")
    ap.add_argument("--alias-conditions", default=None, metavar="A,B[,C]",
                    help="treat these conditions as repeats of ONE condition. "
                         "For the case the original failure was found in: two "
                         "labels carrying byte-identical prompts (human:<name> "
                         "and human_consensus with a single annotator). Only "
                         "pass conditions you have checked are the same prompt "
                         "— nothing here can verify that, and aliasing two "
                         "genuinely different sketches would report their real "
                         "difference as harness noise.")
    ap.add_argument("--json", action="store_true",
                    help="also write determinism.json into the run directory")
    args = ap.parse_args()

    rows = load(args.run)
    if args.alias_conditions:
        alias = args.alias_conditions.split(",")
        pos = {c: i for i, c in enumerate(alias)}
        missing = [c for c in alias if c not in {r["condition"] for r in rows}]
        if missing:
            raise SystemExit(f"[error] not in this run: {', '.join(missing)}")
        merged = "|".join(alias)
        rows = [r for r in rows if r["condition"] in pos]
        for r in rows:
            # keep each source condition's own repeats distinct from the others'
            r["rollout_idx"] = int(r["rollout_idx"]) * len(alias) + pos[r["condition"]]
            r["condition"] = merged
        print(f"\n[alias] {' and '.join(alias)} folded into one condition "
              f"`{merged}`; their rollout indices are the repeats.")
    if args.conditions:
        wanted = set(args.conditions.split(","))
    else:
        # TextOnlyGuessPolicy draws a fresh sample per rollout index BY DESIGN,
        # so its variation is the measurement, not a defect. Including it would
        # put a ~30% "flip rate" next to the oracle's few percent and read as
        # though the harness were far worse than it is.
        wanted = {r["condition"] for r in rows if r.get("policy") != "text_guess"}
    dropped = sorted({r["condition"] for r in rows} - wanted)
    rows = [r for r in rows if r["condition"] in wanted]
    if not rows:
        raise SystemExit(f"[error] no rows left after --conditions {args.conditions}")

    per_group, has_hashes = analyse(rows, args.field)
    n_flip = sum(1 for g in per_group if g["flipped"])
    lo, hi = wilson(n_flip, len(per_group))

    print(f"\n=== determinism: {args.run} — field `{args.field}` ===")
    print(f"conditions: {', '.join(sorted(wanted))}"
          + (f"   (dropped as stochastic by design: {', '.join(dropped)})" if dropped else ""))
    print(f"groups (scene x condition) with >=2 repeats: {len(per_group)}")
    print(f"\nFLIP RATE  {n_flip}/{len(per_group)} = {fmt_pct(n_flip / len(per_group))}"
          f"   95% CI [{fmt_pct(lo)}, {fmt_pct(hi)}]")

    for key, title in [("condition", "by condition"), ("suite", "by suite"),
                       ("tier", "by tier")]:
        print(f"\n  {title}")
        for name, b in breakdown(per_group, key).items():
            print(f"    {name:<22} {b['n_flipped']:>3}/{b['n_groups']:<4} "
                  f"{fmt_pct(b['rate']):>7}   CI [{fmt_pct(b['ci95'][0])}, {fmt_pct(b['ci95'][1])}]")

    print("\nPAIRWISE DISAGREEMENT (scenes differing between two rollout indices)")
    pw = pairwise(rows, args.field)
    for cond, p in pw.items():
        print(f"  {cond:<22} over {p['n_scenes']} scenes, {p['n_pairs']} index pairs: "
              f"min {p['min']}, median {p['median']}, mean {p['mean']:.2f}, max {p['max']}")

    print(f"\nHEADLINE SPREAD (`{args.field}` rate if only one rollout index were used)")
    hs = headline_spread(rows, args.field)
    for cond, h in hs.items():
        per = "  ".join(f"{i}:{fmt_pct(v)}" for i, v in h["by_rollout_idx"].items())
        print(f"  {cond:<22} min {fmt_pct(h['min'])}  max {fmt_pct(h['max'])}  "
              f"spread {h['spread_pp']:.1f}pp")
        print(f"    {per}")

    print("\nMECHANISM")
    if has_hashes:
        # The partition below is ORDERED and mutually exclusive, so a later row
        # reading 0 does NOT mean that fingerprint agreed -- it means those
        # groups were already claimed by an earlier row. Print the raw per
        # fingerprint counts first; without them "start identical, final state
        # differs: 0" reads as though the final states matched.
        for h in HASH_FIELDS:
            n = sum(1 for g in per_group if not g.get(h + "_same", True))
            print(f"  {h:<22} differs within group: {n}/{len(per_group)}")
        print()
    if not has_hashes:
        print("  state fingerprints not in this results.csv — it predates the "
              "init_state_hash / init_warmstart_hash / final_state_hash columns.\n"
              "  Flip rate above is still valid; where the repeats parted is not "
              "answerable from it.")
        mech = None
    else:
        mech = mechanism(per_group)
        for k, v in mech.items():
            share = fmt_pct(v["n"] / len(per_group))
            print(f"  {k:<40} {v['n']:>4} groups ({share:>6})   "
                  f"outcome flipped in {v['n_flipped']}")

    # Which outcome columns move at all. A flip that shows up on
    # `correct_instance_grasped` parted at the grasp; one that shows up only on
    # `success_sustained` parted later, at the place or inside the success
    # window. Cheap, and it saves re-running the script per column.
    print("\nWHICH STAGE MOVES")
    for f in BOOL_FIELDS + STR_FIELDS + ["z_pick", "z_place", "n_steps"]:
        if f not in rows[0]:
            continue
        gs, _ = analyse(rows, f)
        k = sum(1 for g in gs if g["flipped"])
        mark = "  <-- " + args.field if f == args.field else ""
        print(f"  {f:<26} {k:>4}/{len(gs)} groups{mark}")

    # The soft signal, and on the evidence so far the more honest one: the
    # boolean is a threshold over a continuous quantity, so agreement on it
    # hides how far apart the two runs actually finished.
    spreads = sorted(g["dist_spread"] for g in per_group
                     if g["dist_spread"] is not None)
    if spreads:
        nz = [s for s in spreads if s > 0]
        print(f"\nTERMINAL POSITION SPREAD (max - min of terminal_dist_xy per group)")
        print(f"  {len(nz)}/{len(spreads)} groups did not finish in the same place")
        if nz:
            mid = spreads[len(spreads) // 2]
            p90 = spreads[min(len(spreads) - 1, int(0.9 * len(spreads)))]
            print(f"  median {mid * 1000:.2f} mm   p90 {p90 * 1000:.2f} mm   "
                  f"max {spreads[-1] * 1000:.2f} mm")
            worst = max(per_group, key=lambda g: g["dist_spread"] or -1)
            print(f"  widest: {worst['suite']}/{worst['dir']} [{worst['condition']}]"
                  f"  {'(outcome flipped)' if worst['flipped'] else '(outcome held)'}")

    flipped = [g for g in per_group if g["flipped"]]
    if flipped:
        print(f"\nFLIPPED GROUPS ({len(flipped)})")
        for g in flipped:
            print(f"  {g['suite']}/{g['dir']:<12} {g['condition']:<22} "
                  f"tier={g['tier']:<12} {' '.join(g['values'])}")
    else:
        print("\nNo group flipped. On this sample the scripted path is reproducible.")

    if args.json:
        out = dict(run=args.run, field=args.field, conditions=sorted(wanted),
                   dropped_conditions=dropped, n_groups=len(per_group),
                   n_flipped=n_flip, flip_rate=n_flip / len(per_group),
                   flip_rate_ci95=[lo, hi],
                   by_condition=breakdown(per_group, "condition"),
                   by_suite=breakdown(per_group, "suite"),
                   by_tier=breakdown(per_group, "tier"),
                   pairwise=pw, headline_spread=hs,
                   mechanism=mech, has_state_hashes=has_hashes,
                   flipped_groups=[{k: g[k] for k in
                                    ("suite", "dir", "condition", "tier", "values")}
                                   for g in flipped])
        path = os.path.join(RUN_ROOT, args.run, "determinism.json")
        json.dump(out, open(path, "w"), indent=2)
        print(f"\nwrote {path}")
    print()


if __name__ == "__main__":
    main()
