"""Aggregate pi0.5 rollout runs into the numbers, the figure and the LaTeX
tables the reports need. Read-only over results.csv; writes analysis.json next to
each run, plus a tables fragment and figures under report/.

    # legacy single-arm behaviour (the 3-rollout 2026-08 baseline)
    python scripts/analyze_pi05_baseline.py

    # the two prompt arms of the v2 dataset, side by side
    python scripts/analyze_pi05_baseline.py \
        --arms explicit=pi05_explicit_532 ambiguous=pi05_ambiguous_532 \
        --tables report/pi05_baselines_v2/tables.tex \
        --figdir report/pi05_baselines_v2/figures

An arm whose results.csv is absent is not an error: its blocks come out empty,
its table rows print as `---`, and the report still compiles. That is deliberate
-- the report is written before the GPU run and filled by re-running this."""
import argparse
import collections
import csv
import json
import os
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUN_ROOT = os.path.join(_REPO, "outputs", "rollouts")

TIERS = ["control", "referential", "directional", "both"]
SUITES = ["spatial", "object", "goal"]
SUITE_DIR = {"spatial": "validation_set_spatial", "object": "validation_set_object",
             "goal": "validation_set_goal"}
# dataviz reference palette, categorical slots 1-3 (validated all-pairs, both modes)
C = {"success": "#2a78d6", "object": "#eb6834", "dest": "#1baf7a"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#d8d7d2"

_meta_cache = {}


def meta_of(suite, dir_):
    key = (suite, dir_)
    if key not in _meta_cache:
        p = os.path.join(_REPO, "outputs", SUITE_DIR[suite], dir_, "meta.json")
        _meta_cache[key] = json.load(open(p))
    return _meta_cache[key]


# Chance level, per scene. A policy given an explicit caption is told the target
# CATEGORY but not the instance, so on a scene with k same-category candidates
# the best it can do by guessing is 1/k. These floors describe the SCENE (how
# many candidates the tier puts in front of the gripper) and are therefore the
# same for both prompt arms -- what changes between the arms is whether the
# caption narrows the field to that category at all.
def chance(suite, dir_):
    m = meta_of(suite, dir_)
    n_pick = 1 + len(m.get("siblings") or [])
    n_place = 1 + len(m.get("other_baskets") or m.get("other_plates")
                      or m.get("other_dests") or [])
    return 1.0 / n_pick, 1.0 / n_place, 1.0 / (n_pick * n_place)


def pct(rs, field):
    rs = list(rs)
    return (100.0 * sum(1 for r in rs if r[field] == "True") / len(rs)) if rs else None


EMPTY = dict(n_rollouts=0, n_scenes=0, success=None, correct_object=None,
             grasped_any=None, correct_destination=None, chance_object=None,
             chance_dest=None, chance_joint=None, fail_wrong_object=None,
             fail_right_object=None, fail_no_grasp=None)


def block(rs):
    rs = list(rs)
    if not rs:
        return dict(EMPTY)
    ch = [chance(r["suite"], r["dir"]) for r in rs]
    fails = [r for r in rs if r["success_sustained"] != "True"]
    n = len(rs)
    return dict(
        n_rollouts=n, n_scenes=len({(r["suite"], r["dir"]) for r in rs}),
        success=pct(rs, "success_sustained"),
        correct_object=pct(rs, "correct_instance_grasped"),
        grasped_any=pct(rs, "grasped_any"),
        correct_destination=pct(rs, "correct_destination"),
        chance_object=100.0 * sum(c[0] for c in ch) / n,
        chance_dest=100.0 * sum(c[1] for c in ch) / n,
        chance_joint=100.0 * sum(c[2] for c in ch) / n,
        # failure decomposition, as a share of ALL rollouts in the block
        fail_wrong_object=100.0 * sum(
            1 for r in fails if r["grasped_any"] == "True"
            and r["correct_instance_grasped"] != "True") / n,
        fail_right_object=100.0 * sum(
            1 for r in fails if r["correct_instance_grasped"] == "True") / n,
        fail_no_grasp=100.0 * sum(1 for r in fails if r["grasped_any"] != "True") / n,
    )


def wrong_grasp_mix(rows):
    """What did it grab when it grabbed the wrong thing? A same-category sibling
    is referential confusion -- the failure a sketch addresses. Anything else is
    not."""
    wrong = collections.Counter()
    for r in rows:
        if r["grasped_any"] != "True" or r["correct_instance_grasped"] == "True":
            continue
        m = meta_of(r["suite"], r["dir"])
        g = r["grasped_instance"]
        sib = set(m.get("siblings") or [])
        dests = set([m["destination"]] + list(m.get("other_baskets")
                                              or m.get("other_plates")
                                              or m.get("other_dests") or []))
        wrong["sibling" if g in sib else "destination" if g in dests else "other"] += 1
    n = sum(wrong.values())
    return dict(n=n, **{k: dict(n=v, pct=round(100.0 * v / n, 1))
                        for k, v in wrong.items()})


def load_rows(run_dir, expect_prompt_type=None):
    path = os.path.join(run_dir, "results.csv")
    if not os.path.exists(path):
        return []
    rows = [r for r in csv.DictReader(open(path)) if r["skipped"] != "True"]
    if expect_prompt_type:
        # A run whose rows say `ambiguous` analysed as the explicit arm would
        # mislabel a whole table. Refuse rather than relabel. Rows written before
        # the prompt_type column existed carry "" and are taken at their word.
        seen = {r.get("prompt_type", "") for r in rows} - {""}
        if seen and seen != {expect_prompt_type}:
            sys.exit(f"[error] {path} holds prompt_type {sorted(seen)}, "
                     f"analysed as {expect_prompt_type!r}")
    return rows


def analyse(rows):
    return dict(
        overall=block(rows),
        by_suite={s: block([r for r in rows if r["suite"] == s]) for s in SUITES},
        by_tier={t: block([r for r in rows if r["tier"] == t]) for t in TIERS},
        by_suite_tier={f"{s}|{t}": block([r for r in rows if r["suite"] == s
                                          and r["tier"] == t])
                       for s in SUITES for t in TIERS},
        control=block([r for r in rows if r["tier"] == "control"]),
        other_tiers=block([r for r in rows if r["tier"] != "control"]),
        wrong_object_grasp_mix=wrong_grasp_mix(rows),
        by_template=by_template(rows),
    )


def template_of(suite, dir_, prompt_type):
    m = meta_of(suite, dir_)
    return (m.get("prompt_template_id") or {}).get(prompt_type)


def by_template(rows):
    """Success per ambiguous-caption template. The ambiguous arm draws on 20
    wordings, which is what stops one unlucky sentence from becoming the result
    -- but only if the spread is actually inspected. A template far off the
    others is a lexical artefact, not a finding about ambiguity, and this is
    where it shows up. Empty for the explicit arm, whose captions are authored
    per scene rather than drawn from a bank."""
    groups = {}
    for r in rows:
        t = template_of(r["suite"], r["dir"], r.get("prompt_type") or "explicit")
        if t and t != "authored":
            groups.setdefault(t, []).append(r)
    return {t: block(rs) for t, rs in sorted(groups.items())}


def f(x, nd=1):
    return "---" if x is None else f"{x:.{nd}f}"


# ------------------------------------------------------------------ latex ----
def suite_rows(a):
    out = []
    for s in SUITES:
        b = a["by_suite"][s]
        out.append(f"{s.capitalize()} & {b['n_scenes']} & {b['n_rollouts']} & "
                   f"{f(b['success'])} & {f(b['correct_object'])} & "
                   f"{f(b['correct_destination'])} \\\\")
    out.append("\\midrule")
    b = a["overall"]
    out.append(f"All & {b['n_scenes']} & {b['n_rollouts']} & {f(b['success'])} & "
               f"{f(b['correct_object'])} & {f(b['correct_destination'])} \\\\")
    return "\n".join(out)


def tier_rows(a):
    out = []
    for t in TIERS:
        b = a["by_tier"][t]
        out.append(f"{t.capitalize()} & {b['n_scenes']} & {b['n_rollouts']} & "
                   f"{f(b['success'])} & {f(b['correct_object'])} & "
                   f"{f(b['chance_object'])} & {f(b['correct_destination'])} & "
                   f"{f(b['chance_dest'])} \\\\")
    return "\n".join(out)


def write_tables(path, arms):
    lines = ["% generated by scripts/analyze_pi05_baseline.py -- do not edit by hand"]
    for name, a in arms.items():
        key = name.capitalize()
        lines += [f"\\newcommand{{\\tab{key}Suites}}{{%\n{suite_rows(a)}}}",
                  f"\\newcommand{{\\tab{key}Tiers}}{{%\n{tier_rows(a)}}}"]
        for macro, val in [("Overall", a["overall"]["success"]),
                           ("Control", a["control"]["success"]),
                           ("Other", a["other_tiers"]["success"]),
                           ("Grasped", a["overall"]["grasped_any"])]:
            lines.append(f"\\newcommand{{\\num{key}{macro}}}{{{f(val)}}}")
        mix = a["wrong_object_grasp_mix"]
        sib = mix.get("sibling", {}).get("pct")
        lines.append(f"\\newcommand{{\\num{key}Sibling}}{{{f(sib)}}}")
        lines.append(f"\\newcommand{{\\num{key}WrongN}}{{{mix.get('n', 0)}}}")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    open(path, "w").write("\n".join(lines) + "\n")
    print("wrote", path)


# ----------------------------------------------------------------- figure ----
def tier_figure(path, arms):
    """Sustained success by tier, one bar group per tier, one bar per arm. The
    v1 figure carried three metrics for one arm; with two arms that becomes
    six bars a group and stops being readable, so this keeps the metric the
    report quotes and puts the arms side by side."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    names = list(arms)
    fig, ax = plt.subplots(figsize=(7.6, 3.2), dpi=200)
    x = np.arange(len(TIERS))
    w = 0.8 / max(len(names), 1)
    colours = [C["success"], C["object"], C["dest"]]
    for i, name in enumerate(names):
        vals = [arms[name]["by_tier"][t]["success"] or 0.0 for t in TIERS]
        off = (i - (len(names) - 1) / 2) * w
        bars = ax.bar(x + off, vals, w * 0.9, label=f"{name} prompt",
                      color=colours[i % len(colours)], edgecolor="white",
                      linewidth=1.0, zorder=3)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 1.4, f"{v:.0f}", ha="center",
                    va="bottom", fontsize=7.6, color=INK2, zorder=4)

    ax.set_xticks(x)
    ax.set_xticklabels([f"{t}\n{arms[names[0]]['by_tier'][t]['n_scenes']} scenes"
                        for t in TIERS], fontsize=8.4)
    ax.set_ylabel("sustained success (%)", fontsize=8.6, color=INK2)
    ax.set_ylim(0, 106)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.tick_params(axis="y", labelsize=8, colors=INK2, length=0)
    ax.tick_params(axis="x", length=0, colors=INK)
    ax.grid(axis="y", color=GRID, linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.legend(frameon=False, fontsize=8.0, ncol=len(names), loc="upper center",
              bbox_to_anchor=(0.5, 1.15), labelcolor=INK2, columnspacing=1.4,
              handlelength=1.6)
    fig.tight_layout()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    print("wrote", path)


# ------------------------------------------------------------------- main ----
def report(name, a):
    print(f"\n===== {name} =====")
    b = a["overall"]
    print(f"  overall  scenes={b['n_scenes']:3} rollouts={b['n_rollouts']:4} "
          f"succ={f(b['success']):>5} obj={f(b['correct_object']):>5} "
          f"dest={f(b['correct_destination']):>5} grasped_any={f(b['grasped_any']):>5}")
    print("  -- by suite --")
    for s in SUITES:
        b = a["by_suite"][s]
        print(f"  {s:12} scenes={b['n_scenes']:3} rollouts={b['n_rollouts']:4} "
              f"succ={f(b['success']):>5} obj={f(b['correct_object']):>5} "
              f"dest={f(b['correct_destination']):>5}")
    print("  -- by tier --")
    for t in TIERS:
        b = a["by_tier"][t]
        print(f"  {t:12} scenes={b['n_scenes']:3} rollouts={b['n_rollouts']:4} "
              f"succ={f(b['success']):>5} obj={f(b['correct_object']):>5} "
              f"(chance {f(b['chance_object']):>5}) dest={f(b['correct_destination']):>5} "
              f"(chance {f(b['chance_dest']):>5})")
    mix = a["wrong_object_grasp_mix"]
    print(f"  wrong-object grasps n={mix['n']}: " + ", ".join(
        f"{k}={v['pct']}%" for k, v in mix.items() if isinstance(v, dict)))
    bt = a.get("by_template") or {}
    if bt:
        vals = [b["success"] for b in bt.values() if b["success"] is not None]
        print(f"  -- by caption template ({len(bt)} templates, spread "
              f"{f(min(vals))}-{f(max(vals))}) --" if vals else "  -- by template --")
        for t, b in bt.items():
            print(f"  {t:20} scenes={b['n_scenes']:3} rollouts={b['n_rollouts']:4} "
                  f"succ={f(b['success']):>5} obj={f(b['correct_object']):>5}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", default=["baseline=pi05_baseline"],
                    metavar="NAME=RUNDIR",
                    help="arm label and run directory under outputs/rollouts/. "
                         "NAME `explicit` or `ambiguous` also asserts the run's "
                         "prompt_type column matches.")
    ap.add_argument("--tables", default=None,
                    help="write a LaTeX macro fragment here")
    ap.add_argument("--figdir", default=os.path.join(_REPO, "report", "pi05_baseline",
                                                     "figures"))
    ap.add_argument("--no-figure", action="store_true")
    args = ap.parse_args()

    arms = {}
    for spec in args.arms:
        name, _, run = spec.partition("=")
        run_dir = run if os.path.isabs(run) else os.path.join(RUN_ROOT, run)
        rows = load_rows(run_dir, name if name in ("explicit", "ambiguous") else None)
        if not rows:
            print(f"[warn] no rows for arm {name!r} at {run_dir} -- tables will read ---")
        a = analyse(rows)
        arms[name] = a
        if rows:
            json.dump(a, open(os.path.join(run_dir, "analysis.json"), "w"), indent=2)
        report(name, a)

    if args.tables:
        write_tables(args.tables if os.path.isabs(args.tables)
                     else os.path.join(_REPO, args.tables), arms)
    if not args.no_figure:
        figdir = args.figdir if os.path.isabs(args.figdir) else os.path.join(_REPO, args.figdir)
        tier_figure(os.path.join(figdir, "fig_tier_breakdown.png"), arms)


if __name__ == "__main__":
    main()
