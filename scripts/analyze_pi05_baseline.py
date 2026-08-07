"""Aggregate outputs/rollouts/pi05_baseline/results.csv into the numbers and the
figure report/pi05_baseline/ needs. Read-only over the run artefacts."""
import collections, csv, json, os, sys

REPO = "/workspace/aaron/sketch_prompted_vla"
RUN = os.path.join(REPO, "outputs", "rollouts", "pi05_baseline")
FIGDIR = os.path.join(REPO, "report", "pi05_baseline", "figures")

TIERS = ["control", "referential", "directional", "both"]
SUITES = ["spatial", "object", "goal"]
# dataviz reference palette, categorical slots 1-3 (validated all-pairs, both modes)
C = {"success": "#2a78d6", "object": "#eb6834", "dest": "#1baf7a"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#d8d7d2"

rows = [r for r in csv.DictReader(open(os.path.join(RUN, "results.csv")))
        if r["skipped"] != "True"]

SUITE_DIR = {"spatial": "validation_set_spatial", "object": "validation_set_object",
             "goal": "validation_set_goal"}

# Chance level, per scene. A text-only policy is told the CATEGORY but not the
# instance, so on a scene with k same-category candidates the best it can do by
# guessing is 1/k. This is the floor the "referential ambiguity" half of the gap
# is measured against -- without it, a low correct-object rate on an ambiguous
# tier is uninterpretable, because part of it is irreducible by construction.
_meta_cache = {}


def meta_of(suite, dir_):
    key = (suite, dir_)
    if key not in _meta_cache:
        p = os.path.join(REPO, "outputs", SUITE_DIR[suite], dir_, "meta.json")
        _meta_cache[key] = json.load(open(p))
    return _meta_cache[key]


def chance(suite, dir_):
    m = meta_of(suite, dir_)
    n_pick = 1 + len(m.get("siblings") or [])
    n_place = 1 + len(m.get("other_baskets") or m.get("other_plates")
                      or m.get("other_dests") or [])
    return 1.0 / n_pick, 1.0 / n_place, 1.0 / (n_pick * n_place)


def pct(rs, field):
    rs = list(rs)
    return (100.0 * sum(1 for r in rs if r[field] == "True") / len(rs)) if rs else None


def block(rs):
    rs = list(rs)
    if not rs:
        return dict(n_rollouts=0, n_scenes=0, success=None, correct_object=None,
                    grasped_any=None, correct_destination=None,
                    chance_object=None, chance_dest=None, chance_joint=None,
                    fail_wrong_object=None, fail_right_object=None, fail_no_grasp=None)
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


by_suite = {s: block([r for r in rows if r["suite"] == s]) for s in SUITES}
by_tier = {t: block([r for r in rows if r["tier"] == t]) for t in TIERS}
by_suite_tier = {f"{s}|{t}": block([r for r in rows if r["suite"] == s and r["tier"] == t])
                 for s in SUITES for t in TIERS}
control = block([r for r in rows if r["tier"] == "control"])
other = block([r for r in rows if r["tier"] != "control"])
overall = block(rows)

# What did it grab when it grabbed the wrong thing? A same-category sibling is
# referential confusion -- the failure a sketch addresses. Anything else is not.
wrong = collections.Counter()
for r in rows:
    if r["grasped_any"] != "True" or r["correct_instance_grasped"] == "True":
        continue
    m = meta_of(r["suite"], r["dir"])
    g = r["grasped_instance"]
    sib = set(m.get("siblings") or [])
    dests = set([m["destination"]] + list(m.get("other_baskets") or m.get("other_plates")
                                          or m.get("other_dests") or []))
    wrong["sibling" if g in sib else "destination" if g in dests else "other"] += 1
n_wrong = sum(wrong.values())
wrong_mix = {k: dict(n=v, pct=round(100.0 * v / n_wrong, 1)) for k, v in wrong.items()}

out = dict(wrong_object_grasp_mix=dict(n=n_wrong, **wrong_mix),
           overall=overall, by_suite=by_suite, by_tier=by_tier,
           by_suite_tier=by_suite_tier, control=control, other_tiers=other,
           tier_gap_pp=(round(control["success"] - other["success"], 1)
                        if control["success"] is not None and other["success"] is not None
                        else None))
json.dump(out, open(os.path.join(RUN, "analysis.json"), "w"), indent=2)


def f(x):
    return "---" if x is None else f"{x:.1f}"


print("== overall ==", json.dumps(overall, indent=2))
print("\n== by suite ==")
for s in SUITES:
    b = by_suite[s]
    print(f"  {s:8} scenes={b['n_scenes']:3} rollouts={b['n_rollouts']:3} "
          f"succ={f(b['success']):>5} obj={f(b['correct_object']):>5} "
          f"dest={f(b['correct_destination']):>5} grasped_any={f(b['grasped_any']):>5}")
print("\n== by tier ==")
for t in TIERS:
    b = by_tier[t]
    print(f"  {t:12} scenes={b['n_scenes']:3} rollouts={b['n_rollouts']:3} "
          f"succ={f(b['success']):>5} obj={f(b['correct_object']):>5} "
          f"(chance {f(b['chance_object']):>5}) dest={f(b['correct_destination']):>5} "
          f"(chance {f(b['chance_dest']):>5})")

print("\n== failure decomposition (% of all rollouts in block) ==")
for name, b in [("ALL", overall)] + [(t, by_tier[t]) for t in TIERS]:
    print(f"  {name:12} success={f(b['success']):>5} | fail_right_obj={f(b['fail_right_object']):>5} "
          f"fail_wrong_obj={f(b['fail_wrong_object']):>5} fail_no_grasp={f(b['fail_no_grasp']):>5}")
print(f"\n== wrong-object grasps (n={n_wrong}) ==")
for k, v in sorted(wrong_mix.items(), key=lambda kv: -kv[1]["n"]):
    print(f"  {k:12} {v['n']:4}  {v['pct']:5.1f}%")

print(f"\ncontrol {f(control['success'])}%  vs other tiers {f(other['success'])}%  "
      f"gap {out['tier_gap_pp']}pp")

print("\n== LaTeX: results by suite ==")
for s in SUITES:
    b = by_suite[s]
    print(f"{s.capitalize():8} & {b['n_scenes']} & {b['n_rollouts']} & {f(b['success'])} & "
          f"{f(b['correct_object'])} & {f(b['correct_destination'])} \\\\")
print(f"All      & {overall['n_scenes']} & {overall['n_rollouts']} & {f(overall['success'])} & "
      f"{f(overall['correct_object'])} & {f(overall['correct_destination'])} \\\\")

print("\n== LaTeX: tier ==")
for name, b in (("Control", control), ("Other tiers", other)):
    print(f"{name:12} & {b['n_scenes']} & {b['n_rollouts']} & {f(b['success'])} & "
          f"{f(b['correct_object'])} \\\\")

print("\n== LaTeX: all four tiers ==")
for t in TIERS:
    b = by_tier[t]
    print(f"{t.capitalize():12} & {b['n_scenes']} & {b['n_rollouts']} & {f(b['success'])} & "
          f"{f(b['correct_object'])} & {f(b['correct_destination'])} \\\\")

# ------------------------------------------------------------------ figure --
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

series = [("Sustained success", "success", C["success"]),
          ("Correct object grasped", "correct_object", C["object"]),
          ("Correct destination", "correct_destination", C["dest"])]

fig, ax = plt.subplots(figsize=(8.4, 3.9), dpi=200)
x = np.arange(len(TIERS))
w = 0.26
for i, (label, key, col) in enumerate(series):
    vals = [by_tier[t][key] or 0.0 for t in TIERS]
    bars = ax.bar(x + (i - 1) * w, vals, w * 0.92, label=label, color=col,
                  edgecolor="white", linewidth=1.0, zorder=3)
    # every bar direct-labelled: the relief the aqua slot's sub-3:1 contrast obliges
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 1.4, f"{v:.0f}", ha="center",
                va="bottom", fontsize=7.6, color=INK2, zorder=4)

# The published average as a reference line. Labelled in the legend rather than
# on the plot: an in-plot annotation sits at ~97% and collides with whichever
# bar happens to be tallest, which changes as the roster grows.
pub = 96.85
ref = ax.axhline(pub, color="#8a8983", linestyle=(0, (4, 3)), linewidth=1.2, zorder=1)
ref.set_label("published $\\pi_{0.5}$-LIBERO average (96.9)")

ax.set_xticks(x)
ax.set_xticklabels([f"{t}\n{by_tier[t]['n_scenes']} scenes, "
                    f"{by_tier[t]['n_rollouts']} rollouts" for t in TIERS], fontsize=8.4)
ax.set_ylabel("percent", fontsize=8.6, color=INK2)
ax.set_ylim(0, 106)
ax.set_yticks([0, 25, 50, 75, 100])
ax.tick_params(axis="y", labelsize=8, colors=INK2, length=0)
ax.tick_params(axis="x", length=0, colors=INK)
ax.grid(axis="y", color=GRID, linewidth=0.7, zorder=0)
ax.set_axisbelow(True)
for side in ("top", "right", "left"):
    ax.spines[side].set_visible(False)
ax.spines["bottom"].set_color(GRID)
ax.legend(frameon=False, fontsize=8.0, ncol=4, loc="upper center",
          bbox_to_anchor=(0.5, 1.17), labelcolor=INK2, columnspacing=1.4,
          handlelength=1.6)
fig.tight_layout()
os.makedirs(FIGDIR, exist_ok=True)
p = os.path.join(FIGDIR, "fig_tier_breakdown.png")
fig.savefig(p, bbox_inches="tight", facecolor="white")
print("\nwrote", p)
