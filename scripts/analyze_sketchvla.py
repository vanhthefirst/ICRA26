"""Analysis of the four SketchPromptVLA evaluation arms against the pi0.5 baselines.

Consumes the rollout CSVs written by `examples/libero/eval_sketchvla.py` (model
repo) and `scripts/rollout_sketch.py` (this repo), and emits `analysis.json`, a
LaTeX macro file, and the report figures.

Contrasts are UNPAIRED. `init_state_hash` is identical across all four arms, but
`init_warmstart_hash` agrees only on rollout 0: the scene env is opened once and
reused for all rollouts of that scene, and `env.set_init_state()` restores
qpos/qvel without clearing MuJoCo's `qacc_warmstart`/`act`, which therefore carry
over from whatever the previous rollout left behind. Rollout k>0 of an arm is
consequently not the same draw as rollout k of another arm. The pairing rate is
measured and reported rather than assumed; see `pairing` in the JSON.

Run anywhere: stdlib / numpy / pandas / matplotlib, no simulator.

    python scripts/analyze_sketchvla.py \
        --arms explicit_sketch=sketchvla_pcla_explicit_sketch \
               explicit_blank=sketchvla_pcla_explicit_blank \
               ambiguous_sketch=sketchvla_pcla_ambiguous_sketch \
               ambiguous_blank=sketchvla_pcla_ambiguous_blank \
        --baselines explicit=pi05_explicit_532 ambiguous=pi05_ambiguous_532 \
        --tables report/finetuned_eval/tables.tex \
        --figdir report/finetuned_eval/figures
"""

import argparse
import json
import math
import pathlib

import numpy as np
import pandas as pd

TIERS = ["control", "referential", "directional", "both"]
MODAL_TARGET = "akita_black_bowl_1"


def parse_pairs(items):
    out = {}
    for it in items:
        if "=" not in it:
            raise SystemExit("expected name=run_id, got %r" % (it,))
        k, v = it.split("=", 1)
        out[k] = v
    return out


def load_arm(root, run_id, suite, targets):
    path = pathlib.Path(root) / "outputs" / "rollouts" / run_id / "results.csv"
    if not path.exists():
        raise SystemExit("missing %s" % (path,))
    df = pd.read_csv(path)
    df = df[(df.suite == suite) & (~df.skipped.astype(bool))].copy()
    df["target"] = df.dir.map(targets)
    df["target_grasped"] = df.grasped_instance == df.target
    if df.target.isna().any():
        raise SystemExit("%s has scenes absent from the manifest" % (run_id,))
    return df


def wilson(k, n):
    if n == 0:
        return (float("nan"),) * 3
    p = k / n
    z = 1.959963985
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, c - h, c + h


def two_proportion(k1, n1, k2, n2):
    """Difference of two independent proportions, with a Wald CI and a pooled z test."""
    p1, p2 = k1 / n1, k2 / n2
    diff = p1 - p2
    se = math.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
    pooled = (k1 + k2) / (n1 + n2)
    se0 = math.sqrt(pooled * (1 - pooled) * (1 / n1 + 1 / n2))
    z = diff / se0 if se0 > 0 else 0.0
    pval = math.erfc(abs(z) / math.sqrt(2))
    return {
        "p1": p1, "p2": p2, "diff": diff,
        "ci_lo": diff - 1.959963985 * se, "ci_hi": diff + 1.959963985 * se,
        "z": z, "p_value": pval,
    }


def arm_summary(df):
    n = len(df)
    reached = df.nearest_destination.notna()
    return {
        "n": int(n),
        "n_scenes": int(df.dir.nunique()),
        "success": wilson(int(df.success_sustained.sum()), n)[0],
        "success_ci": wilson(int(df.success_sustained.sum()), n)[1:],
        "grasped_any": float(df.grasped_any.mean()),
        "target_grasped": float(df.target_grasped.mean()),
        # `correct_destination` is only defined once the object reaches a plate,
        # and that denominator differs by a factor of three between the two
        # models, so I report it conditionally rather than over all rollouts.
        "reached_destination": float(reached.mean()),
        "correct_destination_given_reached": (
            float(df.correct_destination[reached].mean()) if reached.any() else float("nan")),
        "by_tier": {
            t: {
                "n": int((df.tier == t).sum()),
                "success": float(df.success_sustained[df.tier == t].mean()),
                "grasped_any": float(df.grasped_any[df.tier == t].mean()),
                "target_grasped": float(df.target_grasped[df.tier == t].mean()),
            } for t in TIERS if (df.tier == t).any()
        },
    }


def position_prior(df):
    """Does the arm select by conditioning, or always reach the same place?

    Split on whether the scene's target is the modal instance. A policy driven by
    its conditioning should score similarly on both halves; a policy reaching a
    fixed table position scores on the modal half only.
    """
    g = df[df.grasped_any]
    modal = g[g.target == MODAL_TARGET]
    other = g[g.target != MODAL_TARGET]
    return {
        "n_grasps_modal_target": int(len(modal)),
        "n_grasps_other_target": int(len(other)),
        "hits_modal_target": int((modal.grasped_instance == modal.target).sum()),
        "hits_other_target": int((other.grasped_instance == other.target).sum()),
        "grabs_modal_when_target_is_other": (
            float((other.grasped_instance == MODAL_TARGET).mean()) if len(other) else float("nan")),
    }


def pairing(arms):
    """Fraction of rows whose pre-rollout state matches the first arm, row for row."""
    key = ["dir", "rollout_idx"]
    names = list(arms)
    base = arms[names[0]].sort_values(key).reset_index(drop=True)
    out = {}
    for nm in names:
        s = arms[nm].sort_values(key).reset_index(drop=True)
        out[nm] = {
            "init_state": float((s.init_state_hash.values == base.init_state_hash.values).mean()),
            "init_warmstart": float(
                (s.init_warmstart_hash.values == base.init_warmstart_hash.values).mean()),
        }
    return out


def fig_tiers(arms, baselines, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    series = [
        ("$\\pi_{0.5}$ explicit", baselines["explicit"], "#3b6ea5"),
        ("$\\pi_{0.5}$ ambiguous", baselines["ambiguous"], "#9dc3e6"),
        ("Sketch-VLA explicit + sketch", arms["explicit_sketch"], "#b4544f"),
        ("Sketch-VLA ambiguous + blank", arms["ambiguous_blank"], "#e2b0ad"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 2.7))
    x = np.arange(len(TIERS))
    w = 0.2
    for ax, col, title in ((axes[0], "target_grasped", "Grasped the designated object"),
                           (axes[1], "success_sustained", "Sustained task success")):
        for i, (lab, df, c) in enumerate(series):
            vals = [df[df.tier == t][col].mean() * 100 if (df.tier == t).any() else 0 for t in TIERS]
            ax.bar(x + (i - 1.5) * w, vals, w, label=lab, color=c)
            # A zero bar and an absent bar look identical, and every zero here is
            # a measured zero over 168 rollouts, so it gets printed.
            for xi, v in zip(x, vals):
                if v == 0:
                    ax.text(xi + (i - 1.5) * w, 1.5, "0", ha="center", fontsize=6, color=c)
        ax.set_xticks(x)
        ax.set_xticklabels([t.capitalize() for t in TIERS], fontsize=8)
        ax.set_ylabel("%", fontsize=8)
        ax.set_title(title, fontsize=9)
        ax.tick_params(labelsize=8)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_ylim(0, 100)
    axes[0].legend(fontsize=7, frameon=False, ncol=2, loc="upper right")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def fig_selection(arms, baselines, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = [
        ("$\\pi_{0.5}$ explicit", baselines["explicit"]),
        ("$\\pi_{0.5}$ ambiguous", baselines["ambiguous"]),
        ("SVLA expl+sketch", arms["explicit_sketch"]),
        ("SVLA expl+blank", arms["explicit_blank"]),
        ("SVLA amb+sketch", arms["ambiguous_sketch"]),
        ("SVLA amb+blank", arms["ambiguous_blank"]),
    ]
    fig, ax = plt.subplots(figsize=(6.6, 2.5))
    y = np.arange(len(rows))
    modal, other = [], []
    for _, df in rows:
        pp = position_prior(df)
        modal.append(100 * pp["hits_modal_target"] / max(pp["n_grasps_modal_target"], 1))
        other.append(100 * pp["hits_other_target"] / max(pp["n_grasps_other_target"], 1))
    ax.barh(y + 0.19, modal, 0.36, label="target is the modal instance", color="#6d8fb5")
    ax.barh(y - 0.19, other, 0.36, label="target is any other instance", color="#b4544f")
    for yi, (m, o) in enumerate(zip(modal, other)):
        for dy, v, n in ((0.19, m, "n_grasps_modal_target"), (-0.19, o, "n_grasps_other_target")):
            if v == 0:
                ax.text(0.6, yi - dy, "0 of %d" % (position_prior(rows[yi][1])[n],),
                        va="center", fontsize=7, color="#b4544f")
    ax.set_yticks(y)
    ax.set_yticklabels([r[0] for r in rows], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("grasps that hit the designated object (%)", fontsize=8)
    ax.tick_params(labelsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=7, frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def pct(x):
    return "---" if x is None or (isinstance(x, float) and math.isnan(x)) else "%.1f" % (100 * x)


def write_tables(res, path):
    L = ["% generated by scripts/analyze_sketchvla.py -- do not edit by hand"]

    def mac(name, value):
        L.append("\\newcommand{\\%s}{%s}" % (name, value))

    body = []
    order = [("Sketch-VLA, explicit + sketch", "explicit_sketch"),
             ("Sketch-VLA, explicit + blank", "explicit_blank"),
             ("Sketch-VLA, ambiguous + sketch", "ambiguous_sketch"),
             ("Sketch-VLA, ambiguous + blank", "ambiguous_blank")]
    for lab, k in order:
        a = res["arms"][k]
        body.append("%s & %d & %s & %s & %s \\\\" % (
            lab, a["n"], pct(a["success"]), pct(a["grasped_any"]), pct(a["target_grasped"])))
    body.append("\\midrule")
    for lab, k in [("$\\pi_{0.5}$-LIBERO, explicit", "explicit"),
                   ("$\\pi_{0.5}$-LIBERO, ambiguous", "ambiguous")]:
        a = res["baselines"][k]
        body.append("%s & %d & %s & %s & %s \\\\" % (
            lab, a["n"], pct(a["success"]), pct(a["grasped_any"]), pct(a["target_grasped"])))
    mac("tabArms", "%\n" + "\n".join(body))

    body = []
    for t in TIERS:
        cells = []
        for src, k in [("arms", "explicit_sketch"), ("arms", "ambiguous_blank"),
                       ("baselines", "explicit"), ("baselines", "ambiguous")]:
            d = res[src][k]["by_tier"].get(t)
            cells.append(pct(d["target_grasped"]) if d else "---")
        n = res["arms"]["explicit_sketch"]["by_tier"].get(t, {}).get("n", 0)
        body.append("%s & %d & %s \\\\" % (t.capitalize(), n, " & ".join(cells)))
    mac("tabTiers", "%\n" + "\n".join(body))

    for name, c in res["contrasts"].items():
        stem = "num" + "".join(w.capitalize() for w in name.split("_"))
        mac(stem + "Diff", "%+.1f" % (100 * c["diff"],))
        mac(stem + "Lo", "%+.1f" % (100 * c["ci_lo"],))
        mac(stem + "Hi", "%+.1f" % (100 * c["ci_hi"],))
        mac(stem + "P", "%.2f" % (c["p_value"],) if c["p_value"] >= 0.01 else "<0.01")

    pp = res["position_prior"]
    mac("numOtherGrasps", str(sum(v["n_grasps_other_target"] for v in pp["arms"].values())))
    mac("numOtherHits", str(sum(v["hits_other_target"] for v in pp["arms"].values())))
    mac("numModalPullExpl", pct(pp["arms"]["explicit_sketch"]["grabs_modal_when_target_is_other"]))
    mac("numBaselineOtherHits", pct(
        pp["baselines"]["explicit"]["hits_other_target"]
        / max(pp["baselines"]["explicit"]["n_grasps_other_target"], 1)))
    mac("numPairInit", pct(min(v["init_state"] for v in res["pairing"].values())))
    mac("numPairWarm", pct(min(v["init_warmstart"] for v in res["pairing"].values())))
    mac("numRollouts", str(sum(a["n"] for a in res["arms"].values())))

    pathlib.Path(path).write_text("\n".join(L) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--suite", default="spatial")
    ap.add_argument("--arms", nargs="+", required=True)
    ap.add_argument("--baselines", nargs="+", required=True)
    ap.add_argument("--manifest", default="outputs/validation_manifest_all.json")
    ap.add_argument("--tables")
    ap.add_argument("--figdir")
    ap.add_argument("--json")
    args = ap.parse_args()

    man = json.loads((pathlib.Path(args.root) / args.manifest).read_text())
    targets = {r["dir"]: r["target"] for r in man if r["suite"] == args.suite}

    arms = {k: load_arm(args.root, v, args.suite, targets)
            for k, v in parse_pairs(args.arms).items()}
    bases = {k: load_arm(args.root, v, args.suite, targets)
             for k, v in parse_pairs(args.baselines).items()}

    def contrast(a, b, col):
        return two_proportion(int(a[col].sum()), len(a), int(b[col].sum()), len(b))

    res = {
        "suite": args.suite,
        "arms": {k: arm_summary(v) for k, v in arms.items()},
        "baselines": {k: arm_summary(v) for k, v in bases.items()},
        "pairing": pairing(arms),
        "position_prior": {
            "arms": {k: position_prior(v) for k, v in arms.items()},
            "baselines": {k: position_prior(v) for k, v in bases.items()},
        },
        "contrasts": {
            "sketch_vs_blank_explicit": contrast(
                arms["explicit_sketch"], arms["explicit_blank"], "target_grasped"),
            "sketch_vs_blank_ambiguous": contrast(
                arms["ambiguous_sketch"], arms["ambiguous_blank"], "target_grasped"),
            "caption_sketchvla": contrast(
                arms["explicit_sketch"], arms["ambiguous_sketch"], "target_grasped"),
            "caption_baseline": contrast(
                bases["explicit"], bases["ambiguous"], "target_grasped"),
            "model_gap": contrast(
                bases["explicit"], arms["explicit_sketch"], "target_grasped"),
        },
    }

    if args.figdir:
        d = pathlib.Path(args.figdir)
        d.mkdir(parents=True, exist_ok=True)
        fig_tiers(arms, bases, d / "fig_tier_breakdown.png")
        fig_selection(arms, bases, d / "fig_selection.png")
    if args.tables:
        pathlib.Path(args.tables).parent.mkdir(parents=True, exist_ok=True)
        write_tables(res, args.tables)
    if args.json:
        pathlib.Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        pathlib.Path(args.json).write_text(json.dumps(res, indent=2))
    print(json.dumps(res["contrasts"], indent=2))


if __name__ == "__main__":
    main()
