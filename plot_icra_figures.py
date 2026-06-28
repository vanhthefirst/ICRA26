"""
plot_icra_figures.py

Three publication-ready figures for ICRA submission.

  fig_interaction.pdf      Factorial interaction: depth x conditioning type
  fig_calvin_survival.pdf  CALVIN analysis: cumulative delta + conditional survival
  fig_summary_heatmap.pdf  Unified model summary heatmap

Input: probe_results.json  (produced by run_table1.py)

All paper constants (Tables 2, 3, 4) are hardcoded and annotated with source.
"""

import json
import math
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

# ── Paper-sourced constants ────────────────────────────────────────────────────
# Source: Table 3, D-LAPA CoRL 2026 submission
_LIBERO_LONG = {
    "LAPA":    77.8,
    "Model 1": 79.8,
    "Model 2": 83.2,
    "Model 3": 80.2,
    "Model 4": 84.2,
    "Model 5": 77.4,
}

# Source: Table 4, D-LAPA CoRL 2026 submission
# Per-task success rate (%) across a 5-task sequential chain
_CALVIN = {
    "LAPA":    [86.8, 61.2, 44.8, 34.1, 22.2],
    "Model 2": [84.2, 65.5, 46.0, 34.3, 24.6],
    "Model 4": [84.6, 63.2, 44.9, 34.0, 23.9],
}
# Average tasks completed (Table 4) -- reported directly, not derivable from per-task rates
_CALVIN_AVG = {"LAPA": 2.13, "Model 2": 2.35, "Model 4": 2.31}

# Source: Table 2, D-LAPA CoRL 2026 submission
_HAS_DEPTH   = {1: True,  2: True,  3: False, 4: True,  5: False}
_SUPERVISION = {1: "Disc", 2: "Disc", 3: "Disc", 4: "Cont", 5: "Cont"}
_RGB_INPUT   = {1: "Index", 2: "Feature", 3: "Feature", 4: "Feature", 5: "Feature"}

# ── Shared colours ─────────────────────────────────────────────────────────────
_C_BASELINE = "#636363"
_C_M2       = "#2166ac"   # blue
_C_M4       = "#1b7837"   # green
_C_DISC     = "#e08214"   # orange  (discrete supervision line)
_C_CONT     = "#1b7837"   # green   (continuous supervision line, matches M4)
_C_DEPTH    = "#2166ac"
_C_NODEPTH  = "#d6604d"

plt.rcParams.update({
    "font.family":       "serif",
    "font.size":         9,
    "axes.linewidth":    0.8,
    "xtick.major.size":  3.5,
    "ytick.major.size":  3.5,
    "axes.spines.top":   False,
    "axes.spines.right": False,
})

PROBE_JSON = "probe_results.json"


def _load(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def _delta_r2(probe: dict) -> dict:
    base = probe["base LAPA (proxy reference)"]["r2"]
    return {k: probe[f"base LAPA \u2295 Model {k}"]["r2"] - base for k in range(1, 6)}


# ── Figure 1: 2x2 Factorial Interaction ───────────────────────────────────────

def fig_interaction(out: str = "figures/fig_interaction.pdf") -> None:
    """
    Two-panel figure showing the factorial interaction between depth image
    availability and supervision type.

    Left  -- Interaction plot (classic 2x2 line chart).
      Model mapping (Table 2):
        Discrete + No depth    -> Model 3 (80.2%)
        Discrete + Has depth   -> Model 2 (83.2%)   Delta = +3.0 pp
        Continuous + No depth  -> Model 5 (77.4%)
        Continuous + Has depth -> Model 4 (84.2%)   Delta = +6.8 pp
      Non-parallel lines reveal a super-additive interaction: depth imagery
      yields 2.3x more gain under continuous feature distillation.
      *M1 uses discrete RGB index conditioning (not continuous RGB features)
      and lies outside the 2x2; it is shown in the right panel.

    Right -- Decomposed effect sizes for three key ablation contrasts:
      (i)  RGB conditioning quality (M1 -> M2, both with depth): +3.4 pp
      (ii) Depth image effect under discrete supervision (M3 -> M2): +3.0 pp
      (iii)Depth image effect under continuous distillation (M5 -> M4): +6.8 pp

    Source: Tables 2 and 3, D-LAPA CoRL 2026 submission.
    """
    fig, (ax_int, ax_eff) = plt.subplots(1, 2, figsize=(11.0, 4.4))

    # ── Left: interaction plot ────────────────────────────────────────────────
    x_vals   = [0.0, 1.0]
    x_labels = ["Depth image absent", "Depth image present"]

    disc_y = [_LIBERO_LONG["Model 3"], _LIBERO_LONG["Model 2"]]
    cont_y = [_LIBERO_LONG["Model 5"], _LIBERO_LONG["Model 4"]]

    ax_int.plot(x_vals, disc_y, color=_C_DISC, linewidth=2.2,
                marker="s", markersize=8, zorder=4, label="Discrete supervision (CE)")
    ax_int.plot(x_vals, cont_y, color=_C_CONT, linewidth=2.2,
                marker="o", markersize=8, zorder=4, label="Continuous distillation (MSE+cos)")
    ax_int.axhline(_LIBERO_LONG["LAPA"], color=_C_BASELINE,
                   linestyle="--", linewidth=1.3, zorder=2, label="LAPA baseline (77.8%)")

    # Left-side labels: plain text with no leader lines so nothing obscures the values
    ax_int.text(-0.06, disc_y[0] + 0.85, "M3  80.2%",
                fontsize=8.5, color=_C_DISC, ha="right", va="bottom")
    ax_int.text(-0.06, cont_y[0] - 0.85, "M5  77.4%",
                fontsize=8.5, color=_C_CONT, ha="right", va="top")

    # Right-side labels: arrow with head pointing directly at the data point
    ax_int.annotate("M4  84.2%",
                    xy=(1.0, cont_y[1]),
                    xytext=(1.08, cont_y[1] + 1.3),
                    fontsize=8.5, color=_C_CONT, va="bottom",
                    arrowprops=dict(arrowstyle="-|>", color=_C_CONT, lw=1.0,
                                    mutation_scale=10))
    ax_int.annotate("M2  83.2%",
                    xy=(1.0, disc_y[1]),
                    xytext=(1.08, disc_y[1] - 1.4),
                    fontsize=8.5, color=_C_DISC, va="top",
                    arrowprops=dict(arrowstyle="-|>", color=_C_DISC, lw=1.0,
                                    mutation_scale=10))

    # Depth effect labels at the midpoint of each line
    depth_eff_disc = disc_y[1] - disc_y[0]
    depth_eff_cont = cont_y[1] - cont_y[0]
    ax_int.text(0.50, (disc_y[0] + disc_y[1]) / 2 + 0.7,
                f"+{depth_eff_disc:.1f} pp", ha="center", fontsize=8.5,
                color=_C_DISC, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=_C_DISC, lw=0.7))
    ax_int.text(0.50, (cont_y[0] + cont_y[1]) / 2 - 1.5,
                f"+{depth_eff_cont:.1f} pp", ha="center", fontsize=8.5,
                color=_C_CONT, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=_C_CONT, lw=0.7))

    ax_int.set_xticks(x_vals)
    ax_int.set_xticklabels(x_labels, fontsize=9.5)
    ax_int.set_xlim(-0.55, 1.55)
    ax_int.set_ylim(74.5, 88.0)
    ax_int.set_ylabel("LIBERO-LONG success rate (%, Table 3)")
    ax_int.set_title("Depth \u00d7 Conditioning Type Interaction", pad=7)
    ax_int.legend(fontsize=8, loc="upper left", framealpha=0.92, edgecolor="#cccccc")
    ax_int.grid(axis="y", linewidth=0.4, color="#d8d8d8", zorder=0)
    ax_int.text(0.01, 0.02,
                "*M1: RGB-index conditioning -- see right panel.",
                transform=ax_int.transAxes, fontsize=7, color="#666666", va="bottom")

    # ── Right: effect-size bars ───────────────────────────────────────────────
    contrasts = [
        ("RGB conditioning\n(M1 \u2192 M2)",
         _LIBERO_LONG["Model 2"] - _LIBERO_LONG["Model 1"], "#9ecae1"),
        ("Depth image effect\ndiscrete (M3 \u2192 M2)",
         _LIBERO_LONG["Model 2"] - _LIBERO_LONG["Model 3"], _C_DISC),
        ("Depth image effect\ncontinuous (M5 \u2192 M4)",
         _LIBERO_LONG["Model 4"] - _LIBERO_LONG["Model 5"], _C_CONT),
    ]

    x_eff      = np.arange(len(contrasts))
    values_eff = [c[1] for c in contrasts]
    colors_eff = [c[2] for c in contrasts]

    bars = ax_eff.bar(x_eff, values_eff, color=colors_eff, width=0.52,
                      zorder=3, linewidth=0)
    ax_eff.axhline(0, color="black", linewidth=0.9, zorder=4)

    for bar, val in zip(bars, values_eff):
        ax_eff.text(bar.get_x() + bar.get_width() / 2, val + 0.15,
                    f"+{val:.1f} pp", ha="center", va="bottom",
                    fontsize=9.5, fontweight="bold")

    ax_eff.set_xticks(x_eff)
    ax_eff.set_xticklabels([c[0] for c in contrasts], fontsize=9)
    ax_eff.set_xlim(-0.65, 2.65)
    ax_eff.set_ylim(0, 10.5)
    ax_eff.set_ylabel("Performance gain (pp, LIBERO-LONG, Table 3)")
    ax_eff.set_title(
        "Ablation Effect Sizes\n(*M1: RGB-index conditioning, depth image present)",
        pad=7)
    ax_eff.grid(axis="y", linewidth=0.4, color="#d8d8d8", zorder=0)

    fig.tight_layout(pad=1.8)
    fig.savefig(out, dpi=300, bbox_inches="tight")
    fig.savefig(out.replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {out}")


# ── Figure 2: CALVIN Analysis ──────────────────────────────────────────────────

def fig_calvin_survival(out: str = "figures/fig_calvin_survival.pdf") -> None:
    """
    Two-panel CALVIN reanalysis (Table 4).

    Left  -- Grouped bar chart of delta success rate vs. LAPA baseline at each
             task position. Bars show gain/loss over LAPA; positive = outperforms.
             Computed as: delta_k = Model_k(task) - LAPA(task).
             Key finding: both depth models sacrifice Task-1 performance slightly
             (-2 to -3 pp) but recover strongly at Task 2 (+2 to +4 pp), with
             Model 2 sustaining a positive delta through Task 5.

    Right -- Per-step conditional transition probability:
             P(succeed at task k | succeeded at task k-1)
             Computation: P(k|k-1) = P(k) / P(k-1).
             Key finding: LAPA's weakest link is the T4->T5 transition (65.1%);
             both depth models hold 70-72% at that step.

    Source: Table 4, D-LAPA CoRL 2026 submission.
    """
    fig, (ax_bar, ax_cond) = plt.subplots(1, 2, figsize=(10.5, 4.0))

    tasks     = np.arange(1, 6)
    task_lbls = [f"Task {t}" for t in tasks]
    lapa_rates = np.array(_CALVIN["LAPA"])
    m2_delta   = np.array(_CALVIN["Model 2"]) - lapa_rates
    m4_delta   = np.array(_CALVIN["Model 4"]) - lapa_rates

    # ── Left: delta bar chart (gain over LAPA) ───────────────────────────────
    bw  = 0.32
    x   = np.arange(len(tasks))
    ax_bar.bar(x - bw / 2, m2_delta, width=bw, color=_C_M2, label="Model 2", zorder=3)
    ax_bar.bar(x + bw / 2, m4_delta, width=bw, color=_C_M4, label="Model 4", zorder=3)
    ax_bar.axhline(0, color=_C_BASELINE, linewidth=1.2, linestyle="--",
                   zorder=2, label="LAPA baseline (0)")

    # Value labels on each bar
    for xi, (d2, d4) in enumerate(zip(m2_delta, m4_delta)):
        va2, yo2 = ("bottom", 0.08) if d2 >= 0 else ("top", -0.08)
        va4, yo4 = ("bottom", 0.08) if d4 >= 0 else ("top", -0.08)
        ax_bar.text(xi - bw / 2, d2 + yo2, f"{d2:+.1f}",
                    ha="center", va=va2, fontsize=7.5, color=_C_M2, fontweight="bold")
        ax_bar.text(xi + bw / 2, d4 + yo4, f"{d4:+.1f}",
                    ha="center", va=va4, fontsize=7.5, color=_C_M4, fontweight="bold")

    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(task_lbls)
    ax_bar.set_ylabel("\u0394 success rate vs. LAPA (pp)")
    ax_bar.set_ylim(min(m2_delta.min(), m4_delta.min()) - 1.5,
                    max(m2_delta.max(), m4_delta.max()) + 2.0)
    ax_bar.set_title("CALVIN: Gain / Loss over LAPA Baseline", pad=7)
    ax_bar.grid(axis="y", linewidth=0.4, color="#d8d8d8", zorder=0)
    ax_bar.legend(fontsize=8.5, framealpha=0.92, edgecolor="#cccccc")

    # ── Right: conditional transition probability ─────────────────────────────
    style = {
        "LAPA":    dict(color=_C_BASELINE, ls="--", marker="D", lw=1.8, ms=5.5),
        "Model 2": dict(color=_C_M2,       ls="-",  marker="o", lw=2.0, ms=5.5),
        "Model 4": dict(color=_C_M4,       ls="-",  marker="^", lw=2.0, ms=6.0),
    }

    x_labels = ["Task 1\n(first)", "T1\u2192T2", "T2\u2192T3", "T3\u2192T4", "T4\u2192T5"]

    def conditional_probs(rates):
        r = np.array(rates) / 100.0
        cond = [r[0]]
        for k in range(1, len(r)):
            cond.append(r[k] / r[k - 1] if r[k - 1] > 0 else 0.0)
        return np.array(cond) * 100.0

    cond_data = {name: conditional_probs(rates) for name, rates in _CALVIN.items()}

    for name, cond in cond_data.items():
        kw = style[name]
        ax_cond.plot(tasks, cond, color=kw["color"], linestyle=kw["ls"],
                     linewidth=kw["lw"], zorder=3, label=name)
        ax_cond.scatter(tasks, cond, color=kw["color"], marker=kw["marker"],
                        s=kw["ms"] ** 2, zorder=4)

    lapa_cond = cond_data["LAPA"]
    ax_cond.annotate(
        f"LAPA weakest:\n{lapa_cond[4]:.1f}%",
        xy=(5, lapa_cond[4]),
        xytext=(4.1, lapa_cond[4] - 9.5),
        fontsize=7.5, color=_C_BASELINE,
        arrowprops=dict(arrowstyle="-|>", color=_C_BASELINE, lw=0.8,
                        mutation_scale=9),
    )

    ax_cond.set_xticks(tasks)
    ax_cond.set_xticklabels(x_labels, fontsize=8.5)
    ax_cond.set_ylabel("P(task k | succeeded at task k\u22121) (%)")
    ax_cond.set_ylim(52, 100)
    ax_cond.set_title("CALVIN: Per-Step Conditional Survival", pad=7)
    ax_cond.grid(linewidth=0.4, color="#d8d8d8", zorder=0)
    ax_cond.legend(fontsize=8.5, framealpha=0.92, edgecolor="#cccccc")
    ax_cond.text(0.02, 0.04,
                 "P(task 1) = raw rate\nP(task k\u22121\u2192k) = P(k) / P(k\u22121)",
                 transform=ax_cond.transAxes, fontsize=7, color="#555555", va="bottom")

    fig.tight_layout(pad=1.5)
    fig.savefig(out, dpi=300, bbox_inches="tight")
    fig.savefig(out.replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {out}")


# ── Figure 3: Full Model Summary Heatmap ──────────────────────────────────────

def _fmt_3sf(val: float, signed: bool = False) -> str:
    """Format val to 3 significant figures, capped at 4 decimal places."""
    if val == 0:
        return "+0.00" if signed else "0.00"
    magnitude = math.floor(math.log10(abs(val)))
    dp = int(max(0, min(4, 2 - magnitude)))
    fmt = f"{{:+.{dp}f}}" if signed else f"{{:.{dp}f}}"
    return fmt.format(val)

def fig_summary_heatmap(probe: dict, out: str = "figures/fig_summary_heatmap.pdf") -> None:
    """
    Unified model-comparison heatmap (Tables 2-4 + probe_results.json).

    Colour convention (right numeric block only):
      Blue gradient   -- absolute performance (R^2, LIBERO-L %, CALVIN avg)
      Red-yellow-green diverging, centred at 0 -- delta columns
        0 -> yellow; positive -> green; negative -> red
        Both axes symmetric so equal-magnitude gains share equal saturation.
    LAPA baseline delta cells show '--' (reference, not a measurement).

    Sources:
      probe_results.json  (our reproduction)
      Tables 2, 3, 4 -- D-LAPA CoRL 2026 submission.
    """
    model_names = ["LAPA", "Model 1", "Model 2", "Model 3", "Model 4", "Model 5"]
    dr2         = _delta_r2(probe)
    base_r2     = probe["base LAPA (proxy reference)"]["r2"]

    r2_all = {
        "LAPA":    base_r2,
        "Model 1": probe["base LAPA \u2295 Model 1"]["r2"],
        "Model 2": probe["base LAPA \u2295 Model 2"]["r2"],
        "Model 3": probe["base LAPA \u2295 Model 3"]["r2"],
        "Model 4": probe["base LAPA \u2295 Model 4"]["r2"],
        "Model 5": probe["base LAPA \u2295 Model 5"]["r2"],
    }

    lapa_ll  = _LIBERO_LONG["LAPA"]
    lapa_cal = _CALVIN_AVG["LAPA"]
    _calvin_available = {"LAPA", "Model 2", "Model 4"}

    rows = []
    for name in model_names:
        k = int(name[-1]) if name != "LAPA" else None
        is_lapa = (name == "LAPA")
        rows.append({
            "name":        name,
            "is_lapa":     is_lapa,
            "has_depth":   _HAS_DEPTH[k]   if k else None,
            "supervision": _SUPERVISION[k]  if k else None,
            "rgb_input":   _RGB_INPUT[k]    if k else None,
            "r2":          r2_all[name],
            "dr2":         (dr2[k] if k else 0.0),
            "ll_pct":      _LIBERO_LONG[name],
            "ll_delta":    _LIBERO_LONG[name] - lapa_ll,
            "cal_avg":     _CALVIN_AVG.get(name, None),
            "cal_delta":   (_CALVIN_AVG[name] - lapa_cal) if name in _calvin_available else None,
        })

    n_rows  = len(rows)
    fig_h   = 0.60 * n_rows + 2.6
    fig, ax = plt.subplots(figsize=(11.5, fig_h))
    ax.axis("off")

    col_headers = [
        "Model", "Depth\nimage", "Superv.", "RGB\ninput",
        "R\u00b2", "\u0394R\u00b2",
        "LIBERO-L\n(%)", "\u0394 LL\n(pp)",
        "CALVIN\navg", "\u0394 CALVIN\navg",
    ]
    n_cat      = 4
    col_widths = [0.13, 0.07, 0.07, 0.08,  0.08, 0.09, 0.10, 0.09, 0.10, 0.10]
    col_x      = [0.01]
    for w in col_widths[:-1]:
        col_x.append(col_x[-1] + w)

    row_h    = 1.0 / (n_rows + 1.8)
    num_keys = ["r2", "dr2", "ll_pct", "ll_delta", "cal_avg", "cal_delta"]
    delta_keys = {"dr2", "ll_delta", "cal_delta"}   # centred at 0
    abs_keys   = {"r2", "ll_pct", "cal_avg"}        # sequential blue scale

    cmaps = {
        "r2":        plt.cm.Blues,
        "dr2":       plt.cm.RdYlGn,
        "ll_pct":    plt.cm.Blues,
        "ll_delta":  plt.cm.RdYlGn,
        "cal_avg":   plt.cm.Blues,
        "cal_delta": plt.cm.RdYlGn,
    }

    # Absolute column range: min/max across all non-None rows
    def _abs_range(key):
        vals = [r[key] for r in rows if r.get(key) is not None]
        return (min(vals), max(vals)) if vals else (0.0, 1.0)

    # Delta column: symmetric around 0 using max absolute value
    def _delta_max_abs(key):
        vals = [abs(r[key]) for r in rows
                if r.get(key) is not None and not r["is_lapa"]]
        return max(vals) if vals else 1.0

    col_abs_range = {k: _abs_range(k) for k in abs_keys}
    col_delta_abs = {k: _delta_max_abs(k) for k in delta_keys}

    def _norm_abs(val, lo, hi):
        if hi == lo:
            return 0.5
        return (val - lo) / (hi - lo)

    def _norm_diverging(val, max_abs):
        """Map val to [0,1]: 0=most red, 0.5=yellow, 1=most green."""
        if max_abs == 0:
            return 0.5
        ratio = np.clip(val / max_abs, -1.0, 1.0)
        return 0.5 + 0.40 * ratio

    # ── Header row ────────────────────────────────────────────────────────────
    header_y = 1.0 - row_h * 0.5
    for ci, (hdr, cx) in enumerate(zip(col_headers, col_x)):
        ax.text(cx + col_widths[ci] / 2, header_y, hdr,
                ha="center", va="center", fontsize=8.5, fontweight="bold",
                transform=ax.transAxes)

    div_x = col_x[n_cat]
    ax.plot([div_x - 0.004, div_x - 0.004], [0.08, 1.0],
            color="#999999", lw=0.9, transform=ax.transAxes, zorder=5)
    ax.plot([0.0, 1.0], [1.0 - row_h, 1.0 - row_h],
            color="#aaaaaa", lw=0.9, transform=ax.transAxes, zorder=5)

    # ── Data rows ─────────────────────────────────────────────────────────────
    for ri, row in enumerate(rows):
        y_center = 1.0 - row_h * (ri + 1.5)
        y_bot    = 1.0 - row_h * (ri + 2)

        bg = "#f7f7f7" if ri % 2 == 0 else "#ffffff"
        rect = mpatches.FancyBboxPatch(
            (0.0, y_bot), 1.0, row_h,
            boxstyle="square,pad=0", linewidth=0,
            facecolor=bg, transform=ax.transAxes, zorder=1
        )
        ax.add_patch(rect)

        if ri < n_rows - 1:
            ax.plot([0.0, 1.0], [y_bot, y_bot],
                    color="#dddddd", lw=0.5, transform=ax.transAxes, zorder=2)

        # Categorical columns
        bold = row["name"] in ("Model 2", "Model 4")
        ax.text(col_x[0] + col_widths[0] / 2, y_center, row["name"],
                ha="center", va="center", fontsize=8.5,
                fontweight="bold" if bold else "normal",
                transform=ax.transAxes, zorder=3)

        has_d = row["has_depth"]
        if has_d is None:
            sym, col = u"\u2014", "#888888"
        elif has_d:
            sym, col = "Yes", _C_DEPTH
        else:
            sym, col = "No", _C_NODEPTH
        ax.text(col_x[1] + col_widths[1] / 2, y_center, sym,
                ha="center", va="center", fontsize=8.5,
                color=col, fontweight="bold", transform=ax.transAxes, zorder=3)

        ax.text(col_x[2] + col_widths[2] / 2, y_center,
                row["supervision"] or u"\u2014",
                ha="center", va="center", fontsize=8,
                transform=ax.transAxes, zorder=3)

        ax.text(col_x[3] + col_widths[3] / 2, y_center,
                row["rgb_input"] or u"\u2014",
                ha="center", va="center", fontsize=8,
                transform=ax.transAxes, zorder=3)

        # Numeric columns
        for ci, num_key in enumerate(num_keys):
            actual_ci = n_cat + ci
            val = row.get(num_key)
            cx  = col_x[actual_ci]
            cw  = col_widths[actual_ci]

            # LAPA delta cells: show dash, no colour
            if row["is_lapa"] and num_key in delta_keys:
                ax.text(cx + cw / 2, y_center, u"\u2014",
                        ha="center", va="center", fontsize=9.5,
                        color="#222222", fontweight="bold",
                        transform=ax.transAxes, zorder=3)
                continue

            if val is None:
                ax.text(cx + cw / 2, y_center, "N/A",
                        ha="center", va="center", fontsize=8,
                        color="#aaaaaa", transform=ax.transAxes, zorder=3)
                continue

            # Colour mapping
            if num_key in abs_keys:
                lo, hi = col_abs_range[num_key]
                nv = _norm_abs(val, lo, hi)
                cell_c = cmaps[num_key](0.15 + 0.70 * nv)
            else:
                max_abs = col_delta_abs[num_key]
                nv = _norm_diverging(val, max_abs)
                cell_c = cmaps[num_key](nv)

            cell_rect = mpatches.FancyBboxPatch(
                (cx + 0.004, y_bot + 0.006), cw - 0.008, row_h - 0.012,
                boxstyle="round,pad=0.002", linewidth=0,
                facecolor=cell_c, alpha=0.80,
                transform=ax.transAxes, zorder=2
            )
            ax.add_patch(cell_rect)

            luminance = 0.299 * cell_c[0] + 0.587 * cell_c[1] + 0.114 * cell_c[2]
            txt_color = "white" if luminance < 0.52 else "#222222"

            signed = num_key in ("dr2", "ll_delta", "cal_delta")
            txt = _fmt_3sf(val, signed=signed)

            ax.text(cx + cw / 2, y_center, txt,
                    ha="center", va="center", fontsize=8.5,
                    color=txt_color, transform=ax.transAxes, zorder=4)

    # ── Colour-scale legend: spread across the full numeric block ────────────
    legend_y  = 0.038
    bar_h     = 0.022
    # Numeric block spans col_x[n_cat] to ~0.93; split it into two halves
    num_start = col_x[n_cat] + 0.01
    num_end   = 0.93
    half_span = (num_end - num_start - 0.04) / 2   # gap of 0.04 between bars
    bar_w1    = half_span
    bar_w2    = half_span

    def _draw_gradient(ax_obj, x0, y0, w, h, cmap, label):
        grad = np.linspace(0, 1, 256).reshape(1, -1)
        ax_obj.imshow(grad, aspect="auto", cmap=cmap,
                      extent=[x0, x0 + w, y0, y0 + h],
                      transform=ax_obj.transAxes, zorder=6, alpha=0.88)
        ax_obj.text(x0 + w / 2, y0 + h + 0.008, label,
                    ha="center", va="bottom", fontsize=8,
                    transform=ax_obj.transAxes, color="#333333")

    _draw_gradient(ax, num_start, legend_y, bar_w1, bar_h,
                   plt.cm.Blues, "Absolute values (blue = higher)")
    _draw_gradient(ax, num_start + bar_w1 + 0.04, legend_y, bar_w2, bar_h,
                   plt.cm.RdYlGn, "Gains / losses vs. LAPA (green = higher, 0 = yellow)")

    ax.set_title(
        "D-LAPA Stage-2.5 Model Summary\n"
        "(R\u00b2 from probe_results.json; LIBERO-LONG / CALVIN from Tables 2\u20134, CoRL 2026)",
        fontsize=9.5, pad=8
    )

    fig.tight_layout(pad=0.5)
    fig.savefig(out, dpi=300, bbox_inches="tight")
    fig.savefig(out.replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {out}")


# ── Entry point ────────────────────────────────────────────────────────────────

def main(probe_json: str = PROBE_JSON) -> None:
    probe = _load(probe_json)
    fig_interaction()
    fig_calvin_survival()
    fig_summary_heatmap(probe)
    print("\nAll done.  Output files:")
    for f in ["fig_interaction.pdf", "fig_calvin_survival.pdf", "fig_summary_heatmap.pdf"]:
        print(f"  {f}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else PROBE_JSON)
