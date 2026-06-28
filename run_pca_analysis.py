"""
PCA of the 1024-d depth encoder features (z_depth_feature_pred_model_k)
for each Stage-2.5 model, plus the ground-truth depth feature as a reference.

Memory strategy
---------------
Three shard-by-shard passes per model; the full training matrix is never
materialised in RAM.

  Pass 1 (_fit_scaler):  StandardScaler.partial_fit on training rows.
                         Peak: one shard ≈ 64 MB (8192 × 1024 × float64).

  Pass 2 (_fit_ipca):    IncrementalPCA.partial_fit on scaled training rows,
                         buffered to ensure batch ≥ n_components.
                         Peak: buffer (≤ MIN_IPCA_BATCH rows) ≈ 4 MB.

  Pass 3 (_scores_test): transform scaled test rows shard by shard.
                         Peak: one shard ≈ 64 MB.

Scientific question
-------------------
Does depth training concentrate the geometric signal (Spearman ρ with |Δt|)
into the dominant variance directions of the encoder's output?  If Models 2
and 4's top principal components show high |ρ| while Models 3 and 5 show
near-zero |ρ| at all ranks, depth conditioning organises the encoder's output
around geometric content — something the UMAP or aggregate R² cannot show.

Outputs
-------
  pca_results.json       per-model PC correlation spectra and explained variance
  fig_pca_analysis.pdf   two-panel publication figure:
                           left  — cumulative Σ|ρ_k| vs. PC rank (smooth,
                                   monotonically increasing; replaces noisy
                                   per-PC plot)
                           right — scree (cumulative explained variance) with
                                   80 / 90 / 95 % reference lines

Usage
-----
  python run_pca_analysis.py all_models_val_libero10_manifest.json

  To re-generate the figure from an existing pca_results.json without
  re-running the expensive 3-pass analysis, pass the JSON directly:

    python run_pca_analysis.py pca_results.json
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.decomposition import IncrementalPCA
from sklearn.preprocessing import StandardScaler

from data_loader import video_disjoint_split

MANIFEST_DEFAULT  = "all_models_val_libero10_manifest.json"
N_COMPONENTS      = 50
MIN_IPCA_BATCH    = max(N_COMPONENTS * 2, 500)   # must be >= n_components

_MODELS = [
    ("Model 2 (Disc, depth)",    "z_depth_feature_pred_model2", "#2166ac", "-"),
    ("Model 4 (Cont, depth)",    "z_depth_feature_pred_model4", "#1b7837", "-"),
    ("Model 3 (Disc, no depth)", "z_depth_feature_pred_model3", "#e08214", "--"),
    ("Model 5 (Cont, no depth)", "z_depth_feature_pred_model5", "#d6604d", "--"),
    ("GT depth feat. (ref.)",    "z_depth_feature_gt",          "#7b2d8b", ":"),
]

plt.rcParams.update({
    "font.family": "serif", "font.size": 9, "axes.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False,
    "pdf.fonttype": 42, "ps.fonttype": 42,
})


def _scan(manifest_path):
    with open(manifest_path) as f:
        manifest = json.load(f)

    shard_info, video_ids, mags = [], [], []
    offset = 0
    for part in manifest["parts"]:
        shard = torch.load(part["path"], map_location="cpu", weights_only=False)
        n = len(shard["video_id"])
        shard_info.append({"path": part["path"], "offset": offset, "n": n})
        video_ids.extend(shard["video_id"])
        mags.append(shard["magnitude"].numpy().astype(np.float32))
        del shard
        offset += n

    y = np.concatenate(mags)
    train_mask, test_mask = video_disjoint_split(video_ids)
    return shard_info, y, np.where(train_mask)[0], np.where(test_mask)[0]


def _local_idx(info, global_idx):
    """Return sorted local indices within this shard for the given sorted global indices."""
    off, n = info["offset"], info["n"]
    mask = (global_idx >= off) & (global_idx < off + n)
    return global_idx[mask] - off


def _fit_scaler(shard_info, key, train_idx):
    """Pass 1: StandardScaler.partial_fit on training rows only."""
    train_idx = np.sort(train_idx)
    scaler = StandardScaler()
    for info in shard_info:
        local = _local_idx(info, train_idx)
        if not len(local):
            continue
        shard = torch.load(info["path"], map_location="cpu", weights_only=False)
        X = shard[key].numpy()[local].astype(np.float64)
        del shard
        scaler.partial_fit(X)
        del X
    return scaler


def _fit_ipca(shard_info, key, scaler, train_idx, n_components):
    """
    Pass 2: IncrementalPCA.partial_fit on scaled training rows.
    Buffers shard batches until MIN_IPCA_BATCH rows are accumulated
    to satisfy IncrementalPCA's minimum-batch requirement.
    """
    train_idx = np.sort(train_idx)
    ipca      = IncrementalPCA(n_components=n_components)
    buf, buf_n = [], 0

    for info in shard_info:
        local = _local_idx(info, train_idx)
        if not len(local):
            continue
        shard = torch.load(info["path"], map_location="cpu", weights_only=False)
        X = scaler.transform(shard[key].numpy()[local].astype(np.float64))
        del shard
        buf.append(X); buf_n += len(X)
        del X

        if buf_n >= MIN_IPCA_BATCH:
            ipca.partial_fit(np.concatenate(buf))
            buf, buf_n = [], 0

    if buf:
        batch = np.concatenate(buf)
        if len(batch) >= n_components:
            ipca.partial_fit(batch)

    return ipca


def _scores_test(shard_info, key, scaler, ipca, test_idx):
    """Pass 3: transform test rows shard by shard, return score matrix."""
    test_idx = np.sort(test_idx)
    parts = []
    for info in shard_info:
        local = _local_idx(info, test_idx)
        if not len(local):
            continue
        shard = torch.load(info["path"], map_location="cpu", weights_only=False)
        X = scaler.transform(shard[key].numpy()[local].astype(np.float64))
        del shard
        parts.append(ipca.transform(X))
        del X
    return np.concatenate(parts)


def _spearman(x, y):
    r = spearmanr(x, y)
    return float(r.statistic if hasattr(r, "statistic") else r.correlation)


def _pca_spectrum(shard_info, key, train_idx, test_idx, y_test):
    print(f"    pass 1/3: fitting scaler ...", flush=True)
    scaler = _fit_scaler(shard_info, key, train_idx)

    print(f"    pass 2/3: fitting IncrementalPCA (n_components={N_COMPONENTS}) ...", flush=True)
    ipca = _fit_ipca(shard_info, key, scaler, train_idx, N_COMPONENTS)

    print(f"    pass 3/3: transforming test set ...", flush=True)
    scores = _scores_test(shard_info, key, scaler, ipca, test_idx)

    rhos = np.array([_spearman(scores[:, k], y_test) for k in range(N_COMPONENTS)])
    return {
        "spearman_rho":      rhos.tolist(),
        "explained_variance": ipca.explained_variance_ratio_.tolist(),
    }


def run(manifest_path):
    print(f"Scanning manifest: {manifest_path}", flush=True)
    shard_info, y, train_idx, test_idx = _scan(manifest_path)
    print(f"  n_train={len(train_idx):,}  n_test={len(test_idx):,}", flush=True)
    y_test = y[test_idx]

    results = {}
    for label, key, *_ in _MODELS:
        print(f"  [{label}]  key={key}", flush=True)
        results[label] = _pca_spectrum(shard_info, key, train_idx, test_idx, y_test)
        top5 = np.round(np.abs(results[label]["spearman_rho"][:5]), 3).tolist()
        print(f"    top-5 |ρ|: {top5}", flush=True)

    with open("pca_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Saved → pca_results.json", flush=True)
    _plot(results)


def _plot(results: dict) -> None:
    """
    Left panel  — Cumulative Σ|ρ_k| with |Δt| vs. PC rank.
                  Smooth and monotonically increasing; replaces the per-PC plot
                  which is inherently noisy (individual PCs fluctuate around
                  their mean).  The final value is the total geometric correlation
                  captured by the top-N_COMPONENTS PCs; a steeper early rise
                  means the signal is concentrated in the dominant variance
                  directions.

    Right panel — Cumulative explained variance (scree) with 80 / 90 / 95 %
                  reference lines so the "k PCs for X% variance" threshold
                  is immediately readable.

    Legend below both panels; does not overlap the plot area.
    """
    fig, (ax_rho, ax_var) = plt.subplots(1, 2, figsize=(11.0, 4.8))

    for label, _, color, ls in _MODELS:
        if label not in results:
            continue
        rhos    = np.abs(results[label]["spearman_rho"])
        cum_rho = np.cumsum(rhos)
        evr     = np.array(results[label]["explained_variance"])
        k       = np.arange(1, len(rhos) + 1)
        ax_rho.plot(k, cum_rho, color=color, linestyle=ls, linewidth=1.8,
                    label=label, zorder=4)
        ax_var.plot(k, np.cumsum(evr) * 100, color=color, linestyle=ls,
                    linewidth=1.8, label=label, zorder=4)

    for pct in (80, 90, 95):
        ax_var.axhline(pct, color="#b0b0b0", linewidth=0.7, linestyle=":", zorder=1)
        ax_var.text(N_COMPONENTS + 0.8, pct, f"{pct}%",
                    fontsize=7, va="center", color="#888888")

    ax_rho.set_xlabel("Principal component rank  (sorted by variance explained)",
                      fontsize=9.5)
    ax_rho.set_ylabel("Cumulative $\\sum|\\rho_k|$ with $|\\Delta t|$", fontsize=9.5)
    ax_rho.set_title(
        "Geometric Signal Accumulation across PC Space\n"
        f"(depth encoder output, 1024-d; top {N_COMPONENTS} PCs by variance)",
        pad=7,
    )
    ax_rho.grid(axis="y", linewidth=0.35, color="#e0e0e0", zorder=0)
    ax_rho.set_xlim(0, N_COMPONENTS + 1)

    ax_var.set_xlabel("Principal component rank", fontsize=9.5)
    ax_var.set_ylabel("Cumulative explained variance (%)", fontsize=9.5)
    ax_var.set_title(
        "Variance Concentration (Scree)\n"
        "(how concentrated is the depth encoder's output variance?)",
        pad=7,
    )
    ax_var.grid(axis="y", linewidth=0.35, color="#e0e0e0", zorder=0)
    ax_var.set_xlim(0, N_COMPONENTS + 4)   # extra room for % labels

    handles, labels = ax_rho.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, fontsize=8.5,
               framealpha=0.93, edgecolor="#cccccc", bbox_to_anchor=(0.5, -0.04))

    fig.tight_layout(pad=1.2, rect=[0, 0.09, 1, 1])
    out = "figures/fig_pca_analysis.pdf"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    fig.savefig(out.replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out}")


if __name__ == "__main__":
    manifest    = sys.argv[1] if len(sys.argv) > 1 else MANIFEST_DEFAULT
    json_exists = Path("pca_results.json").exists()
    pdf_exists  = Path("figures/fig_pca_analysis.pdf").exists()

    if json_exists and pdf_exists:
        print("Both pca_results.json and fig_pca_analysis.pdf already exist — nothing to do.")
    elif json_exists and not pdf_exists:
        print("Found pca_results.json — generating fig_pca_analysis.pdf only.")
        with open("pca_results.json") as f:
            _plot(json.load(f))
    else:
        run(manifest)