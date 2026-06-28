"""
Reproduces Supplemental Figure 1 (§.1):
  Six-panel UMAP of concat representations, coloured by |Δt|,
  annotated with Ridge R² (from probe_results.json) and Moran's I.

Moran's I formula (supplemental §.1):
  I = (N/W) · [Σ_{i,j} w_{ij}(x_i−x̄)(x_j−x̄)] / Σ_i(x_i−x̄)²
  with row-standardised k-NN weights → W = N →
  I = [Σ_i x_i · mean_{j∈NN(i)} x_j] / Σ_i x_i²

UMAP strategy: scaler is fitted on training rows (shard-by-shard, partial_fit),
then applied to the full test set. UMAP is fitted and transformed on the full
scaled test set (~27k points), avoiding the disconnected-graph issue that arises
from subsampling.

Usage:
    python run_umap.py all_models_val_libero10_manifest_local.json [probe_results.json]
"""

import gc
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import KDTree
from sklearn.preprocessing import StandardScaler
import torch
import umap

from data_loader import load_split_info
from probe import ridge_probe

MANIFEST_DEFAULT = "all_models_val_libero10_manifest_local.json"
RESULTS_JSON     = "probe_results.json"
OUTPUT_FIG       = "figures/supplemental_fig1_umap.pdf"

UMAP_SEED      = 42
UMAP_NEIGHBORS = 15
UMAP_MIN_DIST  = 0.1
MORANS_K       = 15
N_PERMUTATIONS = 199


# ── Moran's I ─────────────────────────────────────────────────────────────────

def morans_i(
    coords: np.ndarray,
    values: np.ndarray,
    k: int = MORANS_K,
    n_permutations: int = N_PERMUTATIONS,
    seed: int = UMAP_SEED,
) -> tuple:
    tree = KDTree(coords)
    _, nn_idx = tree.query(coords, k=k + 1)
    nn_idx = nn_idx[:, 1:]

    x = values - values.mean()

    def _stat(z: np.ndarray) -> float:
        return float(np.dot(z, z[nn_idx].mean(axis=1)) / np.dot(z, z))

    I_obs = _stat(x)
    rng = np.random.default_rng(seed)
    perm_vals = np.array([_stat(rng.permutation(x)) for _ in range(n_permutations)])
    p_value = float((np.sum(perm_vals >= I_obs) + 1) / (n_permutations + 1))
    return I_obs, p_value


# ── Shard-by-shard scaler fit + test assembly + UMAP ─────────────────────────

def embed_test(
    manifest: dict,
    train_mask: np.ndarray,
    test_mask: np.ndarray,
    extract_fn,
) -> np.ndarray:
    """
    1. Fit StandardScaler shard-by-shard on train rows (no full train matrix needed).
    2. Assemble the full scaled test set shard-by-shard.
    3. Fit and transform UMAP on the full scaled test set.

    Fitting UMAP on the full test set (~27k points) rather than a subsample
    avoids the disconnected nearest-neighbour graph and produces cleaner embeddings.
    Peak RAM ≈ (n_test × dim × 4 bytes) for the assembled test matrix.
    """
    scaler = StandardScaler()
    X_test_parts: list = []
    offset = 0

    for part_meta in manifest["parts"]:
        n = part_meta["num_samples"]
        shard_tr = train_mask[offset:offset + n]
        shard_te = test_mask [offset:offset + n]

        shard = torch.load(part_meta["path"], map_location="cpu", weights_only=False)
        X_tr, X_te = extract_fn(shard, shard_tr, shard_te)
        del shard
        gc.collect()

        if X_tr.shape[0] > 0:
            scaler.partial_fit(X_tr)
        X_test_parts.append(X_te)
        offset += n
        del X_tr, X_te

    X_test = scaler.transform(np.concatenate(X_test_parts, axis=0))
    del X_test_parts
    gc.collect()

    reducer = umap.UMAP(
        n_neighbors=UMAP_NEIGHBORS,
        n_components=2,
        min_dist=UMAP_MIN_DIST,
        metric="euclidean",
        random_state=UMAP_SEED,
        verbose=False,
    )
    embedding = reducer.fit_transform(X_test)
    del X_test
    gc.collect()

    return embedding


# ── Extract functions (shard → (X_tr, X_te)) ─────────────────────────────────

def _simple(field: str):
    def fn(shard, tr, te):
        arr = shard[field].numpy()
        return arr[tr].copy(), arr[te].copy()
    return fn

def _concat(depth_field: str):
    def fn(shard, tr, te):
        rgb = shard["z_rgb_feature_input"].numpy()
        dep = shard[depth_field].numpy()
        return (np.concatenate([rgb[tr], dep[tr]], axis=1),
                np.concatenate([rgb[te], dep[te]], axis=1))
    return fn


# ── Panel specification ───────────────────────────────────────────────────────

PANEL_META = [
    ("(a)", "base LAPA RGB (4096-d, proxy for finetuned)",  "base LAPA (proxy reference)", 4096, _simple("z_rgb_feature_input")),
    ("(b)", "base LAPA \u2295 Model 1 (5120-d)",           "base LAPA \u2295 Model 1",     5120, _concat("z_depth_feature_pred_model1")),
    ("(c)", "base LAPA \u2295 Model 2 (5120-d)",           "base LAPA \u2295 Model 2",     5120, _concat("z_depth_feature_pred_model2")),
    ("(d)", "base LAPA \u2295 Model 3 (5120-d)",           "base LAPA \u2295 Model 3",     5120, _concat("z_depth_feature_pred_model3")),
    ("(e)", "base LAPA \u2295 Model 4 (5120-d)",           "base LAPA \u2295 Model 4",     5120, _concat("z_depth_feature_pred_model4")),
    ("(f)", "base LAPA \u2295 Model 5 (5120-d)",           "base LAPA \u2295 Model 5",     5120, _concat("z_depth_feature_pred_model5")),
]


# ── Probe loading ─────────────────────────────────────────────────────────────

def load_or_run_probes(manifest: dict, train_mask, test_mask, y) -> dict:
    if Path(RESULTS_JSON).exists():
        with open(RESULTS_JSON) as f:
            return json.load(f)

    print("  probe_results.json not found — running Ridge probes inline …")
    from data_loader import assemble_feature

    y_train, y_test = y[train_mask], y[test_mask]
    results = {}
    for _, _, feat_key, dim, extract_fn in PANEL_META:
        from data_loader import assemble_feature
        X_tr, X_te = assemble_feature(manifest, train_mask, test_mask, dim, extract_fn)
        out = ridge_probe(X_tr, y_train, X_te, y_test)
        results[feat_key] = {"r2": out["r2"]}
        del X_tr, X_te
        gc.collect()
        print(f"    {feat_key}: R²={out['r2']:.3f}")
    return results


# ── Figure ────────────────────────────────────────────────────────────────────

def plot_panels(panels: list, magnitude_test: np.ndarray, output_path: str) -> None:
    mag_norm = (magnitude_test - magnitude_test.min()) / (
        magnitude_test.max() - magnitude_test.min() + 1e-9
    )
    vmin, vmax = 0.2, 1.0
    cmap = plt.cm.plasma

    # 2×3 grid + dedicated narrow column for the colorbar
    fig = plt.figure(figsize=(14, 9))
    gs = fig.add_gridspec(
        2, 4,
        width_ratios=[1, 1, 1, 0.045],
        hspace=0.32, wspace=0.22,
        left=0.03, right=0.91, top=0.88, bottom=0.04,
    )
    panel_axes = [fig.add_subplot(gs[r, c]) for r in range(2) for c in range(3)]
    cbar_ax    = fig.add_subplot(gs[:, 3])

    scatter_kw = dict(
        c=mag_norm, cmap=cmap, vmin=vmin, vmax=vmax,
        s=0.8, alpha=0.45, rasterized=True, linewidths=0,
    )

    for ax, (label, title, _, _, _, embed, r2, I, p) in zip(panel_axes, panels):
        ax.scatter(embed[:, 0], embed[:, 1], **scatter_kw)
        ax.set_title(title, fontsize=7.5, pad=4)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_linewidth(0.5)
        ax.text(0.03, 0.04, label, transform=ax.transAxes,
                fontsize=9, fontweight="bold", va="bottom")
        ax.text(0.97, 0.97,
                f"$R^2 = {r2:.3f}$\n$I = {I:.3f}$",
                transform=ax.transAxes, fontsize=7.5,
                ha="right", va="top",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.85, ec="none"))

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label(r"$|\Delta t|$ (normalised)", fontsize=9, rotation=270, labelpad=14)
    cbar.ax.tick_params(labelsize=8)

    fig.suptitle(
        "Deployment-equivalent concat representation\n"
        r"(5120-d, base LAPA $\oplus$ Stage-2.5 depth-encoder feature"
        r" — finetuned LAPA unavailable, base LAPA used as proxy)",
        fontsize=9, y=0.96,
    )
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure saved → {output_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main(manifest_path: str = MANIFEST_DEFAULT) -> None:
    print(f"Loading manifest and computing split: {manifest_path}")
    manifest, train_mask, test_mask, y = load_split_info(manifest_path)
    y_test = y[test_mask]
    print(f"Split — n_train={train_mask.sum():,}  n_test={test_mask.sum():,}\n")

    probe_results = load_or_run_probes(manifest, train_mask, test_mask, y)

    panels = []
    for label, title, feat_key, dim, extract_fn in PANEL_META:
        print(f"  UMAP for {feat_key} (dim={dim}) …")
        embedding = embed_test(manifest, train_mask, test_mask, extract_fn)

        print(f"    Moran's I …")
        I_obs, p_val = morans_i(embedding, y_test)

        r2 = probe_results[feat_key]["r2"]
        panels.append((label, title, feat_key, dim, extract_fn, embedding, r2, I_obs, p_val))
        print(f"    R²={r2:.3f}  I={I_obs:.3f}  p={p_val:.4f}")

    plot_panels(panels, y_test, OUTPUT_FIG)


if __name__ == "__main__":
    manifest = sys.argv[1] if len(sys.argv) > 1 else MANIFEST_DEFAULT
    main(manifest)