"""
Reproduces Table 1 — Held-out |Δt| Ridge R² probe on LIBERO-10.

Memory-efficient: each probe row loads shards one at a time and pre-allocates
only (n_train, dim) + (n_test, dim) arrays. The full (138090, dim) matrix is
never materialised. Peak RAM ≈ 2.7 GB for the widest (5120-d) rows.

Field mapping (confirmed via check_pt_shapes.py + probe diagnostic):
  z_rgb_feature_input  → Pre-VQ LAPA LAQ (SSv2),  R²≈0.454
  z_rgb_indices_input  → Post-VQ LAPA LAQ (SSv2),  32-d one-hot
  z_depth_feature_gt   → Pre-VQ depth tok. (SSv2), 1024-d
  z_depth_indices_gt   → Post-VQ depth tok. (SSv2), 32-d one-hot
  concat(z_rgb_feature_input, z_depth_feature_pred_model_k) → right block proxy

Usage:
    python run_table1.py all_models_val_libero10_manifest_local.json
"""

import gc
import json
import sys
from pathlib import Path

import numpy as np
import torch

from data_loader import load_split_info, assemble_feature, one_hot_indices
from probe import ridge_probe

MANIFEST_DEFAULT = "all_models_val_libero10_manifest_local.json"
RESULTS_OUT = "probe_results.json"


# ── Extract functions (shard → (X_tr, X_te)) ──────────────────────────────────

def _simple(field: str):
    def fn(shard, tr, te):
        arr = shard[field].numpy()
        return arr[tr].copy(), arr[te].copy()
    return fn

def _onehot(field: str):
    def fn(shard, tr, te):
        arr = one_hot_indices(shard[field])
        return arr[tr].copy(), arr[te].copy()
    return fn

def _concat(depth_field: str):
    def fn(shard, tr, te):
        rgb = shard["z_rgb_feature_input"].numpy()
        dep = shard[depth_field].numpy()
        return (np.concatenate([rgb[tr], dep[tr]], axis=1),
                np.concatenate([rgb[te], dep[te]], axis=1))
    return fn


# ── Row specifications ─────────────────────────────────────────────────────────

def row_specs() -> list:
    return [
        # ── left block ────────────────────────────────────────────────────────
        ("Pre-VQ LAPA LAQ (SSv2)",        4096, _simple("z_rgb_feature_input")),
        ("Post-VQ LAPA LAQ (SSv2)",         32, _onehot("z_rgb_indices_input")),
        ("Pre-VQ depth tok. (SSv2)",       1024, _simple("z_depth_feature_gt")),
        ("Post-VQ depth tok. (SSv2)",        32, _onehot("z_depth_indices_gt")),
        # ── right block (z_rgb_feature_input used as proxy for finetuned LAPA) ─
        ("base LAPA (proxy reference)",    4096, _simple("z_rgb_feature_input")),
        ("base LAPA ⊕ Model 1",           5120, _concat("z_depth_feature_pred_model1")),
        ("base LAPA ⊕ Model 2",           5120, _concat("z_depth_feature_pred_model2")),
        ("base LAPA ⊕ Model 3",           5120, _concat("z_depth_feature_pred_model3")),
        ("base LAPA ⊕ Model 4",           5120, _concat("z_depth_feature_pred_model4")),
        ("base LAPA ⊕ Model 5",           5120, _concat("z_depth_feature_pred_model5")),
    ]


# ── Table printing ─────────────────────────────────────────────────────────────

def print_table(summary: list, n_test: int) -> None:
    left_rows  = summary[:4]
    right_rows = summary[4:]
    col_w = 36
    sep = "─" * (col_w * 2 + 24)

    print()
    print("Table 1 (reproduced) — Held-out |Δt| Ridge R² probe on LIBERO-10")
    print(f"n_test={n_test:,}, video-disjoint 80/20, seed 42")
    print()
    print(f"{'LAQ latent actions (Stage-1 targets)':<{col_w}} {'dim':>5}  {'R²':>6}"
          f"  {'Joint RGB ⊕ depth (proxy base)':<{col_w}} {'dim':>5}  {'R²':>6}")
    print(sep)

    for i in range(max(len(left_rows), len(right_rows))):
        ls = (f"{left_rows[i][0]:<{col_w}} {left_rows[i][1]:>5}  {left_rows[i][2]:>6.3f}"
              if i < len(left_rows) else " " * (col_w + 10))
        rs = (f"  {right_rows[i][0]:<{col_w}} {right_rows[i][1]:>5}  {right_rows[i][2]:>6.3f}"
              if i < len(right_rows) else "")
        print(ls + rs)

    print(sep)
    print()
    print("Paper right-block values (uses finetuned LAPA as RGB base — not available here):")
    for name, dim, r2 in [
        ("Finetuned LAPA (reference)", 4096, 0.619),
        ("FT LAPA ⊕ Model 1",         5120, 0.616),
        ("FT LAPA ⊕ Model 2",         5120, 0.618),
        ("FT LAPA ⊕ Model 3",         5120, 0.616),
        ("FT LAPA ⊕ Model 4",         5120, 0.626),
        ("FT LAPA ⊕ Model 5",         5120, 0.619),
    ]:
        print(f"  {name:<{col_w}} {dim:>5}  {r2:>6.3f}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main(manifest_path: str = MANIFEST_DEFAULT) -> dict:
    print(f"Loading manifest and computing split: {manifest_path}")
    manifest, train_mask, test_mask, y = load_split_info(manifest_path)
    y_train = y[train_mask]
    y_test  = y[test_mask]
    print(f"Split — n_train={train_mask.sum():,}  n_test={test_mask.sum():,}\n")

    specs   = row_specs()
    summary = []

    print(f"{'Row':<2}  {'Feature':<36} {'dim':>5}  {'R²':>6}  {'ρ':>6}  {'α':>8}")
    print("─" * 70)

    for i, (name, dim, extract_fn) in enumerate(specs, 1):
        print(f"{i:>2}. {name:<36} {dim:>5}  ", end="", flush=True)

        X_train, X_test = assemble_feature(manifest, train_mask, test_mask, dim, extract_fn)
        out = ridge_probe(X_train, y_train, X_test, y_test)

        del X_train, X_test
        gc.collect()

        summary.append((name, dim, out["r2"]))
        print(f"{out['r2']:>6.3f}  {out['rho']:>6.3f}  {out['alpha']:>8.1e}")

    print_table(summary, n_test=int(test_mask.sum()))

    save_data = {name: {"r2": r2, "dim": dim} for name, dim, r2 in summary}
    Path(RESULTS_OUT).write_text(json.dumps(save_data, indent=2))
    print(f"\nSaved → {RESULTS_OUT}")
    return save_data


if __name__ == "__main__":
    manifest = sys.argv[1] if len(sys.argv) > 1 else MANIFEST_DEFAULT
    main(manifest)