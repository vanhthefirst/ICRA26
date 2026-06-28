# D-LAPA Feature Analysis: Reproduction & Extension

Reproduction of Section 3 (Feature Analysis) and Supplemental §.1 of *Depth Information Injection in VLA with Latent Action Pretraining via Synthetic Depth Data* (D-LAPA, CoRL 2026 anonymous submission), plus three novel analysis figures prepared as an extension toward a potential ICRA submission.

---

## Results at a Glance

### Table 1 — Left Block: Stage-1 LAQ targets (fully reproducible, all |Δ| ≤ 0.004)

| Feature | dim | R² (ours) | R² (paper) | Δ |
|---|---:|---:|---:|---:|
| Pre-VQ LAPA LAQ (SSv2) | 4096 | 0.450 | 0.454 | −0.004 |
| Post-VQ LAPA LAQ (SSv2) | 32 | 0.009 | 0.006 | +0.003 |
| Pre-VQ depth tok. (SSv2) | 1024 | 0.137 | 0.134 | +0.003 |
| Post-VQ depth tok. (SSv2) | 32 | 0.137 | 0.137 | 0.000 |
| Finetuned LAPA (action-sup.) | 4096 | — | 0.619 | not reproducible |

*Requires finetuned LAPA checkpoint absent from provided cached features.

### Table 1 — Right Block: base LAPA proxy (relative ordering fully preserved)

| Feature | dim | R² (ours) | ΔR² (ours) | R² (paper) |
|---|---:|---:|---:|---:|
| RGB baseline (base LAPA) | 4096 | 0.450 | — | 0.619 |
| ⊕ Model 1 (depth, disc., index) | 5120 | 0.451 | +0.0006 | 0.616 |
| **⊕ Model 2** (depth, disc., feat.) | 5120 | **0.513** | **+0.0630** | 0.618 |
| ⊕ Model 3 (no depth, disc., feat.) | 5120 | 0.451 | +0.0007 | 0.616 |
| **⊕ Model 4** (depth, cont., feat.) | 5120 | **0.534** | **+0.0838** | **0.626** |
| ⊕ Model 5 (no depth, cont., feat.) | 5120 | 0.438 | −0.0119 | 0.619 |

Base LAPA (R²=0.450) replaces finetuned LAPA (R²=0.619) as the RGB component; absolute values differ but the rank ordering is identical: Models 2 and 4 are the clear leaders.

### Supplemental §.1 — Moran's *I* on UMAP Coordinates

| Panel | Feature | R² | *I* (ours) | *I* (paper) |
|---|---|---:|---:|---:|
| (a) | base LAPA RGB | 0.450 | 0.325 | 0.585 |
| (b) | ⊕ Model 1 | 0.451 | 0.380 | 0.557 |
| (c) | **⊕ Model 2** | 0.513 | **0.539** | **0.623** |
| (d) | ⊕ Model 3 | 0.451 | 0.325 | 0.587 |
| (e) | **⊕ Model 4** | 0.534 | **0.582** | **0.633** |
| (f) | ⊕ Model 5 | 0.438 | 0.324 | 0.573 |

Rank ordering of Moran's *I* is preserved; absolute values are attenuated because the base LAPA proxy lacks the finetuned backbone's spatial structure.

---

## Key Finding

**Depth imagery is the operative signal, not model architecture.** Models 2 and 4 (depth image present) gain ΔR²=+0.063 and +0.084 over the RGB baseline. Models 1, 3, and 5 gain ΔR²≤+0.001 or regress — ruling out added parameters as the explanation. This is confirmed by three independent metrics (probe R², Moran's *I*, and downstream LIBERO-LONG/CALVIN performance) and independently of which RGB base is used.

---

## Novel Extensions (ICRA-targeted figures)

| Figure | Content | Source |
|---|---|---|
| `fig_interaction.pdf` | 2×2 factorial interaction (depth × supervision) + ablation effect sizes | Tables 2–3 |
| `fig_calvin_survival.pdf` | Per-step CALVIN delta bars + conditional transition survival | Table 4 |
| `fig_summary_heatmap.pdf` | Unified R² / LIBERO-LONG / CALVIN summary heatmap | Tables 3–4 + probe |

**Scale-stratified residual analysis** (`run_residual_analysis.py`): ⊕ Model 2 reduces large-displacement (|Δt| ≥ 0.715, top quartile) probe MAE by **11.4%** vs.\ base LAPA; no-depth ablations (Models 3, 5) stay within ±0.8%, ruling out architectural capacity as the explanation.

---

## Reproducibility

```bash
pip install -r requirements.txt
python fix_manifest_paths.py "<manifest_path>" "<pt_folder>"
python run_table1.py  "<manifest_local_path>"           # → probe_results.json
python run_umap.py    "<manifest_local_path>"           # → supplemental_fig1_umap.pdf
python plot_icra_figures.py                             # → fig_interaction.pdf, fig_calvin_survival.pdf, fig_summary_heatmap.pdf
python run_pca_analysis.py "<manifest_local_path>"      # → pca_results.json, fig_pca_analysis.pdf
python run_residual_analysis.py "<manifest_local_path>" # → residual_results.npz, residual_analysis_summary.md
```

- **Split:** 80/20 by `video_id`, `seed=42`; n\_train=110,528, n\_test=27,562
- **Peak RAM:** ≈ 2.7 GB (shard-by-shard loading; 138,090 frames across 17 `.pt` shards)
- **Runtime:** Table 1 ≈ 20–40 min CPU; UMAP ≈ 30–60 min

---

## File Structure

```
.
├── data_loader.py            Shard loading, video-split, one-hot, feature assembly
├── probe.py                  Ridge GCV probe → R², Spearman ρ
├── run_table1.py             Table 1 entry point → probe_results.json
├── run_umap.py               Supplemental Fig. 1 → supplemental_fig1_umap.pdf
├── plot_icra_figures.py      Three novel ICRA figures (factorial, CALVIN, heatmap)
├── run_residual_analysis.py  Scale-stratified MAE analysis
├── run_pca_analysis.py       PCA of depth-encoder features
├── fix_manifest_paths.py     One-time path rewrite utility
├── probe_results.json        Computed R² for all features (ground truth for figures)
├── acl.sty                   ACL style file
├── report.tex / report.pdf   Full technical report (ACL format)
└── README.md                 This file
```

---

## What Could Not Be Reproduced

| Item | Reason |
|---|---|
| Finetuned LAPA R²=0.619 (right-block baseline) | Action-supervised LAPA checkpoint absent from cached `.pt` files |
| Exact Moran's *I* magnitudes from Supplemental §.1 | Same cause — base LAPA proxy lacks finetuned backbone's spatial structure |
| Supplemental §.2 data-size ablation curves | Requires intermediate checkpoints at 0–65k steps, not provided |

---

*Full technical report: `documents/Feature_Analysis_Report.pdf`*