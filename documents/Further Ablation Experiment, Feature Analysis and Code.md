# Further Ablation Experiment, Feature Analysis and Code

*Supplemental material for: Depth Information Injection in VLA with Latent Action Pretraining via Synthetic Depth Data*

---

## .1 Deployment-Equivalent Feature: Concatenated RGB ⊕ Depth

**Moran's I on UMAP coordinates.** Spatial autocorrelation of $|\Delta t|$ on the 2D UMAP coordinates. With row-standardised $k=15$ nearest-neighbour weights $w_{ij}$:

$$I = \frac{N}{W} \frac{\sum_{i,j} w_{ij}(x_i - \bar{x})(x_j - \bar{x})}{\sum_i (x_i - \bar{x})^2}, \quad W = \sum_{i,j} w_{ij}$$

where $x_i = |\Delta t_i|$. $I \to 1$ means neighbouring projected points share similar $|\Delta t|$; $I \to 0$ means the projection is colour-random. $p$-values are obtained from 199 label permutations.

> **Figure 1:** Deployment-equivalent concat representation (5120-d, Finetuned LAPA ⊕ Stage-2.5 depth-encoder feature). Six-panel UMAP snapshot — panel (a) is the action-supervised finetuned-LAPA RGB feature alone (4096-d), panels (b)–(f) are FT LAPA ⊕ Models 1–5 (5120-d each). Each panel is annotated with Ridge $R^2$ (from Table 1 of the main paper) and Moran's $I$.
>
> | Panel | Feature | $R^2$ | $I$ |
> |---|---|---|---|
> | (a) | Finetuned LAPA RGB (4096-d, baseline) | 0.619 | 0.585 |
> | (b) | Model 1 + Finetuned LAPA (5120-d) | 0.616 | 0.557 |
> | (c) | Model 2 + Finetuned LAPA (5120-d) | 0.618 | 0.623 |
> | (d) | Model 3 + Finetuned LAPA (5120-d) | 0.616 | 0.587 |
> | (e) | Model 4 + Finetuned LAPA (5120-d) | 0.626 | 0.633 |
> | (f) | Model 5 + Finetuned LAPA (5120-d) | 0.619 | 0.573 |
>
> Colorbar: $|\Delta t|$ (normalised), range 0.2–1.0.

The figure provides further evidence that features from our pretrained encoder, trained exclusively on human demonstrations from the Something-Something V2 (SSv2) dataset, can improve movement magnitude prediction within LIBERO demonstrations. Model 1, which takes action indices as input, yields no improvement, suggesting that the pretrained index outputs from LAPA carry limited spatial information. Notably, Model 2 and Model 4, both trained solely on SSv2 data, still manage to improve predictions over features from a LAPA model finetuned on LIBERO, even though their inputs (pretrained LAPA features) are totally unrelated to the LIBERO dataset. The improvements manifest in both prediction accuracy and smoothness, indicating better generalisation in movement magnitude estimation and a stronger grasp of spatial concepts. By contrast, Model 3 and Model 5, which do not take depth images as input, show no comparable improvement, pointing to depth imagery rather than the model parameters themselves as the true source of the gains.

---

## .2 Datasize Ablation: Probe $R^2$ vs. SSv2 Fraction

> **Figure 2:** Probe Ridge $R^2_\text{test}$ (left) and Ridge Spearman $\rho_\text{test}$ (right) vs. LAPA-LAQ training-set size (% of SSv2) at two pretraining checkpoints (15k and 65k optimizer steps). Probe quality rises monotonically with datasize over the 5–80% range under both probe families.

In this experiment, we examine whether the SSv2 training data size influences movement magnitude prediction. We find that prediction accuracy under the Ridge regression probe improves consistently as the dataset size increases.

---

## .3 Code and Data Availability

All code, configuration files, feature-cache builders, probe runners, and figure-generation scripts can be found at:

[https://anonymouscorl26.github.io/](https://anonymouscorl26.github.io/)

The repository contains step-by-step reproducibility instructions, including environment setup, dataset preparation, feature extraction, probe training, and figure rendering — sufficient to reproduce every number and figure in this paper from the public LAPA and LIBERO checkpoints.
