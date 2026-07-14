# Step 3 — The Fisher-Rao Metric

## The Idea

Every statistical model comes with a built-in geometry. The **Fisher information** measures
how much a distribution changes when you tweak its parameter — it tells you how
"distinguishable" two parameter values are.

For a negative binomial, the Fisher information with respect to the mean **μ** is:

```
I(μ) = 1 / Var(x) = θ / (μ(μ + θ))
```

This is beautiful. The weight is `1 / variance`. When the variance is low (quiet gene), the
weight is high. When the variance is high (noisy gene), the weight is low. This is exactly the
heteroscedastic correction the data demands — and it falls out of first principles.

The figure below shows the intuition:

```
                    Fisher Information I(μ) = 1/Var
                         ^
                     1.0 |***
                         |  ****
                         |      ******
                         |            *************
                     0.0 +---------------------------→  μ (gene mean)
                         0    10     50      500

Quiet, rarely-expressed genes → HIGH weight (each molecule matters)
Loud, highly-expressed genes  → LOW weight  (mostly noise)
```

## Pulling it Back to the Latent Space

We don't compute this in gene space (20,000 dimensions is too many). Instead:

1. A VAE compresses each cell into a low-dimensional latent code **z** ∈ ℝᵈ (d ≈ 10)
2. The decoder maps **z** → gene proportions **ρ(z)**
3. We pull the Fisher metric *back* through the decoder to the latent space

The metric at a point **z** is a **d × d** matrix (only 10×10 per cell):

```
M(z) = J_ρ(z)ᵀ · diag(1/Var(x_g)) · J_ρ(z)
```

where **J_ρ** is the Jacobian: how much each gene proportion changes when z moves.

## What the Metric Tells Us

- **Distance**: two cells that are close in Fisher-Rao are *statistically indistinguishable*
  given their count models. Cells that are far apart are reliably different.
- **Anisotropy**: the metric is highly directional (median condition number ≈ 100× Euclidean).
  Moving in some directions changes the predicted gene expression a lot; in others, barely at
  all.
- **Volume element**: the log-determinant of M(z) acts like a local "magnification factor" —
  some regions get stretched, others compressed.

## The Neighbor Distance

For comparing nearby cells, we use the **symmetric midpoint metric**:

```
δ²(i, j) = (z_i - z_j)ᵀ · (M_i + M_j)/2 · (z_i - z_j)
```

This is a second-order-accurate geodesic surrogate — it approximates the true shortest
distance along the curved manifold without needing to compute a geodesic path.

## Why This is Efficient

The key trick: `d` (≈10) is tiny compared to `G` (≈2,000 genes). We compute `J_ρ` using
**forward-mode** automatic differentiation (`jacfwd`). Forward mode is cheap when the output
dimension (G) is large but the input dimension (d) is small. The per-cell metric is only a
d×d matrix. Nothing G×G is ever materialized. The whole pancreas atlas (16,000 cells) takes
seconds.

Next: [Step 4 — The Negative-Binomial VAE](04-nbvae.md)
