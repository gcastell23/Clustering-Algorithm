# Step 9 — Datasets and Results

## The Five Datasets

| Dataset | Cells | Cell Types | Structure | Why We Use It |
|---|---|---|---|---|
| **PBMC 3k** | 2,638 | 8 immune types | Discrete | Fast development, clean NB demo |
| **Paul15** | 2,730 | 8 myeloid | **Continuous** | Differentiation trajectory; topology showcase |
| **Pancreas atlas** | 16,382 | 14 types, 9 technologies | Mixed discrete | Rare types, batch integration, marker-gene labels |
| **Segerstolpe** | 2,394 | 13 types (healthy/T2D) | Disease contrast | Single technology; diabetes covariate shift |
| **Simulation** | 5,000 | 9 types (4 rare) | Planted discrete | Labels we control; clean causal test |
| **CITE-seq PBMC** | 4,193 | 6 protein-defined | Discrete orthogonal | Protein-only labels; breaks circularity |

All datasets are open access with confirmed integer UMI counts.

## Headline Results

### The Count-Aware Latent Rescues Rare Types

| Dataset | PCA+Leiden rare F1 | IGMC rare F1 | Fold improvement |
|---|---|---|---|
| Pancreas | 0.002 | 0.48 | **240×** |
| Simulation (planted) | 0.04 | 0.45 | **11×** |
| Segerstolpe | 0.00 | 0.10 | **∞** |

### Markov Stability (No Resolution Knob)

| Dataset | PCA+Leiden ARI | IGMC-Markov ARI |
|---|---|---|
| Pancreas (marker labels) | 0.47 | **0.91** |
| PBMC 3k | 0.86 | 0.84 |
| Paul15 | 0.36 | 0.58 |

### Conformal Coverage (Target: 90%)

| Dataset | Achieved Coverage |
|---|---|
| PBMC 3k | 0.907 |
| Pancreas | 0.900 |
| Paul15 | 0.922 |
| Segerstolpe | 0.918 |

### Topology: Discrete vs. Continuous

| Dataset | Discreteness | Conformal Ambiguity |
|---|---|---|
| PBMC 3k | 0.84 | ~1% |
| Pancreas | 0.95 | ~1% |
| Paul15 (continuum) | 0.63 | 27% |

### PhD-Strength Results

| Finding | Number |
|---|---|
| Multi-seed: count-aware ARI vs PCA (pancreas, 95% CI) | 0.82 [0.78,0.86] vs 0.50 [0.48,0.52] |
| CITE-seq DC F1: Fisher-Rao vs Euclidean (same latent) | 0.88 vs 0.27 |
| scIB overall: IGMC vs scANVI (label-supervised) | 0.716 vs 0.733 (3rd of 6) |
| Scale-nested simultaneous coverage (target 0.90) | 0.98 (coherent+Bonferroni) |
| T2D beta-cell OT displacement vs null | 2.7×, p < 0.004 |

## Honest Limitations

- PBMC labels are circular (derived from PCA+Louvain itself) — not a fair benchmark
- The Fisher-Rao metric's edge is setting-dependent; the robust win is the count-aware latent
  + Markov stability
- The scale-nested conformal guarantee is conservative (union bound is loose under nesting)
- No dedifferentiation claim — the diabetic beta-cells retain their identity
- Largest dataset is 16k cells; atlas scale (100k+) remains future work

Next: [Step 10 — PhD-Level Extensions](10-extensions.md)
