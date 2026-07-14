# Step 11 — Code Layout and How to Reproduce

## Code Modules

| File | Does | Lines |
|---|---|---|
| `data.py` | Load & preprocess datasets, QC, HVG, raw counts | ~195 |
| `simulate.py` | Planted-ground-truth NB simulation generator | ~83 |
| `nbvae.py` | Batch-conditional NB-VAE, PyTorch, CPU | ~149 |
| `geometry.py` | Fisher-Rao pullback metric (jacfwd+vmap) | ~131 |
| `graph.py` | Fisher-Rao kNN graph + Leiden | ~98 |
| `markov_stability.py` | exp(−tL) flow graph + VI-plateau selection | ~194 |
| `topology.py` | Vietoris-Rips persistence + transitionality | ~216 |
| `conformal.py` | Mondrian split-conformal prediction sets | ~121 |
| `evaluate.py` | ARI, NMI, rare-F1, Hungarian matching | ~93 |
| `pipeline.py` | End-to-end integrator, cache writer | ~230 |
| `train_cache.py` | VAE + metric cache layer | ~39 |
| `sim_experiment.py` | Metric ablation on planted simulations | ~78 |
| `ablation_markov.py` | Markov on Euclidean vs FR graph | ~66 |
| `multiseed.py` | Multi-seed VAE retraining with 95% CIs | ~188 |
| `citeseq.py` | CITE-seq protein-only labels | ~123 |
| `scib_metrics.py` | scIB battery from primitives | ~147 |
| `benchmark_integration.py` | Multi-method benchmark runner | ~122 |
| `scale_nested_conformal.py` | Scale-nested conformal + coherence theorem | ~389 |
| `advanced.py` | OT of disease, spectral/RMT, Ricci curvature | ~158 |
| `compute_advanced.py` | Runner for advanced computations | ~56 |
| `figures.py` | Figs 1-11 (4 panels each) | ~1,121 |
| `figures_ws.py` | Figs 12-15 (PhD extensions) | ~401 |
| `style.py` | Okabe-Ito palette, Nature-grade design | ~134 |

## Key Engineering Decisions

1. **`jacfwd` (forward-mode AD)** — d ≈ 10 << G ≈ 2,000, so forward mode is efficient
2. **Vectorized top-k** — `np.argpartition` over blocks instead of per-row loops
3. **Stratified subsampling** — Markov stability preserves rare types (≥30 cells each)
4. **Conformal class-mismatch tolerance** — calibration classes not in training are dropped
5. **Adaptive kernel bandwidth** — per-cell sigma = distance to kth neighbor

## Reproduce Everything

```bash
# 1. Build processed datasets
python src/data.py

# 2. Train NB-VAE + compute Fisher-Rao metric once per dataset (~25 min)
python src/train_cache.py

# 3. Run full pipeline on each dataset
python src/pipeline.py pbmc3k pancreas paul15 segerstolpe

# 4. Simulation experiment
python src/sim_experiment.py

# 5. Generate main figures (Figs 1-11)
python src/figures.py all

# 6. Generate extension figures (Figs 12-15)
python src/figures_ws.py all

# 7. Compile manuscript
cd paper && pdflatex main && bibtex main && pdflatex main && pdflatex main
```

## PhD-Strength Runs

```bash
# Multi-seed confidence intervals
python src/multiseed.py pbmc3k paul15 segerstolpe pancreas --nseed 10

# CITE-seq orthogonal ground truth
python src/citeseq.py
# then: python src/multiseed.py citeseq_pbmc --nseed 6

# Integration benchmark
python src/benchmark_integration.py pancreas 200

# Scale-nested conformal
python src/scale_nested_conformal.py pancreas pbmc3k paul15 --splits 25

# Advanced analyses (OT, spectral, curvature)
python src/compute_advanced.py
```

## Dependencies

Python 3.13, PyTorch (CPU), scanpy 1.12, anndata, numpy, scipy, scikit-learn, matplotlib,
ripser, umap-learn, python-igraph + leidenalg, POT (optimal transport), scikit-misc,
harmonypy, scvi-tools.

No GPU required. All 22 modules run on a laptop CPU.

## Repository Layout

```
src/                22 Python modules
paper/              LaTeX manuscript (main.tex, refs.bib, main.pdf)
figures/            15 figures × 3 formats (PNG, SVG, PDF)
website/            Self-contained showcase site (index.html)
docs/               Handoff, synthesis, theory, lit review, run plans
results/<dataset>/  Cached outputs (npz, json)
learning/           This guide
```
