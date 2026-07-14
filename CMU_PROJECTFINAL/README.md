# IGMC — Information-Geometric Multiscale Clustering of single-cell RNA-seq

*The right ruler: information-geometric, multiscale and topologically-aware clustering of
single-cell transcriptomes with conformal confidence.*

CMU Pre-College Computational Biology final project (Topic 15: single-cell RNA-seq clustering).
Team: Gabriella · Megan · Emma · Rajarshi.

## The idea in one line
A negative-binomial VAE already carries the correct geometry — the **Fisher–Rao pullback metric**
`M(z) = Jᶠ(z)ᵀ I_NB(f(z)) Jᶠ(z)` — and that *single* object does four jobs at once: it is the
inter-cell distance, it is scanned across scales by Markov stability (no resolution knob), it is the
filtration probed by persistent homology (discrete type vs. continuum), and it supplies the
non-conformity score for conformal prediction (calibrated per-cell confidence).

## Headline results (all reproducible from `src/`)
| Finding | Number |
|---|---|
| Rare islet types recovered by count-aware latent vs. PCA+Leiden (pancreas) | **F1 0.48 vs 0.002** (≈200×) |
| Markov-stability clustering, **no resolution knob**, vs PCA+Leiden (pancreas, marker labels) | **ARI 0.906 vs 0.472** |
| Fisher–Rao best on planted ground truth (simulation, no circularity) | **rare-F1 0.45 vs 0.04** for PCA |
| Conformal coverage vs 90% target (pbmc / pancreas / paul15) | 0.907 / 0.900 / 0.922 |
| Discreteness index: continuum (paul15) vs discrete (pbmc / pancreas) | **0.63 vs 0.84 / 0.95** |
| Conformal ambiguity on a continuum vs discrete types | **27% vs 1–0%** |
| Disease covariate shift localizes to β/γ-cells (T2D, coverage drop) | γ −0.12, β −0.055 |
| Optimal-transport β-cell displacement (healthy→T2D) vs within-healthy null | **2.7× null, p<0.004** (250 bootstraps) |
| Diffusion spectral gap → metastable states (pancreas) | 11 (≈ 14 true types); MP: 63 signal PCs |
| **Orthogonal (CITE-seq protein) rare-type recovery**: Fisher–Rao vs same-latent Euclidean | **DC F1 0.88 vs 0.27** (PCA 0.78) |
| **Scale-nested conformal**: simultaneous coverage (target 0.90) / coherence with vs without projection | **0.98 / 100% vs 90–97%** |
| **Multi-seed** count-aware ARI vs PCA (pancreas, 95% CI) | **0.82 [0.78,0.86] vs 0.50 [0.48,0.52]** |
| **scIB benchmark** overall (pancreas): IGMC latent rank / score | **3rd of 6 — 0.716** (behind label-supervised scANVI 0.733, Harmony 0.723; > scVI/ComBat/PCA) |

## PhD-track extensions (WS1 rigor + WS2 theory)
The master's results above are stress-tested and extended (see `docs/HANDOFF.md`, `docs/RUN_PLAN_WS1_WS2.md`):
- **Multi-seed confidence intervals** (`src/multiseed.py`): the NB-VAE is retrained from ≥10 seeds; every
  headline metric is reported as mean ± 95% CI (Fig. 12). The count-aware advantage is not a lucky seed.
- **Orthogonal ground truth** (`src/citeseq.py`): a 10x CITE-seq PBMC dataset labels cell types from
  **surface protein only**, breaking the label-circularity objection. Honest finding: PCA stays strong on
  common types, but the Fisher–Rao geometry best recovers the rare dendritic-cell population — measured
  against a ground truth no RNA pipeline could inflate (Fig. 14).
- **Multi-method benchmark** (`src/scib_metrics.py`, `src/benchmark_integration.py`): a transparent scIB
  battery vs PCA, ComBat, Harmony, scVI, scANVI on the pancreas atlas (Fig. 15).
- **Scale-nested conformal prediction** (`src/scale_nested_conformal.py`, `docs/THEORY_scale_nested_conformal.md`):
  a conformal predictor over the Markov-stability taxonomy that is per-scale valid, simultaneously valid
  across scales, and **provably coherent** (fine calls never contradict their coarse super-type) via an
  isotonic-threshold construction — the WS2 theorem (Fig. 13).

## Interdisciplinary depth (master's-level)
Beyond the four core pillars, the same geometry supports two further graduate-level lenses, both on
real data: **optimal transport** (Wasserstein-2 displacement of disease, Waddington-OT lineage;
Fig. 10) and **spectral graph theory + random-matrix theory** (diffusion eigenspectrum, metastable
states, Marchenko–Pastur signal/noise separation; Fig. 11). We also tested **Ollivier–Ricci curvature**
(optimal transport + Riemannian geometry) but report honestly that it is uninformative here — the
batch-conditional latent separates types so cleanly that <7% of graph edges cross a boundary, leaving
no "bridges" for curvature to detect — so that figure was dropped rather than shown with weak claims.

## Repository layout
```
src/
  data.py            standardized loading + QC + HVG for all datasets (raw counts kept)
  simulate.py        planted-ground-truth NB simulation
  nbvae.py           batch-conditional Negative-Binomial VAE (PyTorch)
  geometry.py        Fisher–Rao pullback metric via forward-mode autodiff (jacfwd+vmap)
  graph.py           Fisher–Rao kNN graph (candidate-and-rerank) + Leiden
  markov_stability.py exact continuous-time Markov stability, VI-plateau scale selection
  topology.py        persistent homology + transitionality (discrete vs continuous)
  conformal.py       Mondrian (class-conditional) split-conformal prediction sets
  evaluate.py        ARI/NMI/silhouette + rare-population F1
  pipeline.py        end-to-end per-dataset run, caches everything for figures
  train_cache.py     trains the VAE + metric once per dataset
  sim_experiment.py  metric ablation on planted simulations
  figures.py         the eight publication figures (each 4 panels)
  style.py           Nature-grade matplotlib design system (CVD-safe palette)
paper/               LaTeX manuscript (main.tex, refs.bib) -> main.pdf (17 pp, 8 figures)
website/             self-contained showcase site (index.html)
figures/             fig1..fig8 (png + pdf)
docs/                literature review (60+ papers), synthesis, ONE_PAGER.md
results/<dataset>/   cached latents, metrics, Markov scans, conformal, topology
```

## Datasets (all open access, integer counts, < 1 GB total)
- **PBMC 3k** (10x, 2 638 cells) — discrete immune types, fast development.
- **Paul15** (mouse myeloid, 2 730) — a *continuous* differentiation trajectory.
- **Pancreas atlas** (16 382 cells, 14 types, 9 technologies) — rare islet types + batch.
- **Segerstølpe** (2 394) — healthy vs type-2-diabetes islets (disease covariate shift).
- **Simulation** — planted rare populations down to 0.4% abundance.

## Reproduce
```bash
python src/data.py                 # build processed datasets
python src/train_cache.py          # train NB-VAE + Fisher-Rao metric per dataset
python src/pipeline.py pbmc3k pancreas paul15 segerstolpe
python src/sim_experiment.py
python src/figures.py all
cd paper && pdflatex main && bibtex main && pdflatex main && pdflatex main
```
Environment: Python 3.13, PyTorch (CPU), scanpy, ripser, python-igraph/leidenalg, umap-learn.

## Honest scope
The robust wins are the count-aware latent (rare-cell rescue), Markov stability (no-knob
accuracy), the discrete/continuous topology test, and calibrated conformal confidence — including a
disease-localization-by-coverage result. The Fisher–Rao metric is the unifying theoretical object
and is best on clean/planted data; on some batch-corrected real atlases its edge over Euclidean is
modest, and the *raw* decoder pullback (without the NB Fisher weighting) can be ill-behaved — the
Fisher information regularizes it. Against the orthogonal CITE-seq protein ground truth, standard PCA
remains competitive on common PBMC types (so its edge there is real, not just circular); the count-aware
Fisher–Rao geometry's win is specifically on the *rare* population. We state these limits plainly in the
manuscript.
```
