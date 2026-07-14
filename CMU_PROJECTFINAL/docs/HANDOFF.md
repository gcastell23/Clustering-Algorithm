# IGMC — Research Handoff (master's → PhD)

**Read this first. It is the single source of truth for a fresh session.** It describes the project,
exactly what exists and works, what was tried and failed (do not repeat), the engineering landmines,
and a concrete master's→PhD roadmap. Everything below is real and already committed to git.

Repo root: `.../CMU/final-project/igmc/` (a local git repo; `data/` and `results/*.npz` are gitignored).

---
## 0. Project identity
**IGMC — Information-Geometric Multiscale Clustering of single-cell RNA-seq.**
Thesis: a negative-binomial VAE already carries the correct geometry — the **Fisher–Rao pullback
metric** `M(z) = J_f(z)ᵀ I_NB(f(z)) J_f(z)`, with `I_NB(μ)=1/Var(x)` — and that *one* object does four
jobs: (1) inter-cell distance, (2) multiscale scale selection via Markov stability, (3) discrete-vs-
continuous shape via persistent homology, (4) calibrated per-cell confidence via conformal prediction.
Master's-level extensions added: optimal transport of disease + spectral/diffusion geometry.

Team (authors, **first names only** — do not add surnames; user's surname is Mandal):
Gabriella (data/QC/integration), Megan (preprocessing/baselines), Emma (clustering/annotation/eval),
Rajarshi (geometry/Markov/topology/transport/benchmarking).

Deliverables that exist now: `paper/main.pdf` (21 pp, 11 figures), `website/index.html` (figure-first
gallery, SVG+PNG downloads, dark/light), `docs/ONE_PAGER.md`, `README.md`, `PLAN.md`, `RESEARCH_PLAN.md`.

---
## 1. Environment
- **Python 3.13** (`C:\Users\Rajarshi\AppData\Local\Programs\Python\Python313\python.exe`). Full stack
  installed: numpy 2.4, scipy, scikit-learn, pandas 3.0, matplotlib, torch 2.8 (CPU), scanpy 1.12,
  anndata, ripser, umap-learn, python-igraph, leidenalg, POT (optimal transport), scikit-misc.
- Machine: **8 cores, 12.6 GB RAM, CPU only.** LaTeX (MiKTeX) present. No GPU.
- Always `export PYTHONIOENCODING=utf-8` before running (Windows console is cp1252; unicode in prints
  will crash otherwise).
- Long jobs: run in background and poll a logfile (stdout is buffered; the `.npz`/`.png` files landing
  on disk are the reliable progress signal).

---
## 2. Repository map (`src/`, 17 modules)
| file | what it does |
|---|---|
| `data.py` | loads + QC + HVG for pbmc3k, paul15, pancreas (scIB atlas), segerstolpe (T2D). Keeps raw counts in `.layers['counts']`, ground-truth in `.obs['celltype']`, batch in `.obs['batch']`, disease in `.obs['disease']`. |
| `simulate.py` | planted-ground-truth NB simulation (distinct rare types, low counts → PCA loses them). |
| `nbvae.py` | **batch-conditional** Negative-Binomial VAE (PyTorch). Decoder is BatchNorm-free (LayerNorm) so it is `torch.func`-friendly. |
| `geometry.py` | Fisher–Rao pullback metric via **`torch.func.jacfwd` + `vmap`** (forward mode; see landmine #1). `use_fisher` flag toggles Euclidean-pullback ablation. `fr_knn_indices`, midpoint distances. |
| `graph.py` | Fisher–Rao kNN graph (candidate-and-rerank) + Leiden. |
| `markov_stability.py` | exact continuous-time Markov stability via eigendecomp of the normalised Laplacian; RB-modularity flow-graph identity; VI-plateau scale selection. `flow_graph` uses **vectorised top-k** (landmine #2). |
| `topology.py` | Vietoris–Rips persistence (ripser) under Fisher–Rao distance; `transitionality` (neighbourhood purity) = discreteness index. |
| `conformal.py` | Mondrian (class-conditional) split-conformal, LAC score; robust to classes absent from the train split (landmine #3). |
| `evaluate.py` | ARI/NMI/silhouette + **rare-type F1** (Hungarian-matched, rarest tertile). |
| `advanced.py` | Ollivier-Ricci curvature (DROPPED, see §5), OT-of-disease (Wasserstein), spectral geometry + Marchenko–Pastur. |
| `pipeline.py` | end-to-end per dataset; reuses `_cache.npz`; writes `core.npz`, `markov.npz`, `graph_*.npz`, `summary.json`. |
| `train_cache.py` | trains VAE + metric once per dataset → `results/<ds>/_cache.npz`. |
| `compute_advanced.py` | runs+caches curvature / OT / spectral. |
| `sim_experiment.py` | metric ablation on planted simulation → `results/simulation/`. |
| `ablation_markov.py` | Markov on Euclidean vs Fisher–Rao graph → `results/ablation_markov.json`. |
| `figures.py` | the 11 hero figures (`all_figures()`); `_figure_curvature_unused` is dead. |
| `style.py` | **Okabe–Ito** palette + marker redundancy; `savefig()` exports PNG+SVG+PDF. |

**Reproduce from scratch:**
```
python src/data.py
python src/train_cache.py                 # ~25 min (pancreas VAE is the long pole)
python src/pipeline.py pbmc3k pancreas paul15 segerstolpe
python src/sim_experiment.py
python src/compute_advanced.py
python src/figures.py all
cd paper && pdflatex main && bibtex main && pdflatex main && pdflatex main
```

---
## 3. What works (current results — single source of truth)
Datasets: **PBMC3k** (2638, discrete immune), **Paul15** (2730, myeloid *continuum*), **scIB pancreas**
(16382, 14 types, 9 techs), **Segerstølpe** (2394, healthy vs T2D), **simulation**. ~346 MB downloaded.

Headline numbers (seed 0):
- **Markov stability = best no-knob clustering.** Pancreas (marker labels): ARI **0.906** vs PCA+Leiden
  0.472, NB-VAE+Euclid+Leiden 0.764. PBMC 0.842, Paul15 0.578.
- **Count-aware latent rescues rare types.** Pancreas rare-type F1: NB-VAE latent **0.48** vs PCA **0.002**.
  Simulation (planted, no circularity): Fisher–Rao **0.45** vs PCA **0.04**.
- **Conformal coverage** ≈ target 0.90 everywhere (0.907 / 0.900 / 0.922 / 0.918).
- **Topology** discreteness: continuum Paul15 **0.63** vs discrete PBMC 0.84 / pancreas 0.95; conformal
  ambiguity 27% (continuum) vs ~1–0% (discrete) — topology and confidence agree.
- **Optimal transport (Fig 10):** T2D β-cell displacement **2.7× within-healthy null, p<0.004** (250 boot).
- **Spectral (Fig 11):** diffusion spectral gap → 11 metastable states (≈14 true); Marchenko–Pastur → 63
  signal PCs (pancreas).

11 figures (each 4 panels), all Okabe–Ito, PNG+SVG+PDF, in `figures/` and mirrored to `website/figures/`.

---
## 4. Full per-dataset metrics
```
                 ARI     rareF1   (k)
pbmc3k    PCA    0.860   0.822    7    IGMC-Markov 0.842/0.877   FR 0.593/0.896
pancreas  PCA    0.472   0.002   14    IGMC-Markov 0.906/0.110   NBVAE+Euc 0.764/0.476
paul15    PCA    0.358   0.168    8    IGMC-Markov 0.578/0.155   NBVAE+Euc 0.324/0.668
seger     PCA    0.536   0.000   13    IGMC-Markov 0.520/0.003   FR 0.504/0.129
```

---
## 5. What did NOT work / honest limitations (DO NOT re-sell these as wins)
1. **PBMC3k labels are circular** — they come from a PCA+Louvain pipeline, so PCA "wins" there. Never use
   PBMC as the headline accuracy benchmark. Use pancreas (marker labels) + simulation.
2. **The Fisher–Rao metric's edge over Euclidean is modest and setting-dependent.** It clearly helps on
   clean/planted data and on PBMC; on the batch-corrected pancreas, plain Euclidean-on-latent + Leiden is
   competitive or better on rare-F1. The robust wins are the *count-aware latent* + *Markov stability*.
3. **Ollivier-Ricci curvature was DROPPED.** The batch-conditional latent separates types so cleanly that
   <7% of graph edges cross a type boundary → no "bridges" → curvature is ~constant positive and does not
   correlate with transitionality (r≈−0.02). Code kept in `advanced.py` but unused. Would only work on a
   denser/less-separated graph (e.g., a genuine trajectory before VAE separation).
4. **No T2D dedifferentiation.** The Segerstølpe β-cell identity is essentially unchanged (classifier
   p(β) 0.94→0.92, ns). The honest disease result is a *covariate-shift coverage drop* localized to β/γ
   cells + the OT displacement — NOT dedifferentiation. Do not claim dedifferentiation without new data.
5. **Raw decoder pullback JᵀJ is numerically ill-behaved** on pancreas (fragments to k=109). The NB Fisher
   weighting regularizes it — this is a feature to explain, not a bug to hide.
6. ~~**Single seed.**~~ **ADDRESSED (WS1).** `src/multiseed.py` retrains the NB-VAE from multiple seeds and
   reports mean ± 95% CI (Fig. 12, `results/multiseed_summary.json`). Effect sizes are large; CIs separate.
7. **Small scale.** Largest real dataset is 16k cells; Markov uses a stratified ~3.5k subsample on pancreas.
   (WS3, still open.)
8. ~~**No orthogonal ground truth.**~~ **ADDRESSED (WS1).** `src/citeseq.py` uses 10x CITE-seq protein as an
   orthogonal label source (Fig. 14). **Honest finding:** against protein truth, PCA stays competitive on
   common PBMC types (its edge is real, not purely circular); the Fisher–Rao geometry's win is specifically
   the rare dendritic-cell type (DC F1 0.88 vs 0.27 for the same latent under a Euclidean ruler). Do not
   overclaim "circularity broken / IGMC wins" — report the nuance.

### WS1/WS2 progress log (this session)
- **Conformal quantile bug FIXED** (`src/conformal.py::conformal_quantile`): the old
  `np.quantile(...,lvl,"higher")` returned `max(score)` when the level saturated, under-covering rare
  classes. Now returns the correct R-th order statistic, or 1.0 (always-include) when the group is too
  small. Affects rare-class coverage in Fig. 5 numbers if the pipeline is re-run.
- **Scale-nested conformal (WS2 theorem) DONE** (`src/scale_nested_conformal.py`,
  `docs/THEORY_scale_nested_conformal.md`, Fig. 13). Marginal split-conformal per taxonomic level;
  Props 1–2 standard (Vovk; union bound), **Thm 3 (isotonic-threshold coherence preserving validity) is
  the novel piece**. Validated held-out over 25 splits. Positioning verified against scConform
  (Corbetta 2024, NOT Bioinformatics 2025 — that's Lopez-De-Castro), Ding 2023, Baheri 2025, van der
  Laan 2024, Gupta 2022 (all in `docs/verified_citations.bib`; each existence-checked).
- **Multi-method benchmark** (`src/scib_metrics.py`, `src/benchmark_integration.py`, Fig. 15): scIB
  battery vs PCA/ComBat/Harmony/scVI/scANVI on pancreas. Scanorama omitted (annoy needs a C++ compiler,
  absent here — reported honestly).
- **Engineering:** max 2 concurrent torch jobs on this 8-core box (3+ thrashes to a standstill); pin
  OMP/MKL/OPENBLAS + torch threads to 4. First fit_nbvae per process pays ~120s one-time init.

---
## 6. Engineering landmines (will cost hours if rediscovered)
1. **Metric Jacobian: use `jacfwd`, never `jacrev`.** Reverse mode materialises the G×G softmax Jacobian
   and OOMs (4 GB). Forward mode (d≈10 ≪ G≈2000) is cheap.
2. **`markov_stability.flow_graph` must use vectorised top-k** (argpartition over blocks). A per-row Python
   loop is O(n³)-ish in practice and hangs (killed a 5000-cell run). Pancreas Markov subsamples to ~3.5k.
3. **Conformal must tolerate classes in calibration but absent from the train split** (`conformal.py`
   already handles this; needed for healthy-only calibration).
4. **matplotlib + Arial cannot render unicode sub/superscripts** (`λₖ`, `W₂²`, `Jᵀ`) — they show as boxes.
   Use mathtext (`$\lambda_k$`, `$W_2^2$`, `$J^\top$`).
5. **scanpy HVG seurat_v3 needs `scikit-misc`** (installed).
6. **Never `git add data/`** (302 MB h5ad) — it times out. `.gitignore` already excludes `data/` and
   `results/*.npz`.
7. Result macros in `paper/main.tex` are hand-set from `summary.json`; re-running with new seeds shifts
   numbers — update the macros.

---
## 7. The master's → PhD roadmap (the actual forward work)
A PhD / top-venue (Nature Methods, NeurIPS, ICML) version needs **theory, scale, rigorous benchmarking,
and an orthogonally-validated biological discovery** — not more pipeline. Prioritised workstreams:

### WS1 — Break the circularity & benchmark rigorously (do first; highest reviewer value)
- Get **CITE-seq** data (e.g., 10x PBMC CITE-seq, or Hao 2021 multimodal) — protein levels are an
  *orthogonal* ground truth for cell type, killing the "labels are circular" objection.
- **Multi-seed** everything (≥10 VAE seeds) → report mean ± 95% CI; add seed as a random effect.
- Full **scIB 14-metric battery** vs Harmony, scVI, scANVI, Scanorama, Seurat — bio-conservation +
  batch-removal, on ≥3 real atlases. This is the table reviewers demand.
- Systematic **splatter/dyngen** simulation sweeps: rare-recall vs abundance × noise × dimension × #types;
  likelihood misspecification (NB vs ZINB vs Poisson emission).

### WS2 — Theory (a PhD needs theorems, not just methods)
- **Scale-nested conformal prediction:** construct + prove simultaneous coverage across the Markov-stability
  hierarchy of partitions (a genuinely novel contribution — conformal on a *nested family* of Mondrian
  groups). This is the single most publishable theoretical piece.
- **Identifiable geometry:** decoder-ensemble expected metric (Syrota 2024); prove/measure that
  marginalising K decoders reduces geodesic/cluster variance; report ARI/VI vs K.
- **Characterise the NB Fisher–Rao pullback:** relate it to the closed-form multinomial Fisher–Rao (√p
  sphere / Hellinger, `d=2·arccos Σ√(p_i q_i)`); derive when they coincide; the dual α-geometry of the NB
  exponential family; explain analytically why the Fisher weighting regularises JᵀJ.
- **Topological significance:** Fasy-style bootstrap confidence sets for the persistence diagram → turn the
  discreteness index into a hypothesis test with a calibrated p-value per population.

### WS3 — Scale to atlas size (top venues expect ≥10⁵–10⁶ cells)
- Sparse action of `exp(-tL)` (Chebyshev/Lanczos) instead of dense eigendecomp; landmark/Nyström graph
  geodesics; minibatch metric computation; **GPU** (batched `jacfwd` on CUDA — needs a GPU machine).
- Demonstrate on a real large atlas: **HPAP** islet (open via CZ cellxgene), Tabula Sapiens, or HLCA lung.
- Certified approximation-error report + wall-clock/memory scaling curve.

### WS4 — Biology + orthogonal validation (the discovery that lifts it above "a method")
- Obtain a **real HNF1A/MODY or T1D islet** dataset (HPAP / PANC-DB / cellxgene; the original checkpoint
  target was HNF1A-MODY). Segerstølpe alone is too subtle.
- A **conformally-certified aberrant cell state**, characterised by marker programs (AUCell / decoupleR),
  validated in an **independent cohort** and, ideally, spatial or protein data.

### WS5 — Software & reproducibility
- Refactor `src/` into a pip-installable package (`igmc/`), unit tests, CI, API docs, a tutorial notebook,
  released trained models + Zenodo DOI. Currently research scripts, not a library.

### WS6 — Positioning & writing
- Sharpen the delta vs **GAIA** (closed-form simplex FR, gene space, single-scale), **scVI** (L2 on the
  latent it defines), **PyGenStability** (Euclidean graph), **scTDA** (Mapper on genes), **scConform**
  (conformal on a fixed softmax). The core contribution to defend is the *unification* + the NB-decoder
  pullback + scale-nested conformal.
- Rewrite `paper/main.tex` into a full Nature Methods / NeurIPS submission with the theory + large-scale +
  validated-biology results.

**Suggested order:** WS1 (rigor) → WS2 (theory) in parallel with WS3 (scale) → WS4 (biology) → WS5/WS6.
WS1 + one WS2 theorem + WS4 discovery is the minimal PhD-defensible / top-venue package.

**STATUS (this session):** WS1 substantially done — multi-seed CIs (Fig. 12), CITE-seq orthogonal
ground truth (Fig. 14), scIB multi-method benchmark (Fig. 15). WS2 flagship done — scale-nested
conformal with a proved coherence theorem (Fig. 13, `docs/THEORY_scale_nested_conformal.md`).
**Still open:** WS3 (atlas-scale ≥10⁵ cells), WS4 (an orthogonally-validated biological discovery —
the highest-value remaining item), WS5 (pip package + CI), full scIB dyngen/splatter sweeps.

---
## 8. Non-negotiable guardrails (these made the current work trustworthy — keep them)
- **Honesty is absolute.** Real, committed data only; invent/inflate/cherry-pick nothing. If an analysis
  doesn't work (like curvature), report it and drop it — do not dress up a null result. Every figure panel
  must come from real output.
- **Build → render → inspect the actual image → critique as designer+engineer+scientist → fix → repeat**
  until it clears the publication bar. Do a fresh-eyes cold review before finalising each figure.
- **Okabe–Ito palette, shape/position redundancy, minimal text, information in the visuals.** Export SVG+PNG.
- **Commit to git after each figure / milestone** so progress is saved.
- Keep this file, `README.md`, and the memory (`~/.claude/projects/<proj>/memory/`) updated as you go.
