# IGMC — Information-Geometric Multiscale Clustering of single-cell RNA-seq

**Working title:** *The Right Ruler: Information-Geometric, Multiscale, Topologically-Aware
Clustering of Single-Cell Transcriptomes with Conformal Confidence*

Target venue framing: Nature Methods / NeurIPS. Precollege deliverable: website + figures + manuscript.

## The one-sentence thesis
Standard scRNA-seq clustering measures cell–cell dissimilarity with Euclidean distance on a
generic latent embedding, which is the *wrong ruler*: it ignores that count noise is
mean-dependent (negative binomial). We replace it with the **Fisher–Rao metric of the NB
likelihood pulled back through a generative decoder**, cluster the resulting statistical
manifold with **Markov-stability multiscale community detection** (no resolution knob),
use **persistent homology** to decide whether each population is a discrete type or a point on
a continuum, and attach a **conformal prediction set** (calibrated confidence) to every cell.

## The four pillars (with the math)

### Pillar 1 — NB-VAE + Fisher–Rao pullback metric (the core)
- Custom Negative-Binomial VAE in PyTorch. Encoder q(z|x); decoder z → ρ(z) (softmax over
  genes) → μ_g = ℓ_i · ρ_g(z), gene-specific inverse-dispersion θ_g. NB(μ, θ):
  Var = μ + μ²/θ.
- **Fisher information of NB wrt its mean μ** (derived): I(μ) = θ / (μ(μ+θ)) = 1/Var(x).
  So the "correct ruler" downweights genes/regions by their count variance — exactly the
  heteroscedastic correction we want.
- **Pullback metric on latent space** z ∈ R^d:
      G(z) = J_μ(z)ᵀ diag(I(μ_g(z))) J_μ(z),  J_μ = ∂μ/∂z  (genes × d).
  This is a Riemannian metric measuring *statistical distinguishability* of cells.
- Local squared Fisher–Rao distance between neighbours: δ²(z,z') ≈ (z−z')ᵀ Ḡ (z−z')
  with Ḡ = (G(z)+G(z'))/2 (symmetrised midpoint). Build kNN graph on this.
- **Efficiency** (answers checkpoint blocker): d is small (~10–20). Compute G(z) with
  `torch.func.jacrev` + `vmap` in minibatches → per-cell G is only d×d. No full Jacobian
  materialised. Optionally a graph-geodesic (Dijkstra on local-metric edges) for global dist.

### Pillar 2 — Markov-stability multiscale clustering (no resolution knob)
- Random-walk Laplacian L = I − D⁻¹A on the Fisher–Rao kNN graph. Transition P(t)=exp(−tL).
- Stationary π_i = d_i/2m. Autocovariance / clustered stability at Markov time t:
      r(t;H) = trace( Hᵀ [ Π P(t) − ππᵀ ] H ),  Π=diag(π).
  Maximise over partitions H via Leiden on the (symmetrised) flow matrix at each t.
- Sweep t over ~50 log-spaced scales. **Robust scales = plateaus**: (i) long stretches where
  the optimal number of communities is constant, and (ii) low **variation of information**
  VI(t) across optimisation seeds and VI(t,t') block structure. Plateaus = the "real" clusterings.
- Exact P(t) via eigdecomp of L_sym for n ≲ 8k; stratified subsample or sparse Chebyshev for
  larger (pancreas). Classic 3-panel Markov-stability figure.

### Pillar 3 — Persistent homology: discrete type vs continuous trajectory
- Per population, Vietoris–Rips persistence (ripser) in the Fisher–Rao metric.
- Discrete compact type: H0 bars die fast at one scale, negligible H1. Continuous trajectory:
  long H0 filament persistence and/or H1 loops (branches/cycles).
- **Topological discreteness score** τ combining (a) H0 death-gap / multimodality and
  (b) normalised max-H1 persistence. Validate: paul15 (continuous, low τ) vs PBMC discrete
  types (high τ). Also a global "manifold shape" diagnostic.

### Pillar 4 — Conformal prediction (calibrated confidence per cell)
- Split / Mondrian (class-conditional) conformal on the latent z. Classifier p̂(y|x) (kNN or
  softmax). APS nonconformity (Romano et al.). Per-class threshold q̂ gives sets with
  coverage ≥ 1−α. Output prediction set per cell; |set|>1 = ambiguous, |set|=0 = novel/OOD.
- Disease story: aberrant/intermediate cells get larger or empty sets → method *admits*
  uncertainty. Coverage calibration curve + set-size UMAP.

## Datasets (all open, integer counts confirmed)
1. **PBMC3k** (2700×32738, 10x UMIs) — discrete immune types, fast dev, clean NB demo.
2. **paul15** (2730×3451, mouse myeloid) — CONTINUOUS differentiation → topology showcase.
3. **Pancreas scIB** (16382×19093, 14 types, 9 techs) — islet biology, rare types
   (epsilon=32, mast=42, macrophage=79), batch integration, ground-truth labels.
4. **Disease islet (T2D/MODY)** — compact real dataset TBD (lit review sourcing) for the
   beta-cell aberrant-state stress test + conformal ambiguity story.

## Baselines
Euclidean-Leiden (standard), PCA+Leiden, k-means, raw-latent Leiden (ablation: our graph
minus Fisher–Rao). Metrics: ARI, NMI, silhouette, **rare-population recall/F1**, calibration.

## Deliverables
- `src/` modules (data, nbvae, geometry, graph, markov_stability, topology, conformal, evaluate, baselines).
- Results tables + 5–6 multi-panel figures (each 4 sub-panels, publication grade).
- Full Nature-style manuscript (LaTeX) + references.
- Showcase website. + plain-language 1-pager for teammates.

## Build order
data → nbvae → geometry → graph → markov_stability → topology → conformal → evaluate →
figures → manuscript → website → 1-pager. Test each on PBMC3k first, then scale.
