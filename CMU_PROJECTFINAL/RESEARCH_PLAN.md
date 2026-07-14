# Research plan & literature synthesis

This documents how we surveyed the field, what we learned, and the resulting plan. Full outputs:
`docs/litreview_raw.txt` (60+ papers, 15 subfields) and `docs/synthesis.md` (novelty + critique).

## 1. How we read the literature (systematic, 60+ papers)
We ran a 15-agent parallel survey, one expert per subfield, each finding real papers (mix of
foundational classics and 2021–2026 high-impact work) and returning structured summaries with
citations, key contributions, quantitative findings, gaps, and methods to adopt. Subfields:

1. scRNA-seq foundations & droplet methods  2. clustering algorithms & benchmarks
3. deep generative models (scVI/DCA)         4. information geometry & Fisher–Rao
5. pullback metrics of generative models     6. count models / normalization
7. Markov stability & multiscale communities 8. diffusion maps / random walks
9. persistent homology / TDA                 10. TDA in single-cell (scTDA)
11. conformal prediction                     12. batch integration / scIB
13. UMAP/tSNE critiques                       14. pancreatic islet biology & MODY/T2D
15. trajectory inference (discrete vs continuous)

A final synthesis agent produced the novelty thesis, adversarial positioning against the closest
competitors, and a Nature-readiness critique. Must-cite list (30 papers) → `paper/refs.bib`.

## 2. What the literature established (load-bearing facts)
- **Correct noise model:** droplet UMIs follow a negative binomial, *not* zero-inflated
  (Svensson 2020; Townes 2019). `Var = μ + μ²/θ`.
- **Standard pipeline is distorting:** log(CPM+1) + PCA injects spurious variance that hurts rare
  populations (Townes 2019; Ahlmann-Eltze & Huber 2023).
- **The metric is derivable:** the multinomial/NB likelihood induces a Fisher–Rao metric; the
  pullback framework `M = Jᵀ I J` already names NB/Poisson decoders (Arvanitidis 2022).
- **Closest competitors:** scVI (learns the NB latent but clusters it with plain L2); GAIA
  (closed-form *simplex* Fisher–Rao in gene space, single-scale, no confidence); PyGenStability
  (Markov stability on a Euclidean graph); scTDA (Mapper on genes); scConform (conformal around a
  fixed softmax). None unify the four, and none pull the NB metric through a generative decoder.
- **Islet biology:** four 2016 atlases fix the cell-type taxonomy & markers; T2D/MODY β-cells show
  subtle state changes and (in lineage-tracing models) dedifferentiation (Talchai 2012; Weng 2023).

## 3. Graduate-curriculum synthesis (the math we imported)
- **Information geometry** (Amari; Rao; Čencov): Fisher metric as the canonical Riemannian metric on
  a statistical manifold; pullback through a smooth map.
- **Spectral graph theory / dynamical systems** (Delvenne–Yaliraki–Barahona): random-walk Laplacian,
  `P(t)=exp(−tL)`, stability as clustered autocovariance; the reversible-chain identity that turns it
  into RB-modularity on a flow graph (our tractable implementation).
- **Algebraic topology** (Carlsson; Cohen-Steiner–Edelsbrunner–Harer): Vietoris–Rips persistence,
  `H₀/H₁`, stability of persistence diagrams.
- **Distribution-free inference** (Vovk; Sadinle; Angelopoulos–Bates; Tibshirani): split/Mondrian
  conformal, class-conditional coverage, covariate shift.
- **Numerical ML** (forward-mode autodiff): `jacfwd`+`vmap` to get the decoder Jacobian without the
  G×G softmax Jacobian — the trick that makes the metric cheap.

## 4. The plan we executed (and why it is Nature-shaped)
Build one pipeline where **distance, scale, shape, and confidence are all read off the same
Fisher–Rao pullback metric**, then stress-test it on discrete (PBMC), continuous (Paul15),
rare+batch (islet atlas), disease (T2D), and planted-truth (simulation) data. Deliverables:
a rigorous ablation (three rulers on a shared latent + planted truth to defeat circularity), a
fair benchmark on marker-gene labels, a no-knob multiscale result, a calibrated-coverage audit, and
a disease application via covariate-shift conformal. See `PLAN.md` for the equations and `README.md`
for the results table.

## 5. Honest self-critique (what a reviewer would push on, and our answer)
- *"Is it just scVI + Leiden?"* — No: we replace L2 with the NB Fisher–Rao pullback, remove the
  resolution knob (Markov stability), add a topological discrete/continuous test, and calibrated
  coverage. Ablations isolate each piece.
- *"Is the metric identifiable?"* — It depends on the decoder; a decoder ensemble (Syrota 2024)
  would marginalize it. Flagged as future work.
- *"Circular labels?"* — Addressed with planted-ground-truth simulations (Fig. 8).
- *"Does Fisher–Rao really help?"* — Best on planted truth and clean data; modest/mixed on some
  batch-corrected atlases. We report this rather than hide it.
