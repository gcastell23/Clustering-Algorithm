# Run plan — WS1 (rigor) + WS2 (theory)

Written before heavy compute so intent is visible. Executed autonomously; honesty guardrails from
`HANDOFF.md §8` apply. Caches from the prior session are already on disk (seed 0), so we build on them.

## Environment reality check (done)
- Python 3.13, torch 2.8 CPU, scanpy 1.12, POT, ripser — all present.
- Baselines installable **without a C++ compiler**: **Harmony** (harmonypy 0.2.0 ✓), **ComBat** (scanpy ✓),
  **scVI/scANVI** (scvi-tools, installing). **Scanorama is NOT available** (its `annoy` dep needs MSVC,
  absent on this machine) — will be reported as an honest omission, not hidden.
- CITE-seq ground truth: 10x `5k_pbmc_protein_v3` filtered_feature_bc_matrix.h5 (17 MB) and
  `pbmc_10k_protein_v3` (22 MB) both return HTTP 200 with no login. Downloads stay small.

## Priority order (why this order)
Highest reviewer value first; independent streams run concurrently (compute in background, theory inline).

### M1 — Multi-seed confidence intervals  [WS1, closes the #1 rigor gap]
Retrain the NB-VAE for seeds 0–9 per dataset; recompute ARI / rare-F1 / conformal coverage for
PCA+Leiden, NB-VAE+Euclid, IGMC-FR, (light) IGMC-Markov. Report mean ± 95% CI (bootstrap + t).
**Claim to test:** the count-aware-latent advantage over PCA is real, not a seed fluke (CIs separated).
Long pole = pancreas (16k). Runs in background from the start.

### M2 — Scale-nested conformal prediction  [WS2 flagship theory; CPU-light, uses existing caches]
Formalize + prove simultaneous coverage across the Markov-stability partition hierarchy:
- Thm 1: each scale is a valid Mondrian conformal predictor (foundation).
- Thm 2: allocation Σ α_ℓ = α ⇒ **simultaneous (family-wise) coverage** ≥ 1−α over all scales (union bound).
- Thm 3: a nesting-aware construction yields **coherent** sets (fine-scale calls never contradict their
  coarse-scale supertype) while preserving the guarantee — the genuinely novel structural piece.
- Empirics on real hierarchies (pancreas/pbmc/paul15): per-scale coverage, simultaneous coverage vs
  guarantee, and a per-cell "adaptive resolvable scale" that should track the topology (continuum cells
  resolve coarser). Turn into one figure. Adversarially verify the proofs with independent agents.

### M3 — CITE-seq orthogonal ground truth  [WS1, kills the circularity objection]
Download 10x PBMC CITE-seq; split RNA (Gene Expression) vs ADT (Antibody Capture). Build cell-type
labels **from protein only** (CLR-normalized ADT → cluster/gate on CD3/CD4/CD8/CD19/CD14/CD16/CD56…).
Run IGMC + baselines on RNA counts; evaluate against protein labels — labels now independent of any RNA
pipeline. Report honestly whichever way it comes out.

### M4 — Multi-method integration benchmark  [WS1, "the table reviewers demand"]
Own implementation of the scIB metric battery (bio-conservation: NMI, ARI, ASW_type, isolated-label F1;
batch-removal: ASW_batch, graph connectivity, kBET-lite/iLISI) — implemented from primitives for
transparency. Embeddings compared on pancreas (9 techs): PCA, ComBat, Harmony, scVI, scANVI (if installed),
and our count-aware latent / Fisher–Rao. Honest note on Scanorama omission.

### M5 — Figures (each through build→render→inspect→critique→fix)
New: multi-seed CI figure, scale-nested conformal figure, CITE-seq orthogonal-truth figure, benchmark
figure/table. Okabe–Ito, SVG+PNG+PDF, mirrored to website gallery.

### M6 — Write-up + housekeeping
Fold theorems + WS1 results into `paper/main.tex` (new theory + rigor sections, updated macros), update
`README.md`, `HANDOFF.md`, memory. Commit after every milestone.

## Guardrails (unchanged)
Real committed data only; invent/inflate/cherry-pick nothing; drop what fails (as curvature was dropped);
every figure panel from real output; first-name authors; Okabe–Ito + SVG/PNG.
