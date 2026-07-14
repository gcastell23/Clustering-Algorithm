# Step 10 — PhD-Level Extensions

## WS1: Rigor (Making Results Defensible)

### Multi-Seed Retraining

The original results used one VAE seed. A fair critic asks: "was it luck?"

`src/multiseed.py` retrains the NB-VAE from scratch across ≥6 seeds and recomputes every
metric. Results are reported as **mean ± 95% bootstrap CI**.

The outcome: on the pancreas atlas, the count-aware latent's ARI is 0.82 [0.78, 0.86] vs.
PCA's 0.50 [0.48, 0.52]. Intervals don't overlap. The wins are real.

### CITE-seq Orthodox Ground Truth

A standing objection to any scRNA-seq benchmark: the "ground truth" labels come from a
clustering pipeline on the same RNA data. Circular.

`src/citeseq.py` breaks the circle. On a 10x 5k-PBMC CITE-seq dataset:

1. Cell-type labels are defined from **surface protein only** (CLR-normalized antibody
   counts, Leiden, canonical marker gating)
2. RNA counts are then clustered independent of protein
3. Results are scored against protein-defined labels

Honest finding: PCA stays competitive on common types (ARI 0.75), so its edge is real, not
just circular. But **Fisher-Rao wins the rare dendritic-cell type** (F1 0.88 vs. 0.27 for the
same latent with a Euclidean ruler).

### scIB Multi-Method Benchmark

`src/scib_metrics.py` implements the full scIB battery from primitives (NMI, ARI, ASW,
isolated-label F1, cLISI, iLISI, graph connectivity). `src/benchmark_integration.py` compares
the NB-VAE latent against PCA, ComBat, Harmony, scVI, and scANVI at 10 latent dimensions.

Result on pancreas: IGMC overall 0.716 — 3rd of 6, behind label-supervised scANVI (0.733) and
Harmony (0.723), ahead of scVI, ComBat, and PCA. IGMC achieves the 2nd-best batch mixing.

## WS2: Theory — Scale-Nested Conformal Prediction

This is the novel theoretical contribution. The idea:

### Step 1: Build a Taxonomy

The Markov-stability scan produces partitions at many scales. Which cell types walk together
across Markov time? Average their co-membership over all time points, then do hierarchical
clustering on the resulting distance:

```
D(type_A, type_B) = 1 − mean_t [ P(A and B in same community at time t) ]
```

Cut the dendrogram at geometrically-spaced numbers of clusters. This gives a nested taxonomy:
fine types at level 0, coarser super-types at level 1, coarser still at level 2, etc.

On the pancreas, this produces biologically sensible super-groups: endocrine (alpha, beta,
gamma, delta, epsilon), exocrine (acinar, ductal), immune, and stromal cells.

### Step 2: Per-Level Conformal Predictors

Build one conformal predictor per taxonomic level. Level 0 predicts fine cell types. Level 1
predicts super-types. Each is marginally valid at its own level.

### Step 3: Simultaneous Validity

Allocate the total miscoverage budget α across levels so Σ α_ℓ = α. By the union bound:

```
P( any level misclassifies ) ≤ Σ P(level ℓ misclassifies) ≤ Σ α_ℓ = α
```

This gives a **family-wise guarantee**: the whole hierarchy of predictions is simultaneously
valid.

### Step 4: The Coherence Theorem (New)

Without intervention, a cell might be confidently "beta" at level 0 but "exocrine" at level 1
— fine and coarse calls contradict. This is incoherent.

The fix: **isotonic up-the-tree projection of thresholds**. Starting from the finest level:

```
q̃_0 = q_0
q̃_1 = max(q_1, q̃_0)    ← parent threshold ≥ child threshold
q̃_2 = max(q_2, q̃_1)
...
```

This only *increases* thresholds, which only *enlarges* sets, which only *improves* coverage
— so per-level and simultaneous validity are preserved. But now, with the nesting of
probabilities (super-class probability ≥ child probability) and the nesting of thresholds:

```
p_super(h|x) ≥ p_fine(g|x)   and   q̃_super ≥ q̃_fine
```

If the fine set includes g, then:

```
p_super(parent(g)|x) ≥ p_fine(g|x) ≥ 1 − q̃_fine ≥ 1 − q̃_super
```

So parent(g) is in the coarse set. Coherence holds. This is proven in three propositions and
one theorem in the paper's appendix.

### Empirical Validation

On held-out cells over 25 splits:
- Simultaneous coverage: 0.98 (target 0.90) with the union-bound allocation; 0.90 exactly
  when each level runs at full α
- Coherence rate: 100% with isotonic projection; 90-97% without
- A rare pancreatic type covered at only 0.39 at the finest scale is rescued to 0.99 once
  pooled into a well-populated super-type at coarser scales

Next: [Step 11 — Code and Reproduce](11-code-and-reproduce.md)
