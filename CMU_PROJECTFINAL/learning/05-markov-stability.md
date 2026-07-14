# Step 5 — Removing the Resolution Knob

## The Problem with Resolution Parameters

Standard Leiden clustering requires picking a resolution. 0.5 gives 8 clusters, 1.0 gives 14,
2.0 gives 30. Different choices give materially different conclusions. And there's no
principled way to pick — you just try values until the result "looks right."

But cell identity is genuinely **hierarchical**: types contain subtypes, which contain states.
A single resolution flattens this hierarchy into one answer.

## The Insight: Let a Random Walker Decide

Imagine a random walker on the Fisher-Rao kNN graph. It jumps from cell to cell proportional
to edge weight (how statistically similar the cells are).

- **Short walk** (small Markov time `t`): the walker rarely leaves its immediate neighborhood.
  Communities it identifies are fine-grained — individual subtypes.

- **Long walk** (large `t`): the walker explores broadly. Communities it identifies are
  coarse — major cell types.

- **Intermediate times**: communities at intermediate granularity.

The key question: at which times are the partitions *stable* and *real*?

## Markov Stability

For a partition **H** (an assignment of cells to communities), the **stability at time t**
measures how much of the walk's autocorrelation is captured by H:

```
r(t, H) = trace( Hᵀ · [Π P(t) − ππᵀ] · H )
```

where:
- **P(t) = exp(−t · L_rw)** is the continuous-time transition matrix
- **L_rw = I − D⁻¹A** is the random-walk Laplacian
- **π** is the stationary distribution (how often the walker visits each cell)
- **Π = diag(π)**

At each time `t`, we optimize H to maximize `r(t, H)`.

## The Beautiful Identity

Because the walk is reversible on a symmetric graph, `Π P(t)` is symmetric and non-negative,
and the graph with adjacency `2m · Π P(t)` has the same degrees as the original graph A.
Therefore:

> **Maximizing r(t, H) is exactly RB-modularity maximization at resolution 1 on the flow graph
> A'(t) = 2m · Π P(t).**

This means we can use the Leiden algorithm *directly* — we build A'(t), run Leiden, and get
the optimal partition. No new optimizer needed.

## Building A'(t) Efficiently

We build it from the eigendecomposition of the symmetric normalized Laplacian:

```
L_sym = I − D^(−½) A D^(−½) = U · Λ · Uᵀ

A'(t) = W · diag(exp(−t Λ)) · Wᵀ,    where W = D^½ U
```

Because high eigenvalues decay as `exp(−t λ)`, we can truncate to keep only the smallest
eigenvalues and sparsify the result to the top edges per row. Total cost is `O(n · n_eig)` per
time point, not `O(n³)`.

## Finding Robust Scales

We sweep `t` over ~30 log-spaced values. At each t we:

1. Build the flow graph A'(t)
2. Run Leiden with 5 independent random seeds
3. Record the number of communities and the **variation of information (VI)** across seeds

**Robust scales** are Markov times where:
- The number of communities forms a **plateau** (stays constant over a range of t)
- The across-seed **VI dips** (different optimizers agree)

These plateaus are the "real" clusterings — they emerge from the data, not from a hand-turned
knob.

## What We Get

- **No resolution parameter**: the number of communities is discovered, not set
- **A full hierarchy**: the cluster tree from fine to coarse is visualized in an alluvial
  diagram
- **Accuracy**: on the pancreas atlas, the most robust plateau hits ARI 0.91 against
  marker-gene ground truth, compared to 0.47 for PCA+Leiden

On large datasets, the Markov scan runs on a **stratified subsample** (~3,500 cells) that
preserves every cell type (even rare ones with as few as 30 cells).

Next: [Step 6 — Topology: Discrete vs. Continuous](06-topology.md)
