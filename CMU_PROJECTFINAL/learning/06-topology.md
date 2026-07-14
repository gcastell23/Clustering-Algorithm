# Step 6 — Telling Types from Transitions

## The Question

Some groups of cells are **discrete, well-separated types** — like T-cells vs. B-cells. They're
compact blobs with a clear gap between them.

Others are stops along a **continuous trajectory** — like stem cells differentiating through
progenitor stages into mature cells. They bleed smoothly into each other with no gap.

Standard clustering forces both into the same mold: a hard partition with no "this might be an
intermediate" signal. Can we tell the difference?

## The Idea: Topology Knows Shape

**Persistent homology** is a mathematical tool that tracks how the shape of data changes as
you vary a scale parameter. Imagine slowly inflating balls around each data point:

- At a tiny radius, each point is its own component
- As the radius grows, nearby points merge into connected components (**H₀ bars**)
- At larger radii, loops and voids may form and then fill in (**H₁ bars**, **H₂ bars**)

The **persistence diagram** is the fingerprint of the data's shape.

## What We Look For

### In a Discrete Population

```
H₀ diagram (discrete types)
death │
      │         ●          ← all cells merge at one scale
      │       ●              (compact blob)
      │     ●
      │   ●
      │ ●●●●●●●●●●●          ← many short bars = rapid merging within type
      └─────────────────→ birth
```

- Most H₀ bars are very short (cells quickly connect within their type)
- The last few bars are *much* longer (the gaps between types)
- There is a clear **death gap** — a big jump where the last components merge
- H₁ is negligible (no persistent loops/cycles)

### In a Continuous Trajectory

```
H₀ diagram (continuous)
death │
      │                 ●
      │              ●
      │           ●           ← bars spread evenly
      │        ●                (no sharp gap)
      │     ●
      │  ●
      └─────────────────→ birth

H₁ diagram (continuous)
death │
      │                     ●  ← significant loop
      │                  ●       (differentiation cycle)
      │               ●
      └─────────────────→ birth
```

- H₀ bars are spread more smoothly (no single merge scale)
- There is no sharp death gap
- H₁ may show persistent cycles (branching/looping topology)

## The Transitionality Index

Beyond global shape, we measure **per-cell transitionality**. For each cell:

1. Find its `k` Fisher-Rao nearest neighbors
2. Check how many belong to the same population

```
purity(cell) = fraction of neighbors with the same label
transitional = purity < threshold
```

A high-purity cell sits deep inside a type's core. A low-purity cell sits on the boundary —
its Fisher-Rao neighborhood spans multiple populations. These are the **transitional cells**,
the bridges that knit types together into a continuum.

## The Discreteness Index

Putting it together:

```
discreteness_index = 1 − transitional_fraction
continuity_index   = transitional_fraction
```

Results on real data:

| Dataset | Structure | Discreteness | Continuity | Ambiguity |
|---------|-----------|-------------|------------|-----------|
| PBMC 3k | Discrete immune | 0.84 | 0.16 | ~1% |
| Pancreas | Mixed discrete | 0.95 | 0.05 | ~1% |
| Paul15 | Continuous myeloid | 0.63 | 0.37 | 27% |

The numbers agree with biology: immune cells are sharp types, islet cells are discrete, and
the myeloid differentiation track is a genuine continuum. The conformal prediction
ambiguity (how often a cell sits between types) moves in lockstep with the topological
continuity index — two independent approaches give the same answer because they read from the
same geometry.

Next: [Step 7 — Conformal Prediction](07-conformal-prediction.md)
