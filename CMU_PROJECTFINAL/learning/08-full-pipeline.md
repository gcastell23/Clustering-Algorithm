# Step 8 — How Everything Fits Together

## The Central Idea

One piece of math — the Fisher-Rao pullback metric — does four jobs:

```
                    ┌──────────────┐
                    │  Raw Counts  │
                    └──────┬───────┘
                           │
                    ┌──────▼───────┐
                    │   NB-VAE    │   train once
                    │ z ← Encoder │   z: 10-dimensional latent code
                    │ z → Decoder │   decoder: z → gene proportions ρ(z)
                    └──────┬──────┘
                           │
                    ┌──────▼───────┐
                    │ Fisher-Rao   │   compute once
                    │ pullback     │   M(z) = J_ρ(z)ᵀ·(1/Var)·J_ρ(z)
                    │ metric M(z) │   per cell: 10×10 matrix
                    └──────┬──────┘
                           │
           ┌───────────────┼───────────────────┐
           │               │                   │
    ┌──────▼──────┐ ┌──────▼──────┐  ┌─────────▼──────────┐
    │  Distance    │ │   Scale     │  │      Shape         │
    │              │ │             │  │                    │
    │ FR kNN graph │ │ exp(−tL)    │  │  Vietoris-Rips     │
    │ candidate &  │ │ flow graph  │  │  persistence       │
    │ rerank       │ │ VI plateaus │  │  under FR distance │
    │              │ │             │  │                    │
    │ Which cells  │ │ How many    │  │  Discrete blob     │
    │ are near?    │ │ clusters?   │  │  or continuum?     │
    └──────┬──────┘ └──────┬──────┘  └─────────┬──────────┘
           │               │                   │
           │        ┌──────▼──────┐            │
           │        │  Hierarchy  │◄───────────┘
           │        │  (taxonomy) │  types that walk together
           │        └──────┬──────┘  across Markov time are kin
           │               │
    ┌──────▼───────────────▼──────┐
    │     Conformal Prediction    │
    │                             │
    │  Per cell: prediction set   │
    │  |set|=1 → confident        │
    │  |set|>1 → ambiguous        │
    │  |set|=0 → novel            │
    │  90% coverage guarantee     │
    └─────────────────────────────┘
```

All four readouts — distance, scale, topology, and confidence — are computed from the **same**
Fisher-Rao geometry. They agree with each other because they share a foundation.

## The Agreement

The modules independently converge on the same answers:

- Populations that topology flags as "continuous" (high transitionality) are the same ones
  where conformal prediction gives large, ambiguous sets
- Populations that topology flags as "discrete" (compact, low transitionality) are the same
  ones where conformal gives confident singletons
- The Markov-stability hierarchy (which types merge at which scales) is biologically sensible:
  endocrine cells cluster together, exocrine together, immune together
- The Fisher-Rao distance systematically stretches between-type gaps and contracts within-type
  distances — it *sharpens* boundaries

## The Code Flow

```
1. data.py          → Load raw data, QC, HVG, store .layers['counts']
2. nbvae.py         → Train VAE → latent z, decoder function
3. geometry.py      → Compute M(z) per cell from decoder Jacobian
4. graph.py         → Build FR kNN graph (candidate-and-rerank)
5. pipeline.py      → Run everything:
   ├─ Leiden on Euclidean & FR graphs
   ├─ Markov stability scan
   ├─ Conformal prediction (multiple α)
   ├─ Persistent homology (global + per-cell)
   ├─ UMAP embeddings (Euclidean + FR graphs)
   └─ Save: core.npz, markov.npz, graph_*.npz, summary.json
6. evaluate.py      → ARI, NMI, rare-type F1, per-type F1
7. figures.py       → Generate Figs 1-11 from cached results
8. figures_ws.py    → Generate Figs 12-15 (PhD extensions)
```

## Reuse and Caching

The pipeline is designed for reuse:
- After `pipeline.py` runs once, all figures can be regenerated without retraining
- `_cache.npz` stores the VAE latent + metric (reuse across seeds, ablations)
- `core.npz` stores all downstream outputs (UMAP, labels, conformal sets, topology)
- `summary.json` stores the headline metrics as human-readable JSON

Next: [Step 9 — Datasets and Results](09-datasets-and-results.md)
