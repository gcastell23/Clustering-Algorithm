# Step 2 — Why Euclidean Distance is the Wrong Ruler

## Count Noise is Mean-Dependent

In scRNA-seq, RNA counts are noisy. But the noise isn't the same everywhere — it follows a
**negative binomial** pattern:

```
Var(x) = μ + μ²/θ
```

where **μ** is the gene's mean expression and **θ** controls overdispersion.

What this means in plain language:

- A gene with mean **100** has variance ~100 + 10,000/θ. A difference of 5 here is mostly
  noise — like measuring a shout with a rattling microphone.

- A gene with mean **0.5** has variance ~0.5 + 0.25/θ. A difference of 1 here is a *huge*
  signal — like hearing a whisper in a quiet room.

## What Euclidean Distance Does Wrong

Euclidean distance says: "a difference of 5 in gene A equals a difference of 5 in gene B." It
treats every unit as equally meaningful, which is exactly wrong when noise levels differ by a
factor of 100.

```
Euclidean:   d(cell_i, cell_j) = √( Σ_g (x_ig - x_jg)² )
```

Every gene gets equal weight, regardless of its noise level.

## Who Gets Hurt Most

The cells that suffer most are the **rare, lowly-sampled populations** — often the most
biologically interesting ones (rare immune subtypes, stem cells, diseased cells). These cells
have:

- Fewer total RNA molecules (shallow libraries)
- Expression concentrated in fewer genes
- Noisy high-variance genes dominating the Euclidean distance

Standard log-normalization + PCA papers over this but doesn't fundamentally fix it. The
spurious variance injected by normalizing rare cells into a common scale can actually make
things worse.

## What We Need

A ruler that says: **"A small difference in a quiet gene matters a lot. A big difference in a
loud gene might mean nothing."**

Statistics already has such a ruler. It's called the Fisher information.

Next: [Step 3 — The Fisher-Rao Metric](03-fisher-rao-metric.md)
