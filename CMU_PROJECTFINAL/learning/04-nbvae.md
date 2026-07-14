# Step 4 — The Negative-Binomial VAE

## What a VAE Does

A **Variational Autoencoder** (VAE) is a neural network that:

1. **Encodes** each cell's 20,000-gene expression into a small code **z** (about 10 numbers)
2. **Decodes** that code back into a full gene-expression distribution

The key: the reconstruction is *probabilistic*. The decoder outputs a distribution over
possible gene counts, not a single number. If a cell has exactly 3 molecules of a gene but the
model predicts a mean of 3.2, that's still a good fit — the model captures the *uncertainty*.

## Why Negative Binomial?

Actual RNA count data follows a pattern: lots of zeros, a few moderate counts, rare very high
counts. The negative binomial distribution captures this exactly because its variance grows
with the mean:

```
x_ig ~ NB(mean = library_size × ρ_g(z), inverse_dispersion = θ_g)
```

Each gene gets its own θ_g — genes that are more overdispersed (noisier) get a smaller θ.

## Batch Conditioning

Cells in scRNA-seq come from different samples, labs, and technologies. A cell from lab A
might look different from a cell in lab B even if they're the same type — this is technical
**batch effect**.

The IGMC VAE feeds the batch label (one-hot encoded) to both the encoder and the decoder.
This pushes the technical variation into the batch variable and *out* of the latent code z, so
z captures only biological variation.

```
Encoder:  [gene_expression, batch_onehot] → z_mean, z_logvar
Decoder:  [z, batch_onehot] → gene_proportions (softmax over 2000 genes)
```

## The Architecture

```
    x (counts)                     b (batch)
        │                              │
        ▼                              │
   ┌─────────┐                         │
   │ Encoder ◄─────────────────────────┘
   │ MLP+LN  │
   └────┬────┘
        │
   z_mean, z_logvar
        │
   z = z_mean + ε·exp(z_logvar/2)   ← reparameterization trick
        │
        ▼            b
   ┌─────────┐       │
   │ Decoder ◄───────┘
   │ MLP+LN  │
   └────┬────┘
        │
   ρ = softmax(output)     ← gene proportions on the simplex
        │
   μ_g = library_size × ρ_g
        │
   x ~ NB(μ, θ)             ← reconstruction
```

Note: the decoder uses **LayerNorm** instead of BatchNorm. LayerNorm is compatible with
`torch.func.jacfwd` (automatic differentiation over a single input), which we need to compute
the Fisher-Rao metric Jacobian. BatchNorm would break this because it depends on the batch
statistics.

## Training

The loss function has two parts:

```
loss = reconstruction_error + β × KL_divergence
```

- **Reconstruction error**: negative log-likelihood of the observed counts under the NB
  distribution predicted by the decoder
- **KL divergence**: how far the latent distribution is from a standard normal N(0, I). This
  acts as a regularizer — it keeps z compact and well-behaved.
- **β**: a warmup parameter. Starts at 0 (no KL pressure), gradually rises to 1 over ~50
  epochs. This prevents the model from ignoring the reconstruction and collapsing z to N(0,I)
  before it's learned anything useful.

The model trains for ~200-300 epochs with Adam, gradient clipping, and weight decay. It runs
on CPU (no GPU needed for ~16k cells).

## Output

After training, the model gives us:

- **z** for every cell: a 10-dimensional latent code
- **θ** per gene: the gene-specific inverse dispersion
- The decoder function `z, batch → gene_proportions(z)` — this is what we differentiate to
  get the Fisher-Rao metric

Next: [Step 5 — Markov Stability Clustering](05-markov-stability.md)
