# Step 7 — Calibrated Confidence for Every Cell

## The Problem

Standard clustering says "this is a T-cell" with 100% implied confidence for every cell. Even
for:

- A cell sitting exactly between a T-cell and a B-cell
- A cell of a type the model has never seen before
- A cell that's genuinely ambiguous (transitional state)

There is **no honesty**. No "I'm not sure." No "this might be one of two things." No "I've
never seen anything like this." Just a hard label.

## What Conformal Prediction Gives You

Instead of one label, every cell gets a **prediction set**:

```
C(cell) = { "CD4 T" }                          ← confident
C(cell) = { "CD4 T", "CD8 T" }                 ← ambiguous
C(cell) = { }                                  ← novel (matches nothing)
C(cell) = { "CD4 T", "CD8 T", "B", "NK", ... } ← don't know (matches everything)
```

The crucial part: these sets come with a **mathematical guarantee**. If we say we're aiming
for 90% coverage, then:

```
P( true label ∈ C(cell) ) ≥ 90%
```

This guarantee is **finite-sample** and **distribution-free**. It holds for any data
distribution, on any dataset, as long as the data points are exchangeable (order doesn't
matter). It works out of the box.

## How It Works

### Step 1: Split the Data

```
All cells → Training set (for training a classifier)
          → Calibration set (for calibrating the thresholds)
```

### Step 2: Train a Classifier

On the training set, train any classifier that outputs class probabilities. We use multinomial
logistic regression on the NB-VAE latent z. Output:

```
p̂(y=CD4 T | cell) = 0.92
p̂(y=CD8 T | cell) = 0.04
p̂(y=B      | cell) = 0.02
...
```

### Step 3: Measure How Wrong the Classifier Is

For each cell in the calibration set, compute a **nonconformity score**:

```
s(cell, true_label) = 1 − p̂(true_label | cell)
```

- If the classifier assigned 95% probability to the truth → score = 0.05 (good, easy cell)
- If it assigned only 10% → score = 0.90 (bad, the classifier was confused)

### Step 4: Set Per-Class Thresholds

For *each class separately* (the **Mondrian** or class-conditional approach):

```
q_y = the ⌈(n_y + 1)(1−α)⌉-th smallest score among calibration cells of class y
```

For example, if we want 90% coverage (α = 0.10) and have 100 calibration T-cells:

```
R = ⌈(100 + 1) × 0.90⌉ = ⌈90.9⌉ = 91
q_Tcell = the 91st smallest calibration score for T-cells
```

This threshold tells us: what's the worst score a T-cell can have while still being a
reasonable T-cell candidate?

### Step 5: Predict for New Cells

For a new cell, include every class where the probability is high enough:

```
C(cell) = { y : p̂(y | cell) ≥ 1 − q_y }
```

## Why Class-Conditional Matters

Without class-conditional calibration, the guarantee is only **marginal** — it holds on
average across all classes. A 10,000-cell common type could carry coverage while a 30-cell
rare type gets zero coverage, and the average still meets 90%.

The Mondrian approach guarantees coverage **per class**:

```
P( true label ∈ C(cell) | true label = y ) ≥ 1 − α    for every class y
```

This includes the rare epsilon cells, the rare mast cells, every population.

## What the Set Sizes Mean

| Set size | Interpretation |
|---|---|
| `|C| = 1` | Confident. Only one label is plausible. |
| `|C| > 1` | Ambiguous. Multiple labels are plausible. The cell sits between types. |
| `|C| = 0` | Novel/OOD. No known label is plausible. This cell is unlike anything in the training data. |

## The Disease Application

On the diabetes dataset, we calibrate the confidence layer on **healthy cells only**, then
evaluate on **diabetic cells**. This is a covariate shift — the diabetic cells come from a
different distribution:

- Overall coverage drops only slightly: 90.7% → still meets the 90% target
- But **beta-cells** (the insulin makers) and **gamma-cells** are hit hardest: β coverage drops
  to 87.8%, γ to 81.0%
- Exocrine ductal and acinar cells stay at target

The conformal layer **localizes the disease effect to the endocrine compartment** — the cells
diabetes actually damages — without ever being shown a disease label. It reports: "I'm less
confident about these cells specifically," and that honest uncertainty is itself a biological
signal.

Next: [Step 8 — The Full Pipeline](08-full-pipeline.md)
