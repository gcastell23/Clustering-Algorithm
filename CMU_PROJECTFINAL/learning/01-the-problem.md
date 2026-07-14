# Step 1 — The Problem

## What is scRNA-seq?

Every cell has the same DNA, but different cells use different genes. A brain cell switches on
one set of genes. A muscle cell switches on another. scRNA-seq measures which genes each cell
is using, one cell at a time.

The result is a giant spreadsheet:

```
           Gene_1  Gene_2  ...  Gene_20000
  Cell_A       0      15  ...           3
  Cell_B     120       2  ...           0
  Cell_C       0       0  ...         200
```

Each number is an integer: how many molecules of that gene were detected.

## The Goal

Sort the cells into groups. T-cells here, B-cells there, beta-cells over there. This is called
**clustering**, and it's how scientists make cell-type atlases, discover new cell types, and
study disease.

## The Standard Pipeline

The community uses the same recipe almost everywhere:

```
Raw counts → Log-normalize → PCA → kNN graph → Leiden clustering
```

This pipeline has **four problems**, each one inherited by the next step:

1. **The ruler is wrong.** Euclidean distance treats every unit of gene expression the same,
   but count noise is mean-dependent — a gene used a lot is naturally much noisier than one
   barely expressed. The wrong ruler corrupts rare populations most.

2. **The resolution is arbitrary.** Leiden requires a resolution parameter. 0.5 gives 8
   clusters, 1.0 gives 14, 2.0 gives 30. Different people pick different values and get
   different answers. There's no principled way to choose.

3. **Types and transitions are conflated.** Some populations are sharp, well-separated types.
   Others are stops along a smooth continuum (differentiation, activation). Standard
   clustering forces a hard partition on both and can't say which is which.

4. **There is no confidence.** Every cell gets a hard label — "this is a T-cell" — with
   zero uncertainty, even for a cell sitting ambiguously between types or belonging to a
   state the model has never seen.

The IGMC project replaces this four-step pipeline with **one method** built on **one geometric
object** that solves all four problems.

Next: [Step 2 — Why Euclidean distance is wrong](02-count-noise.md)
