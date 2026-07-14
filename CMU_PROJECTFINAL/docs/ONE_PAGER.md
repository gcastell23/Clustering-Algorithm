# Our project, in plain English (one page, no jargon)

## The big picture
Every cell in your body has the same DNA, but different cells *use* different genes — a brain
cell, a muscle cell, and an immune cell each switch on a different set of genes. There's a
machine that can read, one cell at a time, **which genes each cell is using**. Run it on a
piece of tissue and you get a giant spreadsheet: one row per cell, one column per gene, and each
number is "how much of this gene did this cell use."

The first thing scientists want from that spreadsheet is: **sort the cells into groups** ("these
2,000 cells are all the same type; those 500 are a different type"). That sorting is called
*clustering*, and it's how we make maps of tissues and find out what goes wrong in disease.

## The problem we noticed
To sort cells into groups, the computer has to decide **how far apart two cells are**. The
standard tool measures distance with a plain ruler — like measuring everything in inches.

But cell data is noisy in a sneaky way: **genes that are used a lot are naturally much noisier
than genes that are used a little.** So a plain ruler is misleading — it treats a wobble in a
loud gene the same as a wobble in a quiet gene, even though one is just noise and the other is
real. Using the wrong ruler makes the computer **lump rare cell types in with common ones and
lose them.** Rare cells are often the most interesting ones (they can be the diseased cells).

On top of that, the standard tool has three more weak spots:
1. You have to **turn a knob by hand** to decide how many groups there are — and different people
   turn it differently, getting different answers.
2. It **can't tell the difference** between a real, distinct cell type and a cell that's *in the
   middle of changing* from one type into another (a smooth transition).
3. It gives you an answer with **no honesty about how sure it is** — even for a cell it basically
   guessed on.

## Our idea (one idea that fixes all four)
We build a smart statistical model of the data first. That model automatically "knows" how noisy
each gene is. From it we get a **smarter ruler** that stretches and shrinks correctly — it knows
that a small difference in a quiet gene can matter a lot, and a big difference in a loud gene
might be nothing. The beautiful part: **this one smart ruler also solves the other three
problems**:

- **No more knob.** Instead of picking the number of groups, we let a "random walker" wander
  around the data. Groupings that stay stable no matter how long the walker wanders are the real
  ones. The number of groups *emerges* instead of being set by hand.
- **Types vs. transitions.** We use a branch of math called *topology* (the math of shape) to ask
  of each group: is this a solid, separate blob (a real type), or a bridge between two blobs (a
  transition)? 
- **Honest confidence.** Every cell gets a **calibrated confidence label**: "I'm sure this is a
  T-cell," or "this could be one of two things," or "this doesn't match anything I know." And we
  can *prove* the confidence is honest (if we say we're 90% sure, we're right about 90% of the time).

## What we found
- On a real map of the **pancreas** (the organ that fails in diabetes), our method finds the
  **rare cell types that the standard tool almost completely misses** — the standard tool scored
  near zero on the rarest types; ours found them.
- The "no-knob" method **rediscovers the known cell types on its own**, and even shows the natural
  family tree of how fine groups combine into coarse ones.
- On **diabetic** pancreas, we taught the "honesty meter" on healthy cells only, then turned it
  loose on diabetic cells. It stayed accurate overall — but it got **least confident exactly on the
  insulin-making beta-cells and their neighbors**, the cells diabetes actually damages. Without ever
  being told which cells were sick, the method **pointed to where the disease is.**

## Why it matters
We turned four separate, shaky guessing steps into **one honest, self-consistent method** built on
a single mathematical idea. It finds the rare cells that matter, stops making people guess, tells
real types from transitions, and never pretends to be more certain than it is.

## Who did what
- **Gabriella** — data loading, quality control, and cross-dataset integration.
- **Megan** — the preprocessing pipeline and the standard-method baselines.
- **Emma** — clustering, cell-type annotation, and the evaluation harness.
- **Rajarshi** — the smart-ruler math (information geometry), the random-walk and topology
  modules, and benchmarking.

*(One line if someone asks what it's called: **IGMC — Information-Geometric Multiscale
Clustering.**)*
