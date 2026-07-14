# Scale-nested conformal prediction on the Markov-stability hierarchy

**WS2 theory contribution.** Formal statements and proofs, with honest positioning against prior
work. Implemented in `src/scale_nested_conformal.py`; validated (held-out, 25 splits) in
`results/scale_nested_conformal.json` and Fig. 13. All results are finite-sample and
distribution-free (exchangeability only).

## What is and is not new (read first)
Two of the three ingredients are standard and we present them as such:
- **Per-scale validity (Prop. 1)** is textbook split / Mondrian conformal (Vovk et al. 2005;
  Sadinle et al. 2019; class-conditional variants Ding et al. 2023).
- **Simultaneous coverage (Prop. 2)** is a standard Bonferroni / union-bound allocation across a
  family of predictors (the same across-scale device appears in Baheri & Amiri Shahbazi 2025).

The genuinely new element is **the coherent construction (Thm. 3)**: an isotonic up-the-tree
projection of the per-scale conformal thresholds that makes the family of prediction sets
*coherent across scales* (a fine call never contradicts its coarse super-type) **while provably
preserving per-scale coverage**. It sits between isotonic-in-conformal calibration (van der Laan &
Alaa 2024) and nested/monotone conformal sets (Gupta, Kuchibhotla & Ramdas 2022), but we did not
find this exact construction. The overall *package* — a Markov-stability multiscale-community
hierarchy used as the nested conformal taxonomy for single-cell resolution selection — is, as a
combination, new; the closest prior art is scConform (Corbetta et al. 2024), which does conformal
cell-type annotation with class-conditional coverage and an ontology-graph for coherent sets, but
with a **fixed** ontology (or k-means-on-scores), marginal/class-conditional coverage, and **no**
simultaneous guarantee across a resolution scan.

## Setup and notation
Fine label space $\mathcal Y$, $|\mathcal Y|=K$. A rooted **taxonomy** with levels $\ell=1,\dots,L$
(level 1 finest, $L$ coarsest) partitions $\mathcal Y$ at each level into super-classes
$\mathcal G_\ell$, with coarsening maps $\pi_\ell:\mathcal Y\to\mathcal G_\ell$ and parent maps
$\rho_\ell:\mathcal G_\ell\to\mathcal G_{\ell+1}$ obeying the **nesting** identity
$\pi_{\ell+1}=\rho_\ell\circ\pi_\ell$ ($\pi_1=\mathrm{id}$). We build it from the integrated Markov
co-clustering distance $D_{ab}=1-\tfrac1T\sum_t\sum_c P(a\in c\mid t)P(b\in c\mid t)$ between types
$a,b$ (average-linkage dendrogram cut into distinct nested levels).

A classifier trained on a **disjoint** split outputs fine-class probabilities $p(y\mid x)$;
super-class probabilities aggregate them, $p_\ell(g\mid x)=\sum_{y:\pi_\ell(y)=g}p(y\mid x)$. LAC
score $s_\ell(x,g)=1-p_\ell(g\mid x)$ (Sadinle et al. 2019).

**Calibration.** Given an exchangeable calibration sample $(X_i,Y_i)_{i=1}^n$ and target level
$\alpha_\ell$, the **marginal** threshold is the conformal quantile
$q_\ell=\big\lceil(n+1)(1-\alpha_\ell)\big\rceil\text{-th smallest of }\{s_\ell(X_i,\pi_\ell(Y_i))\}$
($=1$ if the rank exceeds $n$). Prediction set $C_\ell(x)=\{g:p_\ell(g\mid x)\ge 1-q_\ell\}$. We use
marginal calibration by default because it yields *informative* sets; a class-conditional (Mondrian)
variant uses a separate $q_\ell(g)$ per super-class (guaranteeing class-wise coverage but forcing
uninformative full sets for classes with $n_g\lesssim\alpha_\ell^{-1}$ calibration cells).

---

## Proposition 1 (per-level validity — standard)
For each level $\ell$, if the calibration and test points are exchangeable,
$\Pr\big(\pi_\ell(Y_{n+1})\in C_\ell(X_{n+1})\big)\ge 1-\alpha_\ell.$
*Proof.* The event is $\{s_\ell(X_{n+1},\pi_\ell(Y_{n+1}))\le q_\ell\}$; by exchangeability the rank
of the test score among the $n+1$ scores is uniform, giving
$\lceil(n+1)(1-\alpha_\ell)\rceil/(n+1)\ge 1-\alpha_\ell$ (Vovk et al. 2005). $\blacksquare$ The
class-conditional version (per-super-class threshold) is Sadinle et al. 2019 / Ding et al. 2023.

## Proposition 2 (simultaneous coverage via union-bound allocation — standard)
For any allocation $\alpha_\ell\ge0$ with $\sum_\ell\alpha_\ell=\alpha$,
$\Pr(\exists\ell:\pi_\ell(Y_{n+1})\notin C_\ell(X_{n+1}))\le\alpha.$
*Proof.* Boole's inequality over the per-level miscoverage events, each $\le\alpha_\ell$ by
Prop. 1. $\blacksquare$
**Remark (conservativeness).** Nesting makes the miscoverage events strongly positively dependent,
so the bound is loose: empirically simultaneous coverage $\approx0.98$ at $\alpha=0.1$. It is not
vacuous, though — running every level at the full $\alpha$ (no allocation) drops simultaneous
coverage to the target with **no** margin (empirically $\approx0.90$). We report the provable
allocation and its slack; a tighter valid joint threshold is future work.

---

## Coherence (the new part)
Call the family **coherent** if for all $x$ and $\ell<L$, $\rho_\ell(C_\ell(x))\subseteq C_{\ell+1}(x)$:
if a fine super-class is accepted, so is its coarse parent (no orphaned fine calls).

**Lemma A (monotone aggregation).** For $g\in\mathcal G_\ell$, parent $h=\rho_\ell(g)$:
$p_{\ell+1}(h\mid x)\ge p_\ell(g\mid x)$, since nesting gives
$\{y:\pi_\ell(y)=g\}\subseteq\{y:\pi_{\ell+1}(y)=h\}$ and probabilities are nonnegative. $\blacksquare$

**Lemma B (isotonic monotonization preserves the exchangeability quantile).** Replacing a threshold
$q$ by any $\tilde q\ge q$ enlarges $\{s\le\tilde q\}\supseteq\{s\le q\}$, hence enlarges $C_\ell$ and
cannot lower the coverage $\Pr(\pi_\ell(Y_{n+1})\in C_\ell(X_{n+1}))$; so the Prop. 1 bound survives
any upward projection of the thresholds. $\blacksquare$

**Theorem 3 (coherent construction preserving validity — new).** Define projected thresholds by a
fine→coarse pass $\tilde q_1=q_1$, $\tilde q_{\ell+1}=\max(q_{\ell+1},\ \tilde q_\ell)$ (in the
Mondrian case, per parent: $\tilde q_{\ell+1}(h)=\max(q_{\ell+1}(h),\max_{g:\rho_\ell(g)=h}\tilde
q_\ell(g))$). Then $C_\ell(x)=\{g:p_\ell(g\mid x)\ge1-\tilde q_\ell\}$ is **coherent** and still
satisfies Props. 1–2.
*Proof.* Validity: $\tilde q_\ell\ge q_\ell$, so Lemma B preserves each per-level bound and hence the
union bound. Coherence: if $g\in C_\ell(x)$ then, with $h=\rho_\ell(g)$,
$p_{\ell+1}(h\mid x)\ge p_\ell(g\mid x)\ge1-\tilde q_\ell\ge1-\tilde q_{\ell+1}$ (Lemma A and the
upward pass), so $h\in C_{\ell+1}(x)$. $\blacksquare$
**Price of coherence.** The projection only enlarges sets; empirically the cost is small, while
*without* it coherence is frequently violated (0–91 % of cells coherent, dropping to 0 % when sets
are large), versus exactly 100 % with it.

## The adaptive resolvable scale
$r(x)=\min\{\ell:|C_\ell(x)|=1\}$ — the finest scale at which a cell has a singleton call — is a
per-cell, calibrated resolution selector (computed in the per-scale operating regime so singletons
exist). Prediction: continuum cells resolve only coarsely (large $r$); discrete-type cells resolve
finely. We test this against the topological discreteness index; the two independent lenses agree.

## Assumptions and honest limits
- Exchangeability of calibration + test; classifier trained on a disjoint split. Both hold here.
- **Guarantee vs. informativeness trade-off is explicit.** The family-wise Bonferroni regime
  ($\alpha_\ell=\alpha/L$) is deliberately conservative — fine-scale sets are large and rarely
  singletons; that is the price of a provable simultaneous guarantee. The per-scale regime
  ($\alpha_\ell=\alpha$) is only per-level valid but gives informative sets; we use it for the
  resolvable-scale analysis and state this.
- Marginal coverage is the default; class-conditional (Mondrian) coverage is available but makes
  rare fine types uninformative (full sets). Either way, rare types under-served at the fine scale
  are recovered at coarser scales where they are pooled into well-populated super-classes — the
  hierarchy is the remedy, not a patch.
- The union bound (Prop. 2) is conservative under nesting; we report the empirical slack.

## References (verified; see docs/verified_citations.bib)
Vovk, Gammerman & Shafer 2005; Sadinle, Lei & Wasserman 2019; Angelopoulos & Bates 2021; Ding,
Angelopoulos, Bates, Jordan & Tibshirani 2023 (class-conditional CP); Baheri & Amiri Shahbazi 2025
(multi-scale CP / union bound); Gupta, Kuchibhotla & Ramdas 2022 (nested CP); van der Laan & Alaa
2024 (self-calibrating / isotonic CP); Corbetta, Finos, Geistlinger & Risso 2024 (scConform);
Delvenne, Yaliraki & Barahona 2010 and Arnaudon et al. 2023 (Markov stability).
