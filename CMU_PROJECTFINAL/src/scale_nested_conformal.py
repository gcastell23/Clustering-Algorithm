"""
scale_nested_conformal.py — conformal prediction on the Markov-stability hierarchy.

WS2 theory contribution.  The Markov-stability scan produces cell communities at a range of
scales (fine -> coarse as Markov time grows).  This induces a *taxonomy* of cell types: which
fine types belong to the same coarse super-type.  We give every cell a whole FAMILY of
conformal prediction sets, one per taxonomic level, and prove three things (see
docs/THEORY_scale_nested_conformal.md and paper/main.tex):

  Thm 1  each level is a valid Mondrian split-conformal predictor:
             P(Y_l in C_l(X) | level-l group) >= 1 - alpha_l.
  Thm 2  with an allocation  sum_l alpha_l = alpha,  the family is SIMULTANEOUSLY valid:
             P( exists l : Y_l(X) not in C_l(X) ) <= alpha        (union bound over levels).
  Thm 3  the aggregated-probability construction with an isotonic up-the-tree projection of the
         calibrated thresholds is COHERENT:  pi_l( C_{l+1}(x) ) subset C_l(x)  for every x,
         i.e. a fine-scale call never contradicts its coarse-scale super-type — while the
         projection only enlarges sets and therefore preserves the Thm 1 coverage.

Construction.  A single classifier gives fine-label probabilities p(y|x).  Super-class
probabilities aggregate them:  p_l(g|x) = sum_{y : pi_l(y)=g} p(y|x)  (so p_l(g) >= p_{l+1}(child)).
LAC score s_l(x,g)=1-p_l(g|x); Mondrian threshold q_l(g) at level ceil((n_g+1)(1-alpha_l))/n_g.
Coherence needs q_l(parent) >= q_{l+1}(child); we enforce it by
    q~_l(g) = max( q_l(g),  max_{child c of g} q~_{l+1}(c) ),      (fine -> coarse pass)
which only increases thresholds (Lemma: enlarging q enlarges C and cannot lower coverage).
"""
from __future__ import annotations
import os, sys, json
import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from conformal import conformal_quantile
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(HERE, "results")


# ----------------------------------------------------------------------------------------
# 1) Markov taxonomy: nested cell-type hierarchy from the multiscale co-clustering
# ----------------------------------------------------------------------------------------
def markov_comembership_distance(y_ms, mlabels, times):
    """
    Type x type dissimilarity from the *integrated* Markov co-clustering:
        affinity_ab = (1/T) sum_t  sum_c P(a in c | t) P(b in c | t),
        D_ab        = 1 - affinity_ab.
    Two types that are walked into the same community across many Markov times are close;
    types that only merge at the very coarsest scales are far.  Continuous (no ties) so the
    dendrogram cuts cleanly into distinct levels.

    y_ms    : (n_ms,) true labels on the Markov subsample
    mlabels : list over times of (n_ms,) community assignments
    Returns : (types, D) square distance matrix over sorted types.
    """
    y_ms = np.asarray([str(v) for v in y_ms])
    types = np.array(sorted(set(y_ms)))
    K = len(types); tindex = {t: i for i, t in enumerate(types)}
    na = {a: max(1, int(np.sum(y_ms == a))) for a in types}
    T = len(mlabels)
    CMsum = np.zeros((K, K))
    for lab in mlabels:
        lab = np.asarray(lab); comms = np.unique(lab)
        P = np.zeros((K, len(comms)))
        for ci, c in enumerate(comms):
            cells = lab == c
            for a in types:
                P[tindex[a], ci] = np.sum((y_ms == a) & cells) / na[a]
        CMsum += P @ P.T
    CM = CMsum / T
    D = 1.0 - CM
    np.fill_diagonal(D, 0.0)
    D = np.clip(0.5 * (D + D.T), 0.0, None)
    return types, D


def build_taxonomy(types, D, n_levels=4):
    """
    Average-linkage dendrogram on the Markov distance, cut into nested levels of DISTINCT,
    geometrically-spaced sizes (K -> ... -> 2).  Level 0 = finest (identity, every type its
    own class); the trivial 1-class level is dropped.  Returns:
      maps[l] : dict type-name -> level-l super-class id (str), l=0..L-1 (fine -> coarse)
      sizes   : list of #classes per level
      Z       : the linkage matrix (for the dendrogram panel)
    """
    K = len(types)
    Z = linkage(squareform(D, checks=False), method="average")
    want = sorted({int(s) for s in np.round(np.geomspace(K, 2, n_levels)).astype(int)
                   if 2 <= s <= K}, reverse=True)
    if K not in want:
        want = [K] + want
    maps, sizes, seen = [], [], set()
    for s in want:
        if s == K:
            m = {t: f"S{K}_{i}" for i, t in enumerate(types)}; ncl = K
        else:
            cl = fcluster(Z, t=s, criterion="maxclust"); ncl = len(np.unique(cl))
            m = {types[i]: f"S{s}_{cl[i]}" for i in range(K)}
        if ncl in seen:                          # drop collapsed / duplicate levels
            continue
        seen.add(ncl); maps.append(m); sizes.append(ncl)
    return maps, sizes, Z


def parent_maps(maps):
    """From per-level type->superclass maps, derive child->parent maps between adjacent levels.
    Returns list of dicts: parent_of[l] maps a level-l superclass id to its level-(l+1) parent id.
    (levels ordered fine=0 .. coarse=L-1)."""
    types = list(maps[0].keys())
    L = len(maps)
    parent_of = []
    for l in range(L - 1):
        pm = {}
        for t in types:
            pm[maps[l][t]] = maps[l + 1][t]           # consistent because taxonomy is nested
        parent_of.append(pm)
    return parent_of


# ----------------------------------------------------------------------------------------
# 2) Scale-nested Mondrian conformal predictor
# ----------------------------------------------------------------------------------------
class ScaleNestedConformal:
    def __init__(self, maps, alpha=0.1, weights=None, coherent=True, mondrian=False,
                 C=1.0, seed=0):
        self.maps = maps                              # fine=0 .. coarse=L-1
        self.L = len(maps)
        self.alpha = alpha
        # allocation sum_l alpha_l = alpha (default equal)
        w = np.ones(self.L) if weights is None else np.asarray(weights, float)
        self.alpha_l = alpha * w / w.sum()
        self.coherent = coherent
        self.mondrian = mondrian    # False = marginal per-scale (informative); True = class-conditional
        self.clf = LogisticRegression(max_iter=3000, C=C)      # multinomial is the default
        self.scaler = StandardScaler()
        self.seed = seed

    def _superlabels(self, y, l):
        return np.array([self.maps[l][str(v)] for v in y])

    def fit(self, Z, y, calib_frac=0.5):
        rng = np.random.default_rng(self.seed)
        y = np.array([str(v) for v in y])
        # STRATIFIED split so every class is represented in training (avoids train-absent classes
        # -> undefined finest-level super-classes; handoff landmine #3).
        cal, tr = [], []
        for c in np.unique(y):
            ci = np.where(y == c)[0]; rng.shuffle(ci)
            ncal_c = int(round(len(ci) * calib_frac))
            if len(ci) >= 2:
                ncal_c = min(max(ncal_c, 1), len(ci) - 1)   # >=1 in each of calib & train
            else:
                ncal_c = 0                                   # singletons -> training
            cal.extend(ci[:ncal_c].tolist()); tr.extend(ci[ncal_c:].tolist())
        cal, tr = np.array(cal, int), np.array(tr, int)
        Ztr = self.scaler.fit_transform(Z[tr])
        self.clf.fit(Ztr, y[tr])
        self.fine_classes_ = self.clf.classes_
        self._col = {c: j for j, c in enumerate(self.fine_classes_)}
        Zc = self.scaler.transform(Z[cal]); Pc = self.clf.predict_proba(Zc)
        self._cal_idx, self._tr_idx = cal, tr
        # per-level superclass list + aggregation matrix (fine -> superclass)
        self.super_classes_ = []; self.agg_ = []
        for l in range(self.L):
            sc = sorted(set(self.maps[l][c] for c in self.fine_classes_))
            self.super_classes_.append(sc)
            A = np.zeros((len(sc), len(self.fine_classes_)))
            sidx = {g: i for i, g in enumerate(sc)}
            for j, c in enumerate(self.fine_classes_):
                A[sidx[self.maps[l][c]], j] = 1.0
            self.agg_.append(A)
        # calibrate thresholds per level.  Marginal (default): one threshold per level shared by
        # all super-classes -> informative sets.  Mondrian: a separate threshold per super-class
        # -> class-conditional coverage but rare classes force uninformative full sets.
        self.q_ = []
        for l in range(self.L):
            Psup = Pc @ self.agg_[l].T                 # (ncal, n_super)
            ysup = self._superlabels(y[cal], l)
            sidx = {g: i for i, g in enumerate(self.super_classes_[l])}
            # drop any calibration cell whose (true) super-class was absent from training
            keep = np.array([g in sidx for g in ysup])
            cols = np.array([sidx.get(g, 0) for g in ysup])
            s_all = 1.0 - Psup[np.arange(len(cal)), cols]
            q = {}
            if self.mondrian:
                for g in self.super_classes_[l]:
                    m = keep & (ysup == g)
                    q[g] = conformal_quantile(s_all[m], self.alpha_l[l]) if m.sum() else 1.0
            else:
                qm = conformal_quantile(s_all[keep], self.alpha_l[l])
                for g in self.super_classes_[l]:
                    q[g] = qm
            self.q_.append(q)
        if self.coherent:
            self._project_thresholds()
        return self

    def _project_thresholds(self):
        """Isotonic up-the-tree: q~_l(parent) = max(q_l, max child q~).  Fine -> coarse."""
        pm = parent_maps(self.maps)                    # parent_of[l]: level-l id -> level-(l+1) id
        self.q_raw_ = [dict(q) for q in self.q_]
        for l in range(self.L - 1):
            for child, q_child in self.q_[l].items():
                parent = pm[l][child]
                if q_child > self.q_[l + 1].get(parent, 0.0):
                    self.q_[l + 1][parent] = q_child

    def predict(self, Z):
        """Return per-level prediction sets: list over cells of list over levels of set(ids)."""
        P = self.clf.predict_proba(self.scaler.transform(Z))
        sets_by_level = []
        for l in range(self.L):
            Psup = P @ self.agg_[l].T
            sidx = {g: i for i, g in enumerate(self.super_classes_[l])}
            Sl = []
            for i in range(P.shape[0]):
                keep = {g for g in self.super_classes_[l] if Psup[i, sidx[g]] >= 1 - self.q_[l][g]}
                Sl.append(keep)
            sets_by_level.append(Sl)
        return sets_by_level

    def evaluate(self, Z, y):
        y = np.array([str(v) for v in y])
        sets = self.predict(Z)                         # sets[l][i]
        n = len(y); out = {"L": self.L, "alpha": self.alpha,
                           "alpha_l": self.alpha_l.tolist(), "levels": []}
        # per-level coverage + set size
        miscov_any = np.zeros(n, bool)
        incoherent = np.zeros(n, bool)
        resolvable = np.full(n, -1)                    # finest level with singleton correct-ish set
        pm = parent_maps(self.maps)
        for l in range(self.L):
            ysup = self._superlabels(y, l)
            cov = np.array([ysup[i] in sets[l][i] for i in range(n)])
            size = np.array([len(sets[l][i]) for i in range(n)])
            miscov_any |= ~cov
            # per-superclass coverage
            pcc = {}
            for g in self.super_classes_[l]:
                m = ysup == g
                pcc[g] = float(cov[m].mean()) if m.sum() else float("nan")
            out["levels"].append({
                "level": l, "n_super": len(self.super_classes_[l]),
                "alpha_l": float(self.alpha_l[l]),
                "marginal_coverage": float(cov.mean()),
                "mean_set_size": float(size.mean()),
                "frac_singleton": float(np.mean(size == 1)),
                "min_per_class_coverage": float(np.nanmin(list(pcc.values()))),
            })
        # coherence: for each cell/level, parents of fine set subset coarse set
        for l in range(self.L - 1):
            for i in range(n):
                parents = {pm[l][g] for g in sets[l][i]}
                if not parents.issubset(sets[l + 1][i]):
                    incoherent[i] = True
        # adaptive resolvable scale: finest level where the set is a singleton
        for i in range(n):
            r = -1
            for l in range(self.L):
                if len(sets[l][i]) == 1:
                    r = l; break
            resolvable[i] = r
        out["simultaneous_coverage"] = float(np.mean(~miscov_any))
        out["simultaneous_target"] = float(1 - self.alpha)
        out["coherence_rate"] = float(np.mean(~incoherent))
        out["resolvable_level"] = resolvable.tolist()
        return out, sets, resolvable


# ----------------------------------------------------------------------------------------
# 3) Driver: build taxonomy from cache + run on full data
# ----------------------------------------------------------------------------------------
def _split_eval(Z, y, maps, alpha, coherent, alloc, fit_frac, seed):
    """One stratified fit/test split; fit on fit_frac, evaluate coverage on the held-out rest."""
    rng = np.random.default_rng(seed)
    y = np.array([str(v) for v in y])
    fit_i, test_i = [], []
    for c in np.unique(y):
        ci = np.where(y == c)[0]; rng.shuffle(ci)
        nfit_c = int(round(len(ci) * fit_frac))
        if len(ci) >= 2:
            nfit_c = min(max(nfit_c, 1), len(ci) - 1)
        else:
            nfit_c = 1
        fit_i.extend(ci[:nfit_c].tolist()); test_i.extend(ci[nfit_c:].tolist())
    fit_i, test_i = np.array(fit_i, int), np.array(test_i, int)
    snc = ScaleNestedConformal(maps, alpha=alpha, coherent=coherent, seed=seed)
    snc.alpha_l = (np.ones(snc.L) * alpha if alloc == "none"
                   else alpha * np.ones(snc.L) / snc.L)          # Bonferroni vs uncorrected
    snc.fit(Z[fit_i], y[fit_i])
    res, _, resolv = snc.evaluate(Z[test_i], y[test_i])
    return res, snc, test_i, resolv


def run_dataset(name, n_levels=4, alpha=0.1, n_splits=25, fit_frac=0.7, seed0=0):
    """Rigorous held-out evaluation averaged over n_splits random calib/test partitions.
    Returns aggregate stats + one representative predictor (for the figure)."""
    core = np.load(os.path.join(RESULTS, name, "core.npz"), allow_pickle=True)
    mk = np.load(os.path.join(RESULTS, name, "markov.npz"), allow_pickle=True)
    Z = core["Z"]; y = core["y"]
    times = mk["times"]; y_ms = mk["y_ms"]
    mlabels = [mk[f"mlabel__{i}"] for i in range(len(times))]
    types, D = markov_comembership_distance(y_ms, mlabels, times)
    maps, sizes, Zlink = build_taxonomy(types, D, n_levels=n_levels)
    L = len(maps)

    def agg(records, key, sub=None):
        vals = [(r[key] if sub is None else r[key][sub]) for r in records]
        vals = np.asarray(vals, float)
        return {"mean": float(vals.mean()), "std": float(vals.std(ddof=1)) if len(vals) > 1 else 0.0}

    # three configurations, each over n_splits held-out partitions
    configs = {"coherent_bonf": dict(coherent=True, alloc="bonf"),
               "raw_bonf":      dict(coherent=False, alloc="bonf"),
               "uncorrected":   dict(coherent=True, alloc="none")}
    out = {"dataset": name, "alpha": alpha, "n_levels": L, "taxonomy_sizes": sizes,
           "types": types.tolist(), "taxonomy_maps": maps, "n_splits": n_splits, "configs": {}}
    rep = None
    for cname, cfg in configs.items():
        recs = []
        for s in range(n_splits):
            res, snc, test_i, resolv = _split_eval(Z, y, maps, alpha, cfg["coherent"],
                                                   cfg["alloc"], fit_frac, seed0 + s)
            recs.append(res)
            if cname == "coherent_bonf" and s == 0:
                rep = (snc, res)
        d = {"simultaneous_coverage": agg(recs, "simultaneous_coverage"),
             "simultaneous_target": recs[0]["simultaneous_target"],
             "coherence_rate": agg(recs, "coherence_rate"),
             "levels": []}
        for l in range(L):
            d["levels"].append({
                "level": l, "n_super": recs[0]["levels"][l]["n_super"],
                "alpha_l": recs[0]["levels"][l]["alpha_l"],
                "marginal_coverage": agg([r["levels"][l] for r in recs], "marginal_coverage"),
                "min_per_class_coverage": agg([r["levels"][l] for r in recs], "min_per_class_coverage"),
                "mean_set_size": agg([r["levels"][l] for r in recs], "mean_set_size"),
                "frac_singleton": agg([r["levels"][l] for r in recs], "frac_singleton")})
        out["configs"][cname] = d

    # per-cell resolvable scale by K-fold CROSS-FITTING (out-of-fold; no in-sample optimism).
    # Use the PER-SCALE operating regime (each level a valid 1-alpha predictor) so singletons
    # exist -- the family-wise Bonferroni regime (alpha/L) is deliberately too conservative to
    # ever single out a fine type. This predictor is still per-level valid at 1-alpha.
    rng = np.random.default_rng(seed0)
    n = len(y); folds = rng.integers(0, 5, size=n)
    resolv_oof = np.full(n, -1)
    for f in range(5):
        te = folds == f; trc = ~te
        if te.sum() == 0:
            continue
        sncf = ScaleNestedConformal(maps, alpha=alpha, coherent=True, seed=seed0)
        sncf.alpha_l = np.ones(sncf.L) * alpha            # per-scale (not alpha/L)
        sncf.fit(Z[trc], y[trc])
        _, _, rl = sncf.evaluate(Z[te], y[te])
        resolv_oof[te] = rl
    out["resolvable_level_full"] = resolv_oof.tolist()
    out["resolvable_regime"] = "per-scale alpha (each level valid at 1-alpha)"
    # one representative predictor (for taxonomy tree / example sets in the figure)
    snc_rep = ScaleNestedConformal(maps, alpha=alpha, coherent=True, seed=seed0).fit(Z, y)
    return out, snc_rep, Zlink, (Z, y, core)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="*", default=["pancreas", "pbmc3k", "paul15"])
    ap.add_argument("--levels", type=int, default=4)
    ap.add_argument("--splits", type=int, default=25)
    a = ap.parse_args()
    allres = {}
    for nm in (a.names or ["pancreas", "pbmc3k", "paul15"]):
        res, _, _, _ = run_dataset(nm, n_levels=a.levels, n_splits=a.splits)
        allres[nm] = res
        cb = res["configs"]["coherent_bonf"]; unc = res["configs"]["uncorrected"]
        raw = res["configs"]["raw_bonf"]
        print(f"\n=== {nm}  taxonomy {res['taxonomy_sizes']}  (held-out, {res['n_splits']} splits) ===")
        for lv in cb["levels"]:
            print(f"  L{lv['level']} ({lv['n_super']:2d} cls, a={lv['alpha_l']:.3f}): "
                  f"cov={lv['marginal_coverage']['mean']:.3f}+/-{lv['marginal_coverage']['std']:.3f} "
                  f"minPC={lv['min_per_class_coverage']['mean']:.3f} "
                  f"set={lv['mean_set_size']['mean']:.2f} singl={lv['frac_singleton']['mean']:.2f}")
        print(f"  SIMULTANEOUS: coherent+Bonf={cb['simultaneous_coverage']['mean']:.3f} "
              f"(target {cb['simultaneous_target']:.2f}) | "
              f"uncorrected={unc['simultaneous_coverage']['mean']:.3f} "
              f"| coherence: coherent={cb['coherence_rate']['mean']:.3f} raw={raw['coherence_rate']['mean']:.3f}")
    json.dump(allres, open(os.path.join(RESULTS, "scale_nested_conformal.json"), "w"), indent=2)
    print("\nwrote results/scale_nested_conformal.json")
