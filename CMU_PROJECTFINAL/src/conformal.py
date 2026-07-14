"""
conformal.py — calibrated confidence sets for every cell (Mondrian conformal, LAC score).

Split conformal prediction with a class-conditional (Mondrian) calibration, using the
least-ambiguous set-valued (LAC) nonconformity score s(x,y)=1-phat(y|x) (Sadinle et al. 2019).

Per-class threshold (finite-sample corrected):
    q_y = Quantile_{ceil((n_y+1)(1-alpha))/n_y} of { s(x_i,y) : calib cells with label y }.
Prediction set:
    C(x) = { y : phat(y|x) >= 1 - q_y }.
Guarantee: P(y in C(x) | Y=y) >= 1-alpha for every class y (class-conditional coverage).

Interpretation used in the paper:
  |C(x)| = 1  -> confident call
  |C(x)| > 1  -> AMBIGUOUS (cell sits between types, e.g. a transition/aberrant state)
  |C(x)| = 0  -> NOVEL / out-of-distribution (conforms to no known type)
"""
from __future__ import annotations
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


def conformal_quantile(scores, alpha):
    """Finite-sample split-conformal threshold: the R-th smallest score,
    R = ceil((n+1)(1-alpha)).  If R > n the group is too small to certify at level
    (1-alpha): return 1.0 so the class is always included (coverage 1), which is the
    correct behaviour of the exchangeability bound -- NOT max(scores), which under-covers.
    """
    scores = np.asarray(scores, dtype=float)
    n = len(scores)
    if n == 0:
        return 1.0
    R = int(np.ceil((n + 1) * (1.0 - alpha)))
    if R > n:
        return 1.0
    return float(np.sort(scores)[R - 1])


class MondrianConformal:
    def __init__(self, alpha=0.1, C=1.0, seed=0):
        self.alpha = alpha
        self.seed = seed
        self.clf = LogisticRegression(max_iter=2000, C=C, multi_class="multinomial")
        self.scaler = StandardScaler()

    def fit(self, Z, y, calib_frac=0.5):
        """Train classifier on a train split, calibrate thresholds on a calib split.

        Robust to classes present in calibration but absent from training (e.g. when
        calibrating on a healthy subset): such calibration points are dropped, and the
        prediction-set candidates are exactly the classes the classifier was trained on.
        """
        rng = np.random.default_rng(self.seed)
        y = np.asarray(y)
        idx = rng.permutation(len(y))
        ncal = int(len(y) * calib_frac)
        cal, tr = idx[:ncal], idx[ncal:]
        Ztr = self.scaler.fit_transform(Z[tr])
        self.clf.fit(Ztr, y[tr])
        self.classes_ = self.clf.classes_                     # only trainable classes
        col = {c: j for j, c in enumerate(self.clf.classes_)}
        Zcal = self.scaler.transform(Z[cal])
        P = self.clf.predict_proba(Zcal)
        ycal = y[cal]
        keep = np.array([c in col for c in ycal])
        scores = 1 - P[np.arange(len(cal)), [col.get(c, 0) for c in ycal]]
        self.q_ = {}
        for c in self.classes_:
            m = keep & (ycal == c)
            sc = scores[m]
            self.q_[c] = conformal_quantile(sc, self.alpha)
        self._col = col
        self._train_idx, self._cal_idx = tr, cal
        return self

    def predict_sets(self, Z):
        """Return list of prediction sets (arrays of class labels) + probability matrix."""
        P = self.clf.predict_proba(self.scaler.transform(Z))
        sets = []
        for i in range(P.shape[0]):
            keep = [c for c in self.clf.classes_ if P[i, self._col[c]] >= 1 - self.q_[c]]
            sets.append(np.array(keep, dtype=object))
        return sets, P

    def summarize(self, Z, y=None):
        sets, P = self.predict_sets(Z)
        sizes = np.array([len(s) for s in sets])
        out = dict(
            mean_set_size=float(sizes.mean()),
            frac_singleton=float(np.mean(sizes == 1)),
            frac_ambiguous=float(np.mean(sizes > 1)),
            frac_novel=float(np.mean(sizes == 0)),
            sizes=sizes, sets=sets, proba=P, classes=self.clf.classes_,
        )
        if y is not None:
            y = np.asarray(y)
            covered = np.array([y[i] in set(sets[i].tolist()) for i in range(len(y))])
            out["marginal_coverage"] = float(covered.mean())
            # per-class coverage (on all cells)
            pcc = {}
            for c in self.classes_:
                m = y == c
                pcc[str(c)] = float(covered[m].mean()) if m.sum() else float("nan")
            out["per_class_coverage"] = pcc
        return out


if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(__file__))
    from data import load
    from nbvae import fit_nbvae
    A = load("pbmc3k"); y = A.obs["celltype"].astype(str).values
    model, Z, lib = fit_nbvae(A, epochs=150, verbose=False)
    mc = MondrianConformal(alpha=0.1).fit(Z, y)
    s = mc.summarize(Z, y)
    print("alpha=0.1 -> marginal coverage %.3f (target >=0.90)" % s["marginal_coverage"])
    print("mean set size %.2f | singleton %.2f | ambiguous %.2f | novel %.2f"
          % (s["mean_set_size"], s["frac_singleton"], s["frac_ambiguous"], s["frac_novel"]))
