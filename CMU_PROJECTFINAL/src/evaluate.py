"""
evaluate.py — clustering metrics with a focus on RARE populations.

Standard metrics (ARI/NMI/homogeneity) reward getting the big clusters right and are almost
blind to a 15-cell population.  We therefore also report:
  * rare_recall  : mean best-match F1 over the rarest tertile of cell types
  * per_type_f1  : Hungarian-matched F1 for every ground-truth type
which is exactly where an information-geometric ruler is supposed to help.
"""
from __future__ import annotations
import numpy as np
from sklearn.metrics import (adjusted_rand_score, normalized_mutual_info_score,
                             homogeneity_score, completeness_score, silhouette_score,
                             adjusted_mutual_info_score)
from scipy.optimize import linear_sum_assignment


def basic_metrics(y_true, y_pred):
    return dict(
        ARI=float(adjusted_rand_score(y_true, y_pred)),
        NMI=float(normalized_mutual_info_score(y_true, y_pred)),
        AMI=float(adjusted_mutual_info_score(y_true, y_pred)),
        homogeneity=float(homogeneity_score(y_true, y_pred)),
        completeness=float(completeness_score(y_true, y_pred)),
        n_clusters=int(len(np.unique(y_pred))),
    )


def per_type_f1(y_true, y_pred):
    """Best-match F1 per true type via Hungarian assignment on the F1 matrix."""
    types = np.unique(y_true)
    clusters = np.unique(y_pred)
    F = np.zeros((len(types), len(clusters)))
    for a, t in enumerate(types):
        tmask = y_true == t
        for b, c in enumerate(clusters):
            cmask = y_pred == c
            tp = np.sum(tmask & cmask)
            if tp == 0:
                continue
            prec = tp / cmask.sum()
            rec = tp / tmask.sum()
            F[a, b] = 2 * prec * rec / (prec + rec)
    # maximise total F1 (pad to square handled by linear_sum_assignment on -F)
    ri, ci = linear_sum_assignment(-F)
    f1 = {}
    for a in range(len(types)):
        j = ci[list(ri).index(a)] if a in ri else None
        f1[str(types[a])] = float(F[a, j]) if j is not None else 0.0
    return f1


def rare_recall(y_true, y_pred, frac=1 / 3):
    """Mean matched-F1 over the rarest `frac` of cell types (by abundance)."""
    types, counts = np.unique(y_true, return_counts=True)
    order = np.argsort(counts)
    n_rare = max(1, int(np.ceil(len(types) * frac)))
    rare_types = set(types[order][:n_rare].tolist())
    f1 = per_type_f1(y_true, y_pred)
    vals = [f1[str(t)] for t in rare_types]
    return float(np.mean(vals)), {str(t): f1[str(t)] for t in rare_types}


def embedding_silhouette(Z, y_pred, metric="euclidean", sample=4000, seed=0):
    if len(np.unique(y_pred)) < 2:
        return float("nan")
    rng = np.random.default_rng(seed)
    if Z.shape[0] > sample:
        sel = rng.choice(Z.shape[0], sample, replace=False)
        Z, y_pred = Z[sel], np.asarray(y_pred)[sel]
    try:
        return float(silhouette_score(Z, y_pred, metric=metric))
    except Exception:
        return float("nan")


def resolution_sweep(A, y_true, resolutions, objective="modularity", seed=0):
    """Cluster A at each resolution; return list of dicts with metrics + rare recall."""
    from graph import leiden
    out = []
    for r in resolutions:
        yp = leiden(A, resolution=r, seed=seed, objective=objective)
        m = basic_metrics(y_true, yp)
        rr, _ = rare_recall(y_true, yp)
        m.update(resolution=float(r), rare_recall=rr, labels=yp)
        out.append(m)
    return out


def best_at_matched_k(sweep, k_target):
    """Pick the sweep entry whose n_clusters is closest to k_target (tie -> higher ARI)."""
    return sorted(sweep, key=lambda m: (abs(m["n_clusters"] - k_target), -m["ARI"]))[0]
