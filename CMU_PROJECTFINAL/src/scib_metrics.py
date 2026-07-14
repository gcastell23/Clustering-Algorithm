"""
scib_metrics.py — a transparent, dependency-light re-implementation of the scIB metric battery.

We compute these from sklearn/scipy primitives (rather than the `scib` package, whose deps do not
build on this machine) so every number is auditable.  Two families, following Luecken et al. 2022:

BIO-CONSERVATION (does the embedding preserve cell-type structure?  higher = better):
  NMI, ARI            : label vs Leiden clustering optimised over resolution (scIB opt_louvain).
  ASW_celltype        : silhouette by cell type, rescaled to [0,1].
  isolated_label_F1   : best cluster F1 for the label(s) present in the fewest batches.
  cLISI               : cell-type local inverse-Simpson index, normalised so 1 = pure neighbourhoods.

BATCH-CORRECTION (are technical batches mixed?  higher = better):
  ASW_batch           : 1 - |silhouette by batch|, averaged within cell type, rescaled to [0,1].
  graph_connectivity  : mean over types of the largest-connected-component fraction on the kNN graph.
  iLISI               : batch local inverse-Simpson index, normalised so 1 = fully mixed.

Overall = 0.6 * mean(bio) + 0.4 * mean(batch)   (the scIB weighting).
"""
from __future__ import annotations
import numpy as np
from scipy import sparse
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import (silhouette_samples, adjusted_rand_score,
                             normalized_mutual_info_score)


def _knn(emb, k=90):
    nn = NearestNeighbors(n_neighbors=min(k + 1, len(emb))).fit(emb)
    dist, idx = nn.kneighbors(emb)
    return idx[:, 1:], dist[:, 1:]


def _leiden_opt(emb, labels, seed=0):
    """Best NMI/ARI over a Leiden resolution sweep on a Euclidean kNN graph (scIB opt_louvain)."""
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from graph import build_euclidean_graph, leiden
    A = build_euclidean_graph(emb, k=15)
    best = (-1, None)
    for r in np.round(np.geomspace(0.1, 2.5, 12), 3):
        yp = leiden(A, resolution=float(r), seed=seed)
        nmi = normalized_mutual_info_score(labels, yp)
        if nmi > best[0]:
            best = (nmi, yp)
    yp = best[1]
    return {"NMI": float(best[0]), "ARI": float(adjusted_rand_score(labels, yp))}, yp


def _lisi(emb, cats, k=90):
    """Median local inverse-Simpson index of a categorical over kNN neighbourhoods."""
    idx, _ = _knn(emb, k=k)
    cats = np.asarray(cats)
    uniq = {c: i for i, c in enumerate(np.unique(cats))}
    codes = np.array([uniq[c] for c in cats])
    C = len(uniq)
    inv = np.empty(len(emb))
    for i in range(len(emb)):
        nb = codes[idx[i]]
        p = np.bincount(nb, minlength=C) / len(nb)
        inv[i] = 1.0 / np.sum(p ** 2)
    return float(np.median(inv)), C


def _asw_celltype(emb, labels):
    s = silhouette_samples(emb, labels)
    return float((np.mean(s) + 1) / 2)


def _asw_batch(emb, labels, batch):
    """scIB batch ASW: within each cell type, 1-|sil by batch|; average, rescale to [0,1]."""
    labels = np.asarray(labels); batch = np.asarray(batch)
    scores = []
    for t in np.unique(labels):
        m = labels == t
        if m.sum() < 10 or len(np.unique(batch[m])) < 2:
            continue
        s = silhouette_samples(emb[m], batch[m])
        scores.append(np.mean(1 - np.abs(s)))
    return float(np.mean(scores)) if scores else float("nan")


def _graph_connectivity(emb, labels, k=15):
    from scipy.sparse.csgraph import connected_components
    idx, _ = _knn(emb, k=k)
    labels = np.asarray(labels); n = len(labels)
    rows = np.repeat(np.arange(n), idx.shape[1]); cols = idx.ravel()
    A = sparse.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))
    A = A.maximum(A.T)
    fracs = []
    for t in np.unique(labels):
        m = np.where(labels == t)[0]
        if len(m) < 2:
            continue
        sub = A[m][:, m]
        ncomp, lab = connected_components(sub, directed=False)
        _, cnt = np.unique(lab, return_counts=True)
        fracs.append(cnt.max() / len(m))
    return float(np.mean(fracs)) if fracs else float("nan")


def _isolated_label_f1(emb, labels, batch, yp):
    """Best cluster-F1 for the label(s) present in the fewest batches (scIB isolated label)."""
    labels = np.asarray(labels); batch = np.asarray(batch); yp = np.asarray(yp)
    nb = {t: len(np.unique(batch[labels == t])) for t in np.unique(labels)}
    mn = min(nb.values())
    iso = [t for t, v in nb.items() if v == mn]
    f1s = []
    for t in iso:
        tm = labels == t
        best = 0.0
        for c in np.unique(yp):
            cm = yp == c
            tp = np.sum(tm & cm)
            if tp == 0:
                continue
            prec = tp / cm.sum(); rec = tp / tm.sum()
            best = max(best, 2 * prec * rec / (prec + rec))
        f1s.append(best)
    return float(np.mean(f1s)) if f1s else float("nan")


def score_embedding(emb, celltype, batch, seed=0, k=90):
    """Full scIB battery for one embedding. Returns dict of metrics + bio/batch/overall."""
    emb = np.asarray(emb, dtype=np.float64)
    celltype = np.asarray([str(x) for x in celltype]); batch = np.asarray([str(x) for x in batch])
    clu, yp = _leiden_opt(emb, celltype, seed=seed)
    clisi, C = _lisi(emb, celltype, k=k); ilisi, B = _lisi(emb, batch, k=k)
    bio = {
        "NMI": clu["NMI"], "ARI": clu["ARI"],
        "ASW_celltype": _asw_celltype(emb, celltype),
        "isolated_label_F1": _isolated_label_f1(emb, celltype, batch, yp),
        "cLISI": float((C - clisi) / (C - 1)) if C > 1 else 1.0,       # 1 = pure
    }
    bat = {
        "ASW_batch": _asw_batch(emb, celltype, batch),
        "graph_connectivity": _graph_connectivity(emb, celltype),
        "iLISI": float((ilisi - 1) / (B - 1)) if B > 1 else float("nan"),  # 1 = mixed
    }
    biov = np.nanmean([v for v in bio.values()])
    batv = np.nanmean([v for v in bat.values()])
    out = {**{f"bio/{k_}": v for k_, v in bio.items()},
           **{f"batch/{k_}": v for k_, v in bat.items()},
           "bio_conservation": float(biov), "batch_correction": float(batv),
           "overall": float(0.6 * biov + 0.4 * batv)}
    return out
