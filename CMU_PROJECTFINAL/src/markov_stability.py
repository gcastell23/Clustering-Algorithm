"""
markov_stability.py — multiscale community detection with no resolution knob.

We let a continuous-time random walk explore the Fisher-Rao graph and keep the partitions
that stay stable across Markov time.  Formulation (Delvenne-Yaliraki-Barahona 2010):

  M = D^-1 A (random-walk operator), L_rw = I - M, stationary pi_i = d_i/2m.
  Transition P(t) = exp(-t L_rw).  Clustered stability of partition H at time t:
      r(t,H) = trace( H^T [ Pi P(t) - pi pi^T ] H ).

Key identity we use for an exact, optimiser-friendly implementation:
  the chain is reversible (A symmetric) so F(t) := Pi P(t) is SYMMETRIC and non-negative,
  and the graph with adjacency  A'(t) = 2m * F(t)  has the SAME degrees d_i as A.  Hence
      r(t,H) = (1/2m) * [ RB-modularity(gamma=1) of A'(t) ],
  which leidenalg maximises directly.  We build F(t) exactly from the eigendecomposition
  of the symmetric normalised Laplacian:
      L_sym = I - D^-1/2 A D^-1/2 = U diag(Lambda) U^T,
      A'(t) = W diag(exp(-t Lambda)) W^T,   W = D^1/2 U.

Robust scales are the Markov times where (i) the number of communities forms a plateau and
(ii) the variation of information between independent optimiser seeds dips.
"""
from __future__ import annotations
import numpy as np
from scipy import sparse
import igraph as ig
import leidenalg as la


def _laplacian_eig(A):
    """Eigendecomposition of the symmetric normalised Laplacian of adjacency A."""
    A = sparse.csr_matrix(A).astype(np.float64)
    d = np.asarray(A.sum(1)).ravel()
    d = np.clip(d, 1e-12, None)
    twom = d.sum()
    Dm12 = sparse.diags(1.0 / np.sqrt(d))
    Lsym = sparse.eye(A.shape[0]) - Dm12 @ A @ Dm12
    Lsym = 0.5 * (Lsym + Lsym.T)                     # numerical symmetry
    lam, U = np.linalg.eigh(Lsym.toarray())
    lam = np.clip(lam, 0, None)
    W = (np.sqrt(d)[:, None]) * U                    # D^1/2 U
    return lam, U, W, d, twom


def flow_graph(lam, W, t, keep_per_row=40, tol=1e-9, n_eig=None):
    """
    Sparse A'(t) = W diag(exp(-t lam)) W^T, kept to top `keep_per_row` per row.

    Low-rank truncation: modes with large lambda decay as exp(-t lambda) and contribute
    nothing at the Markov times of interest, so we keep only the n_eig smallest-lambda modes
    (eigh returns ascending lambda).  This turns the per-time cost from O(n^3) to O(n^2 n_eig)
    and the top-k extraction is vectorised per block.
    """
    n = W.shape[0]
    K = min(n_eig or n, W.shape[1])
    Wk = W[:, :K]
    E = Wk * np.exp(-t * lam[:K])[None, :]           # (n,K)
    kpr = min(keep_per_row, n - 1)
    bs = 1024
    rows, cols, vals = [], [], []
    for s in range(0, n, bs):
        blk = E[s:s + bs] @ Wk.T                      # (b,n)
        np.clip(blk, 0, None, out=blk)
        b = blk.shape[0]
        # zero the diagonal within this block
        di = np.arange(b)
        blk[di, s + di] = 0.0
        if kpr < n - 1:
            part = np.argpartition(blk, -kpr, axis=1)[:, -kpr:]     # (b,kpr) top indices
        else:
            part = np.tile(np.arange(n), (b, 1))
        rr = np.repeat(np.arange(s, s + b), part.shape[1])
        cc = part.ravel()
        vv = blk[np.repeat(np.arange(b), part.shape[1]), cc]
        m = vv > tol
        rows.append(rr[m]); cols.append(cc[m]); vals.append(vv[m])
    r = np.concatenate(rows); c = np.concatenate(cols); v = np.concatenate(vals)
    Ap = sparse.csr_matrix((v, (r, c)), shape=(n, n))
    Ap = Ap.maximum(Ap.T)                            # symmetrise
    return Ap


def _to_igraph(A):
    A = sparse.triu(A, k=1).tocoo()
    return ig.Graph(n=A.shape[0], edges=list(zip(A.row.tolist(), A.col.tolist())),
                    edge_attrs={"weight": A.data.tolist()}, directed=False)


def variation_of_information(x, y):
    """VI(X,Y) = H(X)+H(Y)-2 I(X,Y), in nats."""
    x = np.asarray(x); y = np.asarray(y); n = len(x)
    from collections import Counter
    def ent(z):
        c = np.array(list(Counter(z).values())) / n
        return -np.sum(c * np.log(c))
    # mutual information
    cont = {}
    for a, b in zip(x, y):
        cont[(a, b)] = cont.get((a, b), 0) + 1
    mi = 0.0
    px = {k: v / n for k, v in Counter(x).items()}
    py = {k: v / n for k, v in Counter(y).items()}
    for (a, b), nab in cont.items():
        pab = nab / n
        mi += pab * np.log(pab / (px[a] * py[b]))
    return ent(x) + ent(y) - 2 * mi


def markov_stability_scan(A, times=None, n_seeds=6, keep_per_row=40, n_eig=None, verbose=True):
    """
    Scan Markov time. Returns dict with arrays over times:
      n_comms (mean), vi (mean pairwise VI across seeds), stability, best_labels (per t).
    """
    if times is None:
        times = np.logspace(-1.5, 2.0, 32)
    lam, U, W, d, twom = _laplacian_eig(A)
    results = {"times": np.asarray(times), "n_comms": [], "vi": [], "stability": [],
               "labels": []}
    for ti, t in enumerate(times):
        Ap = flow_graph(lam, W, t, keep_per_row=keep_per_row, n_eig=n_eig)
        g = _to_igraph(Ap)
        parts, ncs = [], []
        for s in range(n_seeds):
            p = la.find_partition(g, la.RBConfigurationVertexPartition,
                                  weights="weight", resolution_parameter=1.0, seed=s)
            parts.append(np.array(p.membership)); ncs.append(len(set(p.membership)))
        # pairwise VI across seeds
        vis = [variation_of_information(parts[i], parts[j])
               for i in range(n_seeds) for j in range(i + 1, n_seeds)]
        # stability value at the best (seed 0) partition: trace(H^T B(t) H)/2m
        H = parts[0]
        stab = _stability_value(lam, W, d, twom, t, H)
        results["n_comms"].append(float(np.mean(ncs)))
        results["vi"].append(float(np.mean(vis)) if vis else 0.0)
        results["stability"].append(float(stab))
        results["labels"].append(parts[int(np.argmin(ncs))])  # representative
        if verbose and (ti % 6 == 0 or ti == len(times) - 1):
            print("  t=%8.3f  <N>=%5.1f  VI=%.3f  R=%.4f" %
                  (t, np.mean(ncs), np.mean(vis) if vis else 0, stab), flush=True)
    for k in ["n_comms", "vi", "stability"]:
        results[k] = np.asarray(results[k])
    return results


def _stability_value(lam, W, d, twom, t, H):
    """r(t,H) = (1/2m) sum_c [ sum_{i,j in c} A'(t)_ij ] - sum_c (vol_c/2m)^2."""
    labels = np.asarray(H)
    e = np.exp(-t * lam)
    # within-community sum of A'(t): sum_c (sum_{i in c} W_i)^2-weighted by e
    stab = 0.0
    vol_term = 0.0
    for c in np.unique(labels):
        m = labels == c
        wc = W[m].sum(0)                       # sum of rows of W in community c  (len n_eig)
        stab += np.sum(e * wc * wc)            # = sum_{i,j in c} A'(t)_ij
        vol_c = d[m].sum()
        vol_term += (vol_c / twom) ** 2
    return stab / twom - vol_term


def select_robust_scales(results, min_comms=2, vi_quantile=0.25):
    """Return indices of robust Markov times: low VI plateaus with >1 community."""
    vi = results["vi"]; nc = results["n_comms"]
    mask = nc >= min_comms
    if mask.sum() == 0:
        return []
    thr = np.quantile(vi[mask], vi_quantile)
    robust = np.where(mask & (vi <= thr))[0]
    return robust.tolist()


if __name__ == "__main__":
    import sys, os, time as _t
    sys.path.insert(0, os.path.dirname(__file__))
    from data import load
    from nbvae import fit_nbvae
    from geometry import compute_metric_tensors
    from graph import build_fr_graph
    from evaluate import basic_metrics, rare_recall
    A = load("pbmc3k"); y = A.obs["celltype"].astype(str).values
    model, Z, lib = fit_nbvae(A, epochs=150, verbose=False)
    G, *_ = compute_metric_tensors(model, Z, lib)
    Afr = build_fr_graph(Z, G, k=15)
    t0 = _t.time()
    res = markov_stability_scan(Afr, times=np.logspace(-1.5, 2.0, 24), n_seeds=5)
    print("scan time %.1fs" % (_t.time() - t0))
    robust = select_robust_scales(res)
    print("robust scale indices:", robust)
    for i in robust:
        yp = res["labels"][i]
        m = basic_metrics(y, yp); rr, _ = rare_recall(y, yp)
        print("  t=%.2f  N=%d  ARI=%.3f rareF1=%.3f" %
              (res["times"][i], m["n_clusters"], m["ARI"], rr))
