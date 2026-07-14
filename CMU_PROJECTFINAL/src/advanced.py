"""
advanced.py — graduate-level, interdisciplinary analyses on the Fisher-Rao cell manifold.

Three independent theoretical lenses, all computed on the same NB-VAE latent + metric:

1. Ollivier-Ricci curvature (optimal transport + Riemannian geometry).  For an edge (x,y),
   kappa = 1 - W1(m_x, m_y) / d(x,y), where m_x is a lazy random-walk measure on x's Fisher-Rao
   neighbourhood and W1 is the Wasserstein-1 distance (Villani; Ollivier 2009; Sandhu 2015 for
   networks).  Negative curvature marks bridges / transitions; positive marks tight communities.

2. Entropic optimal transport of disease (Waddington-OT lineage; Schiebinger 2019).  Per cell type
   we compute the (squared) Wasserstein-2 distance between healthy and diabetic cells in latent
   space, against a within-condition null, quantifying the transcriptional displacement of disease.

3. Spectral / diffusion geometry + random-matrix null.  Eigenspectrum of the diffusion operator
   (metastable states via spectral gaps; Coifman-Lafon) and a Marchenko-Pastur bound separating
   signal principal components from noise.
"""
from __future__ import annotations
import numpy as np
import ot
from scipy import sparse
from sklearn.neighbors import NearestNeighbors
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from geometry import fr_knn_indices, local_fr_dist2_to_candidates


# ---------------- 1. Ollivier-Ricci curvature ----------------
def _fr_dist_block(Z, G, A_idx, B_idx):
    """Fisher-Rao midpoint distances between two index sets -> (|A|,|B|)."""
    diff = Z[A_idx][:, None, :] - Z[B_idx][None, :, :]
    Gm = 0.5 * (G[A_idx][:, None] + G[B_idx][None, :])
    return np.sqrt(np.clip(np.einsum("aid,aide,aie->ai", diff, Gm, diff), 0, None))


def ollivier_ricci(Z, G, k=12, idle=0.5, max_cells=2600, seed=0):
    """Return per-node and per-edge Ollivier-Ricci curvature on the Fisher-Rao kNN graph."""
    rng = np.random.default_rng(seed)
    n = Z.shape[0]
    if n > max_cells:
        idx = np.sort(rng.choice(n, max_cells, replace=False))
        Z, G = Z[idx], G[idx]
    else:
        idx = np.arange(n)
    n = Z.shape[0]
    knn = fr_knn_indices(Z, G, k=k)                       # (n,k)
    # edge FR distances (x -> each neighbour)
    d_nb = _row_fr(Z, G, knn)                             # (n,k)
    node_curv = np.zeros(n)
    node_cnt = np.zeros(n)
    edges, ekappa = [], []
    for x in range(n):
        Nx = knn[x]
        Sx = np.concatenate(([x], Nx))
        mx = np.concatenate(([idle], np.full(k, (1 - idle) / k)))
        for j, y in enumerate(Nx):
            if y <= x:
                continue
            Ny = knn[y]
            Sy = np.concatenate(([y], Ny))
            my = np.concatenate(([idle], np.full(k, (1 - idle) / k)))
            C = _fr_dist_block(Z, G, Sx, Sy)
            W1 = ot.emd2(mx, my, C)
            dxy = d_nb[x, j]
            kap = 1.0 - W1 / (dxy + 1e-12)
            edges.append((x, y)); ekappa.append(kap)
            node_curv[x] += kap; node_curv[y] += kap
            node_cnt[x] += 1; node_cnt[y] += 1
    node_curv = node_curv / np.clip(node_cnt, 1, None)
    return dict(idx=idx, node_curv=node_curv, edges=np.array(edges),
                edge_curv=np.array(ekappa), knn=knn)


def _row_fr(Z, G, knn):
    n, k = knn.shape
    out = np.zeros((n, k))
    for j in range(k):
        nb = knn[:, j]
        diff = Z - Z[nb]
        Gm = 0.5 * (G + G[nb])
        out[:, j] = np.sqrt(np.clip(np.einsum("nd,nde,ne->n", diff, Gm, diff), 0, None))
    return out


# ---------------- 2. Optimal transport of disease ----------------
def ot_disease(Z, y, disease, min_n=12, reg=None, seed=0):
    """
    Per cell type: squared-W2 (or entropic) OT distance healthy<->T2D vs a within-healthy null.
    Returns per-type effect sizes and, for beta cells, the OT coupling for visualisation.
    """
    rng = np.random.default_rng(seed)
    y = np.asarray([str(v) for v in y]); disease = np.asarray([str(v) for v in disease])
    out = {}
    for c in np.unique(y):
        h = np.where((y == c) & (disease == "healthy"))[0]
        t = np.where((y == c) & (disease == "T2D"))[0]
        if len(h) < min_n or len(t) < min_n:
            continue
        Zh, Zt = Z[h], Z[t]
        def W(a, b):
            M = ot.dist(a, b, metric="sqeuclidean")
            wa = np.full(len(a), 1 / len(a)); wb = np.full(len(b), 1 / len(b))
            return (ot.sinkhorn2(wa, wb, M, reg) if reg else ot.emd2(wa, wb, M))
        w_ht = float(W(Zh, Zt))
        # within-healthy null: split healthy in two
        perm = rng.permutation(len(h)); half = len(h) // 2
        w_null = float(W(Zh[perm[:half]], Zh[perm[half:2 * half]])) if half >= min_n // 2 else np.nan
        out[c] = dict(W_ht=w_ht, W_null=w_null,
                      effect=w_ht - (w_null if np.isfinite(w_null) else 0.0),
                      n_healthy=len(h), n_t2d=len(t))
    # beta coupling for the displacement panel
    coupling = None
    hb = np.where((y == "beta") & (disease == "healthy"))[0]
    tb = np.where((y == "beta") & (disease == "T2D"))[0]
    if len(hb) >= min_n and len(tb) >= min_n:
        M = ot.dist(Z[hb], Z[tb], metric="sqeuclidean")
        P = ot.emd(np.full(len(hb), 1 / len(hb)), np.full(len(tb), 1 / len(tb)), M)
        coupling = dict(hb=hb, tb=tb, P=P)
    return out, coupling


# ---------------- 3. Spectral / diffusion geometry + RMT ----------------
def spectral_geometry(Zlatent, Xlognorm, k=15, n_eig=40):
    """Diffusion eigenspectrum + spectral gaps + Marchenko-Pastur signal cut."""
    n = Zlatent.shape[0]
    nn = NearestNeighbors(n_neighbors=k + 1).fit(Zlatent)
    dist, idxs = nn.kneighbors(Zlatent)
    sig = np.median(dist[:, -1])
    rows = np.repeat(np.arange(n), k); cols = idxs[:, 1:].ravel()
    w = np.exp(-(dist[:, 1:].ravel() ** 2) / (sig ** 2 + 1e-12))
    A = sparse.csr_matrix((w, (rows, cols)), shape=(n, n)); A = A.maximum(A.T)
    d = np.asarray(A.sum(1)).ravel(); Dm12 = sparse.diags(1 / np.sqrt(np.clip(d, 1e-12, None)))
    Lsym = sparse.eye(n) - Dm12 @ A @ Dm12
    from scipy.sparse.linalg import eigsh
    lam = np.sort(eigsh(Lsym, k=min(n_eig, n - 2), which="SM", return_eigenvectors=False))
    lam = np.clip(lam, 0, None)
    diff_eig = 1 - lam                                    # diffusion operator eigenvalues
    gaps = np.diff(lam)
    # Marchenko-Pastur on the log-norm data (per-gene standardised)
    X = np.asarray(Xlognorm)
    X = (X - X.mean(0)) / (X.std(0) + 1e-8)
    nS, p = X.shape
    ev = np.linalg.svd(X, compute_uv=False) ** 2 / nS
    gamma = p / nS
    lam_plus = (1 + np.sqrt(gamma)) ** 2
    n_signal = int(np.sum(ev > lam_plus))
    return dict(lap_eig=lam, diff_eig=diff_eig, gaps=gaps,
                pca_eig=ev, mp_plus=float(lam_plus), gamma=float(gamma), n_signal=n_signal)


if __name__ == "__main__":
    from train_cache import load_cache
    c = load_cache("paul15")
    r = ollivier_ricci(c["Z"], c["G"].astype(np.float64), k=12, max_cells=1500)
    print("Ollivier-Ricci: node curv range %.2f..%.2f, %d edges" %
          (r["node_curv"].min(), r["node_curv"].max(), len(r["edges"])))
