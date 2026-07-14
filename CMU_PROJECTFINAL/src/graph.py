"""
graph.py — neighbour-graph construction on the latent space.

Two graphs:
  * Euclidean kNN (standard baseline).
  * Fisher-Rao kNN: candidate-and-rerank.  We first fetch k_cand Euclidean candidates
    (cheap, sklearn), then re-score them with the local Fisher-Rao distance (midpoint
    metric) and keep the k statistically-nearest.  Cost O(N k_cand d^2), so it scales to
    the pancreas atlas without an N^2 blow-up.

Edges are weighted with an adaptive Gaussian kernel (per-cell bandwidth = distance to the
k-th neighbour), then symmetrised.  We return a scipy CSR adjacency and an igraph.Graph.
"""
from __future__ import annotations
import numpy as np
from scipy import sparse
from sklearn.neighbors import NearestNeighbors
import igraph as ig

from geometry import local_fr_dist2_to_candidates


def _adaptive_kernel(dist2, idx, n, k, symmetrize="mean"):
    """Build symmetric weighted CSR adjacency from per-cell neighbour dist^2 + indices."""
    sigma = np.sqrt(np.clip(dist2[:, min(k, dist2.shape[1]) - 1], 1e-12, None))  # kth-NN scale
    rows = np.repeat(np.arange(n), idx.shape[1])
    cols = idx.ravel()
    d2 = dist2.ravel()
    s_ij = sigma[rows] * sigma[cols] + 1e-12
    w = np.exp(-d2 / s_ij)
    A = sparse.csr_matrix((w, (rows, cols)), shape=(n, n))
    A.setdiag(0); A.eliminate_zeros()
    if symmetrize == "mean":
        A = 0.5 * (A + A.T)
    else:
        A = A.maximum(A.T)
    return A.tocsr()


def build_euclidean_graph(Z, k=15):
    n = Z.shape[0]
    nn = NearestNeighbors(n_neighbors=k + 1).fit(Z)
    dist, idx = nn.kneighbors(Z)
    return _adaptive_kernel(dist[:, 1:] ** 2, idx[:, 1:], n, k)


def build_fr_graph(Z, G, k=15, k_cand=60):
    """Fisher-Rao kNN graph via candidate-and-rerank."""
    n = Z.shape[0]
    k_cand = min(k_cand, n - 1)
    nn = NearestNeighbors(n_neighbors=k_cand + 1).fit(Z)
    _, cand = nn.kneighbors(Z)
    cand = cand[:, 1:]                                     # drop self, (N,k_cand)
    D2 = local_fr_dist2_to_candidates(Z, G, cand)         # (N,k_cand) FR distances
    order = np.argsort(D2, axis=1)[:, :k]                 # k statistically-nearest
    idx = np.take_along_axis(cand, order, axis=1)
    d2 = np.take_along_axis(D2, order, axis=1)
    return _adaptive_kernel(d2, idx, n, k)


def to_igraph(A):
    """scipy CSR (symmetric) -> undirected weighted igraph.Graph."""
    A = sparse.triu(A, k=1).tocoo()
    g = ig.Graph(n=A.shape[0], edges=list(zip(A.row.tolist(), A.col.tolist())),
                 edge_attrs={"weight": A.data.tolist()}, directed=False)
    return g


def leiden(A, resolution=1.0, seed=0, objective="modularity"):
    """Leiden community detection on a weighted adjacency; returns integer labels."""
    import leidenalg as la
    g = to_igraph(A)
    if objective == "cpm":
        part = la.find_partition(g, la.CPMVertexPartition, weights="weight",
                                 resolution_parameter=resolution, seed=seed)
    else:
        part = la.find_partition(g, la.RBConfigurationVertexPartition, weights="weight",
                                 resolution_parameter=resolution, seed=seed)
    return np.array(part.membership)


if __name__ == "__main__":
    import sys, os, time
    sys.path.insert(0, os.path.dirname(__file__))
    from data import load
    from nbvae import fit_nbvae
    from geometry import compute_metric_tensors
    A = load("pbmc3k")
    model, Z, lib = fit_nbvae(A, epochs=60, verbose=False)
    G, *_ = compute_metric_tensors(model, Z, lib)
    t = time.time()
    Aeuc = build_euclidean_graph(Z, k=15)
    Afr = build_fr_graph(Z, G, k=15)
    print("graphs built in %.2fs; nnz euc=%d fr=%d" % (time.time() - t, Aeuc.nnz, Afr.nnz))
    le = leiden(Aeuc); lf = leiden(Afr)
    print("euclidean leiden clusters:", len(np.unique(le)))
    print("fisher-rao leiden clusters:", len(np.unique(lf)))
