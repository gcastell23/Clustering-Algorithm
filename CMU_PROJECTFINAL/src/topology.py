"""
topology.py — persistent homology to decide: discrete cell type or point on a continuum?

Intuition:
  * A discrete cell type is a compact blob: points merge (H0) at a single small scale, and
    it is separated from its neighbours by a large density GAP.  Negligible H1.
  * A continuous trajectory is a filament / branched arc / loop: H0 deaths are spread out,
    there is NO separating gap to neighbours, and cyclic structure yields persistent H1.

We use two complementary, interpretable statistics computed from Vietoris-Rips persistence
(ripser) under the Fisher-Rao distance:

  (1) global manifold H1 signal  -> is the whole dataset a looped/branched continuum?
  (2) per-cluster separation ratio rho_c = (gap to merge with rest) / (within-cluster scale)
      and a continuity score.  Large rho_c => discrete; rho_c ~ 1 => continuous.

Combined into tau_c in [0,1]: high = discrete, low = continuous.
"""
from __future__ import annotations
import numpy as np
from ripser import ripser
from scipy.spatial.distance import pdist, squareform


# ---------- Fisher-Rao distance matrix for a set of cells ----------
def fr_distance_matrix(Zsub, Gsub):
    """Symmetric midpoint Fisher-Rao distance matrix for a subset (m cells)."""
    m = Zsub.shape[0]
    diff = Zsub[:, None, :] - Zsub[None, :, :]              # (m,m,d)
    Gm = 0.5 * (Gsub[:, None, :, :] + Gsub[None, :, :, :])  # (m,m,d,d)
    D2 = np.einsum("ijd,ijde,ije->ij", diff, Gm, diff)
    D2 = np.clip(D2, 0, None)
    D = np.sqrt(D2)
    return 0.5 * (D + D.T)


# ---------- persistence ----------
def persistence(D, maxdim=1):
    """Vietoris-Rips persistence from a distance matrix. Returns dict of H0,H1 arrays."""
    res = ripser(D, maxdim=maxdim, distance_matrix=True)
    dgms = res["dgms"]
    out = {}
    for k, dg in enumerate(dgms):
        dg = dg[np.isfinite(dg[:, 1])]                      # drop essential (inf) bars
        out[f"H{k}"] = dg
    return out


def _lifetimes(dg):
    return (dg[:, 1] - dg[:, 0]) if len(dg) else np.array([])


def h0_death_gap(dgH0, scale):
    """
    Largest relative gap in the sorted H0 death (merge) scales.

    k well-separated discrete clusters -> the k-1 'component-connecting' bars are much longer
    than the intra-cluster bars, leaving a big GAP in the death sequence (high value).
    A continuous manifold merges smoothly -> deaths spread evenly -> small value.
    Returns (gap_ratio, n_significant_components, normalized_deaths).
    """
    deaths = np.sort(dgH0[:, 1])
    deaths = deaths[np.isfinite(deaths) & (deaths > 0)]
    if len(deaths) < 5:
        return 0.0, 1, deaths / (scale + 1e-9)
    dn = deaths / (scale + 1e-9)
    gaps = np.diff(dn)
    gmax = gaps.max()
    ratio = float(gmax / (np.mean(gaps) + 1e-12))
    # number of components separated by the dominant gap (bars above the gap location)
    cut = dn[np.argmax(gaps)]
    n_sig = int(np.sum(dn > cut) + 1)
    return ratio, n_sig, dn


def global_topology(Z, G=None, sample=1200, seed=0):
    """Global H0/H1 of the manifold. Returns diagram + summary stats."""
    rng = np.random.default_rng(seed)
    n = Z.shape[0]
    idx = rng.choice(n, min(sample, n), replace=False)
    if G is not None:
        D = fr_distance_matrix(Z[idx], G[idx])
    else:
        D = squareform(pdist(Z[idx]))
    scale = np.median(D[D > 0])
    dg = persistence(D, maxdim=1)
    h1 = _lifetimes(dg["H1"])
    h0 = _lifetimes(dg["H0"])
    gap_ratio, n_sig, deaths_norm = h0_death_gap(dg["H0"], scale)
    stats = dict(
        scale=float(scale),
        h0_gap_ratio=float(gap_ratio),          # high = discrete, low = continuous
        n_significant=int(n_sig),
        h1_total=float(h1.sum() / scale) if len(h1) else 0.0,
        h1_max=float(h1.max() / scale) if len(h1) else 0.0,
        h1_count=int((h1 > 0.5 * scale).sum()),
        h0_spread=float(np.std(h0) / scale) if len(h0) else 0.0,
        n_points=len(idx),
    )
    dg["deaths_norm"] = deaths_norm
    return dg, stats, idx


def transitionality(Z, labels, G=None, k=30, purity_thresh=0.7):
    """
    Label-free-ish topological boundary detection under the Fisher-Rao graph.

    A cell is TRANSITIONAL if its k Fisher-Rao neighbours span >1 cluster (purity < thresh):
    such cells are the H0 'bridges' that knit otherwise-separate populations together, the
    signature of a continuum rather than discrete types.

    Returns:
      per_cell_purity : (N,) fraction of same-label neighbours
      transitional    : (N,) bool
      continuity_index: dataset fraction of transitional cells (high = continuous)
      per_cluster      : {cluster: {'transitional_frac', 'mean_purity', 'n'}}
    """
    from geometry import fr_knn_indices
    labels = np.asarray([str(x) for x in labels])
    if G is not None:
        knn = fr_knn_indices(Z, G, k=k)
    else:
        from sklearn.neighbors import NearestNeighbors
        knn = NearestNeighbors(n_neighbors=k + 1).fit(Z).kneighbors(Z)[1][:, 1:]
    same = labels[knn] == labels[:, None]
    purity = same.mean(1)
    transitional = purity < purity_thresh
    per_cluster = {}
    for c in np.unique(labels):
        m = labels == c
        per_cluster[c] = dict(transitional_frac=float(transitional[m].mean()),
                              mean_purity=float(purity[m].mean()), n=int(m.sum()))
    return dict(per_cell_purity=purity, transitional=transitional,
                continuity_index=float(transitional.mean()), per_cluster=per_cluster)


def dataset_discreteness(Z, labels, G=None, k=30):
    """Dataset discreteness index = 1 - continuity_index (high = discrete)."""
    tr = transitionality(Z, labels, G=G, k=k)
    return 1.0 - tr["continuity_index"], tr


def cluster_separation(Z, labels, G=None, cluster=None, sample=400, seed=0):
    """
    For a cluster, compute the separation ratio rho = gap / within-scale via H0.

    within-scale = median H0 lifetime inside the cluster (how fast the blob knits together)
    gap          = extra radius needed to connect the cluster to its nearest other cells,
                   i.e. (nearest cross-cluster distance) relative to within-scale.
    rho >> 1 : isolated blob (discrete);  rho ~ 1 : merges immediately (continuous).
    """
    rng = np.random.default_rng(seed)
    labels = np.asarray(labels)
    members = np.where(labels == cluster)[0]
    others = np.where(labels != cluster)[0]
    if len(members) < 5 or len(others) < 5:
        return None
    if len(members) > sample:
        members = rng.choice(members, sample, replace=False)
    # within-cluster persistence
    if G is not None:
        Dw = fr_distance_matrix(Z[members], G[members])
    else:
        Dw = squareform(pdist(Z[members]))
    within = np.median(Dw[Dw > 0])
    dgw = persistence(Dw, maxdim=0)
    h0w = _lifetimes(dgw["H0"])
    knit = float(np.quantile(h0w, 0.9)) if len(h0w) else within  # scale to connect blob
    # nearest cross-cluster distance (gap), in Fisher-Rao if available
    oth = others if len(others) <= 3000 else rng.choice(others, 3000, replace=False)
    if G is not None:
        # midpoint metric member x other
        diff = Z[members][:, None, :] - Z[oth][None, :, :]
        Gm = 0.5 * (G[members][:, None] + G[oth][None, :])
        cross = np.sqrt(np.clip(np.einsum("ijd,ijde,ije->ij", diff, Gm, diff), 0, None))
    else:
        d = Z[members][:, None, :] - Z[oth][None, :, :]
        cross = np.linalg.norm(d, axis=2)
    gap = float(np.min(cross.min(1)))          # closest approach to any other cell
    rho = gap / (knit + 1e-9)
    return dict(cluster=cluster, within=within, knit=knit, gap=gap, rho=float(rho),
               n=len(members))


def discreteness_score(rho, k=1.0, midpoint=1.0):
    """Map separation ratio rho -> tau in [0,1] (high = discrete). Logistic in log(rho)."""
    return float(1.0 / (1.0 + np.exp(-k * (np.log(rho + 1e-9) - np.log(midpoint)))))


def per_cluster_topology(Z, labels, G=None):
    """Separation ratio + discreteness for every cluster."""
    out = {}
    for c in np.unique(labels):
        r = cluster_separation(Z, labels, G=G, cluster=c)
        if r is None:
            continue
        r["tau"] = discreteness_score(r["rho"])
        out[c] = r
    return out


if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(__file__))
    from data import load
    from nbvae import fit_nbvae
    from geometry import compute_metric_tensors
    for name in ["pbmc3k", "paul15"]:
        A = load(name)
        model, Z, lib = fit_nbvae(A, epochs=120, verbose=False)
        G, *_ = compute_metric_tensors(model, Z, lib)
        _, stats, _ = global_topology(Z, G, sample=1000)
        print(f"[{name}] structure={A.uns.get('structure')}  "
              f"H1_total={stats['h1_total']:.3f}  H1_max={stats['h1_max']:.3f}  "
              f"H1_count={stats['h1_count']}  H0_spread={stats['h0_spread']:.3f}")
