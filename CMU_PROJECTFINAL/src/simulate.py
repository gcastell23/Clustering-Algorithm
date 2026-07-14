"""
simulate.py — planted-ground-truth NB single-cell simulation (no label circularity).

Generates a mixture of discrete cell types at controlled abundances (including very rare
populations) plus an optional continuous trajectory, under a negative-binomial count model.
This is the clean causal testbed for the metric ablation: because the labels are planted,
"does the NB Fisher-Rao geometry recover rare populations?" has an unambiguous answer.
"""
from __future__ import annotations
import numpy as np
import anndata as ad
from scipy import sparse


def simulate_rare(n_cells=5000, n_genes=1200, n_types=9, n_marker=30,
                  rare_abundances=(0.004, 0.008, 0.02, 0.05), theta=0.5,
                  lib_mu=6.6, lib_sigma=0.45, de_strength=1.6, seed=0):
    """
    n_types distinct cell types (each with its own marker block); the last
    len(rare_abundances) are RARE.  The difficulty is realistic: shallow libraries
    (lib_mu low) and strong overdispersion (theta small), so log-normalization + PCA amplify
    noise in the few, lowly-sampled rare cells and lose them, while the NB model does not.
    """
    rng = np.random.default_rng(seed)
    n_rare = len(rare_abundances)
    n_common = n_types - n_rare
    common_ab = (1 - sum(rare_abundances)) / n_common
    abund = np.array([common_ab] * n_common + list(rare_abundances))
    abund = abund / abund.sum()
    counts_per_type = rng.multinomial(n_cells, abund)

    base = rng.normal(-1.0, 1.0, n_genes)
    programs = np.tile(base, (n_types, 1))
    perm = rng.permutation(n_genes)
    for k in range(n_types):                          # every type distinct, incl. rare
        mk = perm[(k * n_marker) % n_genes:(k * n_marker) % n_genes + n_marker]
        programs[k, mk] += rng.normal(de_strength, 0.3, len(mk))

    labels, rows = [], []
    for k, nk in enumerate(counts_per_type):
        if nk == 0:
            continue
        rho = np.exp(programs[k]); rho = rho / rho.sum()
        libs = rng.lognormal(lib_mu, lib_sigma, nk)
        mu = libs[:, None] * rho[None, :]
        # NB sample: Gamma-Poisson
        shape = theta
        gam = rng.gamma(shape, mu / shape)
        x = rng.poisson(gam)
        rows.append(x)
        labels += [f"type{k}" + ("_rare" if k >= n_common else "")] * nk
    X = np.vstack(rows).astype(np.float32)
    A = ad.AnnData(X=sparse.csr_matrix(X))
    A.obs["celltype"] = np.array(labels)
    A.obs["batch"] = "sim"
    A.obs_names = [f"cell{i}" for i in range(A.n_obs)]
    A.var_names = [f"gene{j}" for j in range(n_genes)]
    A.obs["celltype"] = A.obs["celltype"].astype("category")
    A.obs["batch"] = A.obs["batch"].astype("category")
    return A


def preprocess_sim(A, n_hvg=1000):
    """Attach the layers/obs the pipeline expects (counts, log1p X, size factors, HVG)."""
    import scanpy as sc
    A = A.copy()
    A.layers["counts_full"] = A.X.copy()
    sc.pp.highly_variable_genes(A, n_top_genes=min(n_hvg, A.n_vars - 1),
                                flavor="seurat_v3", layer="counts_full", subset=True)
    A.layers["counts"] = A.X.copy()
    n_counts = np.asarray(A.layers["counts_full"].sum(1)).ravel() if "counts_full" in A.layers \
        else np.asarray(A.X.sum(1)).ravel()
    sc.pp.normalize_total(A, target_sum=1e4); sc.pp.log1p(A)
    A.obs["n_counts"] = n_counts.astype(np.float32)
    del A.layers["counts_full"]
    A.uns["dataset"] = "simulation"; A.uns["structure"] = "discrete"
    return A


if __name__ == "__main__":
    A = simulate_rare()
    print("sim:", A.shape, A.obs["celltype"].value_counts().to_dict())
