"""
sim_experiment.py — metric ablation on planted-ground-truth simulations.

Because the labels are planted (not derived from any clustering), this is the clean causal
test of whether the NB Fisher-Rao geometry, specifically, recovers rare populations.  We
compare, on the SAME NB-VAE latent, three rulers plus PCA/k-means, and measure rare-type
recall as a function of planted abundance.
"""
from __future__ import annotations
import os, sys, json, time
import numpy as np
from scipy import sparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from simulate import simulate_rare, preprocess_sim
from nbvae import fit_nbvae
from geometry import compute_metric_tensors
from graph import build_euclidean_graph, build_fr_graph
from evaluate import basic_metrics, per_type_f1, resolution_sweep, best_at_matched_k

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(HERE, "results", "simulation")
os.makedirs(RES, exist_ok=True)


def run(seed=0, epochs=250):
    rare_ab = (0.004, 0.008, 0.02, 0.05)
    A = simulate_rare(n_cells=5000, n_genes=1200, n_types=9, rare_abundances=rare_ab, seed=seed)
    A = preprocess_sim(A, n_hvg=1000)
    y = A.obs["celltype"].astype(str).values
    k_true = len(np.unique(y))
    print(f"[sim] {A.shape} {k_true} types; abundances {np.bincount(A.obs['celltype'].cat.codes)}",
          flush=True)

    model, Z, lib, B = fit_nbvae(A, epochs=epochs, seed=seed, verbose=False)
    G, logdet, trace = compute_metric_tensors(model, Z, lib, B=B, use_fisher=True)
    G_euc, _, _ = compute_metric_tensors(model, Z, lib, B=B, use_fisher=False)

    res = np.round(np.geomspace(0.2, 4.0, 18), 3)
    labels = {}
    # PCA+Leiden
    Xln = A.X.toarray() if sparse.issparse(A.X) else np.asarray(A.X)
    from sklearn.decomposition import PCA
    from sklearn.cluster import KMeans
    Xpca = PCA(n_components=50, random_state=seed).fit_transform(Xln)
    labels["PCA+Leiden"] = best_at_matched_k(resolution_sweep(build_euclidean_graph(Xpca), y, res), k_true)["labels"]
    labels["KMeans"] = KMeans(n_clusters=k_true, n_init=10, random_state=seed).fit_predict(Z)
    labels["NBVAE+Euclid"] = best_at_matched_k(resolution_sweep(build_euclidean_graph(Z), y, res), k_true)["labels"]
    labels["IGMC-EucPull"] = best_at_matched_k(resolution_sweep(build_fr_graph(Z, G_euc), y, res), k_true)["labels"]
    labels["IGMC-FR+Leiden"] = best_at_matched_k(resolution_sweep(build_fr_graph(Z, G), y, res), k_true)["labels"]

    # per-type F1 and abundance
    types, counts = np.unique(y, return_counts=True)
    abund = counts / counts.sum()
    out = {"types": types.tolist(), "abundance": abund.tolist(), "methods": {}}
    for m, yp in labels.items():
        f1 = per_type_f1(y, yp)
        bm = basic_metrics(y, yp)
        rareF1 = float(np.mean([f1[str(t)] for t in types[abund < 0.03]]))
        out["methods"][m] = {"ARI": bm["ARI"], "NMI": bm["NMI"], "rareF1": rareF1,
                             "per_type_f1": [f1[str(t)] for t in types]}
        print(f"  {m:16s} ARI={bm['ARI']:.3f} rareF1(<3%)={rareF1:.3f}", flush=True)

    # UMAP for the figure
    from pipeline import _umap_from_graph
    umap_fr = _umap_from_graph(build_fr_graph(Z, G))
    np.savez_compressed(os.path.join(RES, "sim.npz"),
                        Z=Z, y=y, abundance=abund, types=types, umap_fr=umap_fr,
                        **{f"labels__{k}": v for k, v in labels.items()})
    with open(os.path.join(RES, "sim.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("  saved simulation results", flush=True)
    return out


if __name__ == "__main__":
    run()
