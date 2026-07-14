"""
pipeline.py — end-to-end IGMC run per dataset, with caching for figures.

Trains the NB-VAE once, computes the Fisher-Rao metric, builds graphs, runs every clustering
method + baselines, the Markov-stability scan, conformal prediction and persistent-homology
diagnostics, then writes everything to results/<name>/ for the figure scripts.
"""
from __future__ import annotations
import os, sys, json, time, warnings
import numpy as np
import scanpy as sc
from scipy import sparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings("ignore")

from data import load, PROC
from nbvae import fit_nbvae
from geometry import compute_metric_tensors
from graph import build_euclidean_graph, build_fr_graph, leiden
from evaluate import (basic_metrics, rare_recall, per_type_f1, embedding_silhouette,
                      resolution_sweep, best_at_matched_k)
from markov_stability import markov_stability_scan, select_robust_scales
from conformal import MondrianConformal
import topology as topo

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(HERE, "results")


def _umap_from_graph(A, seed=0):
    """UMAP embedding from a precomputed symmetric affinity graph via scanpy plumbing."""
    import anndata as ad
    n = A.shape[0]
    tmp = ad.AnnData(X=sparse.csr_matrix((n, 1)))
    A = sparse.csr_matrix(A)
    tmp.obsp["connectivities"] = A
    D = A.copy(); D.data = 1.0 / (D.data + 1e-9)
    tmp.obsp["distances"] = D
    tmp.uns["neighbors"] = {"connectivities_key": "connectivities",
                            "distances_key": "distances",
                            "params": {"method": "umap", "n_neighbors": 15}}
    sc.tl.umap(tmp, random_state=seed)
    return tmp.obsm["X_umap"]


def run_dataset(name, epochs=250, d_latent=10, k=15, seed=0, markov_times=None,
                markov_subsample=3500, verbose=True):
    t0 = time.time()
    outdir = os.path.join(RESULTS, name)
    os.makedirs(outdir, exist_ok=True)
    A = load(name)
    y = A.obs["celltype"].astype(str).values
    k_true = len(np.unique(y))
    log = {"name": name, "n_cells": int(A.n_obs), "n_genes": int(A.n_vars),
           "n_types": k_true, "n_batches": int(A.obs["batch"].nunique())}
    if verbose:
        print(f"\n===== {name}: {A.n_obs} cells x {A.n_vars} genes, "
              f"{k_true} types, {log['n_batches']} batches =====", flush=True)

    # 1+2) NB-VAE latent + Fisher-Rao metric (reuse cache if present)
    cache_path = os.path.join(outdir, "_cache.npz")
    if os.path.exists(cache_path):
        c = np.load(cache_path, allow_pickle=True)
        Z, G, logdet, trace, lib, theta = (c["Z"], c["G"].astype(np.float64),
                                           c["logdet"], c["trace"], c["lib"], c["theta"])
        G_euc = c["G_euc"].astype(np.float64) if "G_euc" in c.files else None
        if verbose:
            print("  [using cached NB-VAE latent + metric]", flush=True)
    else:
        model, Z, lib, Boh = fit_nbvae(A, d_latent=d_latent, epochs=epochs, seed=seed, verbose=verbose)
        theta = model.theta().detach().cpu().numpy()
        G, logdet, trace = compute_metric_tensors(model, Z, lib, B=Boh, use_fisher=True)
        G_euc, _, _ = compute_metric_tensors(model, Z, lib, B=Boh, use_fisher=False)

    # 3) graphs
    Aeuc = build_euclidean_graph(Z, k=k)
    Afr = build_fr_graph(Z, G, k=k)

    # 4) baselines + ours (resolution sweeps for graph methods)
    res = np.round(np.geomspace(0.15, 3.0, 16), 3)
    labels, metrics = {}, {}

    # standard pipeline: PCA(logX) -> Leiden
    Xln = A.X.toarray() if sparse.issparse(A.X) else np.asarray(A.X)
    from sklearn.decomposition import PCA
    Xpca = PCA(n_components=min(50, Xln.shape[1] - 1), random_state=seed).fit_transform(Xln)
    Apca = build_euclidean_graph(Xpca, k=k)
    sweep_pca = resolution_sweep(Apca, y, res)
    b = best_at_matched_k(sweep_pca, k_true); labels["PCA+Leiden"] = b["labels"]

    # NB-VAE latent + Euclidean Leiden (ablation: same latent, Euclidean ruler)
    sweep_euc = resolution_sweep(Aeuc, y, res)
    b = best_at_matched_k(sweep_euc, k_true); labels["NBVAE+Euclid"] = b["labels"]

    # k-means on latent
    from sklearn.cluster import KMeans
    labels["KMeans"] = KMeans(n_clusters=k_true, n_init=10, random_state=seed).fit_predict(Z)

    # ablation: Euclidean pullback (J^T J, no NB Fisher information)
    if G_euc is not None:
        Aeucp = build_fr_graph(Z, G_euc, k=k)
        sweep_eucp = resolution_sweep(Aeucp, y, res)
        b = best_at_matched_k(sweep_eucp, k_true); labels["IGMC-EucPull"] = b["labels"]

    # ours: NB-VAE latent + Fisher-Rao Leiden
    sweep_fr = resolution_sweep(Afr, y, res)
    b = best_at_matched_k(sweep_fr, k_true); labels["IGMC-FR+Leiden"] = b["labels"]

    # 5) Markov stability on the FR graph (subsample large datasets)
    if markov_times is None:
        markov_times = np.logspace(-1.5, 2.2, 30)
    if A.n_obs > markov_subsample:
        rng = np.random.default_rng(seed)
        # stratified subsample to keep rare types
        sub = []
        for c in np.unique(y):
            ci = np.where(y == c)[0]
            take = min(len(ci), max(30, int(markov_subsample * len(ci) / A.n_obs)))
            sub.append(rng.choice(ci, take, replace=False))
        sub = np.concatenate(sub)
        Afr_ms = build_fr_graph(Z[sub], G[sub], k=k)
        ms = markov_stability_scan(Afr_ms, times=markov_times, n_seeds=5, verbose=verbose)
        ms["subsample_idx"] = sub
        y_ms = y[sub]
    else:
        ms = markov_stability_scan(Afr, times=markov_times, n_seeds=5, verbose=verbose)
        ms["subsample_idx"] = np.arange(A.n_obs)
        y_ms = y
    robust = select_robust_scales(ms)
    # pick the robust scale whose #comms is closest to k_true as the headline Markov partition
    if robust:
        cand = [(i, abs(ms["n_comms"][i] - k_true)) for i in robust]
        best_i = sorted(cand, key=lambda z: z[1])[0][0]
    else:
        best_i = int(np.argmin(ms["vi"]))
    labels["IGMC-Markov"] = ms["labels"][best_i]
    ms["headline_index"] = int(best_i)
    ms["y"] = y_ms

    # 6) evaluate all (on full data unless Markov which is on subsample)
    for m, yp in labels.items():
        yy = y if len(yp) == len(y) else y_ms
        met = basic_metrics(yy, yp)
        rr, rr_detail = rare_recall(yy, yp)
        met["rare_recall"] = rr
        met["silhouette_Z"] = embedding_silhouette(Z if len(yp) == len(y) else Z[ms["subsample_idx"]], yp)
        met["per_type_f1"] = per_type_f1(yy, yp)
        metrics[m] = met
        if verbose:
            print(f"  {m:16s} k={met['n_clusters']:2d} ARI={met['ARI']:.3f} "
                  f"NMI={met['NMI']:.3f} rareF1={met['rare_recall']:.3f}", flush=True)

    # 7) conformal prediction on the latent (ground-truth labels)
    conf = {}
    for alpha in [0.05, 0.1, 0.2]:
        mc = MondrianConformal(alpha=alpha, seed=seed).fit(Z, y)
        s = mc.summarize(Z, y)
        conf[str(alpha)] = {kk: s[kk] for kk in
                            ["mean_set_size", "frac_singleton", "frac_ambiguous",
                             "frac_novel", "marginal_coverage", "per_class_coverage"]}
        if alpha == 0.1:
            conf_sizes = s["sizes"]; conf_sets = s["sets"]; conf_proba = s["proba"]
            conf_classes = s["classes"]
    if verbose:
        c1 = conf["0.1"]
        print(f"  conformal a=0.1: cov={c1['marginal_coverage']:.3f} "
              f"meanset={c1['mean_set_size']:.2f} amb={c1['frac_ambiguous']:.2f} "
              f"novel={c1['frac_novel']:.2f}", flush=True)

    # 8) topology: global persistence + Fisher-Rao boundary detection (transitionality)
    dg_glob, gstats, gidx = topo.global_topology(Z, G, sample=min(1000, A.n_obs), seed=seed)
    tr = topo.transitionality(Z, y, G=G, k=30)             # vs ground truth (validation)
    tr_infer = topo.transitionality(Z, labels["IGMC-FR+Leiden"], G=G, k=30)  # operational
    gstats["discreteness_index"] = float(1.0 - tr["continuity_index"])
    if verbose:
        print(f"  topology: H0_gap={gstats['h0_gap_ratio']:.1f} "
              f"continuity={tr['continuity_index']:.3f} "
              f"discreteness={gstats['discreteness_index']:.3f} "
              f"(structure={A.uns.get('structure')})", flush=True)

    # 9) embeddings for figures
    emb = {}
    emb["umap_euclid"] = _umap_from_graph(Aeuc, seed=seed)
    emb["umap_fr"] = _umap_from_graph(Afr, seed=seed)

    # ---- save ----
    np.savez_compressed(os.path.join(outdir, "core.npz"),
                        Z=Z, G=G.astype(np.float32), logdet=logdet, trace=trace, lib=lib,
                        theta=theta, y=y, Xpca=Xpca.astype(np.float32),
                        umap_euclid=emb["umap_euclid"], umap_fr=emb["umap_fr"],
                        conf_sizes=conf_sizes, conf_proba=conf_proba.astype(np.float32),
                        conf_classes=np.array(conf_classes, dtype=object),
                        purity=tr["per_cell_purity"], transitional=tr["transitional"],
                        dgm_H0=dg_glob["H0"], dgm_H1=dg_glob["H1"], dgm_idx=gidx,
                        **{f"labels__{k_}": v for k_, v in labels.items()})
    sparse.save_npz(os.path.join(outdir, "graph_fr.npz"), sparse.csr_matrix(Afr))
    sparse.save_npz(os.path.join(outdir, "graph_euc.npz"), sparse.csr_matrix(Aeuc))
    np.savez_compressed(os.path.join(outdir, "markov.npz"),
                        times=ms["times"], n_comms=ms["n_comms"], vi=ms["vi"],
                        stability=ms["stability"], headline_index=ms["headline_index"],
                        subsample_idx=ms["subsample_idx"], y_ms=y_ms,
                        robust=np.array(robust, dtype=int),
                        **{f"mlabel__{i}": ms["labels"][i] for i in range(len(ms["times"]))})
    # json-safe summaries
    def _clean(o):
        if isinstance(o, dict): return {k_: _clean(v) for k_, v in o.items()}
        if isinstance(o, (list, tuple)): return [_clean(v) for v in o]
        if isinstance(o, (np.floating, np.integer)): return float(o)
        if isinstance(o, np.ndarray): return o.tolist()
        return o
    with open(os.path.join(outdir, "summary.json"), "w") as f:
        json.dump(_clean({"log": log, "metrics": metrics, "conformal": conf,
                          "topology_global": gstats,
                          "topology_continuity_index": tr["continuity_index"],
                          "topology_per_cluster_truth": tr["per_cluster"],
                          "topology_per_cluster_infer": tr_infer["per_cluster"],
                          "markov_robust_times": ms["times"][robust].tolist() if robust else [],
                          "runtime_sec": time.time() - t0}), f, indent=2)
    if verbose:
        print(f"  saved -> {outdir}  ({time.time() - t0:.0f}s)", flush=True)
    return outdir


if __name__ == "__main__":
    names = sys.argv[1:] or ["pbmc3k", "paul15", "pancreas"]
    ep = {"pbmc3k": 250, "paul15": 250, "pancreas": 200}
    for nm in names:
        run_dataset(nm, epochs=ep.get(nm, 250))
