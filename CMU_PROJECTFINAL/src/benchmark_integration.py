"""
benchmark_integration.py — the multi-method integration benchmark reviewers demand (WS1).

Compares embeddings from PCA, ComBat, Harmony, scVI, scANVI, and our count-aware NB-VAE latent
(IGMC) on a real multi-batch atlas (pancreas, 9 technologies), scored by the transparent scIB
battery in scib_metrics.py.  Scanorama is omitted honestly (its `annoy` dependency needs a C++
compiler unavailable on this machine).  Every method uses n_latent = 10 for a fair comparison;
scANVI additionally sees the cell-type labels (semi-supervised) and so has an inherent advantage.
"""
from __future__ import annotations
import os, sys, json, time, warnings
import numpy as np
import scanpy as sc
from scipy import sparse
warnings.filterwarnings("ignore"); sc.settings.verbosity = 0

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import load
from scib_metrics import score_embedding

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(HERE, "results")
N_LATENT = 10


def _pca(A, n=N_LATENT, seed=0):
    from sklearn.decomposition import PCA
    X = A.X.toarray() if sparse.issparse(A.X) else np.asarray(A.X)
    return PCA(n_components=n, random_state=seed).fit_transform(X)


def embeddings(name, seed=0, scvi_epochs=200):
    A = load(name)
    celltype = A.obs["celltype"].astype(str).values
    batch = A.obs["batch"].astype(str).values
    emb = {}; timing = {}

    t = time.time(); emb["PCA"] = _pca(A, seed=seed); timing["PCA"] = time.time() - t

    # ComBat on log-normalized X, then PCA
    try:
        t = time.time(); Ac = A.copy(); sc.pp.combat(Ac, key="batch")
        from sklearn.decomposition import PCA
        Xc = Ac.X.toarray() if sparse.issparse(Ac.X) else np.asarray(Ac.X)
        emb["ComBat"] = PCA(n_components=N_LATENT, random_state=seed).fit_transform(Xc)
        timing["ComBat"] = time.time() - t
    except Exception as e:
        print("ComBat failed:", e, flush=True)

    # Harmony on the PCA embedding
    try:
        import harmonypy, pandas as pd
        t = time.time()
        meta = pd.DataFrame({"batch": batch})
        ho = harmonypy.run_harmony(emb["PCA"], meta, ["batch"])
        Zc = np.asarray(ho.Z_corr)
        if Zc.shape[0] != len(batch):          # orient to (n_cells, n_latent) regardless of version
            Zc = Zc.T
        emb["Harmony"] = Zc
        timing["Harmony"] = time.time() - t
    except Exception as e:
        print("Harmony failed:", e, flush=True)

    # scVI / scANVI
    try:
        import scvi, torch
        torch.set_num_threads(int(os.environ.get("BENCH_THREADS", "3")))  # share cores politely
        scvi.settings.seed = seed
        Av = A.copy()
        Av.layers["counts"] = (Av.layers["counts"].astype(np.float32))
        scvi.model.SCVI.setup_anndata(Av, layer="counts", batch_key="batch")
        m = scvi.model.SCVI(Av, n_latent=N_LATENT)
        t = time.time(); m.train(max_epochs=scvi_epochs, early_stopping=True)
        emb["scVI"] = m.get_latent_representation(); timing["scVI"] = time.time() - t
        try:
            scanvi = scvi.model.SCANVI.from_scvi_model(m, unlabeled_category="Unknown",
                                                       labels_key="celltype")
            t = time.time(); scanvi.train(max_epochs=max(20, scvi_epochs // 2), early_stopping=True)
            emb["scANVI"] = scanvi.get_latent_representation(); timing["scANVI"] = time.time() - t
        except Exception as e:
            print("scANVI failed:", e, flush=True)
    except Exception as e:
        print("scVI failed:", e, flush=True)

    # ours: cached count-aware NB-VAE latent (Fisher-Rao geometry lives on top of this latent)
    try:
        c = np.load(os.path.join(RESULTS, name, "_cache.npz"), allow_pickle=True)
        emb["IGMC (NB-VAE latent)"] = c["Z"]
    except Exception as e:
        print("IGMC latent load failed:", e, flush=True)

    return emb, celltype, batch, timing


def run(name="pancreas", seed=0, scvi_epochs=200):
    emb, celltype, batch, timing = embeddings(name, seed=seed, scvi_epochs=scvi_epochs)
    out = {"dataset": name, "n_cells": int(len(celltype)), "n_batches": int(len(set(batch))),
           "n_types": int(len(set(celltype))), "timing_sec": timing, "methods": {}}
    for mname, E in emb.items():
        try:
            E = np.asarray(E)
            if E.shape[0] != len(celltype):
                E = E.T
            t = time.time()
            s = score_embedding(E, celltype, batch, seed=seed)
            s["score_sec"] = time.time() - t
            out["methods"][mname] = s
            print(f"[{mname:20s}] bio={s['bio_conservation']:.3f} batch={s['batch_correction']:.3f} "
                  f"overall={s['overall']:.3f} (ARI={s['bio/ARI']:.3f} NMI={s['bio/NMI']:.3f} "
                  f"isoF1={s['bio/isolated_label_F1']:.3f})", flush=True)
        except Exception as e:
            print(f"[{mname:20s}] SCORING FAILED: {type(e).__name__}: {e}", flush=True)
    json.dump(out, open(os.path.join(RESULTS, f"benchmark_integration_{name}.json"), "w"), indent=2)
    print(f"\nwrote results/benchmark_integration_{name}.json")
    return out


if __name__ == "__main__":
    nm = sys.argv[1] if len(sys.argv) > 1 else "pancreas"
    ep = int(sys.argv[2]) if len(sys.argv) > 2 else 200
    run(nm, scvi_epochs=ep)
