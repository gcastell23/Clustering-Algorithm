"""train_cache.py — train the NB-VAE + Fisher-Rao metric once per dataset and cache."""
import os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import load
from nbvae import fit_nbvae
from geometry import compute_metric_tensors

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(HERE, "results")


def cache(name, epochs, seed=0):
    out = os.path.join(RES, name); os.makedirs(out, exist_ok=True)
    A = load(name)
    y = A.obs["celltype"].astype(str).values
    t = time.time()
    model, Z, lib, B = fit_nbvae(A, epochs=epochs, seed=seed, verbose=False)
    G, logdet, trace = compute_metric_tensors(model, Z, lib, B=B, use_fisher=True)
    G_euc, _, _ = compute_metric_tensors(model, Z, lib, B=B, use_fisher=False)  # ablation
    theta = model.theta().detach().cpu().numpy()
    np.savez_compressed(os.path.join(out, "_cache.npz"),
                        Z=Z, G=G.astype(np.float32), G_euc=G_euc.astype(np.float32),
                        logdet=logdet, trace=trace, lib=lib, theta=theta, y=y,
                        B=B.astype(np.float32), n_batch=B.shape[1])
    print(f"[{name}] cached Z{Z.shape} nbatch={B.shape[1]} in {time.time()-t:.0f}s "
          f"(logdet {logdet.min():.1f}..{logdet.max():.1f})", flush=True)


if __name__ == "__main__":
    ep = {"pbmc3k": 300, "paul15": 300, "pancreas": 180, "segerstolpe": 300}
    for nm in (sys.argv[1:] or ["pbmc3k", "paul15", "pancreas", "segerstolpe"]):
        cache(nm, ep.get(nm, 250))


def load_cache(name):
    d = np.load(os.path.join(RES, name, "_cache.npz"), allow_pickle=True)
    return {k: d[k] for k in d.files}
