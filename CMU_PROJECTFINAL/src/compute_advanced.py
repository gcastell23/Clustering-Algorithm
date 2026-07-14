"""compute_advanced.py — run + cache the graduate-level analyses for the figures."""
import os, sys, json, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_cache import load_cache
from advanced import ollivier_ricci, ot_disease, spectral_geometry
from data import load as load_ad

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(HERE, "results")


def run_curvature(names=("pbmc3k", "paul15", "pancreas")):
    for nm in names:
        t = time.time(); c = load_cache(nm)
        r = ollivier_ricci(c["Z"], c["G"].astype(np.float64), k=12, max_cells=2600)
        np.savez_compressed(os.path.join(RES, nm, "curvature.npz"),
                            idx=r["idx"], node_curv=r["node_curv"],
                            edges=r["edges"], edge_curv=r["edge_curv"], y=c["y"][r["idx"]])
        print(f"[curv:{nm}] {len(r['edges'])} edges, node kappa "
              f"{r['node_curv'].min():.2f}..{r['node_curv'].max():.2f} ({time.time()-t:.0f}s)", flush=True)


def run_ot_disease():
    c = load_cache("segerstolpe"); A = load_ad("segerstolpe")
    dis = A.obs["disease"].astype(str).values
    perc, coup = ot_disease(c["Z"], c["y"], dis)
    with open(os.path.join(RES, "segerstolpe", "ot_disease.json"), "w") as f:
        json.dump({k: {kk: (float(vv) if isinstance(vv, (int, float, np.floating)) else vv)
                       for kk, vv in v.items()} for k, v in perc.items()}, f, indent=2)
    if coup is not None:
        np.savez_compressed(os.path.join(RES, "segerstolpe", "ot_coupling.npz"),
                            hb=coup["hb"], tb=coup["tb"], P=coup["P"])
    print(f"[ot:segerstolpe] {len(perc)} types; beta W_ht="
          f"{perc.get('beta', {}).get('W_ht', float('nan')):.3f} "
          f"null={perc.get('beta', {}).get('W_null', float('nan')):.3f}", flush=True)


def run_spectral(names=("pbmc3k", "pancreas")):
    from scipy import sparse
    for nm in names:
        c = load_cache(nm); A = load_ad(nm)
        X = A.X.toarray() if sparse.issparse(A.X) else np.asarray(A.X)
        r = spectral_geometry(c["Z"], X, k=15, n_eig=40)
        np.savez_compressed(os.path.join(RES, nm, "spectral.npz"),
                            lap_eig=r["lap_eig"], diff_eig=r["diff_eig"], gaps=r["gaps"],
                            pca_eig=r["pca_eig"], mp_plus=r["mp_plus"],
                            gamma=r["gamma"], n_signal=r["n_signal"])
        print(f"[spec:{nm}] {r['n_signal']} signal PCs (MP+={r['mp_plus']:.2f}), "
              f"largest gap at eig {int(np.argmax(r['gaps']))+1}", flush=True)


if __name__ == "__main__":
    run_spectral(); run_ot_disease(); run_curvature()
    print("advanced analyses done.", flush=True)
