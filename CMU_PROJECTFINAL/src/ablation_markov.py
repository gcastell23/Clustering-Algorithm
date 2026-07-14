"""
ablation_markov.py — does the Fisher-Rao GRAPH drive the no-knob Markov-stability result?

Using the cached NB-VAE latent, we run Markov-stability multiscale clustering on the Euclidean
graph vs the Fisher-Rao graph and compare the best-plateau partition against ground truth.
This isolates the metric's contribution to the headline (no-resolution-knob) result.
"""
import os, sys, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_cache import load_cache
from graph import build_euclidean_graph, build_fr_graph
from markov_stability import markov_stability_scan, select_robust_scales
from evaluate import basic_metrics, rare_recall

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(HERE, "results")


def best_plateau(scan, y, k_true):
    robust = select_robust_scales(scan)
    if not robust:
        robust = [int(np.argmin(scan["vi"]))]
    cand = sorted(robust, key=lambda i: abs(scan["n_comms"][i] - k_true))
    best = cand[0]
    yp = scan["labels"][best]
    m = basic_metrics(y, yp); rr, _ = rare_recall(y, yp)
    return dict(ARI=m["ARI"], NMI=m["NMI"], n=m["n_clusters"], rareF1=rr,
                t=float(scan["times"][best]))


def run(name, subsample=3500, seed=0):
    c = load_cache(name)
    Z, G, y = c["Z"], c["G"].astype(np.float64), c["y"]
    k_true = len(np.unique(y))
    rng = np.random.default_rng(seed)
    if len(Z) > subsample:
        sub = []
        for cl in np.unique(y):
            ci = np.where(y == cl)[0]
            sub.append(rng.choice(ci, min(len(ci), max(30, int(subsample * len(ci) / len(y)))),
                                  replace=False))
        idx = np.concatenate(sub)
        Z, G, y = Z[idx], G[idx], y[idx]
    times = np.logspace(-1.5, 2.2, 26)
    out = {}
    for gname, A in [("euclidean", build_euclidean_graph(Z, k=15)),
                     ("fisher_rao", build_fr_graph(Z, G, k=15))]:
        scan = markov_stability_scan(A, times=times, n_seeds=5, verbose=False)
        out[gname] = best_plateau(scan, y, k_true)
        print(f"  [{name}] Markov on {gname:10s} graph: "
              f"ARI={out[gname]['ARI']:.3f} rareF1={out[gname]['rareF1']:.3f} "
              f"n={out[gname]['n']} (t={out[gname]['t']:.1f})", flush=True)
    return out


if __name__ == "__main__":
    names = sys.argv[1:] or ["pbmc3k", "pancreas", "paul15"]
    allout = {}
    for nm in names:
        print(f"=== {nm} ===", flush=True)
        allout[nm] = run(nm)
    with open(os.path.join(RES, "ablation_markov.json"), "w") as f:
        json.dump(allout, f, indent=2)
    print("saved results/ablation_markov.json")
