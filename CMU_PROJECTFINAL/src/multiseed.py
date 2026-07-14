"""
multiseed.py — multi-seed confidence intervals for the headline IGMC results.

The #1 rigor gap in the master's version was single-seed reporting.  Here we retrain the
NB-VAE from scratch for seeds 0..N-1 (the dominant stochasticity source), recompute every
downstream metric per seed, and write one JSON per seed so aggregation is incremental and
crash-safe.  A separate aggregation (aggregate_seeds) turns these into mean +/- 95% CI.

Methods compared per seed (all at matched cluster number k = #true types):
  PCA+Leiden, NBVAE+Euclid (same latent, Euclidean ruler), IGMC-EucPull (J^T J pullback),
  IGMC-FR+Leiden (NB Fisher-Rao ruler), and optionally IGMC-Markov (light scan).
Also: split-conformal marginal coverage + ambiguous fraction at alpha=0.1.

Usage:
  python src/multiseed.py <ds1> <ds2> ... --nseed 10 [--markov] [--epochs pancreas=180]
"""
from __future__ import annotations
import os, sys, json, time, argparse
import numpy as np
import scanpy as sc
from scipy import sparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import warnings; warnings.filterwarnings("ignore")

from data import load
from nbvae import fit_nbvae
from geometry import compute_metric_tensors
from graph import build_euclidean_graph, build_fr_graph
from evaluate import (basic_metrics, rare_recall, resolution_sweep, best_at_matched_k,
                      embedding_silhouette)
from conformal import MondrianConformal

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(HERE, "results")

DEFAULT_EPOCHS = {"pbmc3k": 300, "paul15": 300, "pancreas": 180, "segerstolpe": 300,
                  "citeseq_pbmc": 300}
RES_GRID = np.round(np.geomspace(0.15, 3.0, 16), 3)


def run_one_seed(name, A, y, seed, k, do_markov=False, markov_subsample=3500):
    """Train + evaluate one seed; return a flat dict of metrics."""
    t0 = time.time()
    out = {"dataset": name, "seed": int(seed), "n_cells": int(A.n_obs), "k_true": int(k)}

    model, Z, lib, B = fit_nbvae(A, epochs=DEFAULT_EPOCHS.get(name, 250), seed=seed, verbose=False)
    G, logdet, trace = compute_metric_tensors(model, Z, lib, B=B, use_fisher=True)
    G_euc, _, _ = compute_metric_tensors(model, Z, lib, B=B, use_fisher=False)

    # graphs
    Aeuc = build_euclidean_graph(Z, k=15)
    Afr = build_fr_graph(Z, G, k=15)
    Aeucp = build_fr_graph(Z, G_euc, k=15)

    # PCA baseline (deterministic input; Leiden seeded)
    Xln = A.X.toarray() if sparse.issparse(A.X) else np.asarray(A.X)
    from sklearn.decomposition import PCA
    Xpca = PCA(n_components=min(50, Xln.shape[1] - 1), random_state=seed).fit_transform(Xln)
    Apca = build_euclidean_graph(Xpca, k=15)

    methods = {}
    for mkey, Agraph in [("PCA+Leiden", Apca), ("NBVAE+Euclid", Aeuc),
                         ("IGMC-EucPull", Aeucp), ("IGMC-FR+Leiden", Afr)]:
        sweep = resolution_sweep(Agraph, y, RES_GRID, seed=seed)
        b = best_at_matched_k(sweep, k)
        rr, _ = rare_recall(y, b["labels"])
        methods[mkey] = {"ARI": float(b["ARI"]), "NMI": float(b["NMI"]),
                         "rare_recall": float(rr), "n_clusters": int(b["n_clusters"])}

    # optional light Markov stability (subsample large data)
    if do_markov:
        from markov_stability import (markov_stability_scan, select_robust_scales)
        times = np.logspace(-1.5, 2.2, 20)
        if A.n_obs > markov_subsample:
            rng = np.random.default_rng(seed); sub = []
            for c in np.unique(y):
                ci = np.where(y == c)[0]
                take = min(len(ci), max(30, int(markov_subsample * len(ci) / A.n_obs)))
                sub.append(rng.choice(ci, take, replace=False))
            sub = np.concatenate(sub); Gm = build_fr_graph(Z[sub], G[sub], k=15); y_ms = y[sub]
        else:
            Gm = Afr; sub = np.arange(A.n_obs); y_ms = y
        ms = markov_stability_scan(Gm, times=times, n_seeds=3, verbose=False)
        robust = select_robust_scales(ms)
        if robust:
            cand = [(i, abs(ms["n_comms"][i] - k)) for i in robust]
            bi = sorted(cand, key=lambda z: z[1])[0][0]
        else:
            bi = int(np.argmin(ms["vi"]))
        yp = ms["labels"][bi]; m = basic_metrics(y_ms, yp); rr, _ = rare_recall(y_ms, yp)
        methods["IGMC-Markov"] = {"ARI": float(m["ARI"]), "NMI": float(m["NMI"]),
                                  "rare_recall": float(rr), "n_clusters": int(m["n_clusters"])}
    out["methods"] = methods

    # conformal coverage at alpha = 0.1
    mc = MondrianConformal(alpha=0.1, seed=seed).fit(Z, y)
    s = mc.summarize(Z, y)
    out["conformal"] = {"marginal_coverage": float(s["marginal_coverage"]),
                        "frac_ambiguous": float(s["frac_ambiguous"]),
                        "frac_singleton": float(s["frac_singleton"]),
                        "mean_set_size": float(s["mean_set_size"])}
    out["logdet_range"] = [float(logdet.min()), float(logdet.max())]
    out["runtime_sec"] = time.time() - t0
    return out


def run(names, nseed=10, do_markov=False):
    for name in names:
        A = load(name)
        y = A.obs["celltype"].astype(str).values
        k = len(np.unique(y))
        sdir = os.path.join(RESULTS, name, "seeds"); os.makedirs(sdir, exist_ok=True)
        for seed in range(nseed):
            tag = "markov" if do_markov else "core"
            fp = os.path.join(sdir, f"seed_{seed}_{tag}.json")
            if os.path.exists(fp):
                print(f"[{name}] seed {seed} ({tag}) exists, skip", flush=True); continue
            res = run_one_seed(name, A, y, seed, k, do_markov=do_markov)
            with open(fp, "w") as f:
                json.dump(res, f, indent=2)
            fr = res["methods"]["IGMC-FR+Leiden"]; pca = res["methods"]["PCA+Leiden"]
            print(f"[{name}] seed {seed} ({tag}) {res['runtime_sec']:.0f}s | "
                  f"FR ARI={fr['ARI']:.3f} rareF1={fr['rare_recall']:.3f} | "
                  f"PCA ARI={pca['ARI']:.3f} rareF1={pca['rare_recall']:.3f} | "
                  f"cov={res['conformal']['marginal_coverage']:.3f}", flush=True)


def aggregate_seeds(names, tags=("core",)):
    """Aggregate per-seed JSONs into mean +/- 95% CI (bootstrap) per dataset/method/metric."""
    from numpy.random import default_rng
    rng = default_rng(0)
    def ci(vals):
        vals = np.asarray(vals, float)
        if len(vals) == 0:
            return None
        boot = np.array([np.mean(rng.choice(vals, len(vals), replace=True)) for _ in range(5000)])
        return {"mean": float(vals.mean()), "std": float(vals.std(ddof=1)) if len(vals) > 1 else 0.0,
                "ci_lo": float(np.percentile(boot, 2.5)), "ci_hi": float(np.percentile(boot, 97.5)),
                "n": int(len(vals)), "values": [float(v) for v in vals]}
    agg = {}
    for name in names:
        sdir = os.path.join(RESULTS, name, "seeds")
        seeds = []
        for tag in tags:
            import glob
            for fp in sorted(glob.glob(os.path.join(sdir, f"seed_*_{tag}.json"))):
                seeds.append(json.load(open(fp)))
        if not seeds:
            continue
        # merge method dicts across core/markov tags by seed index
        by_seed = {}
        for s in seeds:
            i = s["seed"]; by_seed.setdefault(i, {"methods": {}, "conformal": s.get("conformal")})
            by_seed[i]["methods"].update(s["methods"])
            if s.get("conformal"):
                by_seed[i]["conformal"] = s["conformal"]
        methods = sorted({m for v in by_seed.values() for m in v["methods"]})
        d = {"n_seeds": len(by_seed), "methods": {}}
        for m in methods:
            d["methods"][m] = {}
            for metric in ["ARI", "NMI", "rare_recall", "n_clusters"]:
                vals = [v["methods"][m][metric] for v in by_seed.values() if m in v["methods"]]
                d["methods"][m][metric] = ci(vals)
        d["conformal"] = {}
        for metric in ["marginal_coverage", "frac_ambiguous", "frac_singleton", "mean_set_size"]:
            vals = [v["conformal"][metric] for v in by_seed.values() if v.get("conformal")]
            d["conformal"][metric] = ci(vals)
        agg[name] = d
    out = os.path.join(RESULTS, "multiseed_summary.json")
    json.dump(agg, open(out, "w"), indent=2)
    print("wrote", out)
    return agg


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="*", default=["pbmc3k", "paul15", "segerstolpe", "pancreas"])
    ap.add_argument("--nseed", type=int, default=10)
    ap.add_argument("--markov", action="store_true")
    ap.add_argument("--aggregate", action="store_true")
    a = ap.parse_args()
    names = a.names or ["pbmc3k", "paul15", "segerstolpe", "pancreas"]
    if a.aggregate:
        aggregate_seeds(names, tags=("core", "markov"))
    else:
        run(names, nseed=a.nseed, do_markov=a.markov)
