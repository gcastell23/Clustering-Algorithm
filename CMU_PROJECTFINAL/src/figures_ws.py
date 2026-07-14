"""
figures_ws.py — WS1/WS2 figures (multi-seed CIs, scale-nested conformal, CITE-seq orthogonal
ground truth, multi-method integration benchmark).  Same Okabe-Ito design system as figures.py;
each exports PNG+SVG+PDF via style.savefig and mirrors into the website gallery.
"""
from __future__ import annotations
import os, sys, json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from scipy.cluster.hierarchy import dendrogram

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import style as S
from style import CAT, INK, INK2, MUTED, SEQ_MAG, OI

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(HERE, "results")
FIG = os.path.join(HERE, "figures")
os.makedirs(FIG, exist_ok=True)
S.set_style()

DS_LABEL = {"pbmc3k": "PBMC 3k", "paul15": "Paul15", "segerstolpe": "Segerstolpe",
            "pancreas": "Pancreas", "citeseq_pbmc": "CITE-seq PBMC"}
MC = {"PCA+Leiden": OI["grey"], "NBVAE+Euclid": OI["orange"], "IGMC-EucPull": OI["sky"],
      "IGMC-FR+Leiden": OI["blue"], "IGMC-Markov": OI["green"]}
ML = {"PCA+Leiden": "PCA+Leiden", "NBVAE+Euclid": "NB-VAE+Euclid",
      "IGMC-EucPull": "Euclid pullback", "IGMC-FR+Leiden": "IGMC (Fisher-Rao)",
      "IGMC-Markov": "IGMC (Markov)"}


# ============================================================ FIG 13: scale-nested conformal
def figure_scale_nested():
    J = json.load(open(os.path.join(RES, "scale_nested_conformal.json")))
    sys.path.insert(0, os.path.join(HERE, "src"))
    from scale_nested_conformal import markov_comembership_distance, build_taxonomy

    fig = plt.figure(figsize=(13.2, 8.4))
    gs = GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.34,
                  left=0.055, right=0.985, top=0.9, bottom=0.09)

    # (a) Markov taxonomy dendrogram (pancreas)
    axa = fig.add_subplot(gs[0, 0])
    mk = np.load(os.path.join(RES, "pancreas", "markov.npz"), allow_pickle=True)
    times = mk["times"]; y_ms = mk["y_ms"]; mlabels = [mk[f"mlabel__{i}"] for i in range(len(times))]
    types, D = markov_comembership_distance(y_ms, mlabels, times)
    maps, sizes, Z = build_taxonomy(types, D, n_levels=4)
    _short = {"activated_stellate": "activated stell.", "quiescent_stellate": "quiescent stell."}
    dlabels = [_short.get(t, t.replace("_", " ")) for t in types]
    dn = dendrogram(Z, labels=dlabels, ax=axa, orientation="left",
                    color_threshold=0, above_threshold_color=INK2, leaf_font_size=6.5)
    axa.set_title("Markov co-clustering taxonomy (pancreas)", loc="left", fontweight="bold", fontsize=9)
    axa.set_xlabel("Markov distance  (1 - mean co-membership)", fontsize=7.5)
    for sp in ["top", "right"]:
        axa.spines[sp].set_visible(False)
    S.panel_tag(axa, "a")

    # (b) per-level coverage + min-per-class (pancreas), target line
    axb = fig.add_subplot(gs[0, 1])
    cb = J["pancreas"]["configs"]["coherent_bonf"]; lv = cb["levels"]
    x = np.arange(len(lv)); sz = [J["pancreas"]["taxonomy_sizes"][i] for i in range(len(lv))]
    marg = [l["marginal_coverage"]["mean"] for l in lv]
    mpc = [l["min_per_class_coverage"]["mean"] for l in lv]
    al = lv[0]["alpha_l"]
    axb.axhline(1 - al, color=MUTED, ls="--", lw=1.1, label=f"per-level target 1-α/L={1-al:.3f}")
    axb.plot(x, marg, "-o", color=CAT[0], lw=2, ms=6, label="marginal coverage")
    axb.plot(x, mpc, "-s", color=CAT[5], lw=1.8, ms=5, label="worst-class coverage")
    axb.annotate(f"rarest type: {mpc[0]:.2f}", (x[0], mpc[0]), fontsize=6.5, color=CAT[5],
                 xytext=(6, 2), textcoords="offset points")
    axb.set_xticks(x); axb.set_xticklabels([f"L{i}\n{s} cls" for i, s in enumerate(sz)], fontsize=6.5)
    axb.set_ylabel("coverage"); axb.set_ylim(0.3, 1.04)
    axb.set_title("Per-scale coverage holds; a rare type\nunder-served at fine scale is rescued coarse",
                  loc="left", fontweight="bold", fontsize=9)
    axb.legend(fontsize=6.0, loc="lower right"); axb.grid(alpha=0.3, axis="y")
    S.panel_tag(axb, "b")

    # (c) simultaneous coverage: corrected vs uncorrected, across datasets
    axc = fig.add_subplot(gs[0, 2])
    dss = [d for d in ["pbmc3k", "paul15", "pancreas"] if d in J]
    xc = np.arange(len(dss)); w = 0.36
    cor = [J[d]["configs"]["coherent_bonf"]["simultaneous_coverage"]["mean"] for d in dss]
    cor_e = [J[d]["configs"]["coherent_bonf"]["simultaneous_coverage"]["std"] for d in dss]
    unc = [J[d]["configs"]["uncorrected"]["simultaneous_coverage"]["mean"] for d in dss]
    unc_e = [J[d]["configs"]["uncorrected"]["simultaneous_coverage"]["std"] for d in dss]
    axc.axhline(1 - J[dss[0]]["alpha"], color=MUTED, ls="--", lw=1.2,
                label=f"family target 1-α={1-J[dss[0]]['alpha']:.2f}")
    axc.bar(xc - w/2, cor, w, yerr=cor_e, color=CAT[2], label="union-bound allocation", capsize=2.5,
            edgecolor="white")
    axc.bar(xc + w/2, unc, w, yerr=unc_e, color=CAT[1], label="uncorrected (α each level)", capsize=2.5,
            edgecolor="white")
    axc.set_xticks(xc); axc.set_xticklabels([DS_LABEL[d] for d in dss], fontsize=7)
    axc.set_ylabel("simultaneous coverage"); axc.set_ylim(0.82, 1.0)
    axc.set_title("Family-wise guarantee: the allocation\nbuys a safety margin", loc="left",
                  fontweight="bold", fontsize=9)
    axc.legend(fontsize=6.0, loc="lower left"); axc.grid(alpha=0.3, axis="y")
    S.panel_tag(axc, "c")

    # (d) coherence rate: projection vs raw, across datasets
    axd = fig.add_subplot(gs[1, 0])
    coh = [J[d]["configs"]["coherent_bonf"]["coherence_rate"]["mean"] for d in dss]
    raw = [J[d]["configs"]["raw_bonf"]["coherence_rate"]["mean"] for d in dss]
    axd.bar(xc - w/2, coh, w, color=CAT[3], label="isotonic projection (Thm 3)", edgecolor="white")
    axd.bar(xc + w/2, raw, w, color=OI["grey"], label="no projection", edgecolor="white")
    axd.set_xticks(xc); axd.set_xticklabels([DS_LABEL[d] for d in dss], fontsize=7)
    axd.set_ylabel("fraction coherent cells"); axd.set_ylim(0.75, 1.01)
    axd.set_title("Coherence: fine calls never contradict\ntheir coarse super-type", loc="left",
                  fontweight="bold", fontsize=9)
    axd.legend(fontsize=6.2, loc="lower left"); axd.grid(alpha=0.3, axis="y")
    S.panel_tag(axd, "d")

    # (e) worst-class coverage per level, all datasets -> rare-type rescue generalizes
    axe = fig.add_subplot(gs[1, 1])
    dcols = {"pbmc3k": CAT[0], "paul15": CAT[1], "pancreas": CAT[2]}
    for d in dss:
        lvd = J[d]["configs"]["coherent_bonf"]["levels"]
        xx = np.arange(len(lvd))
        yy = [l["min_per_class_coverage"]["mean"] for l in lvd]
        axe.plot(xx, yy, "-o", color=dcols[d], lw=2, ms=5, label=DS_LABEL[d])
    axe.axhline(1 - al, color=MUTED, ls="--", lw=1.1)
    axe.set_xlabel("scale level  (0 = finest)"); axe.set_ylabel("worst-class coverage")
    axe.set_ylim(0.3, 1.04); axe.set_xticks(range(4))
    axe.set_title("Rare-type rescue generalizes:\nworst-class coverage rises with coarsening",
                  loc="left", fontweight="bold", fontsize=9)
    axe.legend(fontsize=6.5, loc="lower right"); axe.grid(alpha=0.3, axis="y")
    S.panel_tag(axe, "e")

    # (f) efficiency: mean prediction-set size per level (per-scale regime), all datasets
    axf = fig.add_subplot(gs[1, 2])
    for d in dss:
        lvd = J[d]["configs"]["uncorrected"]["levels"]
        xx = np.arange(len(lvd))
        yy = [l["mean_set_size"]["mean"] for l in lvd]
        axf.plot(xx, yy, "-o", color=dcols[d], lw=2, ms=5, label=DS_LABEL[d])
    axf.set_xlabel("scale level  (0 = finest)"); axf.set_ylabel("mean prediction-set size")
    axf.set_xticks(range(4))
    axf.set_title("Sets stay compact; the continuum (Paul15)\ncarries the most residual ambiguity",
                  loc="left", fontweight="bold", fontsize=9)
    axf.legend(fontsize=6.5, loc="upper right"); axf.grid(alpha=0.3, axis="y")
    S.panel_tag(axf, "f")

    fig.suptitle("Scale-nested conformal prediction on the Markov-stability hierarchy",
                 x=0.055, ha="left", fontsize=11.5, fontweight="bold")
    S.savefig(fig, os.path.join(FIG, "fig13_scale_nested_conformal.png"))
    plt.close(fig); print("  fig13 -> fig13_scale_nested_conformal.png")


# ============================================================ FIG 14: CITE-seq orthogonal truth
def figure_citeseq():
    import anndata as ad
    name = "citeseq_pbmc"
    core = np.load(os.path.join(RES, name, "core.npz"), allow_pickle=True)
    summ = json.load(open(os.path.join(RES, name, "summary.json")))
    y = np.array([str(v) for v in core["y"]]); UF = core["umap_fr"]
    A = ad.read_h5ad(os.path.join(HERE, "data", "processed", "citeseq_pbmc.h5ad"))
    adt = list(A.uns["adt_names"])
    clr = A.obsm["protein_clr"]

    fig = plt.figure(figsize=(13.0, 8.4))
    gs = GridSpec(2, 2, figure=fig, hspace=0.34, wspace=0.26,
                  left=0.06, right=0.97, top=0.9, bottom=0.09)

    # (a) FR-UMAP (RNA-derived) colored by PROTEIN cell type
    axa = fig.add_subplot(gs[0, 0])
    pal = S.celltype_palette(y)
    for c in sorted(set(y)):
        m = y == c
        axa.scatter(UF[m, 0], UF[m, 1], s=7, c=pal[c], label=f"{c} ({m.sum()})",
                    alpha=0.85, edgecolors="none", rasterized=True)
    axa.set_xticks([]); axa.set_yticks([]); S.despine(axa, keep=())
    axa.legend(loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=6.5, markerscale=2.0,
               handletextpad=0.2, borderpad=0.0, labelspacing=0.3)
    axa.set_title("RNA Fisher-Rao embedding, colored by\nPROTEIN-defined type (orthogonal)",
                  loc="left", fontweight="bold", fontsize=9)
    S.panel_tag(axa, "a")

    # (b) protein marker heatmap: mean CLR per protein type
    axb = fig.add_subplot(gs[0, 1])
    key = ["CD3", "CD4", "CD8a", "CD19", "CD20", "CD14", "CD16", "CD56", "HLA-DR"]
    cols = [next((i for i, a_ in enumerate(adt) if a_.startswith(k + "_")), None) for k in key]
    types = ["CD4 T", "CD8 T", "B", "NK", "CD14 Mono", "DC"]
    types = [t for t in types if t in set(y)]
    Mtx = np.zeros((len(types), len(key)))
    for i, t in enumerate(types):
        m = y == t
        for j, ci in enumerate(cols):
            Mtx[i, j] = clr[m, ci].mean() if ci is not None else np.nan
    im = axb.imshow(Mtx, aspect="auto", cmap=DIVERGE_LOCAL(), vmin=-2, vmax=5)
    axb.set_xticks(range(len(key))); axb.set_xticklabels(key, rotation=45, ha="right", fontsize=6.5)
    axb.set_yticks(range(len(types))); axb.set_yticklabels(types, fontsize=7)
    cb = fig.colorbar(im, ax=axb, fraction=0.046, pad=0.02); cb.set_label("mean CLR", fontsize=6.5)
    cb.ax.tick_params(labelsize=5.5)
    axb.set_title("Protein markers define the labels\n(independent of RNA)", loc="left",
                  fontweight="bold", fontsize=9)
    S.panel_tag(axb, "b")

    # (c) method benchmark vs protein labels
    axc = fig.add_subplot(gs[1, 0])
    order = [m for m in ["PCA+Leiden", "NBVAE+Euclid", "IGMC-FR+Leiden", "IGMC-Markov"]
             if m in summ["metrics"]]
    metrics = ["ARI", "NMI", "rare_recall"]; mlab = ["ARI", "NMI", "rare-F1 (DC)"]
    xc = np.arange(len(metrics)); w = 0.8 / len(order)
    for i, m in enumerate(order):
        vals = [summ["metrics"][m][k] for k in metrics]
        axc.bar(xc + (i - (len(order) - 1) / 2) * w, vals, w, color=MC[m], label=ML[m],
                edgecolor="white", linewidth=0.6)
    axc.set_xticks(xc); axc.set_xticklabels(mlab, fontsize=8)
    axc.set_ylabel("score vs protein labels"); axc.set_ylim(0, 1.0)
    axc.legend(fontsize=6.5, loc="upper right", ncol=1)
    axc.set_title("RNA clustering vs orthogonal protein truth", loc="left",
                  fontweight="bold", fontsize=9); axc.grid(alpha=0.3, axis="y")
    S.panel_tag(axc, "c")

    # (d) rare-type (DC) recovery by method -- the Fisher-Rao ruler wins on orthogonal truth
    axd = fig.add_subplot(gs[1, 1])
    dord = [m for m in ["PCA+Leiden", "NBVAE+Euclid", "IGMC-EucPull", "IGMC-FR+Leiden", "IGMC-Markov"]
            if m in summ["metrics"]]
    dc = [summ["metrics"][m]["per_type_f1"].get("DC", np.nan) for m in dord]
    xg = np.arange(len(dord))
    bars = axd.bar(xg, dc, 0.62, color=[MC[m] for m in dord], edgecolor="white", linewidth=0.6)
    for x, v in zip(xg, dc):
        axd.text(x, v + 0.01, f"{v:.2f}", ha="center", fontsize=7)
    axd.set_xticks(xg); axd.set_xticklabels([ML[m] for m in dord], rotation=22, ha="right", fontsize=6.8)
    axd.set_ylabel("dendritic-cell F1 (rare, 4.5%)"); axd.set_ylim(0, 1.0)
    axd.set_title("Fisher-Rao best recovers the rare DC type\n(same latent, Euclidean ruler fails)",
                  loc="left", fontweight="bold", fontsize=9); axd.grid(alpha=0.3, axis="y")
    S.panel_tag(axd, "d")

    fig.suptitle("CITE-seq protein as orthogonal truth: PCA stays strong on common types, "
                 "Fisher-Rao wins the rare type",
                 x=0.06, ha="left", fontsize=11.0, fontweight="bold")
    S.savefig(fig, os.path.join(FIG, "fig14_citeseq_orthogonal.png"))
    plt.close(fig); print("  fig14 -> fig14_citeseq_orthogonal.png")


def DIVERGE_LOCAL():
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list("clr", ["#0072B2", "#f0efec", "#D55E00"])


# ============================================================ FIG 12: multi-seed robustness
def figure_multiseed():
    J = json.load(open(os.path.join(RES, "multiseed_summary.json")))
    dss = [d for d in ["pbmc3k", "paul15", "segerstolpe", "pancreas"] if d in J]
    # methods actually present across seeds (core run excludes the expensive per-seed Markov scan;
    # show PCA vs the two count-aware-latent rulers -- the WS1 rigor claim)
    present = set()
    for d in dss:
        present.update(J[d]["methods"].keys())
    methods = [m for m in ["PCA+Leiden", "NBVAE+Euclid", "IGMC-FR+Leiden", "IGMC-Markov"]
               if m in present]

    fig = plt.figure(figsize=(13.0, 8.2))
    gs = GridSpec(2, 2, figure=fig, hspace=0.36, wspace=0.24,
                  left=0.07, right=0.98, top=0.9, bottom=0.1)

    def grouped(ax, metric, ylabel, title, tag):
        xd = np.arange(len(dss)); mm = [m for m in methods]
        w = 0.8 / len(mm)
        for i, m in enumerate(mm):
            means, los, his = [], [], []
            for d in dss:
                c = J[d]["methods"].get(m, {}).get(metric)
                if c is None:
                    means.append(np.nan); los.append(0); his.append(0); continue
                means.append(c["mean"]); los.append(c["mean"] - c["ci_lo"]); his.append(c["ci_hi"] - c["mean"])
            ax.bar(xd + (i - (len(mm) - 1) / 2) * w, means, w, yerr=[los, his], capsize=2,
                   color=MC[m], label=ML[m], edgecolor="white", linewidth=0.5,
                   error_kw={"lw": 0.8, "ecolor": INK2})
        ax.set_xticks(xd); ax.set_xticklabels([DS_LABEL[d] for d in dss], fontsize=7.5)
        ax.set_ylabel(ylabel); ax.set_title(title, loc="left", fontweight="bold", fontsize=9)
        ax.grid(alpha=0.3, axis="y"); S.panel_tag(ax, tag)

    axa = fig.add_subplot(gs[0, 0])
    grouped(axa, "ARI", "ARI (mean, 95% CI)", "Clustering accuracy is robust across seeds", "a")
    axa.legend(fontsize=6.3, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=4)
    axa.set_ylim(0, 1.0)

    axb = fig.add_subplot(gs[0, 1])
    grouped(axb, "rare_recall", "rare-type F1 (mean, 95% CI)",
            "Rare-type recovery is robust across seeds", "b")
    axb.set_ylim(0, 1.0)

    # (c) per-seed strip for the headline dataset (pancreas if present else last)
    axc = fig.add_subplot(gs[1, 0])
    dstrip = "pancreas" if "pancreas" in J else dss[-1]
    for i, m in enumerate(methods):
        c = J[dstrip]["methods"].get(m, {}).get("ARI")
        if c is None:
            continue
        vals = c["values"]
        axc.scatter(np.full(len(vals), i) + np.linspace(-0.12, 0.12, len(vals)), vals,
                    s=26, color=MC[m], alpha=0.8, edgecolors="white", linewidths=0.5, zorder=3)
        axc.plot([i - 0.2, i + 0.2], [c["mean"], c["mean"]], color=INK, lw=2, zorder=4)
    axc.set_xticks(range(len(methods))); axc.set_xticklabels([ML[m] for m in methods],
                                                             rotation=20, ha="right", fontsize=7)
    axc.set_ylabel("ARI"); axc.set_title(f"Per-seed spread ({DS_LABEL[dstrip]})", loc="left",
                                         fontweight="bold", fontsize=9)
    axc.grid(alpha=0.3, axis="y"); S.panel_tag(axc, "c")

    # (d) conformal coverage per dataset vs target
    axd = fig.add_subplot(gs[1, 1])
    xd = np.arange(len(dss))
    means = [J[d]["conformal"]["marginal_coverage"]["mean"] for d in dss]
    los = [means[i] - J[dss[i]]["conformal"]["marginal_coverage"]["ci_lo"] for i in range(len(dss))]
    his = [J[dss[i]]["conformal"]["marginal_coverage"]["ci_hi"] - means[i] for i in range(len(dss))]
    axd.axhline(0.9, color=MUTED, ls="--", lw=1.2, label="target 0.90")
    axd.errorbar(xd, means, yerr=[los, his], fmt="o", color=CAT[2], ms=8, capsize=3,
                 lw=1.2, ecolor=INK2)
    axd.set_xticks(xd); axd.set_xticklabels([DS_LABEL[d] for d in dss], fontsize=7.5)
    axd.set_ylabel("conformal coverage (mean, 95% CI)"); axd.set_ylim(0.87, 0.94)
    axd.legend(fontsize=7); axd.set_title("Calibration is stable across seeds", loc="left",
                                          fontweight="bold", fontsize=9)
    axd.grid(alpha=0.3, axis="y"); S.panel_tag(axd, "d")

    fig.suptitle("Multi-seed confidence intervals: the wins are not a lucky seed",
                 x=0.07, ha="left", fontsize=11.5, fontweight="bold")
    S.savefig(fig, os.path.join(FIG, "fig12_multiseed.png"))
    plt.close(fig); print("  fig12 -> fig12_multiseed.png")


# ============================================================ FIG 15: integration benchmark
def figure_benchmark(name="pancreas"):
    J = json.load(open(os.path.join(RES, f"benchmark_integration_{name}.json")))
    M = J["methods"]
    methods = list(M.keys())
    # highlight our method
    def is_ours(m): return "IGMC" in m or "NB-VAE" in m
    bpal = {}
    base = [OI["grey"], OI["orange"], OI["purple"], OI["sky"], OI["vermillion"], OI["yellow"]]
    bi = 0
    for m in methods:
        bpal[m] = OI["blue"] if is_ours(m) else base[bi % len(base)]; bi += 1 if not is_ours(m) else 0

    fig = plt.figure(figsize=(13.0, 8.4))
    gs = GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.28,
                  left=0.07, right=0.97, top=0.9, bottom=0.12)

    # (a) bio vs batch scatter
    axa = fig.add_subplot(gs[0, 0])
    for m in methods:
        x, y = M[m]["batch_correction"], M[m]["bio_conservation"]
        axa.scatter(x, y, s=150 if is_ours(m) else 90, c=bpal[m],
                    marker="*" if is_ours(m) else "o", edgecolors=INK if is_ours(m) else "white",
                    linewidths=1.0 if is_ours(m) else 0.6, zorder=4 if is_ours(m) else 3)
        axa.annotate(m, (x, y), fontsize=6.5, color=INK2, xytext=(5, 3), textcoords="offset points")
    axa.set_xlabel("batch correction"); axa.set_ylabel("bio-conservation")
    axa.set_title("The bio-conservation vs. batch-removal trade-off", loc="left",
                  fontweight="bold", fontsize=9); axa.grid(alpha=0.3)
    S.panel_tag(axa, "a")

    # (b) overall score ranked
    axb = fig.add_subplot(gs[0, 1])
    order = sorted(methods, key=lambda m: M[m]["overall"], reverse=True)
    yv = np.arange(len(order))[::-1]
    axb.barh(yv, [M[m]["overall"] for m in order], color=[bpal[m] for m in order],
             edgecolor="white", linewidth=0.6)
    for m, y in zip(order, yv):
        axb.text(M[m]["overall"] + 0.005, y, f"{M[m]['overall']:.3f}", va="center", fontsize=6.8)
    axb.set_yticks(yv); axb.set_yticklabels(order, fontsize=7.5)
    axb.set_xlabel("overall scIB score (0.6·bio + 0.4·batch)")
    axb.set_title("Overall integration score", loc="left", fontweight="bold", fontsize=9)
    axb.grid(alpha=0.3, axis="x"); S.panel_tag(axb, "b")

    # (c) bio-conservation breakdown
    def breakdown(ax, prefix, title, tag):
        keys = [k for k in M[methods[0]] if k.startswith(prefix + "/")]
        labs = [k.split("/")[1] for k in keys]
        x = np.arange(len(keys)); w = 0.8 / len(methods)
        for i, m in enumerate(methods):
            vals = [M[m].get(k, np.nan) for k in keys]
            ax.bar(x + (i - (len(methods) - 1) / 2) * w, vals, w, color=bpal[m], label=m,
                   edgecolor="white", linewidth=0.4)
        ax.set_xticks(x); ax.set_xticklabels(labs, rotation=25, ha="right", fontsize=6.5)
        ax.set_ylim(0, 1.0); ax.set_title(title, loc="left", fontweight="bold", fontsize=9)
        ax.grid(alpha=0.3, axis="y"); S.panel_tag(ax, tag)

    axc = fig.add_subplot(gs[1, 0]); breakdown(axc, "bio", "Bio-conservation metrics", "c")
    axc.legend(fontsize=6.0, loc="upper center", bbox_to_anchor=(1.12, -0.28), ncol=len(methods))
    axd = fig.add_subplot(gs[1, 1]); breakdown(axd, "batch", "Batch-correction metrics", "d")

    fig.suptitle(f"Benchmarking the count-aware latent against established integration methods ({DS_LABEL.get(name,name)})",
                 x=0.07, ha="left", fontsize=11.0, fontweight="bold")
    S.savefig(fig, os.path.join(FIG, "fig15_benchmark.png"))
    plt.close(fig); print("  fig15 -> fig15_benchmark.png")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("which", nargs="*", default=["all"])
    a = ap.parse_args()
    w = a.which
    if "all" in w or "13" in w:
        figure_scale_nested()
    if "all" in w or "14" in w:
        figure_citeseq()
    if "all" in w or "12" in w:
        figure_multiseed()
    if "all" in w or "15" in w:
        figure_benchmark()
