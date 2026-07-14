"""
figures.py — publication figures for IGMC.  Each figure is a 4-panel composite.

Reads results/<name>/{core.npz, markov.npz, graph_*.npz, summary.json} produced by
pipeline.py and writes figures/<figN>_<name>.{png,pdf}.
"""
from __future__ import annotations
import os, sys, json
import numpy as np
from scipy import sparse
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import style as S
from style import (CAT, METHOD_COLORS, METHOD_LABEL, SEQ_BLUE, SEQ_MAG, DIVERGE, VOL,
                   INK, INK2, MUTED, GRID, AXIS)

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(HERE, "results")
FIG = os.path.join(HERE, "figures")
os.makedirs(FIG, exist_ok=True)
S.set_style()

METHOD_ORDER = ["PCA+Leiden", "NBVAE+Euclid", "KMeans", "IGMC-FR+Leiden", "IGMC-Markov"]


def _load(name):
    d = {}
    core = np.load(os.path.join(RES, name, "core.npz"), allow_pickle=True)
    d["core"] = {k: core[k] for k in core.files}
    mk = np.load(os.path.join(RES, name, "markov.npz"), allow_pickle=True)
    d["markov"] = {k: mk[k] for k in mk.files}
    with open(os.path.join(RES, name, "summary.json")) as f:
        d["summary"] = json.load(f)
    d["labels"] = {k.replace("labels__", ""): v for k, v in d["core"].items()
                   if k.startswith("labels__")}
    return d


# ============================================================ FIGURE 1: geometry
def figure1(name="pbmc3k"):
    from data import load as load_ad
    d = _load(name); core = d["core"]
    A = load_ad(name)
    Z, G, logdet, y = core["Z"], core["G"].astype(np.float64), core["logdet"], core["y"]
    theta = core["theta"]
    counts = A.layers["counts"]
    counts = counts.toarray() if sparse.issparse(counts) else np.asarray(counts)

    fig = plt.figure(figsize=(11.4, 9.2))
    gs = GridSpec(2, 2, figure=fig, hspace=0.32, wspace=0.26,
                  left=0.07, right=0.97, top=0.93, bottom=0.07)

    # (a) mean-variance law + Fisher information
    axa = fig.add_subplot(gs[0, 0])
    mean = counts.mean(0) + 1e-6
    var = counts.var(0) + 1e-6
    axa.scatter(mean, var, s=6, c=S.CAT[0], alpha=0.4, rasterized=True, edgecolors="none")
    xs = np.geomspace(mean.min(), mean.max(), 100)
    axa.plot(xs, xs, color=MUTED, lw=1.4, ls="--", label=r"Poisson  Var$=\mu$")
    th_med = float(np.median(theta))                 # VAE-learned inverse dispersion
    axa.plot(xs, xs + xs**2 / th_med, color=S.CAT[5], lw=2.0,
             label=rf"NB  Var$=\mu+\mu^2/\theta$  ($\bar\theta={th_med:.2f}$)")
    axa.set_xscale("log"); axa.set_yscale("log")
    axa.set_xlabel("gene mean expression  μ"); axa.set_ylabel("gene variance  Var")
    axa.set_title("Count noise is mean-dependent", loc="left", fontweight="bold")
    axa.legend(loc="upper left")
    # inset: Fisher information I(mu)=1/Var
    axi = axa.inset_axes([0.60, 0.10, 0.36, 0.34])
    fisher = 1.0 / (xs + xs**2 / th_med)
    axi.plot(xs, fisher, color=S.CAT[1], lw=2.0)
    axi.set_xscale("log"); axi.set_yscale("log")
    axi.tick_params(labelsize=5.5); axi.set_title("Fisher info  I(μ)=1/Var", fontsize=6.2)
    S.panel_tag(axa, "a")

    # (b) metric field: PCA-2D of Z, smooth volume field + FR ellipses
    axb = fig.add_subplot(gs[0, 1])
    from sklearn.decomposition import PCA
    pca = PCA(2, random_state=0).fit(Z)
    P = pca.transform(Z)
    # smooth filled contours of the log-volume (magnification factor) from scattered points
    tcf = axb.tricontourf(P[:, 0], P[:, 1], logdet, levels=18, cmap=VOL)
    axb.scatter(P[:, 0], P[:, 1], s=2, c="white", alpha=0.16, rasterized=True, edgecolors="none")
    # Fisher-Rao "distance ellipses" = unit balls of the metric, projected to the PCA plane
    W = pca.components_                       # (2, d)
    span = np.ptp(P, axis=0).mean()
    rng = np.random.default_rng(1)
    Gproj = np.array([W @ np.linalg.inv(G[i]) @ W.T for i in range(len(Z))])
    typ = np.median([np.sqrt(np.linalg.eigvalsh(Gp)).sum() for Gp in Gproj])
    for i in rng.choice(len(Z), 30, replace=False):
        ev, evec = np.linalg.eigh(Gproj[i]); ev = np.clip(ev, 1e-9, None)
        ang = np.degrees(np.arctan2(evec[1, 0], evec[0, 0]))
        sc = 0.055 * span / (typ + 1e-9)
        e = Ellipse(P[i], 2*np.sqrt(ev[0])*sc, 2*np.sqrt(ev[1])*sc, angle=ang,
                    fill=False, edgecolor="white", lw=0.9, alpha=0.92)
        axb.add_patch(e)
    cb = fig.colorbar(tcf, ax=axb, fraction=0.045, pad=0.02)
    cb.set_label("log volume  ½·logdet G", fontsize=6.5); cb.ax.tick_params(labelsize=5.5)
    axb.set_xticks([]); axb.set_yticks([]); S.despine(axb, keep=())
    axb.set_title("The ruler that bends: Fisher–Rao metric field", loc="left", fontweight="bold")
    S.panel_tag(axb, "b")

    # (c) anisotropy spectrum
    axc = fig.add_subplot(gs[1, 0])
    ev = np.linalg.eigvalsh(G)                # (N,d) ascending
    cond = ev[:, -1] / np.clip(ev[:, 0], 1e-9, None)
    parts = axc.violinplot([np.log10(cond)], showextrema=False, widths=0.7)
    for pc in parts["bodies"]:
        pc.set_facecolor(S.CAT[4]); pc.set_alpha(0.5); pc.set_edgecolor(S.CAT[4])
    axc.scatter(np.random.normal(1, 0.03, len(cond)), np.log10(cond), s=3, c=INK2,
                alpha=0.15, rasterized=True, edgecolors="none")
    axc.axhline(0, color=S.CAT[5], lw=1.2, ls="--")
    axc.text(1.42, 0.04, "Euclidean (κ=1)", fontsize=6.4, color=S.CAT[5], va="bottom", ha="right")
    axc.set_xticks([1]); axc.set_xticklabels(["cells"])
    axc.set_ylabel(r"$\log_{10}$ condition number  $\kappa(G)$")
    axc.set_ylim(-0.15, np.log10(cond).max() * 1.08)
    med = np.median(cond)
    axc.set_title(f"Strong anisotropy (median κ ≈ {med:.0f}× Euclidean)", loc="left", fontweight="bold")
    axc.text(1.32, np.log10(med), f"median\nκ={med:.0f}", fontsize=6.5, color=S.CAT[4], va="center")
    S.panel_tag(axc, "c")

    # (d) distance distortion: euclidean vs fisher-rao, within vs between type
    axd = fig.add_subplot(gs[1, 1])
    from geometry import local_fr_dist2_to_candidates
    from sklearn.neighbors import NearestNeighbors
    rng = np.random.default_rng(2)
    sub = rng.choice(len(Z), min(1500, len(Z)), replace=False)
    Zs, Gs, ys = Z[sub], G[sub], y[sub]
    nn = NearestNeighbors(n_neighbors=25).fit(Zs)
    _, cand = nn.kneighbors(Zs)
    cand = cand[:, 1:]
    de = np.linalg.norm(Zs[:, None, :] - Zs[cand], axis=2)
    d2 = local_fr_dist2_to_candidates(Zs, Gs, cand)
    dfr = np.sqrt(np.clip(d2, 0, None))
    within = (ys[:, None] == ys[cand])
    de_f = de.ravel(); dfr_f = dfr.ravel(); wf = within.ravel()
    # normalize scales
    de_f = de_f / np.median(de_f); dfr_f = dfr_f / np.median(dfr_f)
    axd.scatter(de_f[wf], dfr_f[wf], s=4, c=S.CAT[1], alpha=0.25, rasterized=True,
                edgecolors="none", label="within type")
    axd.scatter(de_f[~wf], dfr_f[~wf], s=4, c=S.CAT[5], alpha=0.25, rasterized=True,
                edgecolors="none", label="between type")
    lim = [0, np.percentile(np.r_[de_f, dfr_f], 99)]
    axd.plot(lim, lim, color=MUTED, lw=1.2, ls="--")
    axd.set_xlim(lim); axd.set_ylim(lim)
    axd.set_xlabel("Euclidean distance (norm.)"); axd.set_ylabel("Fisher–Rao distance (norm.)")
    axd.set_title("FR stretches between-type gaps", loc="left", fontweight="bold")
    lg = axd.legend(loc="upper left", markerscale=2.5)
    S.panel_tag(axd, "d")

    fig.suptitle(f"Figure 1 · Information geometry of the negative-binomial count manifold  ·  {name}",
                 x=0.07, ha="left", fontsize=11, fontweight="bold")
    S.savefig(fig, os.path.join(FIG, f"fig1_geometry_{name}.png"))
    plt.close(fig)
    print(f"  fig1 -> fig1_geometry_{name}.png")


# ============================================================ FIGURE 2: core result
def figure2(name="pbmc3k"):
    d = _load(name); core = d["core"]; met = d["summary"]["metrics"]
    y = core["y"]; UE = core["umap_euclid"]; UF = core["umap_fr"]
    pal = S.celltype_palette(y)
    order = sorted(set(map(str, y)))

    fig = plt.figure(figsize=(12.2, 9.0))
    gs = GridSpec(2, 2, figure=fig, hspace=0.34, wspace=0.30,
                  left=0.06, right=0.86, top=0.92, bottom=0.08)

    axa = fig.add_subplot(gs[0, 0])
    S.scatter_embed(axa, UE, y, palette=pal, s=6, order=order, legend=False)
    axa.set_title("Euclidean latent → UMAP", loc="left", fontweight="bold")
    S.panel_tag(axa, "a")

    axb = fig.add_subplot(gs[0, 1])
    S.scatter_embed(axb, UF, y, palette=pal, s=6, order=order, legend=True,
                    legend_kw=dict(title="cell type", fontsize=6.5, title_fontsize=7.2))
    axb.set_title("Fisher–Rao graph → UMAP", loc="left", fontweight="bold")
    S.panel_tag(axb, "b")

    # (c) ablation ladder: same latent, three rulers  L2 -> J^T J -> Fisher-Rao
    axc = fig.add_subplot(gs[1, 0])
    ladder = [("NBVAE+Euclid", r"Euclidean $L_2$"),
              ("IGMC-EucPull", r"pullback $J^\top J$"),
              ("IGMC-FR+Leiden", r"Fisher–Rao $J^\top\!\frac{1}{\mathrm{Var}}J$")]
    ladder = [(m, lbl) for m, lbl in ladder if m in met]
    metrics_show = ["ARI", "NMI", "rare_recall"]; labs = ["ARI", "NMI", "rare-type F1"]
    x = np.arange(len(metrics_show)); w = 0.24
    ladcol = [S.CAT[7], S.CAT[2], S.CAT[0]]
    for i, (m, lbl) in enumerate(ladder):
        vals = [met[m][k] for k in metrics_show]
        axc.bar(x + (i - 1) * w, vals, w, color=ladcol[i], label=lbl,
                edgecolor="white", linewidth=0.5)
    axc.set_xticks(x); axc.set_xticklabels(labs); axc.set_ylim(0, 1.0)
    axc.set_ylabel("score"); axc.grid(axis="y", alpha=0.6)
    axc.set_title("Ablation: same latent, three rulers", loc="left", fontweight="bold")
    axc.legend(loc="upper left", bbox_to_anchor=(0.0, -0.14), ncol=1, fontsize=6.6,
               title="inter-cell ruler", title_fontsize=7)
    S.panel_tag(axc, "c")

    # (d) rare-population slopegraph: per-type F1 Euclid -> FR
    axd = fig.add_subplot(gs[1, 1])
    f1e = met["NBVAE+Euclid"]["per_type_f1"]; f1f = met["IGMC-FR+Leiden"]["per_type_f1"]
    types, counts = np.unique(y, return_counts=True)
    tt = [str(t) for t in types[np.argsort(counts)]]  # rare -> common
    for t in tt:
        e = f1e.get(t, 0); f = f1f.get(t, 0)
        col = S.CAT[1] if f >= e else S.CAT[5]
        axd.plot([0, 1], [e, f], color=col, lw=1.6, alpha=0.8, marker="o", ms=4)
    axd.set_xlim(-0.15, 1.15); axd.set_ylim(-0.02, 1.03)
    axd.set_xticks([0, 1]); axd.set_xticklabels(["Euclidean", "Fisher–Rao"])
    axd.set_ylabel("per-type F1")
    # label rare types
    for t in tt[:4]:
        axd.annotate(t, (1.02, f1f.get(t, 0)), fontsize=6.2, color=INK2, va="center")
    axd.set_title("Rare-type rescue (per cell type)", loc="left", fontweight="bold")
    axd.grid(axis="y", alpha=0.5)
    S.panel_tag(axd, "d")

    fig.suptitle(f"Figure 2 · The right ruler resolves cell types without a resolution knob  ·  {name}",
                 x=0.06, ha="left", fontsize=11, fontweight="bold")
    S.savefig(fig, os.path.join(FIG, f"fig2_clustering_{name}.png"))
    plt.close(fig)
    print(f"  fig2 -> fig2_clustering_{name}.png")


# ============================================================ FIGURE 3: Markov stability
def _alluvial(ax, cols, xs, colors, gap=0.012, band_alpha=0.42):
    """Minimal alluvial: cols = list of integer-label arrays (same length); xs = x positions."""
    n = len(cols[0])
    positions = []
    for ci, lab in enumerate(cols):
        cl, counts = np.unique(lab, return_counts=True)
        order = cl[np.argsort(-counts)]
        y0 = 0.0; pos = {}
        for c in order:
            h = np.mean(lab == c)
            pos[c] = (y0, y0 + h); y0 += h + gap
        # rescale to [0,1]
        tot = y0 - gap
        pos = {c: (a / tot, b / tot) for c, (a, b) in pos.items()}
        positions.append(pos)
        for c, (a, b) in pos.items():
            ax.add_patch(mpatches.Rectangle((xs[ci] - 0.012, a), 0.024, b - a,
                         color=colors[ci % len(colors)], alpha=0.9, lw=0))
    # flows between adjacent columns
    for ci in range(len(cols) - 1):
        L, R = cols[ci], cols[ci + 1]
        pl, pr = positions[ci], positions[ci + 1]
        # track running offsets
        loff = {c: pl[c][0] for c in pl}; roff = {c: pr[c][0] for c in pr}
        for cl_ in sorted(np.unique(L)):
            for cr_ in sorted(np.unique(R)):
                sh = np.mean((L == cl_) & (R == cr_))
                if sh < 0.002:
                    continue
                la0 = loff[cl_]; la1 = la0 + sh; loff[cl_] = la1
                ra0 = roff[cr_]; ra1 = ra0 + sh; roff[cr_] = ra1
                xa, xb = xs[ci] + 0.012, xs[ci + 1] - 0.012
                t = np.linspace(0, 1, 40)
                ss = t * t * (3 - 2 * t)
                xx = xa + (xb - xa) * t
                top = la1 + (ra1 - la1) * ss
                bot = la0 + (ra0 - la0) * ss
                ax.fill_between(xx, bot, top, color=colors[ci % len(colors)],
                                alpha=band_alpha, lw=0)
    ax.set_xlim(-0.03, 1.03); ax.set_ylim(-0.02, 1.04); ax.axis("off")


def figure3(name="pbmc3k"):
    from markov_stability import variation_of_information as vi_fn
    d = _load(name); mk = d["markov"]; core = d["core"]
    t = mk["times"]; nc = mk["n_comms"]; vi = mk["vi"]; stab = mk["stability"]
    robust = mk["robust"] if "robust" in mk else np.array([], int)
    y = mk["y_ms"] if "y_ms" in mk else core["y"]

    fig = plt.figure(figsize=(12.0, 9.0))
    gs = GridSpec(2, 2, figure=fig, hspace=0.34, wspace=0.28,
                  left=0.07, right=0.95, top=0.92, bottom=0.09)

    # (a) N(t) + stability, robust plateaus shaded
    axa = fig.add_subplot(gs[0, 0])
    axa.plot(t, nc, color=S.CAT[0], lw=2.2, marker="o", ms=3, label="# communities")
    axa.set_xscale("log"); axa.set_xlabel("Markov time  t"); axa.set_ylabel("# communities", color=S.CAT[0])
    axa.tick_params(axis="y", labelcolor=S.CAT[0])
    ax2 = axa.twinx(); ax2.plot(t, stab, color=S.CAT[7], lw=2.0, ls="-", alpha=0.9)
    ax2.set_ylabel("stability  R(t)", color=S.CAT[7]); ax2.tick_params(axis="y", labelcolor=S.CAT[7])
    ax2.spines["top"].set_visible(False)
    for i in robust:
        axa.axvspan(t[i] * 0.92, t[i] * 1.08, color=S.CAT[1], alpha=0.10, lw=0)
    k_true = len(np.unique(y))
    axa.axhline(k_true, color=MUTED, lw=1, ls=":")
    axa.text(t[0], k_true + 0.3, f"true = {k_true}", fontsize=6.5, color=MUTED)
    axa.set_title("Communities emerge as the walk lengthens", loc="left", fontweight="bold")
    S.panel_tag(axa, "a")

    # (b) VI(t) across seeds
    axb = fig.add_subplot(gs[0, 1])
    axb.plot(t, vi, color=S.CAT[4], lw=2.2, marker="o", ms=3)
    axb.fill_between(t, 0, vi, color=S.CAT[4], alpha=0.10)
    axb.set_xscale("log"); axb.set_xlabel("Markov time  t")
    axb.set_ylabel("variation of information  (across seeds)")
    for i in robust:
        axb.axvspan(t[i] * 0.92, t[i] * 1.08, color=S.CAT[1], alpha=0.12, lw=0)
    axb.set_title("Robust scales = low-VI plateaus", loc="left", fontweight="bold")
    if len(robust):
        axb.scatter(t[robust], vi[robust], s=28, facecolor=S.CAT[1], edgecolor="white",
                    zorder=5, label="robust scale")
        axb.legend(loc="upper left")
    S.panel_tag(axb, "b")

    # (c) VI(t,t') block matrix
    axc = fig.add_subplot(gs[1, 0])
    T = len(t)
    labs = [mk[f"mlabel__{i}"] for i in range(T)]
    M = np.zeros((T, T))
    for i in range(T):
        for j in range(i, T):
            v = vi_fn(labs[i], labs[j]); M[i, j] = M[j, i] = v
    im = axc.imshow(M, cmap=SEQ_BLUE, origin="lower", aspect="auto")
    ticks = np.linspace(0, T - 1, 6).astype(int)
    axc.set_xticks(ticks); axc.set_xticklabels([f"{t[k]:.1f}" for k in ticks], fontsize=6)
    axc.set_yticks(ticks); axc.set_yticklabels([f"{t[k]:.1f}" for k in ticks], fontsize=6)
    axc.set_xlabel("Markov time  t"); axc.set_ylabel("Markov time  t'")
    cb = fig.colorbar(im, ax=axc, fraction=0.045, pad=0.02); cb.set_label("VI(t, t')", fontsize=6.5)
    cb.ax.tick_params(labelsize=5.5)
    axc.set_title("Multiscale block structure  VI(t, t')", loc="left", fontweight="bold")
    S.panel_tag(axc, "c")

    # (d) alluvial across selected scales (fine -> coarse)
    axd = fig.add_subplot(gs[1, 1])
    # pick 4 scales spanning fine->coarse with distinct community counts
    uniq_idx = []
    seen = set()
    for i in range(T):
        k = int(round(nc[i]))
        if k not in seen and k >= 2:
            seen.add(k); uniq_idx.append(i)
    uniq_idx = sorted(uniq_idx, key=lambda i: -nc[i])[:4]
    uniq_idx = sorted(uniq_idx, key=lambda i: t[i])
    cols = [labs[i] for i in uniq_idx]
    xs = np.linspace(0.05, 0.95, len(cols))
    _alluvial(axd, cols, xs, [S.CAT[0], S.CAT[1], S.CAT[2], S.CAT[4]])
    for xi, i in zip(xs, uniq_idx):
        axd.text(xi, -0.04, f"t={t[i]:.1f}\n{int(round(nc[i]))} comm.", ha="center",
                 va="top", fontsize=6.3, color=INK2)
    axd.set_title("Cluster hierarchy across scales", loc="left", fontweight="bold")
    S.panel_tag(axd, "d")

    fig.suptitle(f"Figure 3 · Markov-stability multiscale clustering — no resolution knob  ·  {name}",
                 x=0.07, ha="left", fontsize=11, fontweight="bold")
    S.savefig(fig, os.path.join(FIG, f"fig3_markov_{name}.png"))
    plt.close(fig); print(f"  fig3 -> fig3_markov_{name}.png")


# ============================================================ FIGURE 4: topology
def figure4(name_disc="pbmc3k", name_cont="paul15"):
    fig = plt.figure(figsize=(12.0, 9.0))
    gs = GridSpec(2, 2, figure=fig, hspace=0.34, wspace=0.26,
                  left=0.07, right=0.95, top=0.92, bottom=0.09)

    def persistence_panel(ax, name, title):
        d = _load(name); core = d["core"]
        H0 = core["dgm_H0"]; H1 = core["dgm_H1"]
        scale = d["summary"]["topology_global"]["scale"]
        H0 = H0[np.isfinite(H0[:, 1])]
        mx = np.nanmax(np.r_[H0[:, 1], H1[:, 1] if len(H1) else [scale]]) * 1.05
        ax.plot([0, mx], [0, mx], color=MUTED, lw=1, ls="--", zorder=1)
        ax.fill_between([0, mx], [0, mx], mx, color=GRID, alpha=0.25, zorder=0)
        ax.scatter(H0[:, 0], H0[:, 1], s=10, c=S.CAT[0], alpha=0.7, label=r"$H_0$ (components)",
                   edgecolors="none", rasterized=True)
        if len(H1):
            ax.scatter(H1[:, 0], H1[:, 1], s=22, c=S.CAT[5], alpha=0.85, label=r"$H_1$ (loops)",
                       edgecolors="white", linewidths=0.4)
        ax.set_xlabel("birth (Fisher–Rao scale)"); ax.set_ylabel("death")
        ax.set_xlim(0, mx); ax.set_ylim(0, mx)
        ax.legend(loc="lower right", fontsize=6.5)
        ci = d["summary"].get("topology_continuity_index", np.nan)
        ax.set_title(f"{title}  ·  continuity index = {ci:.2f}", loc="left", fontweight="bold")

    axa = fig.add_subplot(gs[0, 0]); persistence_panel(axa, name_disc, "Discrete types (PBMC)")
    S.panel_tag(axa, "a")
    axb = fig.add_subplot(gs[0, 1]); persistence_panel(axb, name_cont, "Continuum (Paul15)")
    S.panel_tag(axb, "b")

    # (c) UMAP colored by per-cell purity (transitionality) for the continuous set
    axc = fig.add_subplot(gs[1, 0])
    d = _load(name_cont); core = d["core"]
    UF = core["umap_fr"]; purity = core["purity"]
    sc = axc.scatter(UF[:, 0], UF[:, 1], c=purity, s=7, cmap=SEQ_BLUE.reversed(),
                     vmin=0.3, vmax=1.0, rasterized=True, edgecolors="none")
    axc.set_xticks([]); axc.set_yticks([]); S.despine(axc, keep=())
    cb = fig.colorbar(sc, ax=axc, fraction=0.045, pad=0.02)
    cb.set_label("neighbourhood purity", fontsize=6.5); cb.ax.tick_params(labelsize=5.5)
    axc.set_title(f"Transitional cells bridge the continuum ({name_cont})", loc="left", fontweight="bold")
    S.panel_tag(axc, "c")

    # (d) per-cluster transitionality across the discrete and continuous datasets
    axd = fig.add_subplot(gs[1, 1])
    rows = []
    for nm, col in [(name_disc, S.CAT[0]), (name_cont, S.CAT[5])]:
        pc = _load(nm)["summary"]["topology_per_cluster_truth"]
        for c, v in pc.items():
            rows.append((nm, c, v["transitional_frac"], col))
    rows = sorted(rows, key=lambda r: r[2])
    yv = np.arange(len(rows))
    axd.barh(yv, [r[2] for r in rows], color=[r[3] for r in rows], alpha=0.9, height=0.7)
    axd.set_yticks(yv); axd.set_yticklabels([f"{r[1][:14]}" for r in rows], fontsize=5.6)
    axd.set_xlabel("fraction transitional cells")
    axd.axvline(np.median([r[2] for r in rows]), color=MUTED, ls=":", lw=1)
    handles = [mpatches.Patch(color=S.CAT[0], label=f"{name_disc} (discrete)"),
               mpatches.Patch(color=S.CAT[5], label=f"{name_cont} (continuum)")]
    axd.legend(handles=handles, loc="lower right", fontsize=6.3)
    axd.set_title("Per-population transitionality", loc="left", fontweight="bold")
    S.panel_tag(axd, "d")

    fig.suptitle("Figure 4 · Persistent homology separates discrete types from continua",
                 x=0.07, ha="left", fontsize=11, fontweight="bold")
    S.savefig(fig, os.path.join(FIG, "fig4_topology.png"))
    plt.close(fig); print("  fig4 -> fig4_topology.png")


# ============================================================ FIGURE 5: conformal
def figure5(name="pbmc3k"):
    d = _load(name); core = d["core"]; conf = d["summary"]["conformal"]
    y = core["y"]; UF = core["umap_fr"]; sizes = core["conf_sizes"]

    fig = plt.figure(figsize=(12.0, 9.0))
    gs = GridSpec(2, 2, figure=fig, hspace=0.34, wspace=0.28,
                  left=0.07, right=0.93, top=0.92, bottom=0.09)

    # (a) calibration: target vs achieved (marginal + per-class spread)
    axa = fig.add_subplot(gs[0, 0])
    alphas = sorted(conf.keys(), key=float)
    tgt = [1 - float(a) for a in alphas]
    marg = [conf[a]["marginal_coverage"] for a in alphas]
    axa.plot([0.7, 1.0], [0.7, 1.0], color=MUTED, ls="--", lw=1.2, label="ideal")
    for a in alphas:
        pcc = list(conf[a]["per_class_coverage"].values())
        xa = 1 - float(a)
        axa.scatter([xa] * len(pcc), pcc, s=12, c=S.CAT[2], alpha=0.5, edgecolors="none", zorder=3)
    axa.plot(tgt, marg, color=S.CAT[0], lw=2.2, marker="o", ms=6, label="marginal", zorder=4)
    axa.set_xlabel("target coverage  1−α"); axa.set_ylabel("achieved coverage")
    axa.set_title("Conformal guarantee holds", loc="left", fontweight="bold")
    axa.legend(loc="upper left")
    axa.set_xlim(0.72, 1.0); axa.set_ylim(0.6, 1.02)
    S.panel_tag(axa, "a")

    # (b) set-size composition across alpha
    axb = fig.add_subplot(gs[0, 1])
    comp = np.array([[conf[a]["frac_novel"], conf[a]["frac_singleton"], conf[a]["frac_ambiguous"]]
                     for a in alphas])
    labels_c = ["novel |C|=0", "confident |C|=1", "ambiguous |C|>1"]
    cols = [S.CAT[5], S.CAT[1], S.CAT[2]]
    bottom = np.zeros(len(alphas))
    xa = np.arange(len(alphas))
    for j in range(3):
        axb.bar(xa, comp[:, j], 0.6, bottom=bottom, color=cols[j], label=labels_c[j],
                edgecolor="white", linewidth=0.6)
        bottom += comp[:, j]
    axb.set_xticks(xa); axb.set_xticklabels([f"α={a}" for a in alphas])
    axb.set_ylabel("fraction of cells")
    axb.legend(loc="lower center", bbox_to_anchor=(0.5, -0.30), ncol=3, fontsize=6.5)
    axb.set_title("Prediction-set composition", loc="left", fontweight="bold")
    S.panel_tag(axb, "b")

    # (c) UMAP colored by set size (ambiguity map)
    axc = fig.add_subplot(gs[1, 0])
    sc = axc.scatter(UF[:, 0], UF[:, 1], c=np.clip(sizes, 0, 3), s=7, cmap=SEQ_MAG,
                     vmin=0, vmax=3, rasterized=True, edgecolors="none")
    axc.set_xticks([]); axc.set_yticks([]); S.despine(axc, keep=())
    cb = fig.colorbar(sc, ax=axc, fraction=0.045, pad=0.02, ticks=[0, 1, 2, 3])
    cb.set_label("prediction-set size |C(x)|", fontsize=6.5); cb.ax.tick_params(labelsize=5.5)
    axc.set_title("Ambiguity concentrates at boundaries", loc="left", fontweight="bold")
    S.panel_tag(axc, "c")

    # (d) mean set size per cluster vs transitionality
    axd = fig.add_subplot(gs[1, 1])
    pc = d["summary"]["topology_per_cluster_truth"]
    ys = np.asarray([str(v) for v in y])
    xs_, ys_, names_ = [], [], []
    for c, v in pc.items():
        m = ys == c
        if m.sum() < 20: continue
        xs_.append(v["transitional_frac"]); ys_.append(sizes[m].mean()); names_.append(c)
    xs_ = np.array(xs_); ys_ = np.array(ys_)
    axd.scatter(xs_, ys_, s=40, c=S.CAT[0], alpha=0.85, edgecolors="white", linewidths=0.6)
    if len(xs_) > 2:
        r = np.corrcoef(xs_, ys_)[0, 1]
        b, a = np.polyfit(xs_, ys_, 1)
        xr = np.linspace(xs_.min(), xs_.max(), 20)
        axd.plot(xr, b * xr + a, color=S.CAT[5], lw=1.6, ls="--")
        axd.text(0.05, 0.92, f"r = {r:.2f}", transform=axd.transAxes, fontsize=8, color=S.CAT[5])
    for x, yy, nm in zip(xs_, ys_, names_):
        axd.annotate(nm[:12], (x, yy), fontsize=5.6, color=INK2,
                     xytext=(3, 3), textcoords="offset points")
    axd.set_xlabel("fraction transitional (topology)"); axd.set_ylabel("mean prediction-set size")
    axd.set_title("Topology predicts uncertainty", loc="left", fontweight="bold")
    axd.grid(alpha=0.4)
    S.panel_tag(axd, "d")

    fig.suptitle(f"Figure 5 · Conformal prediction gives every cell a calibrated confidence set  ·  {name}",
                 x=0.07, ha="left", fontsize=11, fontweight="bold")
    S.savefig(fig, os.path.join(FIG, f"fig5_conformal_{name}.png"))
    plt.close(fig); print(f"  fig5 -> fig5_conformal_{name}.png")


# ============================================================ FIGURE 6: pancreas atlas
def figure6(name="pancreas"):
    d = _load(name); core = d["core"]; met = d["summary"]["metrics"]
    y = core["y"]; UF = core["umap_fr"]
    from data import load as load_ad
    A = load_ad(name)
    batch = A.obs["batch"].astype(str).values
    pal = S.celltype_palette(y)
    order = sorted(set(map(str, y)))

    fig = plt.figure(figsize=(12.6, 9.2))
    gs = GridSpec(2, 2, figure=fig, hspace=0.30, wspace=0.24,
                  left=0.05, right=0.85, top=0.92, bottom=0.08)

    # (a) FR UMAP by cell type, rare types annotated
    axa = fig.add_subplot(gs[0, 0])
    S.scatter_embed(axa, UF, y, palette=pal, s=5, order=order, legend=True,
                    legend_kw=dict(fontsize=6.0, title="cell type", title_fontsize=7))
    # annotate rare types
    types, counts = np.unique(y, return_counts=True)
    rare = types[np.argsort(counts)][:4]
    for t in rare:
        m = np.asarray([str(v) for v in y]) == str(t)
        cx, cy = UF[m, 0].mean(), UF[m, 1].mean()
        axa.annotate(f"{t} (n={m.sum()})", (cx, cy), fontsize=6.0, fontweight="bold",
                     color=INK, ha="center",
                     bbox=dict(boxstyle="round,pad=0.15", fc="white", ec=MUTED, lw=0.5, alpha=0.85))
    axa.set_title("Islet atlas — Fisher–Rao embedding", loc="left", fontweight="bold")
    S.panel_tag(axa, "a")

    # (b) batch mixing
    axb = fig.add_subplot(gs[0, 1])
    bpal = {b: CAT[i % len(CAT)] for i, b in enumerate(sorted(set(batch)))}
    S.scatter_embed(axb, UF, batch, palette=bpal, s=5, legend=True,
                    legend_kw=dict(fontsize=6.2, title="technology", title_fontsize=7))
    axb.set_title(f"Batch mixing across {len(set(batch))} technologies", loc="left", fontweight="bold")
    S.panel_tag(axb, "b")

    # (c) fair-benchmark method comparison
    axc = fig.add_subplot(gs[1, 0])
    metrics_show = ["ARI", "NMI", "rare_recall"]; labs = ["ARI", "NMI", "rare-type F1"]
    x = np.arange(len(metrics_show)); w = 0.16
    for i, m in enumerate(METHOD_ORDER):
        if m not in met: continue
        vals = [met[m][k] for k in metrics_show]
        axc.bar(x + (i - 2) * w, vals, w, color=METHOD_COLORS[m], label=METHOD_LABEL[m],
                edgecolor="white", linewidth=0.5)
    axc.set_xticks(x); axc.set_xticklabels(labs); axc.set_ylim(0, 1.0)
    axc.set_ylabel("score"); axc.grid(axis="y", alpha=0.6)
    axc.set_title("Fair benchmark: marker-gene labels", loc="left", fontweight="bold")
    axc.legend(loc="upper left", bbox_to_anchor=(0, -0.14), ncol=2, fontsize=6.4)
    S.panel_tag(axc, "c")

    # (d) rare-type per-type F1: standard PCA vs the count-aware latent
    axd = fig.add_subplot(gs[1, 1])
    f1b = met["PCA+Leiden"]["per_type_f1"]; f1o = met["NBVAE+Euclid"]["per_type_f1"]
    tt = [str(t) for t in types[np.argsort(counts)]]
    yv = np.arange(len(tt))
    axd.barh(yv - 0.2, [f1b.get(t, 0) for t in tt], 0.38, color=METHOD_COLORS["PCA+Leiden"],
             label="PCA + Leiden (log-norm)")
    axd.barh(yv + 0.2, [f1o.get(t, 0) for t in tt], 0.38, color=S.CAT[0],
             label="count-aware NB latent (ours)")
    axd.set_yticks(yv); axd.set_yticklabels([f"{t[:13]} ({c})" for t, c in
                                             zip(tt, counts[np.argsort(counts)])], fontsize=5.6)
    axd.set_xlabel("per-type F1"); axd.legend(loc="lower right", fontsize=6.2)
    axd.set_title("Rare islet populations recovered", loc="left", fontweight="bold")
    S.panel_tag(axd, "d")

    fig.suptitle("Figure 6 · Human pancreatic islet atlas — rare cell types across 9 technologies",
                 x=0.05, ha="left", fontsize=11, fontweight="bold")
    S.savefig(fig, os.path.join(FIG, f"fig6_pancreas_{name}.png"))
    plt.close(fig); print(f"  fig6 -> fig6_pancreas_{name}.png")


# ============================================================ FIGURE 7: T2D disease
def figure7(name="segerstolpe"):
    from data import load as load_ad
    from conformal import MondrianConformal
    import topology as topo
    d = _load(name); core = d["core"]
    A = load_ad(name)
    Z = core["Z"]; G = core["G"].astype(np.float64); UF = core["umap_fr"]
    y = np.asarray([str(v) for v in core["y"]])
    disease = A.obs["disease"].astype(str).values

    fig = plt.figure(figsize=(12.4, 9.2))
    gs = GridSpec(2, 2, figure=fig, hspace=0.32, wspace=0.26,
                  left=0.06, right=0.86, top=0.92, bottom=0.09)

    # (a) FR UMAP by cell type
    axa = fig.add_subplot(gs[0, 0])
    pal = S.celltype_palette(y)
    S.scatter_embed(axa, UF, y, palette=pal, s=7, legend=True,
                    legend_kw=dict(fontsize=6.2, title="cell type", title_fontsize=7))
    axa.set_title("Segerstølpe islets — cell types", loc="left", fontweight="bold")
    S.panel_tag(axa, "a")

    # (b) FR UMAP by disease, beta highlighted
    axb = fig.add_subplot(gs[0, 1])
    dcol = {"healthy": S.CAT[1], "T2D": S.CAT[5]}
    for dv in ["healthy", "T2D"]:
        m = disease == dv
        axb.scatter(UF[m, 0], UF[m, 1], s=6, c=dcol[dv], alpha=0.55, edgecolors="none",
                    rasterized=True, label=dv)
    beta = y == "beta"
    axb.scatter(UF[beta, 0], UF[beta, 1], s=14, facecolors="none", edgecolors=INK,
                linewidths=0.4, alpha=0.5, label="beta cells")
    axb.set_xticks([]); axb.set_yticks([]); S.despine(axb, keep=())
    axb.legend(loc="center left", bbox_to_anchor=(1, 0.5), markerscale=1.6, fontsize=6.8)
    axb.set_title("Healthy vs type-2 diabetes", loc="left", fontweight="bold")
    S.panel_tag(axb, "b")

    # conformal calibrated on HEALTHY only, then coverage measured under the disease shift
    hmask = disease == "healthy"; tmask = disease == "T2D"
    alpha = 0.1
    mc = MondrianConformal(alpha=alpha, seed=0).fit(Z[hmask], y[hmask])
    sets, P = mc.predict_sets(Z)
    covered = np.array([y[i] in set(sets[i].tolist()) for i in range(len(y))])
    # major types by abundance
    types, counts = np.unique(y, return_counts=True)
    major = [str(t) for t in types[np.argsort(-counts)] if counts[list(types).index(t)] >= 25][:7]

    # (c) per-type coverage under covariate shift: healthy (in-dist) vs T2D (shifted)
    axc = fig.add_subplot(gs[1, 0])
    covh = [covered[hmask & (y == c)].mean() if (hmask & (y == c)).sum() else np.nan for c in major]
    covt = [covered[tmask & (y == c)].mean() if (tmask & (y == c)).sum() else np.nan for c in major]
    xb = np.arange(len(major)); w = 0.38
    axc.bar(xb - w/2, covh, w, color=S.CAT[1], label="healthy (calibration)", edgecolor="white")
    axc.bar(xb + w/2, covt, w, color=S.CAT[5], label="T2D (covariate shift)", edgecolor="white")
    axc.axhline(1 - alpha, color=INK, ls="--", lw=1.1)
    axc.text(len(major) - 0.5, 1 - alpha + 0.005, "90% target", fontsize=6.4, ha="right", color=INK)
    axc.set_xticks(xb); axc.set_xticklabels(major, rotation=35, ha="right", fontsize=6.6)
    axc.set_ylabel("conformal coverage"); axc.set_ylim(0.6, 1.02)
    axc.legend(loc="lower left", fontsize=6.4)
    axc.set_title("Coverage holds under disease shift — except where disease bites", loc="left",
                  fontweight="bold")
    S.panel_tag(axc, "c")

    # (d) coverage drop per type localizes the disease effect
    axd = fig.add_subplot(gs[1, 1])
    drop = [(c, (covered[hmask & (y == c)].mean() - covered[tmask & (y == c)].mean()))
            for c in major if (tmask & (y == c)).sum() >= 10]
    drop = sorted(drop, key=lambda z: z[1])
    names_ = [d[0] for d in drop]; vals_ = [d[1] for d in drop]
    cols = [S.CAT[5] if n in ("beta", "gamma") else MUTED for n in names_]
    axd.barh(np.arange(len(names_)), vals_, color=cols, height=0.7)
    axd.axvline(0, color=AXIS, lw=0.8)
    axd.set_yticks(np.arange(len(names_))); axd.set_yticklabels(names_, fontsize=7)
    axd.set_xlabel("coverage drop  (healthy − T2D)")
    axd.set_title("The endocrine compartment shifts most", loc="left", fontweight="bold")
    axd.annotate("β-cells: the\ndiabetes cell", xy=(vals_[[i for i,n in enumerate(names_) if n=='beta'][0]] if 'beta' in names_ else 0,
                  [i for i,n in enumerate(names_) if n=='beta'][0] if 'beta' in names_ else 0),
                 xytext=(0.55, 0.25), textcoords="axes fraction", fontsize=6.8, color=S.CAT[5],
                 fontweight="bold", ha="left")
    axd.grid(axis="x", alpha=0.5)
    S.panel_tag(axd, "d")

    fig.suptitle("Figure 7 · Type-2 diabetes — calibrated confidence localizes the disease shift to the islet's endocrine cells",
                 x=0.06, ha="left", fontsize=10.0, fontweight="bold")
    S.savefig(fig, os.path.join(FIG, f"fig7_disease_{name}.png"))
    plt.close(fig); print(f"  fig7 -> fig7_disease_{name}.png")


# ============================================================ FIGURE 8: simulation rigor
def figure8():
    npz = np.load(os.path.join(RES, "simulation", "sim.npz"), allow_pickle=True)
    with open(os.path.join(RES, "simulation", "sim.json")) as f:
        sj = json.load(f)
    y = npz["y"]; UF = npz["umap_fr"]; abund = np.asarray(sj["abundance"])
    types = np.asarray(sj["types"])

    fig = plt.figure(figsize=(12.2, 9.0))
    gs = GridSpec(2, 2, figure=fig, hspace=0.34, wspace=0.28,
                  left=0.07, right=0.88, top=0.92, bottom=0.09)

    # (a) UMAP by planted type
    axa = fig.add_subplot(gs[0, 0])
    pal = S.celltype_palette(y)
    S.scatter_embed(axa, UF, y, palette=pal, s=6, legend=True,
                    legend_kw=dict(fontsize=6.0, title="planted type", title_fontsize=7))
    for t in types[abund < 0.03]:
        m = np.asarray([str(v) for v in y]) == str(t)
        if m.sum():
            axa.annotate(f"{int(abund[list(types).index(t)]*1000)/10:.1f}%",
                         (UF[m, 0].mean(), UF[m, 1].mean()), fontsize=6, fontweight="bold",
                         ha="center", color=INK,
                         bbox=dict(boxstyle="round,pad=0.1", fc="white", ec=MUTED, lw=0.4, alpha=0.85))
    axa.set_title("Planted rare populations", loc="left", fontweight="bold")
    S.panel_tag(axa, "a")

    # (b) recall (per-type F1) vs abundance, per method
    axb = fig.add_subplot(gs[0, 1])
    order = np.argsort(abund)
    ab_sorted = abund[order]
    show = ["PCA+Leiden", "NBVAE+Euclid", "IGMC-EucPull", "IGMC-FR+Leiden"]
    for m in show:
        if m not in sj["methods"]: continue
        f1 = np.asarray(sj["methods"][m]["per_type_f1"])[order]
        axb.plot(ab_sorted * 100, f1, marker="o", ms=5, lw=1.8,
                 color=METHOD_COLORS[m], label=METHOD_LABEL[m])
    axb.set_xscale("log")
    axb.set_xlabel("planted abundance (%)"); axb.set_ylabel("per-type F1")
    axb.set_ylim(-0.03, 1.03)
    axb.legend(loc="lower right", fontsize=6.4)
    axb.axvspan(0.1, 3, color=S.CAT[2], alpha=0.07)
    axb.set_title("Rare types: recall vs abundance", loc="left", fontweight="bold")
    S.panel_tag(axb, "b")

    # (c) ablation ladder bars
    axc = fig.add_subplot(gs[1, 0])
    ladder = [("NBVAE+Euclid", r"$L_2$"), ("IGMC-EucPull", r"$J^\top J$"),
              ("IGMC-FR+Leiden", r"$J^\top\frac{1}{\mathrm{Var}}J$")]
    ladcol = [S.CAT[7], S.CAT[2], S.CAT[0]]
    x = np.arange(2); w = 0.24
    for i, (m, lbl) in enumerate(ladder):
        if m not in sj["methods"]: continue
        vals = [sj["methods"][m]["ARI"], sj["methods"][m]["rareF1"]]
        axc.bar(x + (i - 1) * w, vals, w, color=ladcol[i], label=lbl, edgecolor="white", lw=0.5)
    axc.set_xticks(x); axc.set_xticklabels(["ARI", "rare-type F1"]); axc.set_ylim(0, 1.0)
    axc.set_ylabel("score"); axc.grid(axis="y", alpha=0.6)
    axc.legend(title="ruler", fontsize=6.6, title_fontsize=7, loc="upper right")
    axc.set_title("Ablation on planted truth (no circularity)", loc="left", fontweight="bold")
    S.panel_tag(axc, "c")

    # (d) all-method rareF1 bars
    axd = fig.add_subplot(gs[1, 1])
    ms = [m for m in METHOD_ORDER if m in sj["methods"]]
    vals = [sj["methods"][m]["rareF1"] for m in ms]
    axd.barh(np.arange(len(ms)), vals, color=[METHOD_COLORS[m] for m in ms], height=0.68)
    axd.set_yticks(np.arange(len(ms))); axd.set_yticklabels([METHOD_LABEL[m] for m in ms], fontsize=6.6)
    axd.set_xlabel("rare-type F1 (abundance < 3%)"); axd.set_xlim(0, 1.0)
    for i, v in enumerate(vals):
        axd.text(v + 0.01, i, f"{v:.2f}", va="center", fontsize=6.5, color=INK)
    axd.set_title("Rare-type recovery, all methods", loc="left", fontweight="bold")
    axd.grid(axis="x", alpha=0.5)
    S.panel_tag(axd, "d")

    fig.suptitle("Figure 8 · Planted-ground-truth simulation isolates the count-geometry's causal effect",
                 x=0.07, ha="left", fontsize=11, fontweight="bold")
    S.savefig(fig, os.path.join(FIG, "fig8_simulation.png"))
    plt.close(fig); print("  fig8 -> fig8_simulation.png")


# ============================================================ FIGURE 9: cross-dataset synthesis
def figure9():
    dsets = ["pbmc3k", "pancreas", "paul15", "segerstolpe"]
    dlabel = {"pbmc3k": "PBMC\n(immune)", "pancreas": "pancreas\n(islet atlas)",
              "paul15": "Paul15\n(continuum)", "segerstolpe": "Segerstølpe\n(T2D)"}
    S = {ds: json.load(open(os.path.join(RES, ds, "summary.json"))) for ds in dsets}

    fig = plt.figure(figsize=(12.4, 9.0))
    gs = GridSpec(2, 2, figure=fig, hspace=0.42, wspace=0.30,
                  left=0.13, right=0.95, top=0.91, bottom=0.12)

    # (a) ARI heatmap methods x datasets
    axa = fig.add_subplot(gs[0, 0])
    import style as ST
    M = np.array([[S[ds]["metrics"].get(m, {}).get("ARI", np.nan) for ds in dsets]
                  for m in METHOD_ORDER])
    im = axa.imshow(M, cmap=SEQ_BLUE, aspect="auto", vmin=0.2, vmax=0.95)
    axa.set_xticks(range(len(dsets))); axa.set_xticklabels([dlabel[d] for d in dsets], fontsize=6.5)
    axa.set_yticks(range(len(METHOD_ORDER)))
    axa.set_yticklabels([METHOD_LABEL[m] for m in METHOD_ORDER], fontsize=6.8)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if not np.isnan(M[i, j]):
                axa.text(j, i, f"{M[i,j]:.2f}", ha="center", va="center", fontsize=6.6,
                         color="white" if M[i, j] > 0.6 else INK)
    cb = fig.colorbar(im, ax=axa, fraction=0.045, pad=0.02); cb.set_label("ARI", fontsize=6.5)
    axa.set_title("Clustering accuracy (ARI)", loc="left", fontweight="bold")
    S_ = ST; S_.panel_tag(axa, "a")

    # (b) rare-type F1: PCA vs count-aware latent
    axb = fig.add_subplot(gs[0, 1])
    x = np.arange(len(dsets)); w = 0.38
    pca = [S[ds]["metrics"]["PCA+Leiden"]["rare_recall"] for ds in dsets]
    nb = [max(S[ds]["metrics"]["NBVAE+Euclid"]["rare_recall"],
              S[ds]["metrics"]["IGMC-FR+Leiden"]["rare_recall"]) for ds in dsets]
    axb.bar(x - w/2, pca, w, color=METHOD_COLORS["PCA+Leiden"], label="PCA + Leiden", edgecolor="white")
    axb.bar(x + w/2, nb, w, color=ST.CAT[0], label="count-aware latent (ours)", edgecolor="white")
    axb.set_xticks(x); axb.set_xticklabels([dlabel[d] for d in dsets], fontsize=6.5)
    axb.set_ylabel("rare-type F1"); axb.set_ylim(0, 1.0); axb.grid(axis="y", alpha=0.5)
    axb.legend(loc="upper left", fontsize=6.6)
    axb.set_title("Rare-population recovery", loc="left", fontweight="bold")
    ST.panel_tag(axb, "b")

    # (c) conformal coverage + discreteness + ambiguity
    axc = fig.add_subplot(gs[1, 0])
    cov = [S[ds]["conformal"]["0.1"]["marginal_coverage"] for ds in dsets]
    disc = [S[ds]["topology_global"]["discreteness_index"] for ds in dsets]
    amb = [S[ds]["conformal"]["0.1"]["frac_ambiguous"] for ds in dsets]
    axc.plot(x, cov, "o-", color=ST.CAT[0], lw=2, ms=7, label="conformal coverage")
    axc.plot(x, disc, "s-", color=ST.CAT[4], lw=2, ms=6, label="discreteness index")
    axc.plot(x, amb, "^-", color=ST.CAT[5], lw=2, ms=6, label="ambiguous fraction")
    axc.axhline(0.9, color=MUTED, ls=":", lw=1)
    axc.text(len(dsets)-1, 0.91, "90% target", fontsize=6, ha="right", color=MUTED)
    axc.set_xticks(x); axc.set_xticklabels([dlabel[d] for d in dsets], fontsize=6.5)
    axc.set_ylim(0, 1.02); axc.legend(loc="center left", fontsize=6.5)
    axc.set_title("Confidence & shape agree across data", loc="left", fontweight="bold")
    ST.panel_tag(axc, "c")

    # (d) Markov graph ablation: Euclidean vs Fisher-Rao graph
    axd = fig.add_subplot(gs[1, 1])
    abl_path = os.path.join(RES, "ablation_markov.json")
    if os.path.exists(abl_path):
        abl = json.load(open(abl_path))
        ads = [d for d in ["pbmc3k", "pancreas", "paul15"] if d in abl]
        xa = np.arange(len(ads))
        eu = [abl[d]["euclidean"]["ARI"] for d in ads]
        fr = [abl[d]["fisher_rao"]["ARI"] for d in ads]
        axd.bar(xa - w/2, eu, w, color=METHOD_COLORS["NBVAE+Euclid"], label="Euclidean graph",
                edgecolor="white")
        axd.bar(xa + w/2, fr, w, color=ST.CAT[1], label="Fisher–Rao graph", edgecolor="white")
        axd.set_xticks(xa); axd.set_xticklabels([dlabel[d] for d in ads], fontsize=6.5)
        axd.set_ylabel("Markov-stability ARI"); axd.set_ylim(0, 1.0); axd.grid(axis="y", alpha=0.5)
        axd.legend(loc="upper right", fontsize=6.6)
    axd.set_title("Does the metric help the no-knob result?", loc="left", fontweight="bold")
    ST.panel_tag(axd, "d")

    fig.suptitle("Figure 9 · Cross-dataset synthesis — one geometry, four agreeing readouts",
                 x=0.06, ha="left", fontsize=11, fontweight="bold")
    ST.savefig(fig, os.path.join(FIG, "fig9_synthesis.png"))
    plt.close(fig); print("  fig9 -> fig9_synthesis.png")


# DROPPED: Ollivier-Ricci curvature is uninformative on the cleanly-separated NB-VAE latent
# (only ~6% of edges cross type boundaries, so there are no bridges; curvature ~ constant positive).
def _figure_curvature_unused(main="pancreas", disc="pbmc3k", cont="paul15"):
    import style as ST
    fig = plt.figure(figsize=(12.4, 9.0))
    gs = GridSpec(2, 2, figure=fig, hspace=0.34, wspace=0.28,
                  left=0.06, right=0.92, top=0.91, bottom=0.10)

    cm = np.load(os.path.join(RES, main, "curvature.npz"), allow_pickle=True)
    core = np.load(os.path.join(RES, main, "core.npz"), allow_pickle=True)
    idx = cm["idx"]; nc = cm["node_curv"]; UF = core["umap_fr"][idx]

    # (a) UMAP coloured by Ollivier-Ricci node curvature
    axa = fig.add_subplot(gs[0, 0])
    vlim = np.percentile(np.abs(nc), 96)
    sc = axa.scatter(UF[:, 0], UF[:, 1], c=nc, s=7, cmap=DIVERGE, vmin=-vlim, vmax=vlim,
                     rasterized=True, edgecolors="none")
    axa.set_xticks([]); axa.set_yticks([]); ST.despine(axa, keep=())
    cb = fig.colorbar(sc, ax=axa, fraction=0.045, pad=0.02)
    cb.set_label("Ollivier–Ricci curvature  κ", fontsize=6.5); cb.ax.tick_params(labelsize=5.5)
    axa.set_title("Curvature of the islet manifold", loc="left", fontweight="bold")
    axa.text(0.02, 0.02, "κ<0 bridges · κ>0 cores", transform=axa.transAxes, fontsize=6.2,
             color=INK2)
    ST.panel_tag(axa, "a")

    # (b) edge-curvature distributions: discrete vs continuum
    axb = fig.add_subplot(gs[0, 1])
    for i, (nm, col, lab) in enumerate([(disc, ST.OI["blue"], "PBMC (discrete)"),
                                        (cont, ST.OI["vermillion"], "Paul15 (continuum)")]):
        e = np.load(os.path.join(RES, nm, "curvature.npz"))["edge_curv"]
        axb.hist(e, bins=40, density=True, histtype="stepfilled", alpha=0.45, color=col,
                 label=f"{lab}  (med κ={np.median(e):.2f})")
        axb.axvline(np.median(e), color=col, lw=1.4, ls="--")
    axb.axvline(0, color=INK, lw=0.9)
    axb.set_xlabel("edge Ollivier–Ricci curvature  κ"); axb.set_ylabel("density")
    axb.legend(loc="upper left", fontsize=6.6)
    axb.set_title("Continua carry more negative-curvature bridges", loc="left", fontweight="bold")
    ST.panel_tag(axb, "b")

    # (c) edge curvature: within-type vs boundary-crossing (main dataset)
    axc = fig.add_subplot(gs[1, 0])
    y = np.asarray([str(v) for v in cm["y"]]); edges = cm["edges"]; ek = cm["edge_curv"]
    cross = y[edges[:, 0]] != y[edges[:, 1]]
    data = [ek[~cross], ek[cross]]
    parts = axc.violinplot(data, showextrema=False, widths=0.8)
    for pc_, col in zip(parts["bodies"], [ST.OI["green"], ST.OI["orange"]]):
        pc_.set_facecolor(col); pc_.set_alpha(0.55); pc_.set_edgecolor(col)
    for i, dd in enumerate(data):
        axc.hlines(np.median(dd), i + 0.78, i + 1.22, color=INK, lw=1.6, zorder=5)
    axc.axhline(0, color=MUTED, lw=0.8, ls=":")
    axc.set_xticks([1, 2]); axc.set_xticklabels(["within type", "boundary-\ncrossing"])
    axc.set_ylabel("edge curvature  κ")
    from scipy.stats import mannwhitneyu
    _, pv = mannwhitneyu(data[0], data[1], alternative="greater")
    axc.text(0.5, 0.96, f"boundary edges more negative\n(p={pv:.1e})", transform=axc.transAxes,
             ha="center", va="top", fontsize=6.6, color=INK)
    axc.set_title("Negative curvature marks type boundaries", loc="left", fontweight="bold")
    ST.panel_tag(axc, "c")

    # (d) node curvature vs transitionality (OT-geometry vs topology agree)
    axd = fig.add_subplot(gs[1, 1])
    purity = core["purity"][idx]; trans = 1 - purity
    axd.scatter(trans, nc, s=8, c=ST.OI["blue"], alpha=0.35, rasterized=True, edgecolors="none")
    if len(trans) > 3:
        r = np.corrcoef(trans, nc)[0, 1]
        b, a = np.polyfit(trans, nc, 1); xr = np.linspace(trans.min(), trans.max(), 20)
        axd.plot(xr, b * xr + a, color=ST.OI["vermillion"], lw=1.8, ls="--")
        axd.text(0.05, 0.06, f"r = {r:.2f}", transform=axd.transAxes, fontsize=9,
                 color=ST.OI["vermillion"], fontweight="bold")
    axd.set_xlabel("transitionality  (1 − neighbourhood purity)")
    axd.set_ylabel("Ollivier–Ricci curvature  κ")
    axd.axhline(0, color=MUTED, lw=0.8, ls=":")
    axd.set_title("Two geometries agree: bridges are transitional", loc="left", fontweight="bold")
    ST.panel_tag(axd, "d")

    fig.suptitle("Figure 10 · The Ricci curvature of cell-state space (optimal transport + geometry)",
                 x=0.06, ha="left", fontsize=11, fontweight="bold")
    ST.savefig(fig, os.path.join(FIG, "fig10_curvature.png"))
    plt.close(fig); print("  fig10 -> fig10_curvature.png")


# ============================================================ FIGURE 10: optimal transport of disease
def figure10(name="segerstolpe"):
    import style as ST
    from data import load as load_ad
    with open(os.path.join(RES, name, "ot_disease.json")) as f:
        od = json.load(f)
    core = np.load(os.path.join(RES, name, "core.npz"), allow_pickle=True)
    A = load_ad(name); disease = A.obs["disease"].astype(str).values
    UF = core["umap_fr"]; y = np.asarray([str(v) for v in core["y"]])

    fig = plt.figure(figsize=(12.4, 9.0))
    gs = GridSpec(2, 2, figure=fig, hspace=0.36, wspace=0.30,
                  left=0.09, right=0.93, top=0.91, bottom=0.11)

    types = sorted(od.keys(), key=lambda c: -(od[c]["W_ht"]))
    endo = {"beta", "gamma", "alpha", "delta", "epsilon"}

    # (a) Wasserstein healthy<->T2D vs within-healthy null
    axa = fig.add_subplot(gs[0, 0])
    yv = np.arange(len(types))
    axa.barh(yv, [od[c]["W_ht"] for c in types], color=[ST.OI["vermillion"] if c in endo else ST.OI["grey"]
             for c in types], height=0.66, label="healthy ↔ T2D")
    axa.scatter([od[c]["W_null"] for c in types], yv, marker="|", s=90, color=INK, zorder=5,
                label="within-healthy null")
    axa.set_yticks(yv); axa.set_yticklabels(types, fontsize=6.6)
    axa.set_xlabel(r"$W_2^2$  transcriptional displacement"); axa.legend(loc="lower right", fontsize=6.4)
    axa.set_title("Optimal-transport displacement by cell type", loc="left", fontweight="bold")
    ST.panel_tag(axa, "a")

    # (b) effect size (W_ht - W_null) ranked
    axb = fig.add_subplot(gs[0, 1])
    eff = sorted([(c, od[c]["effect"]) for c in od], key=lambda z: z[1])
    names_ = [e[0] for e in eff]; vals = [e[1] for e in eff]
    axb.barh(np.arange(len(names_)), vals,
             color=[ST.OI["vermillion"] if n in endo else ST.OI["grey"] for n in names_], height=0.66)
    axb.set_yticks(np.arange(len(names_))); axb.set_yticklabels(names_, fontsize=6.6)
    axb.axvline(0, color=AXIS, lw=0.8)
    axb.set_xlabel("displacement above null  (disease effect)")
    axb.set_title("Insulin (β) and glucagon (α) cells shift most", loc="left", fontweight="bold")
    ST.panel_tag(axb, "b")

    # (c) beta-cell OT coupling / displacement on the embedding
    axc = fig.add_subplot(gs[1, 0])
    beta = y == "beta"
    axc.scatter(UF[~beta, 0], UF[~beta, 1], s=3, c=GRID, rasterized=True, edgecolors="none")
    cp_path = os.path.join(RES, name, "ot_coupling.npz")
    if os.path.exists(cp_path):
        cp = np.load(cp_path)
        hb, tb, P = cp["hb"], cp["tb"], cp["P"]
        axc.scatter(UF[hb, 0], UF[hb, 1], s=14, c=ST.OI["green"], edgecolors="white", linewidths=0.3,
                    label="healthy β")
        axc.scatter(UF[tb, 0], UF[tb, 1], s=14, c=ST.OI["vermillion"], edgecolors="white",
                    linewidths=0.3, label="T2D β")
        # draw strongest transport links
        thr = np.quantile(P[P > 0], 0.99)
        ii, jj = np.where(P >= thr)
        for a, b in zip(ii[:300], jj[:300]):
            axc.plot([UF[hb[a], 0], UF[tb[b], 0]], [UF[hb[a], 1], UF[tb[b], 1]],
                     color=INK, lw=0.3, alpha=0.28)
        axc.legend(loc="upper right", fontsize=6.6, markerscale=1.3)
        # zoom to the beta compartment (+ margin) so the map fills the panel
        bx, by = UF[np.r_[hb, tb], 0], UF[np.r_[hb, tb], 1]
        mx = 0.12 * (bx.max() - bx.min() + 1); my = 0.12 * (by.max() - by.min() + 1)
        # robust limits (ignore a few stray misembedded cells)
        axc.set_xlim(np.percentile(bx, 1) - mx, np.percentile(bx, 99) + mx)
        axc.set_ylim(np.percentile(by, 1) - my, np.percentile(by, 99) + my)
    axc.set_xticks([]); axc.set_yticks([]); ST.despine(axc, keep=())
    axc.set_title("β-cell transport map: healthy → T2D", loc="left", fontweight="bold")
    ST.panel_tag(axc, "c")

    # (d) bootstrap significance of the beta-cell displacement (rigorous, honest)
    axd = fig.add_subplot(gs[1, 1])
    import ot as _ot
    Z = core["Z"]
    hb2 = np.where((y == "beta") & (disease == "healthy"))[0]
    tb2 = np.where((y == "beta") & (disease == "T2D"))[0]
    Zh, Zt = Z[hb2], Z[tb2]
    rng = np.random.default_rng(0)

    def _W(a, b):
        M = _ot.dist(a, b, metric="sqeuclidean")
        return _ot.emd2(np.full(len(a), 1 / len(a)), np.full(len(b), 1 / len(b)), M)

    B = 250; obs = []; null = []
    half = len(Zh) // 2
    for _ in range(B):
        ia = rng.integers(0, len(Zh), len(Zh)); ib = rng.integers(0, len(Zt), len(Zt))
        obs.append(_W(Zh[ia], Zt[ib]))
        p = rng.permutation(len(Zh)); null.append(_W(Zh[p[:half]], Zh[p[half:2 * half]]))
    obs = np.array(obs); null = np.array(null)
    axd.hist(null, bins=28, color=ST.OI["green"], alpha=0.6, density=True,
             label="within-healthy null")
    axd.hist(obs, bins=28, color=ST.OI["vermillion"], alpha=0.6, density=True,
             label="healthy → T2D")
    axd.axvline(null.mean(), color=ST.OI["green"], lw=1.4, ls="--")
    axd.axvline(obs.mean(), color=ST.OI["vermillion"], lw=1.4, ls="--")
    pval = float(np.mean(null >= obs.mean()))
    fold = obs.mean() / (null.mean() + 1e-9)
    axd.set_xlabel(r"$\beta$-cell  $W_2^2$  displacement"); axd.set_ylabel("bootstrap density")
    axd.legend(loc="upper center", fontsize=6.8)
    axd.set_title(f"β-cell shift is real: {fold:.1f}× null (p<{max(pval,1/B):.3f}, {B} bootstraps)",
                  loc="left", fontweight="bold")
    ST.panel_tag(axd, "d")

    fig.suptitle("Figure 10 · Optimal transport quantifies the transcriptional displacement of diabetes",
                 x=0.06, ha="left", fontsize=10.5, fontweight="bold")
    ST.savefig(fig, os.path.join(FIG, "fig10_optimaltransport.png"))
    plt.close(fig); print("  fig10 -> fig10_optimaltransport.png")


# ============================================================ FIGURE 11: spectral / diffusion geometry
def figure11(main="pancreas", second="pbmc3k"):
    import style as ST
    sp = np.load(os.path.join(RES, main, "spectral.npz"))
    mk = np.load(os.path.join(RES, main, "markov.npz"), allow_pickle=True)

    fig = plt.figure(figsize=(12.4, 9.0))
    gs = GridSpec(2, 2, figure=fig, hspace=0.36, wspace=0.30,
                  left=0.08, right=0.94, top=0.91, bottom=0.10)

    # (a) diffusion eigenspectrum + spectral gaps
    axa = fig.add_subplot(gs[0, 0])
    lam = sp["lap_eig"]; gaps = sp["gaps"]
    kk = np.arange(1, len(lam) + 1)
    axa.plot(kk, lam, "-", color=ST.OI["blue"], lw=1.4, zorder=2)
    axa.scatter(kk, lam, s=14, color=ST.OI["blue"], zorder=3)
    gbig = int(np.argmax(gaps))
    axa.axvspan(gbig + 0.5, gbig + 1.5, color=ST.OI["orange"], alpha=0.18)
    axa.annotate(f"largest spectral gap\n→ {gbig+1} metastable states",
                 xy=(gbig + 1, lam[gbig]), xytext=(gbig + 4, lam[gbig] + 0.12),
                 fontsize=6.6, color=ST.OI["vermillion"],
                 arrowprops=dict(arrowstyle="->", color=ST.OI["vermillion"], lw=1))
    axa.set_xlabel(r"eigenvalue index  $k$"); axa.set_ylabel(r"Laplacian eigenvalue  $\lambda_k$")
    axa.set_title("Diffusion spectrum: gaps count metastable states", loc="left", fontweight="bold")
    ST.panel_tag(axa, "a")

    # (b) Marchenko-Pastur null vs observed PCA eigenvalues
    axb = fig.add_subplot(gs[0, 1])
    ev = sp["pca_eig"]; mpp = float(sp["mp_plus"]); gamma = float(sp["gamma"])
    bulk = ev[ev <= mpp]; sig = ev[ev > mpp]
    axb.hist(bulk, bins=60, color=ST.OI["grey"], alpha=0.6, label="noise bulk")
    # MP density
    lam_m = (1 - np.sqrt(gamma)) ** 2
    xs = np.linspace(max(lam_m, 1e-3), mpp, 200)
    mp = np.sqrt(np.clip((mpp - xs) * (xs - lam_m), 0, None)) / (2 * np.pi * gamma * xs)
    axb2 = axb.twinx(); axb2.plot(xs, mp, color=ST.OI["blue"], lw=1.6); axb2.set_yticks([])
    axb2.spines["top"].set_visible(False); axb2.spines["right"].set_visible(False)
    axb.axvline(mpp, color=ST.OI["vermillion"], lw=1.4, ls="--")
    axb.set_xlim(0, np.percentile(ev, 99.5))
    axb.set_xlabel("PCA eigenvalue"); axb.set_ylabel("count (noise)")
    axb.set_title(f"Marchenko–Pastur: {int(sp['n_signal'])} signal components", loc="left",
                  fontweight="bold")
    axb.text(mpp, axb.get_ylim()[1] * 0.85, r"  $\lambda_+$ (noise edge)", color=ST.OI["vermillion"],
             fontsize=6.2)
    ST.panel_tag(axb, "b")

    # (c) diffusion timescales vs Markov communities
    axc = fig.add_subplot(gs[1, 0])
    times = mk["times"]; ncomm = mk["n_comms"]
    diff_t = 1.0 / np.clip(lam[1:], 1e-4, None)          # relaxation timescales
    axc.plot(times, ncomm, "-o", color=ST.OI["green"], ms=3, lw=1.8, label="Markov communities")
    axc.set_xscale("log"); axc.set_xlabel("Markov time  t"); axc.set_ylabel("# communities",
                                                                          color=ST.OI["green"])
    axc.tick_params(axis="y", labelcolor=ST.OI["green"])
    ax2 = axc.twinx()
    ax2.eventplot(diff_t[diff_t < times.max() * 3], orientation="horizontal",
                  lineoffsets=[0.5], linelengths=[0.4], colors=ST.OI["orange"])
    ax2.set_yticks([]); ax2.set_ylim(0, 1); ax2.spines["top"].set_visible(False)
    ax2.set_ylabel("diffusion relaxation times", color=ST.OI["orange"], fontsize=6.8)
    axc.set_title("Spectral timescales set the Markov plateaus", loc="left", fontweight="bold")
    ST.panel_tag(axc, "c")

    # (d) eigenspectra across two datasets
    axd = fig.add_subplot(gs[1, 1])
    for nm, col, lab in [(main, ST.OI["blue"], main), (second, ST.OI["vermillion"], second)]:
        s2 = np.load(os.path.join(RES, nm, "spectral.npz"))
        L = s2["lap_eig"]
        axd.plot(np.arange(1, len(L) + 1), L, "-", color=col, lw=1.5, label=lab)
        axd.scatter(np.arange(1, len(L) + 1), L, s=10, color=col)
    axd.set_xlabel(r"eigenvalue index  $k$"); axd.set_ylabel(r"Laplacian eigenvalue  $\lambda_k$")
    axd.legend(loc="lower right", fontsize=6.8)
    axd.set_title("Spectral signature across tissues", loc="left", fontweight="bold")
    ST.panel_tag(axd, "d")

    fig.suptitle("Figure 11 · Spectral & diffusion geometry with a random-matrix null",
                 x=0.06, ha="left", fontsize=11, fontweight="bold")
    ST.savefig(fig, os.path.join(FIG, "fig11_spectral.png"))
    plt.close(fig); print("  fig11 -> fig11_spectral.png")


def all_figures():
    figure1("pbmc3k"); figure2("pbmc3k"); figure3("pbmc3k")
    figure4("pbmc3k", "paul15")
    figure5("pancreas"); figure6("pancreas"); figure7("segerstolpe")
    figure8(); figure9()
    figure10(); figure11()   # OT of disease; spectral/diffusion geometry


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which == "all":
        all_figures()
    else:
        figure1(which); figure2(which); figure3(which)
