"""
style.py — a Nature/Cell-grade matplotlib design system for IGMC figures.

Palette: Okabe-Ito, the canonical colourblind-safe qualitative set, used with shape/position
redundancy so identity never rides on colour alone. Continuous fields use perceptually-uniform,
CVD-safe maps (cividis / viridis) and a blue-orange diverging map (Okabe-Ito poles, grey midpoint).
Every figure is exported as high-resolution PNG and vector SVG.
"""
from __future__ import annotations
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.font_manager as fm

# ---- Okabe-Ito qualitative palette (categorical), ordered for adjacent contrast ----
OI = {"blue": "#0072B2", "orange": "#E69F00", "green": "#009E73", "purple": "#CC79A7",
      "sky": "#56B4E9", "vermillion": "#D55E00", "yellow": "#F0E442", "black": "#000000",
      "grey": "#999999"}
CAT = [OI["blue"], OI["orange"], OI["green"], OI["purple"], OI["sky"], OI["vermillion"],
       OI["yellow"], OI["grey"]]

# ---- chrome / ink (calm, warm-neutral bias) ----
SURFACE = "#ffffff"; PANEL = "#fbfbfa"
INK = "#111111"; INK2 = "#555555"; MUTED = "#8a8a8a"
GRID = "#e6e6e3"; AXIS = "#b9b9b4"

# ---- fixed method identity (colour + marker, redundant) ----
METHOD_COLORS = {
    "PCA+Leiden":     OI["grey"],
    "NBVAE+Euclid":   OI["orange"],
    "KMeans":         OI["purple"],
    "IGMC-EucPull":   OI["sky"],
    "IGMC-FR+Leiden": OI["blue"],
    "IGMC-Markov":    OI["green"],
}
METHOD_MARKERS = {
    "PCA+Leiden": "s", "NBVAE+Euclid": "D", "KMeans": "v",
    "IGMC-EucPull": "P", "IGMC-FR+Leiden": "o", "IGMC-Markov": "^",
}
METHOD_LABEL = {
    "PCA+Leiden": "PCA + Leiden",
    "NBVAE+Euclid": "NB-VAE + Euclidean",
    "KMeans": "k-means",
    "IGMC-EucPull": r"Euclidean pullback ($J^\top J$)",
    "IGMC-FR+Leiden": "IGMC (Fisher–Rao)",
    "IGMC-Markov": "IGMC (Markov stability)",
}

# ---- continuous / diverging (CVD-safe) ----
SEQ_BLUE = LinearSegmentedColormap.from_list(
    "oi_seq", ["#eaf3fb", "#bcd9f0", "#7fb8e3", "#3f8fd0", "#0072B2", "#004c78"])
VOL = plt.get_cmap("cividis")            # magnification / volume field
SEQ_MAG = LinearSegmentedColormap.from_list(
    "oi_mag", ["#fdf3e3", "#f6d38f", "#E69F00", "#D55E00", "#8f3d00"])
DIVERGE = LinearSegmentedColormap.from_list(
    "oi_div", ["#004c78", "#0072B2", "#7fb8e3", "#f0efec", "#f2b56b", "#E69F00", "#8f3d00"])


def celltype_palette(labels):
    """Stable, distinct colour per category (sorted), extending Okabe-Ito with tints."""
    cats = sorted(set(map(str, labels)))
    ext = [OI["blue"], OI["orange"], OI["green"], OI["purple"], OI["sky"], OI["vermillion"],
           OI["yellow"], OI["grey"], "#004c78", "#8f3d00", "#00654a", "#8a4b6d",
           "#2a90c8", "#b58900", "#5a5a5a", "#c05a8a"]
    return {c: ext[i % len(ext)] for i, c in enumerate(cats)}


def set_style():
    sans = "Arial"
    for cand in ["Arial", "Helvetica", "Helvetica Neue", "Segoe UI", "DejaVu Sans"]:
        if any(cand == f.name for f in fm.fontManager.ttflist):
            sans = cand; break
    mpl.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "font.family": "sans-serif", "font.sans-serif": [sans, "DejaVu Sans"],
        "font.size": 8.0, "axes.titlesize": 9.5, "axes.labelsize": 8.5,
        "xtick.labelsize": 7.0, "ytick.labelsize": 7.0, "legend.fontsize": 7.0,
        "axes.edgecolor": AXIS, "axes.linewidth": 0.8, "axes.labelcolor": INK,
        "axes.titlecolor": INK, "text.color": INK,
        "xtick.color": INK2, "ytick.color": INK2, "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
        "axes.spines.top": False, "axes.spines.right": False,
        "xtick.direction": "out", "ytick.direction": "out",
        "xtick.major.size": 3, "ytick.major.size": 3, "xtick.major.width": 0.8, "ytick.major.width": 0.8,
        "axes.grid": False, "grid.color": GRID, "grid.linewidth": 0.6,
        "legend.frameon": False, "legend.handlelength": 1.2, "legend.handletextpad": 0.5,
        "legend.columnspacing": 1.0, "legend.labelspacing": 0.35,
        "figure.dpi": 130, "savefig.dpi": 340, "savefig.bbox": "tight", "savefig.pad_inches": 0.03,
        "lines.linewidth": 1.8, "lines.solid_capstyle": "round", "lines.antialiased": True,
        "patch.linewidth": 0.0, "mathtext.fontset": "dejavusans",
        "axes.titlepad": 6.0, "axes.labelpad": 3.0, "svg.fonttype": "none",
    })


def panel_tag(ax, tag, dx=-0.02, dy=1.06, size=12):
    ax.text(dx, dy, tag, transform=ax.transAxes, fontsize=size, fontweight="bold",
            va="top", ha="right", color=INK)


def scatter_embed(ax, XY, labels, palette=None, s=5, alpha=0.85, legend=True,
                  order=None, lw=0.0, rasterized=True, legend_kw=None):
    labels = np.asarray([str(x) for x in labels])
    cats = order or sorted(set(labels))
    pal = palette or celltype_palette(labels)
    for c in cats:
        m = labels == c
        ax.scatter(XY[m, 0], XY[m, 1], s=s, c=pal.get(c, MUTED), linewidths=lw,
                   edgecolors="white", alpha=alpha, rasterized=rasterized, label=c)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ["left", "bottom"]:
        ax.spines[sp].set_visible(False)
    if legend:
        lk = dict(loc="center left", bbox_to_anchor=(1.0, 0.5), markerscale=2.2,
                  handletextpad=0.2, borderpad=0.0, labelspacing=0.28)
        if legend_kw: lk.update(legend_kw)
        ax.legend(**lk)
    return ax


def despine(ax, keep=("left", "bottom")):
    for sp in ["top", "right", "left", "bottom"]:
        ax.spines[sp].set_visible(sp in keep)


def savefig(fig, path):
    """Export high-res PNG + vector SVG (for the gallery download buttons)."""
    fig.savefig(path, dpi=340, bbox_inches="tight", facecolor=SURFACE)
    fig.savefig(path.replace(".png", ".svg"), bbox_inches="tight", facecolor=SURFACE)
    fig.savefig(path.replace(".png", ".pdf"), bbox_inches="tight", facecolor=SURFACE)


if __name__ == "__main__":
    set_style(); print("Okabe-Ito style OK; methods:", list(METHOD_COLORS))
