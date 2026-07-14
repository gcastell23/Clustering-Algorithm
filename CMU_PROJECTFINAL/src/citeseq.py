"""
citeseq.py — CITE-seq orthogonal ground truth (WS1: kill the label-circularity objection).

The pbmc3k labels are circular (they come from a PCA+Louvain pipeline, so PCA "wins" by
construction).  Here the cell-type labels come from a COMPLETELY DIFFERENT MODALITY: cell-surface
protein (32-antibody TotalSeq-B panel, 10x 5k PBMC).  Protein levels never touch the RNA counts
that IGMC and the baselines cluster, so they are an orthogonal reference.  If the count-aware
latent recovers the protein-defined partition better than PCA+Leiden, the circularity objection
is answered on its own terms.

Two orthogonal ground truths are produced, both from protein only:
  * protein_cluster  — unsupervised Leiden on CLR-normalized protein (judgment-free partition).
  * protein_celltype — the same clusters annotated by a FIXED canonical-marker decision rule
                       (for named rare-type recall, e.g. CD16+ monocytes, NK, DC).

RNA is processed exactly like the other datasets (raw HVG counts kept for the NB-VAE).
"""
from __future__ import annotations
import os, warnings
import numpy as np
import scanpy as sc
import anndata as ad
from scipy import sparse

warnings.filterwarnings("ignore"); sc.settings.verbosity = 0
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(HERE, "data"); PROC = os.path.join(DATA, "processed")
H5 = os.path.join(DATA, "citeseq_5k_pbmc_protein_v3.h5")

ISOTYPE = ["IgG1_control_TotalSeqB", "IgG2a_control_TotalSeqB", "IgG2b_control_TotalSeqB"]


def _clr(x):
    """Centered log-ratio across proteins within each cell (muon/Seurat CLR, margin=cell)."""
    l = np.log1p(x)
    return l - l.mean(axis=1, keepdims=True)


def _annotate(clr_df, clusters):
    """Canonical PBMC surface-marker gating on each protein cluster's mean CLR.

    Standard flow-cytometry hierarchy (most specific first). The CLR CD3 distribution is
    cleanly bimodal (T clusters ~3-5, non-T ~ -0.1..0.8), so a single 1.5 gate separates the
    T lineage; the remaining gates are the textbook lineage markers. Reproducible, no tuning
    per dataset. Returns dict cluster_id -> celltype.
    """
    def mean_marker(name, sel):
        col = [c for c in clr_df.columns if c.startswith(name + "_")]
        return float(clr_df[col[0]].values[sel].mean()) if col else 0.0
    lab = {}
    for c in np.unique(clusters):
        sel = clusters == c
        g = lambda name: mean_marker(name, sel)
        cd3 = g("CD3")
        if cd3 > 1.5:                                          # T lineage
            lab[c] = "CD8 T" if (g("CD8a") > 1.0 and g("CD8a") > g("CD4")) else "CD4 T"
        elif max(g("CD19"), g("CD20")) > 1.5:                 # B lineage
            lab[c] = "B"
        elif g("CD56") > 1.0 and g("CD16") > 1.0:             # NK (CD16+ CD56+ CD3-)
            lab[c] = "NK"
        elif g("CD14") > 1.0:                                 # classical monocyte
            lab[c] = "CD14 Mono"
        elif g("CD16") > 1.0:                                 # non-classical monocyte
            lab[c] = "CD16 Mono"
        elif g("HLA-DR") > 2.0:                               # dendritic cell (HLA-DR++ lin-)
            lab[c] = "DC"
        else:
            lab[c] = "other"
    return lab


def build_citeseq(n_hvg=2000, protein_resolution=1.2, seed=0):
    from data import _finalize
    A = sc.read_10x_h5(H5, gex_only=False); A.var_names_make_unique(); A.obs_names_make_unique()
    is_gex = (A.var["feature_types"] == "Gene Expression").values
    is_adt = (A.var["feature_types"] == "Antibody Capture").values
    rna = A[:, is_gex].copy(); adt = A[:, is_adt].copy()
    rna.X = rna.X.astype(np.float32)
    adt_counts = adt.X.toarray() if sparse.issparse(adt.X) else np.asarray(adt.X)
    rna.obsm["adt_raw"] = adt_counts.astype(np.float32)
    rna.uns["adt_names"] = list(adt.var_names)
    rna.obs["batch"] = "citeseq"
    # QC + HVG + raw counts (rides adt_raw through cell filtering)
    adata = _finalize(rna, n_hvg=n_hvg)

    # protein side: CLR (drop isotype controls for clustering), Leiden partition
    import pandas as pd
    adt_names = list(adata.uns["adt_names"])
    clr = _clr(adata.obsm["adt_raw"])
    clr_df = pd.DataFrame(clr, columns=adt_names, index=adata.obs_names)
    use = [n for n in adt_names if n not in ISOTYPE]
    prot = ad.AnnData(clr_df[use].values.astype(np.float32),
                      obs=adata.obs[[]].copy())
    sc.pp.scale(prot, max_value=10)
    sc.pp.pca(prot, n_comps=min(20, len(use) - 1), random_state=seed)
    sc.pp.neighbors(prot, n_neighbors=15, random_state=seed)
    sc.tl.leiden(prot, resolution=protein_resolution, random_state=seed, flavor="igraph",
                 n_iterations=2, directed=False)
    clusters = prot.obs["leiden"].astype(int).values
    lab = _annotate(clr_df, clusters)
    adata.obs["protein_cluster"] = pd.Categorical([f"P{c}" for c in clusters])
    adata.obs["protein_celltype"] = pd.Categorical([lab[c] for c in clusters])
    adata.obs["celltype"] = adata.obs["protein_celltype"]        # plug into existing pipeline
    adata.obsm["protein_clr"] = clr_df.values.astype(np.float32)
    adata.uns["adt_names"] = adt_names
    adata.obs["batch"] = adata.obs["batch"].astype("category")
    adata.obs["celltype"] = adata.obs["celltype"].astype("category")
    adata.uns["dataset"] = "citeseq_pbmc"; adata.uns["structure"] = "discrete"
    return adata, clr_df, clusters, lab


if __name__ == "__main__":
    import sys
    os.makedirs(PROC, exist_ok=True)
    A, clr_df, clusters, lab = build_citeseq()
    out = os.path.join(PROC, "citeseq_pbmc.h5ad"); A.write_h5ad(out)
    print(f"built {A.shape} RNA HVG | {A.obsm['protein_clr'].shape[1]} proteins | saved {out}")
    print("\nprotein Leiden clusters:", len(np.unique(clusters)))
    vc = A.obs["protein_celltype"].value_counts()
    print("protein_celltype counts:\n", vc.to_string())
    print("\ncluster -> annotation:")
    for c in sorted(lab): print(f"  P{c}: {lab[c]}  (n={int(np.sum(clusters==c))})")
