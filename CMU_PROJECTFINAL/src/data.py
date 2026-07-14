"""
data.py — dataset loading and standardized preprocessing for IGMC.

Produces AnnData objects with a fixed convention:
  .layers['counts'] : raw integer UMI/read counts for the selected HVGs  (NB-VAE input)
  .X               : log1p library-normalized expression (baselines / visualization)
  .obs['celltype'] : ground-truth cell-type label (category) where available
  .obs['batch']    : batch / technology label (category)
  .obs['n_counts'] : per-cell library size (sum of raw counts, all genes)
  .obs['log_lib']  : log library size (NB-VAE size factor)
  .var['highly_variable'] : HVG flag

All datasets are open-access with confirmed integer counts.
"""
from __future__ import annotations
import os, warnings
import numpy as np
import scanpy as sc
import anndata as ad
from scipy import sparse

warnings.filterwarnings("ignore")
sc.settings.verbosity = 0

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(HERE, "data")
PROC = os.path.join(DATA, "processed")
os.makedirs(PROC, exist_ok=True)
sc.settings.datasetdir = os.path.join(DATA, "raw_downloads")


def _finalize(adata: ad.AnnData, n_hvg: int = 2000, min_genes: int = 200,
              min_cells: int = 3, flavor: str = "seurat_v3") -> ad.AnnData:
    """Shared QC + HVG + normalization. Assumes adata.X is raw integer counts."""
    adata.var_names_make_unique()
    adata.obs_names_make_unique()
    # basic QC
    sc.pp.filter_cells(adata, min_genes=min_genes)
    sc.pp.filter_genes(adata, min_cells=min_cells)
    # mito QC (human/mouse)
    adata.var["mt"] = adata.var_names.str.upper().str.startswith("MT-")
    sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], inplace=True, percent_top=None)
    if adata.var["mt"].sum() > 0:
        adata = adata[adata.obs["pct_counts_mt"] < 20].copy()
    # library size on full gene set BEFORE hvg subset
    counts_full = adata.X.copy()
    n_counts = np.asarray(counts_full.sum(1)).ravel()
    # HVG on a normalized copy but select from raw
    adata.layers["counts_full"] = counts_full
    sc.pp.highly_variable_genes(adata, n_top_genes=n_hvg, flavor=flavor, layer="counts_full"
                                if flavor == "seurat_v3" else None, subset=False)
    adata = adata[:, adata.var["highly_variable"]].copy()
    # store raw counts for HVGs
    hvg_counts = adata.X.copy()
    adata.layers["counts"] = hvg_counts.astype(np.float32) if not sparse.issparse(hvg_counts) \
        else hvg_counts.astype(np.float32)
    # normalized log1p for X (baselines/vis)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    adata.obs["n_counts"] = n_counts.astype(np.float32)
    adata.obs["log_lib"] = np.log(np.clip(n_counts, 1, None)).astype(np.float32)
    # tidy
    for k in ["counts_full"]:
        if k in adata.layers:
            del adata.layers[k]
    adata.X = adata.X.astype(np.float32)
    return adata


def build_pbmc3k(n_hvg: int = 2000) -> ad.AnnData:
    """PBMC3k with ground-truth cell types from scanpy's processed version."""
    raw = sc.read_10x_mtx(os.path.join(DATA, "filtered_gene_bc_matrices/hg19"),
                          var_names="gene_symbols", cache=False)
    raw.X = raw.X.astype(np.float32)
    # labels from the annotated processed dataset (Louvain -> 8 cell types)
    proc = sc.datasets.pbmc3k_processed()
    # match by barcode
    common = raw.obs_names.intersection(proc.obs_names)
    raw = raw[common].copy()
    lab = proc.obs.loc[common, "louvain"].astype(str)
    raw.obs["celltype"] = lab.values
    raw.obs["batch"] = "pbmc3k"
    adata = _finalize(raw, n_hvg=n_hvg)
    adata.obs["celltype"] = adata.obs["celltype"].astype("category")
    adata.obs["batch"] = adata.obs["batch"].astype("category")
    adata.uns["dataset"] = "pbmc3k"
    adata.uns["structure"] = "discrete"
    return adata


def build_paul15(n_hvg: int = 1500) -> ad.AnnData:
    """paul15 mouse myeloid progenitors — a CONTINUOUS differentiation trajectory."""
    p = sc.datasets.paul15()
    p.X = np.asarray(p.X, dtype=np.float32)
    p.obs["celltype"] = p.obs["paul15_clusters"].astype(str).str.replace(r"^\d+", "", regex=True)
    # collapse fine stages into lineages for a cleaner ground truth
    def lineage(x):
        x = x.lower()
        if "ery" in x or "mep" in x: return "Erythroid"
        if "mk" in x: return "Megakaryocyte"
        if "baso" in x: return "Basophil"
        if "eos" in x: return "Eosinophil"
        if "gmp" in x or "neu" in x: return "Neutrophil/GMP"
        if "mo" in x: return "Monocyte"
        if "dc" in x: return "Dendritic"
        if "lymph" in x: return "Lymphoid"
        return x
    p.obs["lineage"] = p.obs["celltype"].map(lineage)
    p.obs["batch"] = "paul15"
    adata = _finalize(p, n_hvg=min(n_hvg, p.shape[1]), min_genes=1, min_cells=1,
                      flavor="seurat_v3")
    adata.obs["celltype"] = adata.obs["celltype"].astype("category")
    adata.obs["lineage"] = adata.obs["lineage"].astype("category")
    adata.obs["batch"] = adata.obs["batch"].astype("category")
    adata.uns["dataset"] = "paul15"
    adata.uns["structure"] = "continuous"
    return adata


def build_pancreas(n_hvg: int = 2000) -> ad.AnnData:
    """scIB human pancreas atlas: 14 cell types across 9 technologies (batches)."""
    A = ad.read_h5ad(os.path.join(DATA, "pancreas_scanvi.h5ad"))
    # counts layer -> round to integers for the NB likelihood
    C = A.layers["counts"]
    C = C.toarray() if sparse.issparse(C) else np.asarray(C)
    C = np.rint(np.clip(C, 0, None)).astype(np.float32)
    raw = ad.AnnData(X=sparse.csr_matrix(C), obs=A.obs.copy(), var=A.var.copy())
    raw.obs["celltype"] = A.obs["celltype"].astype(str)
    raw.obs["batch"] = A.obs["tech"].astype(str)
    adata = _finalize(raw, n_hvg=n_hvg)
    adata.obs["celltype"] = adata.obs["celltype"].astype("category")
    adata.obs["batch"] = adata.obs["batch"].astype("category")
    adata.uns["dataset"] = "pancreas"
    adata.uns["structure"] = "mixed"
    return adata


def build_segerstolpe(n_hvg=2000):
    """
    Segerstolpe healthy-vs-T2D islets (smartseq2 subset of the atlas + disease labels from
    the E-MTAB-5061 SDRF, matched by cell name).  Single technology => clean disease contrast.
    """
    import pandas as pd
    A = ad.read_h5ad(os.path.join(DATA, "pancreas_scanvi.h5ad"))
    sub = A[A.obs["tech"].astype(str) == "smartseq2"].copy()
    C = sub.layers["counts"]
    C = C.toarray() if sparse.issparse(C) else np.asarray(C)
    C = np.rint(np.clip(C, 0, None)).astype(np.float32)
    raw = ad.AnnData(X=sparse.csr_matrix(C), obs=sub.obs.copy(), var=sub.var.copy())
    raw.obs["celltype"] = sub.obs["celltype"].astype(str)
    # disease from SDRF
    sdrf = pd.read_csv(os.path.join(DATA, "seg_sdrf.txt"), sep="\t")
    dmap = dict(zip(sdrf["Source Name"].astype(str),
                    sdrf["Characteristics [disease]"].astype(str)))
    dis = np.array([dmap.get(n, "unknown") for n in raw.obs_names.astype(str)])
    dis = np.where(dis == "normal", "healthy",
                   np.where(dis == "type II diabetes mellitus", "T2D", "unknown"))
    raw.obs["disease"] = dis
    raw.obs["batch"] = "segerstolpe"          # single technology; keep disease as biology
    raw = raw[raw.obs["disease"] != "unknown"].copy()
    adata = _finalize(raw, n_hvg=n_hvg)
    adata.obs["celltype"] = adata.obs["celltype"].astype("category")
    adata.obs["disease"] = adata.obs["disease"].astype("category")
    adata.obs["batch"] = adata.obs["batch"].astype("category")
    adata.uns["dataset"] = "segerstolpe"
    adata.uns["structure"] = "disease"
    return adata


BUILDERS = {"pbmc3k": build_pbmc3k, "paul15": build_paul15, "pancreas": build_pancreas,
            "segerstolpe": build_segerstolpe}


def build_all(overwrite: bool = False):
    for name, fn in BUILDERS.items():
        out = os.path.join(PROC, f"{name}.h5ad")
        if os.path.exists(out) and not overwrite:
            print(f"[skip] {name} exists")
            continue
        print(f"[build] {name} ...", flush=True)
        adata = fn()
        adata.write_h5ad(out)
        rare = adata.obs["celltype"].value_counts()
        print(f"  -> {adata.shape} | {adata.obs['celltype'].nunique()} types | "
              f"{adata.obs['batch'].nunique()} batches | rarest: "
              f"{rare.index[-1]}={rare.iloc[-1]} | saved {out}")


def load(name: str) -> ad.AnnData:
    return ad.read_h5ad(os.path.join(PROC, f"{name}.h5ad"))


if __name__ == "__main__":
    build_all(overwrite=True)
