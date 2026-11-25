"""Memory-efficient scRNA reference build: per-sample QC + subsample, sparse PCA."""
import scanpy as sc, pandas as pd, numpy as np, scipy.io, scipy.sparse as sp, warnings, anndata as ad
warnings.filterwarnings("ignore")
sc.settings.verbosity = 1

BASE = os.environ.get("PSC_DATA", "data") + "/scrna"
samples = ["GSM7883934_PSC005_SN_3pr", "GSM7883938_PSC012_SN_3pr", "GSM7883939_PSC014_SN_3pr"]

parts = []
for s in samples:
    m = scipy.io.mmread(f"{BASE}/{s}_matrix.mtx.gz").T.tocsr().astype(np.float32)
    feats = pd.read_csv(f"{BASE}/{s}_features.tsv.gz", sep="\t", header=None)
    bars = pd.read_csv(f"{BASE}/{s}_barcodes.tsv.gz", sep="\t", header=None)
    a = ad.AnnData(X=sp.csr_matrix(m), obs=pd.DataFrame(index=bars[0].values),
                   var=pd.DataFrame(index=feats[1].values))
    a.var_names_make_unique()
    a.obs["sample"] = s.split("_")[1]
    a.obs_names = s.split("_")[1] + "-" + a.obs_names
    a.var["mt"] = a.var_names.str.startswith("MT-")
    sc.pp.calculate_qc_metrics(a, qc_vars=["mt"], percent_top=None, log1p=False, inplace=True)
    keep = (a.obs.n_genes_by_counts.between(200, 5000)) & \
           (a.obs.total_counts.between(500, 10000)) & \
           (a.obs.pct_counts_mt < 20) & \
           (np.log10(a.obs.n_genes_by_counts) / np.log10(a.obs.total_counts) > 0.8)
    a = a[keep.values].copy()
    sc.pp.normalize_total(a, target_sum=1e4)
    sc.pp.log1p(a)
    if a.n_obs > 15000:
        sc.pp.subsample(a, n_obs=15000, random_state=0)
    print(s, "-> after QC+subsample:", a.shape, flush=True)
    parts.append(a)

adata = ad.concat(parts, join="inner")  # inner join genes, sparse preserved
print("concatenated:", adata.shape, flush=True)

sc.pp.highly_variable_genes(adata, n_top_genes=2000)
sc.tl.pca(adata, n_comps=30, use_highly_variable=True, zero_center=False)
sc.pp.neighbors(adata, n_neighbors=15, n_pcs=30)
sc.tl.umap(adata)
sc.tl.leiden(adata, resolution=0.5, key_added="leiden", flavor="igraph", n_iterations=2, directed=False)
print("clusters:", adata.obs.leiden.nunique(), flush=True)

markers = {
    "Cholangiocyte": ["KRT7","KRT19","SOX9","EPCAM","KRT8"],
    "Hepatocyte": ["ALB","TTR","HAMP","APOA1","CYP2E1"],
    "HSC_Fibroblast": ["COL1A1","COL1A2","ACTA2","PDGFRB","RGS5"],
    "Macrophage": ["CD68","LYZ","C1QA","C1QB","CD163","FCGR3A"],
    "T_cell": ["CD3D","CD3E","TRAC","CD2"],
    "B_cell": ["CD79A","MS4A1","CD19","BANK1"],
    "Plasma_cell": ["MZB1","JCHAIN","XBP1","SDC1"],
    "NK_cell": ["NKG7","GNLY","KLRD1","NCAM1"],
    "Endothelial": ["PECAM1","VWF","CLDN5","CDH5"],
}
X = adata.X
def _mean_per_cell(genes_sel):
    idx = adata.var_names.get_indexer(genes_sel)
    idx = idx[idx >= 0]
    sub = X[:, idx]
    return np.asarray(sub.mean(axis=1)).ravel()
score = pd.DataFrame({ct: _mean_per_cell([g for g in gs if g in adata.var_names])
                      for ct, gs in markers.items()}, index=adata.obs_names)
assign = score.idxmax(axis=1)
assign[score.max(axis=1) < 0.3] = "Other"
adata.obs["celltype"] = assign
print(adata.obs.celltype.value_counts(), flush=True)

# signature matrix: mean log1p expression per celltype (sparse group mean)
genes = adata.var_names.tolist()
sig = {}
for ct in sorted(set(assign)):
    mask = (assign == ct).to_numpy()
    sig[ct] = np.asarray(X[mask].mean(axis=0)).ravel()
sig_df = pd.DataFrame(sig, index=genes)
sig_df.to_csv("/workspace/data/scrna_signature_matrix.csv")
adata.write("/workspace/data/psc_scrna_ref.h5ad")
print("saved signature matrix:", sig_df.shape, flush=True)
