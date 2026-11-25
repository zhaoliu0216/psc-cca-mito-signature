"""Supplementary analyses 1-3 for PSC mitochondrial signature manuscript.
Outputs: publication-ready PDF figures + CSV tables.
"""
import os
import pandas as pd, numpy as np, gzip, warnings, os
import scipy.sparse as sp
from scipy.optimize import nnls
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

plt.rcParams.update({
    "font.family": ["Liberation Sans", "DejaVu Sans"],
    "svg.fonttype": "none", "pdf.fonttype": 42,
    "axes.linewidth": 0.8, "font.size": 9,
})
FIG = "outputs/figures"; TAB = "outputs/tables"
os.makedirs(FIG, exist_ok=True); os.makedirs(TAB, exist_ok=True)

DATA = os.environ.get("PSC_DATA", "data")
GENES4 = ["COX5A", "PSAP", "THBS1", "COX8A"]

# ---------------------------------------------------------------- helpers
def zscore(v):
    return (v - v.mean()) / v.std()

def sig_score(expr_df, genes=GENES4):
    """mean z-score of signature genes present in expr_df (rows=genes)."""
    g = [x for x in genes if x in expr_df.index]
    z = expr_df.loc[g].apply(zscore, axis=1)
    return z.mean(axis=0)

def log2cpm(counts_df):
    lib = counts_df.sum(axis=0)
    return np.log2(counts_df.div(lib, axis=1) * 1e6 + 1)

def delong_auc_ci(y, s, n_boot=2000, seed=0):
    """AUC + bootstrap CI + DeLong variance."""
    from sklearn.metrics import roc_auc_score
    y = np.asarray(y); s = np.asarray(s, dtype=float)
    auc = roc_auc_score(y, s)
    rng = np.random.default_rng(seed)
    n = len(y)
    bs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(np.unique(y[idx])) < 2: continue
        bs.append(roc_auc_score(y[idx], s[idx]))
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return auc, lo, hi

def delong_test(y1, s1, y2, s2):
    """DeLong correlation/covariance test for paired AUCs (same samples, y1==y2)."""
    y = np.asarray(y1)
    s1 = np.asarray(s1, float); s2 = np.asarray(s2, float)
    pos = y == 1; neg = y == 0
    # placement values
    def v_us(s):
        # structural: V10 (pos vs neg), V01
        p, n = s[pos], s[neg]
        V10 = np.array([[1 if a > b else (0.5 if a == b else 0) for b in n] for a in p])
        V01 = 1 - V10
        return V10, V01
    V10_1, V01_1 = v_us(s1)
    V10_2, V01_2 = v_us(s2)
    m, n = V10_1.shape
    X10 = V10_1.mean(1) - V10_2.mean(1)  # per-pos diff
    X01 = V01_1.mean(0) - V01_2.mean(0)
    S10 = X10.var(ddof=1) / m if m > 1 else 0
    S01 = X01.var(ddof=1) / n if n > 1 else 0
    z = (V10_1.mean() - V10_2.mean()) / np.sqrt(S10 + S01)
    p = 2 * stats.norm.sf(abs(z))
    return z, p

def roc_curve_pts(y, s):
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(y, s)
    return fpr, tpr

# ---------------------------------------------------------------- load cohorts
cohorts = {}

# GSE159676 (liver microarray, HuGene)
mat = pd.read_pickle(f"{DATA}/GSE159676_expr.pkl")
mat.index = mat.index.str.strip('"')
mp = pd.read_csv(f"{DATA}/gpl6244_map.csv", dtype={"probe": str}).set_index("probe")["symbol"]
mat.index = mat.index.map(mp)
mat = mat[~mat.index.isna()]
mat = mat.groupby(level=0).mean()
md = pd.read_csv(f"{DATA}/GSE159676_metadata.csv")
grp = md["characteristics_ch1"].str.replace("condition: ", "", regex=False).values
cohorts["GSE159676"] = {"expr": mat, "group": pd.Series(grp, index=mat.columns)}

# GSE177044 (whole blood RNA-seq counts)
cnt = pd.read_csv(f"{DATA}/GSE177044_raw_countsPSC.csv.gz", index_col=0)
cnt = cnt.drop(columns=["gene_name"])
cnt.index = [g.split(".")[0] for g in cnt.index]
gname = pd.read_csv(f"{DATA}/GSE177044_raw_countsPSC.csv.gz", index_col=0)["gene_name"]
cnt.index = gname.values
cnt = cnt.groupby(level=0).sum()
prefix = pd.Series(cnt.columns).str.extract(r"^([A-Za-z]+)")[0].values
cohorts["GSE177044"] = {"expr": log2cpm(cnt), "group": pd.Series(prefix, index=cnt.columns)}

# GSE26566 (CCA microarray Illumina)
mat = pd.read_pickle(f"{DATA}/GSE26566_expr.pkl")
mat.index = mat.index.str.strip('"')
mp = pd.read_csv(f"{DATA}/gpl6104_map.csv").set_index("probe")["symbol"]
mat.index = mat.index.map(mp)
mat = mat[~mat.index.isna()].groupby(level=0).mean()
md = pd.read_csv(f"{DATA}/GSE26566_metadata.csv")
grp = md["source_name_ch1"].str.extract(r"^(Surrounding liver|Cholangiocarcinoma|Normal intrahepatic bile duct)")[0].values
cohorts["GSE26566"] = {"expr": mat, "group": pd.Series(grp, index=mat.columns)}

# GSE32225 (ICC microarray Illumina)
mat = pd.read_pickle(f"{DATA}/GSE32225_expr.pkl")
mat.index = mat.index.str.strip('"')
mp = pd.read_csv(f"{DATA}/gpl8432_map.csv").set_index("probe")["symbol"]
mat.index = mat.index.map(mp)
mat = mat[~mat.index.isna()].groupby(level=0).mean()
md = pd.read_csv(f"{DATA}/GSE32225_metadata.csv")
grp = np.where(md["source_name_ch1"].str.contains("intrahepatic cholangiocarcinoma"), "ICC", "Normal bile duct")
cohorts["GSE32225"] = {"expr": mat, "group": pd.Series(grp, index=mat.columns)}

# TCGA-CHOL
tpm = pd.read_parquet(f"{DATA}/tcga_tpm.parquet").set_index("gene_name")
meta = pd.read_csv(f"{DATA}/tcga_chol_files.csv").set_index("sample")
grp = meta.loc[tpm.columns, "sample_type"].map({"Primary Tumor": "CCA", "Solid Tissue Normal": "Normal"}).values
cohorts["TCGA-CHOL"] = {"expr": np.log2(tpm + 1), "group": pd.Series(grp, index=tpm.columns)}

for k, v in cohorts.items():
    print(k, v["expr"].shape, dict(v["group"].value_counts()))
    print("  expr range:", v["expr"].iloc[:5].values.min().round(2), v["expr"].iloc[:5].values.max().round(2))

# check signature genes present
for k, v in cohorts.items():
    print(k, "genes found:", [g for g in GENES4 if g in v["expr"].index])

import pickle
with open(f"{DATA}/cohorts.pkl", "wb") as f:
    pickle.dump(cohorts, f)
print("cohorts saved")
