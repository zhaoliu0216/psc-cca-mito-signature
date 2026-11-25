"""Analyses 1-3 + publication PDF figures."""
import os
import pandas as pd, numpy as np, pickle, warnings, os
from scipy import stats
from scipy.optimize import nnls
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import logrank_test
import statsmodels.api as sm
warnings.filterwarnings("ignore")

plt.rcParams.update({
    "font.family": ["Liberation Sans", "DejaVu Sans"],
    "pdf.fonttype": 42, "axes.linewidth": 0.8, "font.size": 9,
})
FIG = "outputs/figures"; TAB = "outputs/tables"
os.makedirs(FIG, exist_ok=True); os.makedirs(TAB, exist_ok=True)
DATA = os.environ.get("PSC_DATA", "data")
GENES4 = ["COX5A", "PSAP", "THBS1", "COX8A"]
C_MAIN = "#0279EE"; C_HI = "#FF9400"; C_ACC = "#75A025"; C_PINK = "#FD9BED"

with open(f"{DATA}/cohorts.pkl", "rb") as f:
    cohorts = pickle.load(f)

# log2-transform non-log arrays
for k in ["GSE159676", "GSE26566", "GSE32225"]:
    e = cohorts[k]["expr"]
    cohorts[k]["expr"] = np.log2(e.clip(lower=1))

def zscore(v):
    s = v.std()
    return (v - v.mean()) / s if s and np.isfinite(s) and s > 0 else v * 0

def sig_score(expr, genes=GENES4):
    g = [x for x in genes if x in expr.index]
    return expr.loc[g].apply(zscore, axis=1).mean(axis=0)

def auc_ci(y, s, n_boot=2000, seed=0):
    y = np.asarray(y); s = np.asarray(s, float)
    auc = roc_auc_score(y, s)
    rng = np.random.default_rng(seed); bs = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        if len(np.unique(y[idx])) < 2: continue
        bs.append(roc_auc_score(y[idx], s[idx]))
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return auc, lo, hi

def delong_test(y, s1, s2):
    y = np.asarray(y); s1 = np.asarray(s1, float); s2 = np.asarray(s2, float)
    pos = y == 1; neg = ~pos
    def placements(s):
        p, n = s[pos], s[neg]
        V10 = (p[:, None] > n[None, :]) + 0.5 * (p[:, None] == n[None, :])
        return V10
    V1, V2 = placements(s1), placements(s2)
    m, n = V1.shape
    X10 = V1.mean(1) - V2.mean(1)
    X01 = (1 - V1).mean(0) - (1 - V2).mean(0)
    S10 = X10.var(ddof=1) / m if m > 1 else 0
    S01 = X01.var(ddof=1) / n if n > 1 else 0
    z = (V1.mean() - V2.mean()) / np.sqrt(S10 + S01)
    return z, 2 * stats.norm.sf(abs(z))

# ================================================================ Analysis 1
# ---- expression gradient
groups_plot = [
    ("GSE159676", "Liver tissue healthy", "Healthy liver"),
    ("GSE159676", "Primary sclerosing cholangitis", "PSC"),
    ("GSE26566", "Surrounding liver", "Peri-CCA liver"),
    ("GSE26566", "Cholangiocarcinoma", "CCA"),
    ("GSE32225", "Normal bile duct", "Normal bile duct"),
    ("GSE32225", "ICC", "ICC"),
    ("TCGA-CHOL", "Normal", "Normal (TCGA)"),
    ("TCGA-CHOL", "CCA", "CCA (TCGA)"),
]
rows = []
for gse, g, label in groups_plot:
    c = cohorts[gse]
    s = sig_score(c["expr"])
    for v in s[c["group"] == g].values:
        rows.append({"cohort": gse, "group": label, "score": v})
grad = pd.DataFrame(rows)
order = ["Healthy liver", "PSC", "Peri-CCA liver", "CCA", "Normal bile duct", "ICC", "Normal (TCGA)", "CCA (TCGA)"]
grad["group"] = pd.Categorical(grad["group"], categories=order, ordered=True)
grad.to_csv(f"{TAB}/S_gradient_scores.csv", index=False)

# Mann-Whitney: normal vs PSC, normal vs CCA within cohorts
stat_rows = []
for a, b, lab in [("Healthy liver", "PSC", "GSE159676"), ("Peri-CCA liver", "CCA", "GSE26566"),
                  ("Normal bile duct", "ICC", "GSE32225"), ("Normal (TCGA)", "CCA (TCGA)", "TCGA-CHOL")]:
    x = grad[grad.group == a]["score"]; yv = grad[grad.group == b]["score"]
    u, p = stats.mannwhitneyu(x, yv)
    stat_rows.append({"comparison": f"{a} vs {b}", "cohort": lab, "p_mannwhitney": p})
pd.DataFrame(stat_rows).to_csv(f"{TAB}/S_gradient_stats.csv", index=False)

# ---- ROC CCA vs normal per cohort
roc_specs = [
    ("GSE26566", "Cholangiocarcinoma", "Surrounding liver", "GSE26566: CCA vs peri-tumor liver"),
    ("GSE32225", "ICC", "Normal bile duct", "GSE32225: ICC vs normal bile duct"),
    ("TCGA-CHOL", "CCA", "Normal", "TCGA-CHOL: CCA vs normal"),
]
roc_res = []
fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.4))
for ax, (gse, pos, neg, title) in zip(axes, roc_specs):
    c = cohorts[gse]
    s = sig_score(c["expr"])
    y = (c["group"] == pos).astype(int).values
    keep = c["group"].isin([pos, neg]).values
    yy, ss = y[keep], s[keep].values
    auc, lo, hi = auc_ci(yy, ss)
    fpr, tpr, _ = roc_curve(yy, ss)
    ax.plot(fpr, tpr, color=C_MAIN, lw=1.6, label=f"4-gene signature\nAUC={auc:.3f} ({lo:.3f}-{hi:.3f})")
    ax.plot([0, 1], [0, 1], ls="--", lw=0.8, color="grey")
    ax.set_xlabel("1 - Specificity"); ax.set_ylabel("Sensitivity")
    ax.set_title(title, fontsize=9)
    ax.legend(fontsize=7, loc="lower right", frameon=False)
    roc_res.append({"cohort": gse, "comparison": f"{pos} vs {neg}", "n_pos": int(yy.sum()), "n_neg": int((1-yy).sum()),
                    "AUC": auc, "CI_lo": lo, "CI_hi": hi})
plt.tight_layout(); plt.savefig(f"{FIG}/Fig2_CCA_ROC.pdf"); plt.savefig(f"{FIG}/Fig2_CCA_ROC.png", dpi=200); plt.close()
pd.DataFrame(roc_res).to_csv(f"{TAB}/T1_CCA_ROC.csv", index=False)
print("ROC CCA:", pd.DataFrame(roc_res))

# ---- TCGA survival (cBioPortal clinical: complete OS/DFS)
clin = pd.read_csv(f"{DATA}/tcga_cbio_clinical.csv", index_col=0)
tpm = pd.read_parquet(f"{DATA}/tcga_tpm.parquet").set_index("gene_name")
fmap = pd.read_csv(f"{DATA}/tcga_chol_files.csv").set_index("sample")
tumor_samples = [c for c in tpm.columns if c in fmap.index and fmap.loc[c, "sample_type"] == "Primary Tumor"]
s_all = sig_score(np.log2(tpm + 1))
sig_t = s_all[tumor_samples]
clin = clin.loc[[c[:12] for c in sig_t.index]]
clin.index = sig_t.index
time = clin["OS_MONTHS"].astype(float).values * 30.44  # days
event = clin["OS_STATUS"].astype(str).str.startswith("1").astype(int).values
valid = ~np.isnan(time)
time, event = time[valid], event[valid]
sig_v = sig_t[valid].values
med = np.median(sig_v)
surv = pd.DataFrame({"time": time, "event": event, "sig": sig_v,
                     "high": (sig_v > med).astype(int)})
lr = logrank_test(surv[surv.high == 1]["time"], surv[surv.high == 0]["time"],
                  surv[surv.high == 1]["event"], surv[surv.high == 0]["event"])
cph = CoxPHFitter()
cph.fit(surv[["time", "event", "sig"]], "time", "event")
cox_sum = cph.summary
cox_sum.to_csv(f"{TAB}/T2_TCGA_Cox.csv")
print("TCGA OS Cox HR per SD:", np.exp(cox_sum["coef"]).round(2).values, "p:", cox_sum["p"].values)
print("logrank p (median split):", lr.p_value, "n:", len(surv), "events:", event.sum())

fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.4))
ax = axes[0]
kmf = KaplanMeierFitter()
for h, c, lab in [(1, C_HI, "High signature"), (0, C_MAIN, "Low signature")]:
    sub = surv[surv.high == h]
    kmf.fit(sub["time"], sub["event"], label=lab)
    kmf.plot_survival_function(ax=ax, color=c, ci_show=False, lw=1.6)
ax.set_xlabel("Days"); ax.set_ylabel("Overall survival")
ax.set_title(f"TCGA-CHOL OS (median split)\nlog-rank P = {lr.p_value:.3f}", fontsize=9)
ax.legend(frameon=False, fontsize=8)
ax = axes[1]
hr = np.exp(cox_sum["coef"])["sig"]; lo = np.exp(cox_sum["coef lower 95%"])["sig"]; hi = np.exp(cox_sum["coef upper 95%"])["sig"]
p_cox = cox_sum["p"]["sig"]
ax.errorbar([hr], [0], xerr=[[hr - lo], [hi - lo]], fmt="s", color=C_MAIN, capsize=3, markersize=6)
ax.axvline(1, ls="--", lw=0.8, color="grey")
ax.set_yticks([0]); ax.set_yticklabels(["4-gene signature (per SD)"])
ax.set_xlabel("Hazard ratio (95% CI)")
ax.set_title(f"Cox proportional hazards, OS\nHR = {hr:.2f} ({lo:.2f}-{hi:.2f}), P = {p_cox:.3f}", fontsize=9)
plt.tight_layout(); plt.savefig(f"{FIG}/Fig3_TCGA_survival.pdf"); plt.savefig(f"{FIG}/Fig3_TCGA_survival.png", dpi=200); plt.close()

# ================================================================ Analysis 2
GINDIN = {"EIF3E": -1, "ARHGAP5": -1, "AGRN": 1, "CEBPZOS": -1, "CEP170B": 1, "SBNO2": 1,
          "EIF4A2": -1, "NOTCH1": 1, "GRK2": 1, "HCFC1": 1, "CEBPZ": -1, "PIEZO1": 1,
          "MCRS1": 1, "ACAP3": 1, "GOLGA3": 1, "TECPR1": 1, "FANCL": -1, "AUH": -1,
          "FLII": 1, "MICALL1": 1}
lasch = pd.read_csv(f"{DATA}/laschtowitz_fibrosis_signature.csv")
LASCH = dict(zip(lasch["gene"], np.sign(lasch["log2FC"])))

def weighted_score(expr, wdict):
    g = [x for x in wdict if x in expr.index]
    z = expr.loc[g].apply(zscore, axis=1)
    w = pd.Series({x: wdict[x] for x in g})
    return z.mul(w, axis=0).mean(axis=0)

h2h_specs = [
    ("GSE159676", "Primary sclerosing cholangitis", "Liver tissue healthy", "Liver tissue (GSE159676)\nPSC vs healthy"),
    ("GSE177044", "PSC", "Control", "Whole blood (GSE177044)\nPSC vs control"),
]
h2h_rows = []
fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.8))
for ax, (gse, pos, neg, title) in zip(axes, h2h_specs):
    c = cohorts[gse]
    keep = c["group"].isin([pos, neg]).values
    y = (c["group"][keep] == pos).astype(int).values
    s4 = sig_score(c["expr"])[keep].values
    sg = weighted_score(c["expr"], GINDIN)[keep].values
    sl = weighted_score(c["expr"], LASCH)[keep].values
    specs = [("4-gene mitochondrial signature", s4, C_MAIN),
             ("Gindin 2021 risk DEGs (n=20)", sg, C_HI),
             ("Laschtowitz 2025 fibrosis set (n=431)", sl, C_ACC)]
    aucs = {}
    for lab, s, col in specs:
        auc, lo, hi = auc_ci(y, s)
        fpr, tpr, _ = roc_curve(y, s)
        ax.plot(fpr, tpr, color=col, lw=1.6, label=f"{lab}: {auc:.3f}")
        aucs[lab] = (auc, lo, hi)
    _, p_41 = delong_test(y, s4, sg)
    _, p_42 = delong_test(y, s4, sl)
    ax.plot([0, 1], [0, 1], ls="--", lw=0.8, color="grey")
    ax.set_xlabel("1 - Specificity"); ax.set_ylabel("Sensitivity")
    ax.set_title(title, fontsize=9)
    ax.legend(fontsize=6.5, loc="lower right", frameon=False,
              title=f"4-gene vs Gindin: DeLong P={p_41:.3f}\n4-gene vs Laschtowitz: DeLong P={p_42:.3f}",
              title_fontsize=6.5)
    for lab, s, _ in specs:
        auc, lo, hi = auc_ci(y, s)
        h2h_rows.append({"cohort": gse, "signature": lab, "AUC": auc, "CI_lo": lo, "CI_hi": hi,
                         "DeLong_P_vs_4gene": ""})
    h2h_rows[-3]["DeLong_P_vs_4gene"] = "-"
    h2h_rows[-2]["DeLong_P_vs_4gene"] = p_41
    h2h_rows[-1]["DeLong_P_vs_4gene"] = p_42
plt.tight_layout(); plt.savefig(f"{FIG}/Fig4_head2head.pdf"); plt.savefig(f"{FIG}/Fig4_head2head.png", dpi=200); plt.close()
pd.DataFrame(h2h_rows).to_csv(f"{TAB}/T3_head2head.csv", index=False)
print(pd.DataFrame(h2h_rows))

# ================================================================ Analysis 3
sig_mat = pd.read_csv(f"{DATA}/scrna_signature_matrix.csv", index_col=0)
sig_mat = sig_mat.drop(columns=["Other"])
res_rows = []

# ---- GSE159676 (liver): NNLS deconvolution with liver scRNA reference
c = cohorts["GSE159676"]
expr = c["expr"]
shared = sig_mat.index.intersection(expr.index)
S_df = sig_mat.loc[shared].copy()
B_df = expr.loc[shared].copy()
# gene-wise z-score both signature and bulk (cross-platform harmonization)
S_df = S_df.subtract(S_df.mean(axis=1), axis=0).div(S_df.std(axis=1).replace(0, 1), axis=0)
B_df = B_df.subtract(B_df.mean(axis=1), axis=0).div(B_df.std(axis=1).replace(0, 1), axis=0)
S = S_df.values
B = B_df.T.values  # samples x genes
W = np.zeros((B.shape[0], S.shape[1]))
for i in range(B.shape[0]):
    w, _ = nnls(S, B[i])
    W[i] = w / (w.sum() + 1e-9)
prop = pd.DataFrame(W, index=expr.columns, columns=sig_mat.columns)
prop["group"] = (c["group"] == "Primary sclerosing cholangitis").astype(int)
prop.to_csv(f"{TAB}/T4_deconv_GSE159676.csv")

s4 = sig_score(expr)
y = prop["group"].values
# univariate cell-type differences
cell_p = {}
for ct in sig_mat.columns:
    r = stats.mannwhitneyu(prop[prop.group == 1][ct], prop[prop.group == 0][ct])
    cell_p[ct] = r.pvalue
print("cell-type proportions PSC vs healthy (Mann-Whitney p):", {k: round(v, 3) for k, v in cell_p.items()})

# logistic: unadjusted vs adjusted (covariates = all proportions except collinear; use macrophage+HSC+hepatocyte as main)
covs = ["Macrophage", "HSC_Fibroblast", "Hepatocyte", "Cholangiocyte"]
def logit_fit(y, X_df):
    X_df = X_df.loc[:, X_df.std() > 0] if hasattr(X_df, "std") else X_df
    X = sm.add_constant(X_df)
    ok = X.replace([np.inf, -np.inf], np.nan).notna().all(axis=1) & pd.notna(y)
    X, y = X[ok.values], np.asarray(y)[ok.values]
    m = sm.Logit(y, X).fit(disp=0)
    return m
X_un = pd.DataFrame({"sig": zscore(s4)})
X_adj = pd.DataFrame({"sig": zscore(s4)}, index=prop.index)
for ct in covs:
    X_adj[ct] = zscore(prop[ct])
X_cov_only = X_adj.drop(columns=["sig"])
m1 = logit_fit(y, X_cov_only)
m2 = logit_fit(y, X_adj)
# LRT: does sig add beyond cell composition?
llr = 2 * (m2.llf - m1.llf)
p_lrt = stats.chi2.sf(llr, 1)
b_un = m_un = logit_fit(y, X_un)
res_rows.append({"cohort": "GSE159676 (liver)", "model": "unadjusted", "OR_sig": np.exp(m_un.params["sig"]),
                 "p_sig": m_un.pvalues["sig"], "LRT_P_beyond_composition": ""})
res_rows.append({"cohort": "GSE159676 (liver)", "model": "adjusted (macrophage, HSC, hepatocyte, cholangiocyte)",
                 "OR_sig": np.exp(m2.params["sig"]), "p_sig": m2.pvalues["sig"], "LRT_P_beyond_composition": p_lrt})
print("GSE159676 LRT p (signature beyond composition):", p_lrt, "| adj OR:", np.exp(m2.params["sig"]), "p:", m2.pvalues["sig"])

# ---- GSE177044 (whole blood): blood marker module covariates
c = cohorts["GSE177044"]
expr = c["expr"]
blood_mods = {
    "T_cell": ["CD3D", "CD3E", "TRAC", "CD2"],
    "B_cell": ["CD79A", "MS4A1", "CD19"],
    "NK_cell": ["NKG7", "GNLY", "KLRD1"],
    "Monocyte": ["CD68", "LYZ", "S100A8", "S100A9", "LST1"],
    "Neutrophil": ["CSF3R", "FCGR3B", "CXCR2", "S100A12"],
    "Erythrocyte": ["HBB", "HBA1", "HBA2"],
}
mods = pd.DataFrame({k: expr.loc[[g for g in g_ if g in expr.index]].apply(zscore, axis=1).mean(axis=0)
                     for k, g_ in blood_mods.items()})
keep = c["group"].isin(["PSC", "Control"]).values
y = (c["group"][keep] == "PSC").astype(int).values
s4b = sig_score(expr)[keep].values
mb = mods.loc[keep]
X_un = pd.DataFrame({"sig": zscore(s4b)})
X_adj = pd.DataFrame({"sig": zscore(s4b)}).join(mb.reset_index(drop=True))
m_un = logit_fit(y, X_un)
m2 = logit_fit(y, X_adj)
m1 = logit_fit(y, X_adj.drop(columns=["sig"]))
llr = 2 * (m2.llf - m1.llf)
p_lrt_b = stats.chi2.sf(llr, 1)
res_rows.append({"cohort": "GSE177044 (whole blood)", "model": "unadjusted", "OR_sig": np.exp(m_un.params["sig"]),
                 "p_sig": m_un.pvalues["sig"], "LRT_P_beyond_composition": ""})
res_rows.append({"cohort": "GSE177044 (whole blood)", "model": "adjusted (6 blood lineages)",
                 "OR_sig": np.exp(m2.params["sig"]), "p_sig": m2.pvalues["sig"], "LRT_P_beyond_composition": p_lrt_b})
print("GSE177044 LRT p:", p_lrt_b, "| adj OR:", np.exp(m2.params["sig"]), "p:", m2.pvalues["sig"])
res = pd.DataFrame(res_rows)
res.to_csv(f"{TAB}/T5_adjusted_logistic.csv", index=False)
print(res)

# ---- Figure 5
fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.6))
ax = axes[0]
prop_long = prop.melt(id_vars="group", value_vars=sig_mat.columns, var_name="celltype", value_name="proportion")
import seaborn as sns
sns.boxplot(data=prop_long, x="celltype", y="proportion", hue="group", ax=ax, fliersize=0,
            palette=["#9ec9f5", C_MAIN], linewidth=0.8)
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=7)
ax.set_title("Deconvolved liver cell proportions\nGSE159676 (scRNA reference, n=3 PSC donors)", fontsize=8)
ax.legend(frameon=False, fontsize=7, title="PSC", title_fontsize=7)
ax = axes[1]
labs, ors, los, his, cols = [], [], [], [], []
for _, r in res.iterrows():
    labs.append(f"{r['cohort']}\n{r['model'].split('(')[0].strip()}")
    ors.append(r["OR_sig"])
    # recompute CI from logistic fits is complex; store from fits
# recompute CIs properly
ci_store = {}
for (gse_lab, Xdf, yy) in [("GSE159676 (liver)", X_adj, y), ]:
    pass
# simpler: refit and capture CI
def fit_or_ci(yy, Xdf):
    m = logit_fit(yy, Xdf)
    b = m.params["sig"]; se = m.bse["sig"]
    return np.exp(b), np.exp(b - 1.96 * se), np.exp(b + 1.96 * se), m.pvalues["sig"]
or_u_l, lo_u_l, hi_u_l, p_u_l = fit_or_ci(y, X_un)
or_a_l, lo_a_l, hi_a_l, p_a_l = fit_or_ci(y, X_adj)
or_u_b, lo_u_b, hi_u_b, p_u_b = fit_or_ci(np.array(y), X_un)
or_a_b, lo_a_b, hi_a_b, p_a_b = fit_or_ci(np.array(y), X_adj)
pts = [("Liver unadj.", or_u_l, lo_u_l, hi_u_l, p_u_l, C_MAIN),
       ("Liver adj.", or_a_l, lo_a_l, hi_a_l, p_a_l, C_MAIN),
       ("Blood unadj.", or_u_b, lo_u_b, hi_u_b, p_u_b, C_HI),
       ("Blood adj.", or_a_b, lo_a_b, hi_a_b, p_a_b, C_HI)]
for i, (lab, o, l_, h_, p_, col) in enumerate(pts):
    ax.errorbar(o, i, xerr=[[o - l_], [h_ - o]], fmt="s", color=col, capsize=3, markersize=6)
    ax.text(max(h_, o) * 1.15, i, f"P={p_:.1e}", va="center", fontsize=7)
ax.axvline(1, ls="--", lw=0.8, color="grey")
ax.set_yticks(range(len(pts))); ax.set_yticklabels([p[0] for p in pts], fontsize=8)
ax.set_xlabel("Odds ratio of PSC per 1-SD signature score (95% CI)")
ax.set_xscale("log")
ax.set_title("Signature association before/after\nadjustment for cell composition", fontsize=8)
ax = axes[2]
ax.bar(["Liver\n(GSE159676)", "Whole blood\n(GSE177044)"],
       [-np.log10(p_lrt), -np.log10(p_lrt_b)], color=[C_MAIN, C_HI], width=0.5)
ax.axhline(-np.log10(0.05), ls="--", lw=0.8, color="grey")
ax.text(1.4, -np.log10(0.05), "P=0.05", fontsize=7, va="center")
ax.set_ylabel("-log10(LRT P)")
ax.set_title("Signature added beyond cell composition\n(likelihood-ratio test)", fontsize=8)
plt.tight_layout(); plt.savefig(f"{FIG}/Fig5_deconvolution.pdf"); plt.savefig(f"{FIG}/Fig5_deconvolution.png", dpi=200); plt.close()

# ================================================================ Figure 1 (gradient)
fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8))
ax = axes[0]
pal = {"Healthy liver": "#9ec9f5", "PSC": C_MAIN, "Peri-CCA liver": "#ffd28a", "CCA": C_HI,
       "Normal bile duct": "#c8e6c0", "ICC": C_ACC, "Normal (TCGA)": "#e0b3e8", "CCA (TCGA)": C_PINK}
data = [grad[grad.group == g]["score"].values for g in order]
bp = ax.boxplot(data, tick_labels=order, patch_artist=True, widths=0.6, flierprops=dict(markersize=2))
for patch, g in zip(bp["boxes"], order):
    patch.set_facecolor(pal[g]); patch.set_alpha(0.85)
for i, d in enumerate(data):
    x = np.random.default_rng(0).normal(i + 1, 0.06, len(d))
    ax.plot(x, d, ".", color="black", markersize=3, alpha=0.5)
ax.set_xticklabels(order, rotation=40, ha="right", fontsize=7)
ax.set_ylabel("4-gene signature score (mean z)")
ax.set_title("Signature score across the PSC-to-CCA axis", fontsize=9)
ax = axes[1]
# per-gene log2FC CCA vs normal per cohort
fc_rows = []
for gse, pos, neg, lab in [("GSE26566", "Cholangiocarcinoma", "Surrounding liver", "GSE26566"),
                           ("GSE32225", "ICC", "Normal bile duct", "GSE32225"),
                           ("TCGA-CHOL", "CCA", "Normal", "TCGA-CHOL")]:
    e = cohorts[gse]["expr"]
    for g in GENES4:
        a_ = e.loc[g, cohorts[gse]["group"] == pos].values
        b_ = e.loc[g, cohorts[gse]["group"] == neg].values
        fc_rows.append({"cohort": lab, "gene": g, "log2FC": a_.mean() - b_.mean(),
                        "p": stats.mannwhitneyu(a_, b_).pvalue})
fc = pd.DataFrame(fc_rows)
fc.to_csv(f"{TAB}/S_gene_log2FC_CCA.csv", index=False)
piv = fc.pivot(index="gene", columns="cohort", values="log2FC").loc[GENES4]
piv.plot(kind="bar", ax=ax, color=[C_MAIN, C_HI, C_ACC], width=0.75)
ax.axhline(0, lw=0.8, color="black")
ax.set_ylabel("log2 fold change (CCA vs normal)")
ax.set_title("Per-gene expression change along the axis", fontsize=9)
ax.legend(frameon=False, fontsize=7)
ax.set_xticklabels(GENES4, rotation=0)
plt.tight_layout(); plt.savefig(f"{FIG}/Fig1_CCA_gradient.pdf"); plt.savefig(f"{FIG}/Fig1_CCA_gradient.png", dpi=200); plt.close()

print("ALL DONE")
