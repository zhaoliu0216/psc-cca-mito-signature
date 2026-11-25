# A 4-gene mitochondrial bioenergetic signature tracking the PSC-to-CCA progression axis

Reproducible analysis code for the manuscript:

> **A mitochondrial bioenergetic signature marks the progression axis from primary
> sclerosing cholangitis to cholangiocarcinoma and identifies candidate points of
> therapeutic interception.**

The signature is the mean gene-wise z-score of **COX5A, PSAP, THBS1, COX8A**
computed within each cohort. This repository contains the code used to derive,
validate, and benchmark that signature on public cohorts.

## Repository layout

```
scripts/
  00_parse_platform.py     Stream-parse GPL (Affymetrix/Illumina) soft files into probe->gene maps
  01_prepare_cohorts.py    Build per-cohort expression matrices + group labels; save data/cohorts.pkl
  02_signature_validation.py  Signature scoring, expression gradient, ROC/AUC (+bootstrap CI),
                           paired DeLong tests vs published signatures, NNLS deconvolution
                           against a PSC scRNA reference, adjusted logistic regression +
                           likelihood-ratio test, TCGA-CHOL survival (KM + Cox)
  03_scrna_reference.py    Per-sample QC + subsample + sparse PCA for the PSC scRNA reference
                           (GSE247128 3' scRNA-seq donors)
outputs/                   Figures (PDF/PNG) and tables (CSV) are written here
data/                      Downloaded/raw data (NOT committed — see below)
```

## Data sources (all public)

| Cohort | Accession | Platform | Role in manuscript |
|---|---|---|---|
| Liver tissue: healthy / PSC | [GSE159676](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE159676) | HuGene microarray | Discovery + training cohort; deconvolution target |
| CCA tissue: CCA / surrounding liver / normal bile duct | [GSE26566](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE26566) | Illumina microarray | External validation (n = 163 used; 6 normal bile duct samples excluded from cohort-wise z-scoring) |
| ICC tissue: ICC / normal bile duct | [GSE32225](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE32225) | Illumina microarray | External validation |
| Cholangiocarcinoma (TCGA-CHOL) | [TCGA-CHOL](https://portal.gdc.cancer.gov/) via GDC / [cBioPortal](https://www.cbioportal.org/) | RNA-seq (TPM) | External validation + survival (supplementary) |
| PSC liver single-cell reference | [GSE247128](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE247128) | 10x 3' scRNA-seq | Deconvolution reference (3 donors, 33,346 cells, 9 lineages) |

### Download instructions

1. Download the SOFT/family files for GSE159676, GSE26566, GSE32225 from GEO and
   extract each series matrix; run `00_parse_platform.py` on the GPL tables
   (GPL6244, GPL6104, GPL8432) to build `data/gpl*_map.csv` probe->symbol maps.
2. Build the per-cohort expression pickles expected by `01_prepare_cohorts.py`
   (`GSE159676_expr.pkl`, `GSE26566_expr.pkl`, `GSE32225_expr.pkl` and their
   `*_metadata.csv`; see the loading section of that script for the exact columns).
3. TCGA-CHOL: export TPM matrix + file metadata from GDC, and clinical OS data from
   cBioPortal (`tcga_tpm.parquet`, `tcga_chol_files.csv`, `tcga_cbio_clinical.csv`).
4. GSE247128: download the 10x matrices for the PSC donors listed in
   `03_scrna_reference.py` into `data/scrna/`.
5. Point the pipeline at the data directory:

```bash
export PSC_DATA=./data   # default; any directory with the files above
python scripts/01_prepare_cohorts.py   # builds data/cohorts.pkl
python scripts/02_signature_validation.py
```

## Key preprocessing notes (required to reproduce manuscript numbers)

- GSE26566 / GSE32225 series matrices contain **raw-intensity** values: values are
  log2(x+1)-transformed (the code applies `log2(clip(x, 1))`) before z-scoring.
- GSE159676 is delivered log-scale, but ~0.9% of probe values are on a mixed scale
  (>100): these are re-log2-transformed before z-scoring.
- The cohort-wise z-score for **GSE26566 excludes the 6 normal intrahepatic bile duct
  samples** (z within n = 163), which reproduces the manuscript's external-validation
  statistics exactly (e.g. P = 1.29e-09 for the CCA-vs-surrounding comparison).
- Deconvolution: gene-wise z-scores are computed on **both** the scRNA reference and
  the bulk matrix (cross-platform harmonization) before NNLS; proportions are
  normalized to sum to 1.

## Mapping to manuscript figures / tables

| Script output | Manuscript item |
|---|---|
| `outputs/tables/S_gradient_scores.csv`, `S_gradient_stats.csv` | Stepwise gradient (healthy -> PSC -> peri-tumoural -> CCA) |
| `outputs/tables/T1_CCA_ROC.csv` + ROC figures | External-validation AUCs 0.787 / 0.800 / 0.990 |
| `outputs/tables/T3_head2head.csv` | Paired DeLong tests vs the 20-gene (Gindin) and 431-gene (Laschtowitz) signatures |
| `outputs/tables/T4_deconv_GSE159676.csv`, `T5_adjusted_logistic.csv` | Deconvolution, adjusted OR (0.76) and LRT P = 0.54 |
| `outputs/figures/Fig3_TCGA_survival.*` | Supplementary TCGA-CHOL survival (log-rank P = 0.317; HR = 1.41) |

Note: `02_signature_validation.py` retains an optional whole-blood (GSE177044)
comparison module from an earlier iteration of the work; this cohort is **not** part
of the final manuscript and the module can be skipped.

## Environment

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## License

MIT License — see [LICENSE](LICENSE).
