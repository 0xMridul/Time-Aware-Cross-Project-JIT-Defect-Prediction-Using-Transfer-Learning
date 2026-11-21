# Supplementary Document: Feature Engineering and Kamei Metric Definitions

### 1. Kamei et al. Change Metrics
We extract standard code change metrics proposed by Kamei et al. (IEEE TSE 2013):

| Metric | Dimension | Definition | Transformation |
|:---|:---|:---|:---|
| **Lines Added (`la`)** | Diffusion | Number of lines added by the commit | $\log(1 + x)$, Standardized |
| **Lines Deleted (`ld`)** | Diffusion | Number of lines deleted by the commit | $\log(1 + x)$, Standardized |
| **Files Changed (`nf`)** | Diffusion | Number of modified files | $\log(1 + x)$, Standardized |
| **Directories Changed (`nd`)** | Diffusion | Number of modified directories | $\log(1 + x)$, Standardized |
| **Subsystems Changed (`ns`)** | Diffusion | Number of modified top-level subsystems | $\log(1 + x)$, Standardized |
| **Entropy (`ent`)** | Diffusion | Distribution of modifications across files ($-\sum p_i \log_2 p_i$) | Raw, Standardized |
| **Developers (`ndev`)** | History | Number of developers previously modifying the changed files | $\log(1 + x)$, Standardized |
| **Age (`age`)** | History | Average time interval since previous modification of files | $\log(1 + x)$, Standardized |
| **Unique Changes (`nuc`)** | History | Number of unique changes made to the files | $\log(1 + x)$, Standardized |
| **Author Experience (`aexp`)** | Experience | Cumulative geometric experience of the commit author | $\log(1 + x)$, Standardized |
| **Recent Experience (`arexp`)** | Experience | Author experience weighted toward recent commits | $\log(1 + x)$, Standardized |
| **Subsystem Experience (`asexp`)** | Experience | Author experience within the modified subsystems | $\log(1 + x)$, Standardized |
| **Fix Commit Flag (`fix`)** | Purpose | Binary flag: mined from retrospective Jira resolution status | Binary {0, 1} (Leakage Sensitivity only) |

### 2. Primary 12-Feature vs. 13-Feature Rationale
In our primary analysis, we evaluate the **12 continuous and count-based metrics** ($\mathcal{F}_{12}$). The binary commit purpose flag `fix` is excluded from primary analysis and evaluated strictly in Sensitivity Analysis 1.
- *Reasoning:* In the ApacheJIT benchmark, `fix` is mined from Jira issue tracking records where issues are labeled with `Resolution = Fixed` and `Issue Type = Bug`. Because issue resolution is assigned retrospectively upon bug verification and ticket closure, `fix` is unavailable at commit authoring time. Furthermore, `fix` is present in 27.49% of defect-inducing commits but only 3.73% of clean commits (a 7.37-fold enrichment), introducing post-commit information leakage that artificially inflates apparent ROC-AUC by +0.049 to +0.066. Excluding `fix` is therefore mandatory for realistic pre-commit defect prediction. The 13-feature experiment is preserved strictly as a leakage sensitivity analysis. In addition, for the performance anomaly on Hive under RF CORAL (-0.115 ROC-AUC vs. Pooled CP), covariance rotation on discrete indicators represents a possible contributing factor alongside Hive's extreme base rate (70.30%), rather than a proven causal mechanism.
