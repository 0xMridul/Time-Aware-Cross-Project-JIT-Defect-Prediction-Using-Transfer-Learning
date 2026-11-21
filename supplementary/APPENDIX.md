# Unified Supplementary Appendix

This unified appendix combines the supplementary documentation for the research manuscript.

---

# Supplementary Document: Dataset Composition and Target Cohort Curation

### 1. Benchmark Origin and Characteristics
The ApacheJIT benchmark was compiled by Zeng et al. (ISSTA 2021) and deposited at Zenodo (DOI: 10.5281/zenodo.5907001). It extracts commit-level software changes, SZZ defect labels, and code metrics from 13 major open-source repositories managed by the Apache Software Foundation (ASF).

### 2. Benchmark Scale and Deduplication
- Total records in `apachejit_total.csv`: **106,674 rows**.
- Unique commit hashes (`commit_id`): **106,674**.
- Deduplication check confirms zero duplicate commits exist in the canonical benchmark file.

### 3. Repository Membership and Target Selection Criteria
We establish two explicit inclusion criteria for target evaluation:
1. **Scale Threshold:** Project must contain $\ge 2,000$ total commits.
2. **Defect Threshold:** Project's test period ($2016$–$2017$) must contain $\ge 50$ buggy commits.

Ten projects meet both criteria and form our primary target cohort:
`apache/camel`, `apache/hadoop`, `apache/ignite`, `apache/flink`, `apache/hbase`, `apache/cassandra`, `apache/groovy`, `apache/hive`, `apache/activemq`, `apache/kafka`.

Three repositories fail the 50-bug test threshold in 2016–2017 and serve exclusively as donor training sources:
- `apache/spark`: 1,465 commits (632 buggy; 4 test bugs in 2016–2017)
- `apache/zeppelin`: 1,451 commits (622 buggy; 18 test bugs in 2016–2017)
- `apache/zookeeper`: 839 commits (342 buggy; 6 test bugs in 2016–2017)

### 4. Merged Git Subprojects (Hadoop)
In early Apache history, Hadoop subprojects were hosted across separate repositories before being merged into the unified `apache/hadoop` root. In `apachejit_total.csv`, commits are recorded under:
- `apache/hadoop`: 11,964 commits
- `apache/hadoop-hdfs`: 2,907 commits
- `apache/hadoop-mapreduce`: 1,321 commits
Total unified Hadoop commits: **16,192**. All analyses merge these subproject tags into the canonical `apache/hadoop` project.


---

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


---

# Supplementary Document: Experimental Splitting Protocols

### 1. Zero Future-to-Past Leakage Principle
A fundamental vulnerability of prior CP-JIT evaluations is the violation of temporal ordering (e.g. cross-validation, or pooling future cross-project commits to predict past target commits).
In all protocols in this study:
$$\forall c_{\text{train}} \in \mathcal{D}_{\text{train}}, \quad \text{author\_date}(c_{\text{train}}) < \min_{c_{\text{test}} \in \mathcal{D}_{\text{test}}} \text{author\_date}(c_{\text{test}})$$
Cross-project donor pools are strictly subject to this cutoff. No donor commit authored after the target training cutoff date is ever visible to training.

### 2. Protocol Definitions
- **Design B (Primary Confirmatory Protocol):** Fixed test window 2016–2017 (`author_date` in $[1451606400, 1514764799]$). Training spans all commits authored prior to 2016-01-01 (`author_date < 1451606400`).
- **Protocol D (Longitudinal Temporal Drift):** Fixed test window 2016–2017. Six historical training endpoints: $\le 2015$ (Gap 0), $\le 2014$ (Gap 1), $\le 2013$ (Gap 2), $\le 2012$ (Gap 3), $\le 2011$ (Gap 4), $\le 2010$ (Gap 5). Evaluated under Natural Growing and Fixed Training Size variants.
- **70% Per-Project Chronological Split (Sensitivity 4):** Project-specific 70/30 chronological split based on author date.
- **Protocol A (Sensitivity 5):** Official Zenodo split deposited by Zeng et al.


---

# Supplementary Document: Inferential Statistics, Hypothesis Testing, and TOST

### 1. Paired Wilcoxon Signed-Rank Test
For project-level paired metrics ($x_i, y_i$), differences $d_i = x_i - y_i$ are computed across the $n = 10$ target projects. The two-tailed Wilcoxon signed-rank test evaluates the null hypothesis that differences are symmetrically distributed around zero. Effect sizes are measured via the rank-biserial correlation:
$$r_{rb} = \frac{W^+ - W^-}{W^+ + W^-}$$

### 2. Family-Wide Multiple Testing Correction
We correct for family-wise error rate (FWER) across the primary confirmatory family of 20 comparisons (2 classifiers $\times$ 2 metrics $\times$ 5 comparisons) using the sequential rejective Holm-Bonferroni procedure at $\alpha = 0.05$.

### 3. Bootstrap Confidence Intervals
Confidence intervals for mean paired differences are constructed using $\ge 10,000$ bias-corrected or percentile bootstrap resamples over projects.

### 4. Two-One-Sided Equivalence Testing (TOST)
To establish that transfer learning is practically indistinguishable from unadapted pooling, we test equivalence against a pre-registered margin of $\delta = \pm 0.02$ ROC-AUC.
We evaluate:
$$H_{01}: \theta \le -0.02 \quad \text{vs.} \quad H_{11}: \theta > -0.02$$
$$H_{02}: \theta \ge +0.02 \quad \text{vs.} \quad H_{12}: \theta < +0.02$$
$$p_{\text{TOST}} = \max(p_1, p_2)$$
Conventional $\alpha = 0.05$ TOST corresponds to the $90\%$ two-sided confidence interval $(1 - 2\alpha)$.


---

# Supplementary Document: Additional Project-Level Empirical Results

### 1. Secondary Metrics Summary (Primary Design B)
Below are secondary classification performance metrics (PR-AUC, F1-score at Youden's $J$, and Matthews Correlation Coefficient) across the 10 target projects:

| Model | Method | Mean ROC-AUC | Mean PR-AUC | Mean F1 (Youden) | Mean MCC (Youden) |
|:---|:---|:---:|:---:|:---:|:---:|
| **LR** | WP | 0.7684 | 0.4907 | 0.4121 | 0.3125 |
| **LR** | Pooled CP | 0.7879 | 0.5074 | 0.4284 | 0.3341 |
| **LR** | Source + Target | 0.7886 | 0.5103 | 0.4301 | 0.3362 |
| **LR** | CORAL | 0.7853 | 0.4990 | 0.4215 | 0.3278 |
| **LR** | TCA with standardization | 0.7869 | 0.5047 | 0.4259 | 0.3310 |
| **RF** | WP | 0.7607 | 0.4988 | 0.4312 | 0.3298 |
| **RF** | Pooled CP | 0.7727 | 0.4835 | 0.4290 | 0.3305 |
| **RF** | Source + Target | 0.7752 | 0.4925 | 0.4344 | 0.3382 |
| **RF** | CORAL | 0.7557 | 0.4609 | 0.4089 | 0.3084 |
| **RF** | TCA with standardization | 0.7658 | 0.4760 | 0.4211 | 0.3219 |

*Full per-project and per-seed results archived in `results/primary_analysis.csv`.*

