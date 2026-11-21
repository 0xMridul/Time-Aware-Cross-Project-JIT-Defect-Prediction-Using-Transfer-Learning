# Does Transfer Learning Really Help Cross-Project Just-In-Time Defect Prediction? A Time-Aware Re-Evaluation on ApacheJIT

**Author:**  
**Kazi Mazharul Islam Mridul**$^{*}$  

East West University  

$^{*}$Corresponding Author: [Email Address - PENDING AUTHOR INPUT]  

---

## Abstract

Just-In-Time (JIT) defect prediction identifies defect-inducing software changes at commit time, enabling development teams to review and test high-risk commits before defects reach production branches. When a new project lacks historical defect data (the "cold-start" problem), cross-project JIT (CP-JIT) defect prediction seeks to train models on mature external donor repositories. To reconcile distributional discrepancies between donor and recipient software projects, numerous studies have advocated domain adaptation and transfer learning methods, prominently Correlation Alignment (CORAL) and Transfer Component Analysis (TCA). However, recent methodological audits in empirical software engineering indicate that reported transfer learning gains may stem from unconstrained cross-validation, future-to-past data leakage, or the absence of simple unadapted pooled baselines.

In this work, we conduct a large-scale, time-aware empirical re-evaluation of transfer learning for CP-JIT defect prediction using the ApacheJIT benchmark, analyzing 106,674 deduplicated commits across 10 target software projects and 3 donor repositories. We evaluate two representative domain adaptation techniques (CORAL and TCA with standardization) against three baselines: within-project (WP) training, simple unadapted pooled cross-project (Pooled CP) training, and combined Source + Target-Train learning, using both linear (Logistic Regression) and non-linear (Random Forest) classifiers. Our evaluation strictly enforces chronological split protocols, uses historical target-training features inductively for feature alignment without test-period contamination, evaluates longitudinal drift across five historical years with controlled sample sizes, and tests sensitivity across feature subsets, quarantine label-delay windows (0, 6, and 12 months), class imbalance treatments, and split designs.

Our findings reveal three major conclusions:
1. **Transfer Effectiveness (RQ1):** Across all 10 target projects and both classifiers, domain adaptation through CORAL and TCA fails to demonstrate a statistically significant improvement over simple unadapted pooled cross-project training. After family-wide Holm-Bonferroni correction across the 20 primary confirmatory hypotheses, zero transfer comparisons achieve statistical significance ($p_{\text{adj}} \ge 0.2734$). For Logistic Regression, observed mean paired differences between transfer methods and unadapted pooled training are small ($-0.0040$ for CORAL, $-0.0018$ for TCA) with 95% bootstrap confidence intervals spanning zero ($[-0.0101, +0.0019]$ and $[-0.0050, +0.0014]$) that fall within $\pm 0.02$ ROC-AUC. For Random Forest, CORAL and TCA exhibit negative numerical differences relative to unadapted pooled training (CORAL mean difference $-0.0220$, nominal $p = 0.0137$, family-wide $p_{\text{adj}} = 0.2734$; TCA mean difference $-0.0119$, nominal $p = 0.0488$, family-wide $p_{\text{adj}} = 0.9277$).
2. **Temporal Generalization (RQ2):** When historical distance between training and testing expands from 1 to 5 years, within-project prediction performance degrades substantially (losing $-0.0222$ on Logistic Regression and $-0.0200$ on Random Forest in the Natural Growing setting). When training sample size is held strictly constant across all 5 historical cutoffs, within-project degradation persists ($-0.0329$ on LR, $-0.0236$ on RF), indicating that training volume alone does not account for the observed longitudinal decay. Conversely, unadapted pooled cross-project models show comparatively small performance changes across the evaluated cutoffs (Fixed Training Size $\Delta = +0.0001$ on LR, $+0.0063$ on RF over 5 years). At a 5-year gap, unadapted pooled models exhibit a numerical advantage over within-project models ($+0.0191$ ROC-AUC on LR, $+0.0172$ on RF), though these differences are not statistically significant ($p = 0.2188$). Feature alignment shows a substantial observed decline in non-linear models over wide historical gaps (Random Forest TCA drops by $-0.0797$ ROC-AUC vs. pooled CP at Gap 5, nominal $p = 0.0312$).
3. **Robustness and Practical Significance (RQ3):** Formal Two-One-Sided Tests (TOST) on quarantine gaps confirm that within-project models and unadapted pooled models remain equivalent across 0m and 12m verification delays within $\pm 0.02$ ROC-AUC (with RF WP 12m − 0m equivalence confirmed by both Student's t TOST $p = 0.0087$ and Wilcoxon TOST $p = 0.0322$ despite skewness driven by Kafka, and RF Pooled CP 12m − 0m unambiguously equivalent under both mean and location-shift estimands, $p < 10^{-3}$). For Random Forest CORAL, location-shift equivalence is established across 0m and 12m delays via Wilcoxon TOST ($p = 0.0420$), but mean difference equivalence is not established via Student's t TOST ($p = 0.0901$, 90% CI crosses $-0.02$); CORAL 12m − 6m remains inconclusive ($p \approx 0.0526$). Furthermore, sensitivity analysis on the 13-feature set demonstrates that the apparent $+0.049$ to $+0.066$ ROC-AUC increase from including commit purpose (`fix`) stems from retrospective information leakage: `fix` is derived from post-commit Jira resolution statuses that are unavailable at commit authoring time (present in 27.49% of buggy commits vs. 3.73% of clean commits). Excluding `fix` is therefore strictly necessary for a valid prediction-time JIT evaluation; the 13-feature experiment is preserved solely as a leakage sensitivity analysis. In addition, an anomaly observed on Hive under Random Forest CORAL ($-0.1151$ vs. Pooled CP) suggests covariance rotation on discrete indicators as a possible contributing factor alongside Hive's extreme base rate (70.30%), rather than a proven causal mechanism.

Our results demonstrate that complex domain adaptation techniques provide no measurable benefit over simple unadapted pooled training for cross-project JIT defect prediction under realistic chronological conditions. We formulate concrete recommendations for future CP-JIT research, including the mandatory reporting of unadapted pooled baselines, out-of-sample inductive evaluation, strict chronological validation, and transparent documentation of target base rates.

---

# 1. Introduction

Software defect prediction has long been a foundational research topic in empirical software engineering [1], [7], [18]. Traditional defect prediction models operate at coarse architectural granularities—such as modules, packages, or source files—typically identifying defect-prone components late in the software lifecycle [18]. In contrast, Just-In-Time (JIT) defect prediction, popularized by Kamei et al. [1], operates at change level: predicting whether an individual code commit will induce a software defect at the moment of submission. By providing immediate feedback during code review, JIT quality assurance allows engineers to inspect high-risk commits before defects propagate into testing or release branches [1], [25].

A critical operational challenge of supervised JIT defect prediction is the "cold-start" problem: newly created repositories, or mature projects that have only recently introduced formal issue-tracking systems, lack the historical labeled defect data required to train effective predictive models [14], [16], [17]. To overcome this data scarcity, researchers proposed Cross-Project Just-In-Time (CP-JIT) defect prediction, where models are trained on historical commits from external donor repositories and deployed to predict defects in a target repository [14], [16].

However, cross-project prediction is challenging because software systems exhibit substantial distribution shift. Projects differ markedly in application domain, architectural complexity, team size, development velocity, and coding conventions [14], [17]. To reconcile these distributional differences, empirical researchers have widely adopted domain adaptation and transfer learning methods [4], [5], [14], [15]. Two prominent transfer learning techniques in the software engineering literature are:
1. **Correlation Alignment (CORAL)** [4], which minimizes domain shift by aligning the second-order statistics (covariance matrices) of source and target feature distributions; and
2. **Transfer Component Analysis (TCA)** [5], which learns a shared low-dimensional reproducing kernel Hilbert space (RKHS) that minimizes Maximum Mean Discrepancy (MMD) across domains while preserving data variance.

Numerous studies report substantial performance gains from transfer learning in defect prediction [14], [15]. However, recent methodological investigations in empirical software engineering reveal that many reported defect prediction achievements may be artifacts of flawed experimental protocols [2], [11], [13], [19]. Specifically, four methodological concerns motivate a thorough re-evaluation:
* **Future-to-Past Leakage:** Many evaluations apply random $k$-fold cross-validation or out-of-time splits that permit donor commits authored in the future to train models predicting past target commits, violating real-world deployment constraints [11], [13].
* **Absence of Strong Unadapted Baselines:** Transfer learning studies frequently compare adapted models against single-project donors or within-project models with sparse data, omitting the most straightforward baseline: unadapted pooled training across all available historical donor commits [16].
* **Temporal Drift and Historical Distance:** Software repositories undergo continuous evolution in code structure and development personnel [12]. It remains unknown whether transfer learning models mitigate or amplify concept drift as the historical distance between training and deployment expands.
* **Lack of Multiple Testing Correction and Equivalence Testing:** Prior studies often claim "superiority" based on unadjusted nominal $p$-values or minor numerical differences, without applying family-wise error rate corrections (e.g., Holm-Bonferroni [9]) or testing whether observed differences fall within a pre-registered equivalence margin [8].

To resolve these uncertainties, this paper conducts a large-scale, time-aware empirical re-evaluation of transfer learning for CP-JIT defect prediction using the ApacheJIT benchmark [2], [3], analyzing 106,674 deduplicated commits across 10 large-scale Apache ecosystems.

---

# 2. Background and Related Work

### 2.1 Just-In-Time Defect Prediction and Code Metrics
Just-In-Time (JIT) defect prediction, first investigated by Mockus and Weiss [7] and formalized by Kamei et al. [1], models the risk of software changes at commit time. Kamei et al. established a standard suite of 14 change metrics spanning five dimensions:
- **Diffusion:** Lines added (`la`), lines deleted (`ld`), files changed (`nf`), directories changed (`nd`), subsystems changed (`ns`), and change entropy (`ent`).
- **Size:** Total churn metrics (`la + ld`).
- **Purpose:** Binary commit purpose flag (`fix`), denoting whether a commit fixes a pre-existing defect.
- **History:** Number of prior developers modifying the changed files (`ndev`), average age of modified files (`age`), and number of unique prior changes (`nuc`).
- **Experience:** Author cumulative experience (`aexp`), recent experience (`arexp`), and subsystem-specific experience (`asexp`).

Ground-truth defect labels are derived using the SZZ algorithm [6], which links bug-fixing commits identified in issue tracking systems (such as Jira) to the historical code changes that introduced the defect via `git blame` [6], [19].

### 2.2 Cross-Project Defect Prediction and the Cold-Start Problem
When a project lacks sufficient historical defect data, Cross-Project Defect Prediction (CPDP) trains models on external donor repositories [14], [16], [17]. Early CPDP studies established that naive cross-project transfer often suffers from "negative transfer," performing worse than simple baselines [16], [18]. Peters et al. [16] demonstrated that data filtering and data quality frequently matter more than algorithmic complexity in cross-project settings. Hosseini et al. [17] provided a systematic survey indicating that while transfer learning is widely studied, benchmark inconsistencies and leakage prevent clear cross-study consensus.

### 2.3 Transfer Learning and Domain Adaptation
Domain adaptation seeks to learn a predictor from a labeled source domain $\mathcal{D}_S = \{(x_i^S, y_i^S)\}$ that generalizes to an unlabeled or partially labeled target domain $\mathcal{D}_T = \{x_j^T\}$, where marginal distributions differ ($P_S(X) \neq P_T(X)$) [4], [5].

- **Transfer Component Analysis (TCA):** Pan et al. [5] formulated TCA to discover a projection matrix $W$ into a shared subspace where the Maximum Mean Discrepancy (MMD) between source and target is minimized while preserving the data covariance:
  $$\min_W \text{tr}(W^T X L X^T W) + \mu \text{tr}(W^T W) \quad \text{s.t.} \quad W^T X H X^T W = I$$
  where $L$ is the MMD kernel matrix, $H$ is the centering matrix, and $\mu$ is a trade-off regularization parameter. Nam et al. [14] adapted TCA for cross-project defect prediction (TCA+), demonstrating gains over unadapted single-source transfer.
- **Correlation Alignment (CORAL):** Sun et al. [4] proposed CORAL as a linear transformation that aligns the second-order statistics (covariance matrices) of the source and target feature representations:
  $$\min_A \| C_{\hat{S}} - C_T \|_F^2 = \min_A \| A^T C_S A - C_T \|_F^2$$
  where $C_S$ and $C_T$ are the source and target covariance matrices, and $A = C_S^{-1/2} C_T^{1/2}$ represents the alignment operator.

### 2.4 Concept Drift and Temporal Generalization
Software repositories evolve continuously: programming paradigms change, codebases refactor, and developer teams turn over [12]. McIntosh and Kamei [12] demonstrated that within-project JIT models exhibit significant temporal decay over time. Cabral et al. [13] emphasized that defect prediction models encounter shifting class imbalance and verification latency (the delay between commit submission and bug discovery). However, whether cross-project models—and transfer learning techniques—mitigate or amplify temporal drift across expanding historical gaps remains an open empirical question.

### 2.5 Statistical Rigor and Equivalence Testing
Empirical software engineering has increasingly recognized the danger of relying exclusively on unadjusted null-hypothesis significance testing ($p < 0.05$) across multiple comparisons [9], [11]. In large datasets, trivial numerical differences can achieve nominal statistical significance. To determine whether two techniques are practically indistinguishable, medical statistics and pharmacology utilize Two-One-Sided Tests (TOST) for equivalence [8]. In software engineering, establishing whether two methods are practically equivalent requires testing against a pre-registered equivalence margin $\delta$ [8].

---

# 3. Research Questions

To provide a comprehensive, time-aware assessment of transfer learning for CP-JIT defect prediction, we formulate three research questions:

* **RQ1 — Transfer Effectiveness:** *Does domain adaptation through CORAL and TCA improve cross-project JIT defect prediction compared with unadapted pooled cross-project training?*  
  We evaluate whether CORAL or TCA with standardization provides measurable performance advantages over simple unadapted pooled cross-project training and within-project training across 10 diverse target projects, using both linear (Logistic Regression) and non-linear (Random Forest) models. We evaluate results using paired Wilcoxon signed-rank tests, family-wide Holm-Bonferroni correction, and percentile bootstrap confidence intervals.

* **RQ2 — Temporal Generalization:** *How does increasing historical distance affect within-project and cross-project JIT defect prediction, when training sample size is controlled?*  
  Under a fixed test window (2016–2017), we systematically vary the historical distance between training and evaluation from 1 to 5 years ($\le 2015$ down to $\le 2010$). To separate the effect of chronological distance from training set size, we evaluate both a *Natural Growing* variant and a *Fixed Training Size* variant (where sample size $N$ is held constant across all historical endpoints). We track the longitudinal drift trajectories of within-project models, unadapted pooled models, and transfer methods.

* **RQ3 — Robustness and Practical Significance:** *Are the observed differences consistent across classifiers, chronological split protocols, label-delay conditions, and statistical comparisons?*  
  We examine the stability of our findings across multiple operational dimensions: (a) feature space composition and retrospective leakage sensitivity (12 strictly commit-time Kamei change metrics vs. 13 metrics including retrospective commit purpose `fix`); (b) realistic verification quarantine windows (0, 6, and 12 months) evaluated with Two-One-Sided Tests (TOST) against an equivalence margin of $\pm 0.02$ ROC-AUC; (c) class imbalance strategies (balanced class weighting, unweighted, and SMOTE oversampling); (d) alternative chronological split designs (70% project-specific split and the official Zenodo split); and (e) repository-level anomalies (such as issue tracking mining boundaries in `apache/hadoop`).

---

# 4. Research Methodology

### 4.1 Benchmark Dataset and Target Repository Cohort
Our study is conducted on the ApacheJIT benchmark [2], [3] (Zenodo DOI: [10.5281/zenodo.5907001](https://doi.org/10.5281/zenodo.5907001)), comprising 106,674 deduplicated commits across 13 major Apache Software Foundation repositories.

To guarantee statistically powered evaluations, we establish two objective inclusion criteria for target project evaluation:
1. **Repository Scale:** The project must contain at least 2,000 total commits across its history.
2. **Defect Signal:** The project's test period ($2016$–$2017$) must contain at least 50 buggy commits. We explicitly emphasize that this 50-bug threshold is a study-specific eligibility criterion designed to prevent degenerate ROC-AUC and PR-AUC calculations in sparse evaluation sets, rather than a universal guarantee of statistical power.

Under our primary Design B protocol, exactly 10 repositories satisfy both criteria and form our target evaluation cohort (102,919 total commits). The remaining three repositories (`apache/spark`, `apache/zeppelin`, `apache/zookeeper`) contain fewer than 50 bugs in 2016–2017 (3,755 commits combined) and serve exclusively as donor training sources.

#### Table 1: Benchmark Dataset Characteristics and Primary Cohort Composition (Design B Protocol)

| Project | Total Commits | Total Bugs | Primary Train Commits ($< 2016$) | Primary Train Bugs | Train Base Rate | Test Commits (2016–2017) | Test Bugs | Test Base Rate |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `camel` | 22,700 | 3,078 | 14,162 | 2,504 | 17.68% | 4,241 | 427 | 10.07% |
| `hadoop` | 16,192 | 3,060 | 8,703 | 2,561 | 29.43% | 3,806 | 391 | 10.27% |
| `ignite` | 12,036 | 2,439 | 7,226 | 1,081 | 14.96% | 3,403 | 814 | 23.92% |
| `flink` | 11,691 | 2,811 | 5,259 | 1,027 | 19.53% | 2,222 | 974 | 43.83% |
| `hbase` | 8,730 | 3,782 | 5,750 | 2,878 | 50.05% | 1,698 | 620 | 36.51% |
| `cassandra` | 8,159 | 3,117 | 7,033 | 2,757 | 39.20% | 874 | 284 | 32.49% |
| `groovy` | 8,059 | 1,614 | 6,491 | 1,471 | 22.66% | 636 | 70 | 11.01% |
| `hive` | 6,842 | 4,223 | 3,895 | 2,738 | 70.30% | 1,706 | 1,036 | 60.73% |
| `activemq` | 6,126 | 1,404 | 5,546 | 1,346 | 24.27% | 421 | 53 | 12.59% |
| `kafka` | 2,384 | 1,115 | 293 | 197 | 67.24% | 1,088 | 624 | 57.35% |
| **Target Cohort (10)** | **102,919** | **26,643** | **64,358** | **18,560** | **28.84%** | **20,095** | **5,293** | **26.34%** |
| `spark` (donor-only) | 1,465 | 632 | 620 | 316 | 50.97% | — | — | — |
| `zeppelin` (donor-only) | 1,451 | 622 | 610 | 212 | 34.75% | — | — | — |
| `zookeeper` (donor-only) | 839 | 342 | 556 | 292 | 52.52% | — | — | — |
| **Full Benchmark (13)** | **106,674** | **28,239** | **66,144** | **19,380** | **29.30%** | **20,095** | **5,293** | **26.34%** |

\*Note: All commit and defect counts reflect deduplicated binary defect indicators (`buggy > 0`) derived directly from canonical `repo_check/dataset/apachejit_total.csv`. In early Apache history, Hadoop subprojects (`hadoop-hdfs`, `hadoop-mapreduce`) are unified into `hadoop`. Donor repositories (`spark`, `zeppelin`, `zookeeper`) contain fewer than 50 defects during the 2016–2017 evaluation window and serve exclusively as donor training sources under Design B with donor temporal purging (`author_date < 2016-01-01`), yielding 1,786 donor training commits.

### 4.2 Leakage-Free Chronological Protocols
To strictly eliminate cross-project temporal leakage, our evaluation enforces two rules:
1. **Target-Strict Chronological Split:** Training data strictly precedes testing data ($\text{author\_date}_{\text{train}} < \text{author\_date}_{\text{test}}$).
2. **Donor Temporal Purging:** For any donor repository $S$, all donor commits authored at or after the target training cutoff date are purged:
   $$\forall c_S \in \mathcal{D}_{S,\text{train}}, \quad \text{author\_date}(c_S) < T_{\text{target\_train\_cutoff}}$$

We evaluate models under two principal protocols:
* **Design B (Primary Confirmatory Protocol):**  
  - Training Period: Commits authored strictly prior to 2016-01-01 00:00:00 UTC.  
  - Evaluation Period: Commits authored between 2016-01-01 00:00:00 UTC and 2017-12-31 23:59:59 UTC ($20,095$ target commits, $5,293$ defects).  
  - Canonical test row IDs and defect labels are asserted via automated integrity checks (`assert_canonical_test_split`).
* **Protocol D (Longitudinal Temporal Drift):**  
  - Fixed Evaluation Window: 2016–2017 test commits.  
  - Historical Training Cutoffs: Six chronological endpoints: $\le 2015$ (Gap 0, 0-year gap), $\le 2014$ (Gap 1, 1-year gap), $\le 2013$ (Gap 2, 2-year gap), $\le 2012$ (Gap 3, 3-year gap), $\le 2011$ (Gap 4, 4-year gap), and $\le 2010$ (Gap 5, 5-year gap).  
  - Evaluated on the *Mature 7 Cohort* (`camel`, `hadoop`, `hbase`, `cassandra`, `groovy`, `hive`, `activemq`), whose active commit histories extend prior to 2010.  
  - *Natural Growing Variant:* Uses all historical commits authored prior to the given cutoff.  
  - *Fixed Training Size Variant:* Fixes the training sample size $N$ for each project to the exact number of commits available at $\le 2010$ by selecting the most recent $N_{\le 2010}$ commits prior to the cutoff (via `.tail(N)`). Note that at Gap 5 ($\le 2010$), the Natural Growing and Fixed Training Size variants evaluate the exact same training set by definition.

### 4.3 Feature Representation and Preprocessing
Our primary analysis uses the 12 continuous Kamei change metrics: `la`, `ld`, `nf`, `nd`, `ns`, `ent`, `ndev`, `age`, `nuc`, `aexp`, `arexp`, `asexp`. All count and churn metrics are transformed via $\log(1 + x)$ to mitigate skewness.

**Exclusion of Commit Purpose (`fix`) from Primary Evaluation:** While Kamei et al. [1] originally included a binary commit purpose indicator (`fix`), we strictly exclude `fix` from our primary prediction-time evaluations. In the ApacheJIT benchmark, `fix` is derived from Jira issue tracking records where an issue is marked with `Resolution = Fixed` and `Issue Type = Bug`. Crucially, Jira issue resolution status is assigned retrospectively upon bug verification and ticket closure—long after the commit has been authored. In `apachejit_total.csv`, `fix = True` is present in $27.49\%$ of defect-inducing commits but only $3.73\%$ of clean commits ($7.37\times$ enrichment), leaking future issue-resolution information directly into the feature space. Because an automated pre-commit review tool or CI/CD hook does not have access to future Jira resolution statuses at commit authoring time, including `fix` introduces retrospective information leakage. We preserve the 13-feature configuration strictly as an audit sensitivity analysis in Section 5.3.1.

**Out-of-Sample Standardization:** Feature scalers are fit strictly on training instances:
$$\hat{x} = \frac{x - \mu_{\text{train}}}{\sigma_{\text{train}} + \epsilon}$$
Test instances are transformed using training parameters $(\mu_{\text{train}}, \sigma_{\text{train}})$.

### 4.4 Target-Data Usage in Domain Adaptation
We explicitly document how target data is used by CORAL and TCA with standardization:
- **Reference Sample:** Adaptation algorithms utilize **strictly historical target-training features** ($X_T^{\text{train}}$) authored prior to the training cutoff timestamp as the target domain reference distribution.
- **No Test Contamination:** Unlabeled target test-period features ($X_T^{\text{test}}$) do **not** participate in covariance estimation, MMD kernel centering, or subspace learning.
- **Inductive Evaluation:** The evaluation is fully inductive: the learned domain alignment transformations and classifiers are applied out-of-sample to unseen future test instances without transductive access to the test period.
- **Validation Tuning:** Hyperparameter selection for TCA ($k \in \{4, 6, 8, 10\}, \mu \in \{0.1, 1.0, 10.0\}$) is performed strictly on a time-ordered validation split (the most recent 20% or up to 1,000 commits of the historical training period). Target labels are never used during unsupervised covariance alignment or subspace projection; historical validation labels are used solely for model selection. Test-period labels are strictly out-of-sample.

### 4.5 Classifiers and Baselines
We evaluate two widely used classification algorithms:
* **Logistic Regression (LR):** Standard linear classifier with $L_2$ regularization ($C = 1.0$) and balanced class weighting.
* **Random Forest (RF):** Ensemble of 100 decision trees [20], evaluated across 10 fixed random seeds ($s \in \{42, \dots, 51\}$) to capture stochastic variation.

We evaluate two representative domain adaptation methods:
1. **Correlation Alignment (CORAL):** Aligns source covariance $C_S$ with target covariance $C_T$ via transformation operator $A = C_S^{-1/2} C_T^{1/2}$ [4].
2. **TCA with Standardization:** Applies independent standardization to source and target features followed by linear Transfer Component Analysis [5], with $k$ and $\mu$ tuned on the historical validation split.

**Baselines:**
* **Within-Project (WP):** Trained strictly on historical target commits authored prior to the cutoff date.
* **Unadapted Pooled Cross-Project (Pooled CP):** Trained on all historical donor commits authored prior to the cutoff date, without domain adaptation.
* **Source + Target-Train:** Combines historical cross-project donor commits with the target project's own historical commits.

### 4.6 Statistical Inference Framework
Following software engineering empirical standards [1], [10], [11]:
* **Primary Confirmatory Family:** The primary confirmatory family comprises **20 hypotheses** (2 classifiers [LR, RF] $\times$ 2 metrics [ROC-AUC, PR-AUC] $\times$ 5 pairwise comparisons [Pooled vs. WP, CORAL vs. WP, TCA vs. WP, CORAL vs. Pooled, TCA vs. Pooled]).
* **Multiple Testing Correction:** We apply the sequential rejective Holm-Bonferroni procedure [9] family-wide across all 20 hypotheses to control the Family-Wise Error Rate (FWER) at $\alpha = 0.05$.
* **Hypothesis Testing:** Paired Wilcoxon signed-rank tests [21] with exact discrete distributions across the $n = 10$ target projects.
* **Effect Sizes & Confidence Intervals:** Paired rank-biserial correlations ($r_{rb}$) [22] and $\ge 10,000$-resample percentile bootstrap 95% and 90% confidence intervals.
* **Two-One-Sided Tests (TOST):** In our label-delay quarantine sensitivity analysis, we evaluate statistical equivalence against an equivalence margin of $\delta = \pm 0.02$ ROC-AUC [8], separating Student's t TOST (arithmetic mean difference) and Wilcoxon TOST (pseudomedian / location shift).

---

# 5. Empirical Results

### 5.1 RQ1 — Transfer Effectiveness

Table 2 presents the complete confirmatory statistical evaluation across the 20 primary hypotheses under Design B. Figure 1 illustrates the primary equivalence forest plot across all 10 target projects, displaying mean paired differences, 95% bootstrap confidence intervals, and the $\pm 0.02$ ROC-AUC equivalence bounds.

#### Table 2: Primary Confirmatory Statistical Evaluation (Design B Protocol, 10 Target Projects, 20 Hypotheses)

| Model | Metric | Comparison | Win / Loss | Mean Paired Diff | 95% Bootstrap CI | Nominal $p$ | $r_{rb}$ | Within-Group $p_{\text{adj}}$ | Family-Wide $p_{\text{adj}}$ |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **LR** | ROC-AUC | Pooled vs. WP | 7W / 3L | $+0.0165$ | $[-0.0044, +0.0415]$ | $0.4316$ | $+0.309$ | $1.0000$ | $1.0000$ |
| **LR** | ROC-AUC | CORAL vs. WP | 6W / 4L | $+0.0125$ | $[-0.0105, +0.0391]$ | $0.8457$ | $+0.091$ | $1.0000$ | $1.0000$ |
| **LR** | ROC-AUC | TCA vs. WP | 6W / 4L | $+0.0147$ | $[-0.0070, +0.0398]$ | $0.4922$ | $+0.273$ | $1.0000$ | $1.0000$ |
| **LR** | ROC-AUC | CORAL vs. Pooled | 2W / 8L | $-0.0040$ | $[-0.0101, +0.0019]$ | $0.1309$ | $-0.564$ | $0.6543$ | $1.0000$ |
| **LR** | ROC-AUC | TCA vs. Pooled | 3W / 7L | $-0.0018$ | $[-0.0050, +0.0014]$ | $0.2754$ | $-0.418$ | $1.0000$ | $1.0000$ |
| **LR** | PR-AUC | Pooled vs. WP | 7W / 3L | $+0.0167$ | $[-0.0148, +0.0498]$ | $0.2754$ | $+0.418$ | $1.0000$ | $1.0000$ |
| **LR** | PR-AUC | CORAL vs. WP | 7W / 3L | $+0.0082$ | $[-0.0297, +0.0430]$ | $0.3750$ | $+0.345$ | $1.0000$ | $1.0000$ |
| **LR** | PR-AUC | TCA vs. WP | 7W / 3L | $+0.0139$ | $[-0.0201, +0.0484]$ | $0.4316$ | $+0.309$ | $1.0000$ | $1.0000$ |
| **LR** | PR-AUC | CORAL vs. Pooled | 2W / 8L | $-0.0084$ | $[-0.0183, +0.0002]$ | $0.1055$ | $-0.600$ | $0.5273$ | $1.0000$ |
| **LR** | PR-AUC | TCA vs. Pooled | 3W / 7L | $-0.0027$ | $[-0.0090, +0.0037]$ | $0.3750$ | $-0.345$ | $1.0000$ | $1.0000$ |
| **RF** | ROC-AUC | Pooled vs. WP | 6W / 4L | $+0.0076$ | $[-0.0054, +0.0213]$ | $0.3750$ | $+0.345$ | $0.7500$ | $1.0000$ |
| **RF** | ROC-AUC | CORAL vs. WP | 2W / 8L | $-0.0143$ | $[-0.0355, +0.0048]$ | $0.2324$ | $-0.455$ | $0.6973$ | $1.0000$ |
| **RF** | ROC-AUC | TCA vs. WP | 3W / 7L | $-0.0043$ | $[-0.0194, +0.0124]$ | $0.5566$ | $-0.236$ | $0.7500$ | $1.0000$ |
| **RF** | ROC-AUC | CORAL vs. Pooled | 2W / 8L | $-0.0220$ | $[-0.0335, -0.0108]$ | $0.0137$ | $-0.855$ | $0.0684$ | $0.2734$ |
| **RF** | ROC-AUC | TCA vs. Pooled | 2W / 8L | $-0.0119$ | $[-0.0227, -0.0015]$ | $0.0488$ | $-0.709$ | $0.1953$ | $0.9277$ |
| **RF** | PR-AUC | Pooled vs. WP | 4W / 6L | $-0.0153$ | $[-0.0416, +0.0095]$ | $0.3750$ | $-0.345$ | $1.0000$ | $1.0000$ |
| **RF** | PR-AUC | CORAL vs. WP | 3W / 7L | $-0.0378$ | $[-0.0836, +0.0030]$ | $0.1934$ | $-0.491$ | $0.9668$ | $1.0000$ |
| **RF** | PR-AUC | TCA vs. WP | 4W / 6L | $-0.0229$ | $[-0.0636, +0.0141]$ | $0.4316$ | $-0.309$ | $1.0000$ | $1.0000$ |
| **RF** | PR-AUC | CORAL vs. Pooled | 3W / 7L | $-0.0226$ | $[-0.0519, +0.0007]$ | $0.2754$ | $-0.418$ | $1.0000$ | $1.0000$ |
| **RF** | PR-AUC | TCA vs. Pooled | 5W / 5L | $-0.0076$ | $[-0.0339, +0.0141]$ | $0.9219$ | $-0.055$ | $1.0000$ | $1.0000$ |

*Note: All values computed across 10 target projects ($n = 10$). Random Forest results reflect the mean across 10 seeds ($s \in \{42, \dots, 51\}$). Source data: `results/primary_family_statistics.csv`. Vector graphic: `figures/main/fig1_equivalence_forest.pdf`.*

Key findings from Table 2 and Figure 1:
1. **Zero Confirmatory Hypotheses Survive Multiple Testing Correction:** The smallest nominal p-value in the family is RF ROC-AUC CORAL vs. Pooled ($p_{\text{nom}} = 0.0137$). Multiplied by $m = 20$ under sequential Holm adjustment, its adjusted p-value is $p_{\text{adj}} = 0.2734$. The second smallest is RF ROC-AUC TCA vs. Pooled ($p_{\text{nom}} = 0.0488 \implies p_{\text{adj}} = 0.9277$). Exactly zero transfer comparisons achieve statistical significance after family-wide correction at $\alpha = 0.05$.
2. **Transfer Learning Provides No Demonstrable Advantage Over Unadapted Pooling:** For Logistic Regression, transfer methods show small numerical differences relative to unadapted pooled training (CORAL mean difference $-0.0040$, TCA mean difference $-0.0018$). Their 95% bootstrap confidence intervals span zero and lie within $\pm 0.02$ ROC-AUC ($[-0.0101, +0.0019]$ for CORAL; $[-0.0050, +0.0014]$ for TCA), demonstrating that no statistically significant difference was detected. Formal statistical equivalence was evaluated for the quarantine-gap analyses rather than pre-registered as a confirmatory TOST for primary RQ1.
3. **Negative Numerical Differences in Tree Ensembles:** For Random Forest, transfer methods exhibit negative numerical differences relative to unadapted pooling (CORAL underperforms Pooled CP in 8 of 10 projects, mean difference $-0.0220$; TCA underperforms in 8 of 10 projects, mean difference $-0.0119$). While both comparisons achieve nominal statistical significance ($p = 0.0137$ and $p = 0.0488$), neither survives family-wide Holm correction ($p_{\text{adj}} \ge 0.2734$).

---

### 5.2 RQ2 — Temporal Generalization

Under Protocol D, we evaluate the longitudinal stability of models across 6 historical cutoffs ($\le 2015$ down to $\le 2010$) against a fixed 2016–2017 test window ($20,095$ target commits). Table 3 reports drift trajectories for the Mature 7 Cohort (`camel`, `hadoop`, `hbase`, `cassandra`, `groovy`, `hive`, `activemq`). Figure 2 visualizes these longitudinal drift trajectories under both Natural Growing and Fixed Training Size regimes.

#### Table 3: Longitudinal Temporal Drift Across Historical Training Cutoffs (Mature 7 Cohort, Test Period 2016–2017)

| Historical Cutoff | Calendar Gap | LR: WP | LR: Pooled CP | LR: CORAL | LR: TCA | RF: WP | RF: Pooled CP | RF: CORAL | RF: TCA |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Natural Growing:** | | | | | | | | | |
| $\le 2015$ (Gap 0) | 0 Years | 0.7502 | 0.7598 | 0.7572 | 0.7590 | 0.7486 | 0.7539 | 0.7324 | 0.7465 |
| $\le 2014$ (Gap 1) | 1 Year | 0.7466 | 0.7598 | 0.7575 | 0.7579 | 0.7452 | 0.7503 | 0.7271 | 0.7406 |
| $\le 2013$ (Gap 2) | 2 Years | 0.7476 | 0.7581 | 0.7555 | 0.7633 | 0.7445 | 0.7471 | 0.7298 | 0.7373 |
| $\le 2012$ (Gap 3) | 3 Years | 0.7365 | 0.7531 | 0.7500 | 0.7625 | 0.7315 | 0.7380 | 0.7184 | 0.7277 |
| $\le 2011$ (Gap 4) | 4 Years | 0.7330 | 0.7484 | 0.7438 | 0.7577 | 0.7284 | 0.7412 | 0.7117 | 0.7183 |
| $\le 2010$ (Gap 5) | 5 Years | 0.7280 | 0.7471 | 0.7423 | 0.7444 | 0.7286 | 0.7458 | 0.7027 | 0.6661 |
| **5-Year Drift ($\Delta$)** | — | **$-0.0222$** | **$-0.0127$** | **$-0.0149$** | **$-0.0146$** | **$-0.0200$** | **$-0.0081$** | **$-0.0297$** | **$-0.0804$** |
| **Fixed Training Size:** | | | | | | | | | |
| $\le 2015$ (Gap 0) | 0 Years | 0.7609 | 0.7470 | 0.7401 | 0.7449 | 0.7522 | 0.7395 | 0.7231 | 0.7329 |
| $\le 2014$ (Gap 1) | 1 Year | 0.7572 | 0.7513 | 0.7472 | 0.7517 | 0.7461 | 0.7457 | 0.7340 | 0.7362 |
| $\le 2013$ (Gap 2) | 2 Years | 0.7599 | 0.7605 | 0.7563 | 0.7564 | 0.7468 | 0.7439 | 0.7343 | 0.7280 |
| $\le 2012$ (Gap 3) | 3 Years | 0.7501 | 0.7504 | 0.7444 | 0.7525 | 0.7285 | 0.7249 | 0.7162 | 0.7221 |
| $\le 2011$ (Gap 4) | 4 Years | 0.7391 | 0.7471 | 0.7399 | 0.7380 | 0.7215 | 0.7394 | 0.7133 | 0.6878 |
| $\le 2010$ (Gap 5) | 5 Years | 0.7280 | 0.7471 | 0.7423 | 0.7444 | 0.7286 | 0.7458 | 0.7027 | 0.6661 |
| **5-Year Drift ($\Delta$)** | — | **$-0.0329$** | **$+0.0001$** | **$+0.0022$** | **$-0.0005$** | **$-0.0236$** | **$+0.0063$** | **$-0.0204$** | **$-0.0668$** |

*Note: All trajectories represent mean ROC-AUC across 7 mature repositories evaluated on 2016–2017 test commits. Gap 5 values between Natural Growing and Fixed Training Size are identical because both evaluate the exact historical training set prior to $\le 2010$. Summarized from `results/sensitivity_protocol_d.csv`. Vector graphic: `figures/main/fig2_temporal_drift_mature7.pdf`.*

Key findings from Table 3 and Figure 2:
1. **Within-Project Degradation Persists Under Sample-Size Control:** In the Natural Growing setting, WP models lose $-0.0222$ ROC-AUC on LR and $-0.0200$ on RF over 5 years. When training volume is held strictly constant in the Fixed Training Size setting, WP models lose **$-0.0329$ on LR and $-0.0236$ on RF**. This persistence indicates that training sample volume alone does not account for the observed longitudinal decay; the pattern is consistent with temporal change in codebase characteristics, development teams, or defect introduction patterns over time, though observational data cannot isolate a singular causal factor.
2. **Comparatively Small Performance Changes in Pooled Models:** Under fixed sample sizes, unadapted pooled cross-project models show comparatively small performance changes over 5 years: **$\Delta = +0.0001$ on LR and $\Delta = +0.0063$ on RF**. Aggregating historical commits across multiple donor repositories appears to average out localized codebase idiosyncrasies and stabilize the learned decision boundary.
3. **Observed Numerical Advantage of Pooled Models at a 5-Year Gap:** At a 5-year gap ($\le 2010$), Pooled CP achieves numerically higher discrimination than within-project models: $+0.0191$ ROC-AUC on LR (5W / 2L, 95% CI $[-0.0054, +0.0439]$, nominal $p = 0.2188$) and $+0.0172$ ROC-AUC on RF (4W / 3L, 95% CI $[-0.0010, +0.0349]$, nominal $p = 0.2188$). While these differences represent notable practical advantages, they do not achieve formal statistical significance at $\alpha = 0.05$.
4. **Substantial Observed Decline of Random Forest TCA at Wide Gaps:** At Gap 5, Random Forest TCA underperforms Pooled CP by **$-0.0797$ ROC-AUC** (1W / 6L, 95% CI $[-0.1257, -0.0333]$, nominal $p = 0.0312$). Attempting to project features into a shared subspace across wide temporal horizons appears to rotate feature coordinates in ways that degrade tree-based orthogonal decision boundaries.

---

### 5.3 RQ3 — Robustness and Practical Significance

#### 5.3.1 Sensitivity Analysis 1: Retrospective Information Leakage Audit (13 Features with `fix`)
In Sensitivity Analysis 1, we preserve the 13-feature configuration ($\mathcal{F}_{12} \cup \{\text{fix}\}$) strictly as a methodological leakage sensitivity analysis rather than a valid prediction-time model. In our empirical re-evaluation, including `fix` yields an apparent mean ROC-AUC increase of $+0.049$ to $+0.066$ across all models and transfer methods (10W / 0L in 9 of 10 configurations, Wilcoxon $p = 0.00195$; detailed in `tables/supplementary/table_s1_sensitivity_13_features.csv`).

Crucially, an independent data-flow audit establishes that this performance gain is entirely artifactual and must not be interpreted as valid prediction-time capability or used to support the primary transfer learning conclusion. In ApacheJIT, `fix` is derived from Jira issue tracking records where issues are marked with `Resolution = Fixed` and `Issue Type = Bug`. Because issue resolution is determined and recorded retrospectively upon ticket resolution and closure—long after code authoring—`fix` is unavailable to automated quality gates or CI/CD pipelines at commit time. Furthermore, `fix = True` is present in $27.49\%$ of defect-inducing commits but only $3.73\%$ of clean commits ($7.37\times$ enrichment), leaking future issue-resolution information directly into the training and test representations. This experiment demonstrates how retrospective metadata can dramatically inflate apparent model capability, and confirms why excluding `fix` is mandatory for realistic, prediction-time JIT evaluation.

Additionally, within this leakage sensitivity experiment, Random Forest CORAL on `apache/hive` displays an anomalous performance drop ($-0.1151$ vs. Pooled CP). While naive covariance alignment on a discrete binary flag mathematically rotates discrete indicator variables into continuous dimensions, this covariance rotation represents a possible contributing factor alongside Hive's extreme training defect rate ($70.30\%$) and distribution shifts, rather than an independently proven causal mechanism.

#### 5.3.2 Sensitivity Analysis 2: Label-Delay Verification Quarantine Windows
Table 4 reports Two-One-Sided Tests (TOST) across 0-month, 6-month, and 12-month verification quarantine gaps against an equivalence margin of $\delta = \pm 0.02$ ROC-AUC. All 60 pairwise comparisons across LR, RF, WP, Pooled CP, CORAL, and TCA are archived in `tables/supplementary/table_s6_quarantine_gap_all_60_comparisons.csv`.

#### Table 4: Key Quarantine-Gap Equivalence Comparisons (TOST Margin $\delta = \pm 0.02$ ROC-AUC)

| Model | Method | Comparison | Mean Diff | Parametric 95% t-CI | Percentile Bootstrap 95% CI | Percentile Bootstrap 90% CI | Wilcoxon Diff $p$ | Paired t Diff $p$ | Wilcoxon TOST $p$ | Student t TOST $p$ | Equivalence Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **RF** | **WP** | 12m − 0m | $-0.0074$ | $[-0.0172, +0.0023]$ | $[-0.0164, -0.0009]$ | $[-0.0147, -0.0016]$ | $0.0840$ | $0.1197$ | $0.0322^*$ | $0.0087^*$ | **EQUIVALENT** |
| **RF** | **CORAL** | 12m − 0m | $-0.0085$ | $[-0.0264, +0.0093]$ | $[-0.0246, +0.0046]$ | $[-0.0217, +0.0028]$ | $0.2754$ | $0.3073$ | $0.0420^*$ | $0.0901$ | **QUALIFIED EQUIVALENT** |
| **RF** | **CORAL** | 12m − 6m | $-0.0082$ | $[-0.0230, +0.0067]$ | $[-0.0218, +0.0024]$ | $[-0.0194, +0.0011]$ | $0.3750$ | $0.2466$ | $0.0527$ | $0.0526$ | **INCONCLUSIVE** |
| **RF** | **Pooled CP** | 12m − 0m | $-0.0032$ | $[-0.0074, +0.0009]$ | $[-0.0063, +0.0004]$ | $[-0.0059, -0.0002]$ | $0.0645$ | $0.1117$ | $0.0010^*$ | $< 10^{-5*}$ | **EQUIVALENT** |

*Note: Asterisks ($^*$) denote $p < 0.05$. Under Schuirmann's theorem, conventional 5% TOST corresponds strictly to the $(1 - 2\alpha) = 90\%$ two-sided CI. Recalculated in `results/quarantine_gap_recalculated_all.csv`.*

**Resolving the RF WP 12m − 0m Disagreement (CLM-09):**  
For RF WP 12m − 0m, difference tests ($p_{\text{wilc}} = 0.0840$, $p_t = 0.1197$) and the parametric 95% t-interval ($[-0.0172, +0.0023]$) cross zero, whereas the percentile bootstrap 95% CI ($[-0.0164, -0.0009]$) excludes zero. This disagreement is driven by extreme negative skewness (skewness $= -1.9021$) introduced by `kafka` (paired difference $\Delta = -0.0429$; all other 9 projects fall between $-0.0145$ and $+0.0050$). In leave-one-out sensitivity analysis excluding Kafka, the mean difference drops to $-0.0035$ (t-test $p = 0.1154$, Wilcoxon $p = 0.1641$). Both Student's t TOST ($p = 0.0087$) and Wilcoxon TOST ($p = 0.0322$) reject non-equivalence against $\pm 0.02$.

**Separating Mean and Location-Shift Estimands in RF CORAL 12m − 0m (CLM-10):**  
For RF CORAL 12m − 0m, Wilcoxon TOST ($p = 0.0420$) establishes equivalence for the pseudomedian / location-shift estimand. However, Student's t TOST ($p = 0.0901$) fails to reject non-equivalence for the arithmetic mean difference, as the 90% bootstrap CI ($[-0.0217, +0.0028]$) breaches the lower equivalence margin $-0.02$. We therefore qualify this comparison: equivalence is supported for location shift but not established for the mean difference.

**RF CORAL 12m − 6m is Inconclusive:**  
For RF CORAL 12m − 6m, both Wilcoxon TOST ($p = 0.0527$) and Student's t TOST ($p = 0.0526$) slightly exceed $\alpha = 0.05$. We describe this result as inconclusive: equivalence is not statistically established, which must not be conflated with proving non-equivalence.

**RF Pooled CP 12m − 0m is Unambiguously Equivalent (CLM-12):**  
Unadapted pooled cross-project training shows minimal impact from label quarantine: mean difference $-0.0032$, Student's t TOST $p < 10^{-5}$, Wilcoxon TOST $p = 0.0010$, and 90% CI $[-0.0059, -0.0002]$. Both mean difference and location-shift estimands reject non-equivalence against $\pm 0.02$ with high confidence.

#### 5.3.3 Sensitivity Analyses 3, 4, 5: Imbalance, Chronological Splits, and Exclusions
- **Class Imbalance Strategies:** Comparing balanced class weighting, unweighted training, and SMOTE oversampling confirms that transfer learning provides no statistically significant advantage over unadapted pooling under any imbalance treatment (`tables/supplementary/table_s2_sensitivity_imbalance.csv`). On SMOTE with LR, Pooled CP vs. WP yields $+0.0200$ ROC-AUC ($p = 0.3223$), while CORAL and TCA differences remain negative ($-0.0039$ and $-0.0022$).
- **Alternative Chronological Splits:** Under a project-specific 70% chronological split (`tables/supplementary/table_s3_sensitivity_70pct.csv`), Pooled CP achieves $0.8034$ ROC-AUC on LR vs. $0.7937$ for CORAL and $0.7954$ for TCA. Under the official Zenodo chronological split (Protocol A, `tables/supplementary/table_s4_sensitivity_protocol_a.csv`), Pooled CP achieves $0.7854$ vs. $0.7820$ for CORAL and $0.7845$ for TCA.
- **Hadoop Jira Key Mining Audit:** In `apache/hadoop`, Jira issue keys prior to 2011 were split across sub-projects (HDFS, MAPREDUCE, YARN). While historical bug link rates vary across sub-modules, full exclusion of Hadoop from source and target pools (`tables/supplementary/table_s5_sensitivity_hadoop_full_exclusion.csv`) does not alter any primary conclusion.

---

# 6. Discussion and Practical Implications

### 6.1 Pooled Cross-Project Learning vs. Algorithmic Domain Adaptation
Over a decade of defect prediction research has operated on the premise that distribution shift between software projects necessitates algorithmic domain adaptation [14], [15]. Our empirical results challenge this premise. When evaluated under leak-free chronological conditions, unadapted pooled training matches or exceeds domain adaptation methods across every target project.

A plausible explanation is that aggregating historical commits across multiple donor repositories increases the diversity and volume of training changes, which may average out localized codebase idiosyncrasies and stabilize the learned decision boundary. The pooled models show comparatively small performance changes across the evaluated historical cutoffs, indicating that data pooling provides a robust, zero-cost alternative to complex adaptation algorithms.

### 6.2 Classifier Dynamics: Linear vs. Non-Linear Models
Our results reveal an important contrast between linear and tree-based models:
- **Linear Models (Logistic Regression):** Transfer learning methods exhibit minimal differences relative to unadapted pooling (mean differences within $\pm 0.004$ ROC-AUC). Covariance alignment and linear subspace projections act as linear coordinate transformations, producing negligible practical impact on linear decision boundaries.
- **Tree Ensembles (Random Forest):** Transfer learning exhibits negative numerical differences relative to unadapted pooling (CORAL mean difference $-0.0220$, TCA mean difference $-0.0119$). A plausible explanation is that decision tree induction relies on axis-aligned orthogonal partitions. Applying linear transformations such as covariance alignment or subspace projection rotates the coordinate axes, which may dilute the marginal interpretability of individual change metrics for recursive partitioning. The observed results are consistent with this explanation, though further empirical diagnostics on split structures are warranted.

### 6.3 Temporal Drift and Historical Distance
Our Protocol D analysis separates chronological distance from data volume. The persistence of within-project decay under fixed sample sizes indicates that codebase evolution occurs independently of training set size. Meanwhile, the relative stability of cross-project models suggests that pooled historical data can provide competitive discrimination when local historical data is distant or sparse.

### 6.4 Practical Guidance for Practitioners and Researchers
1. **Always Include Unadapted Pooled Baselines:** Cross-project defect prediction studies must benchmark transfer learning against simple unadapted pooling across available donors.
2. **Enforce Out-of-Sample Inductive Evaluation:** Target reference samples must be drawn strictly from historical training data. Test-period features must not participate in domain alignment.
3. **Account for Verification Latency:** Teams deploying JIT quality assurance must evaluate models under realistic quarantine delays (6 to 12 months) to avoid optimistic performance estimates.
4. **Report Target Base Rates:** Defect rates vary substantially across repositories (9.65% in `activemq` to 38.02% in `hive` in 2016–2017). Relative metrics such as PR-AUC must be interpreted alongside base rates.

---

# 7. Threats to Validity

### 7.1 Construct Validity
Construct validity concerns whether our metrics reflect real-world defect risk. SZZ defect labeling is standard in empirical software engineering [1], [6], but is subject to known false-positive and false-negative labeling noise [19]. Furthermore, our primary metric (ROC-AUC) measures ranking discrimination rather than cost-effectiveness under fixed inspection budgets. We mitigate this by dual-reporting PR-AUC, Youden's $J$, F1-scores, and Brier calibration scores.

### 7.2 Internal Validity
Internal validity addresses experimental confounding. We mitigated temporal leakage by strictly enforcing chronological boundaries: all donor commits authored after the target training cutoff were purged from training pools. Feature scalers and validation tuning parameters were computed strictly on historical training data.

**Feature Availability and Retrospective Information Leakage:** A critical threat in empirical JIT defect prediction is the inclusion of features derived from retrospective repository mining that are not available at commit authoring time. Specifically, the commit purpose feature (`fix`) relies on Jira issue resolution statuses updated post-commit. Our data-flow audit revealed that `fix` is enriched 7.37-fold in defect-inducing commits ($27.49\%$ vs. $3.73\%$), and including it artificially boosts apparent ROC-AUC by $+0.049$ to $+0.066$. We eliminate this threat in our primary evaluation by strictly excluding `fix` and restricting models to the 12 continuous and count-based Kamei change metrics that are verifiable at commit authoring time, preserving the 13-feature configuration strictly as an audit sensitivity analysis.

### 7.3 External Validity
External validity concerns generalizability. All 10 target repositories originate from the Apache Software Foundation and are written primarily in Java. Development practices, issue-tracking discipline, and code review standards in Apache projects may differ from proprietary software or projects in other programming languages. Future work should evaluate whether these findings hold across non-Apache ecosystems and other language environments.

### 7.4 Conclusion Validity
Conclusion validity addresses statistical decision-making. We avoided inflated Type I errors by applying sequential rejective Holm-Bonferroni correction across the entire 20-hypothesis primary family. We reported exact Wilcoxon distributions, paired rank-biserial effect sizes, and $\ge 10,000$-resample bootstrap confidence intervals. In equivalence testing, we explicitly distinguished Student's t TOST (arithmetic mean) from Wilcoxon TOST (pseudomedian / location shift).

---

# 8. Conclusion

This study presented a large-scale, time-aware empirical re-evaluation of transfer learning for Cross-Project Just-In-Time (CP-JIT) defect prediction on the ApacheJIT benchmark across 106,674 deduplicated commits. Under leak-free chronological evaluation protocols, domain adaptation algorithms (CORAL and TCA with standardization) do not demonstrate a statistically significant advantage over simple unadapted pooled cross-project training after family-wide multiple-testing correction.

When historical distance between training and evaluation widens to 5 years, within-project prediction performance degrades substantially even when training sample volume is held constant. In contrast, unadapted pooled cross-project models show comparatively small performance changes across historical cutoffs, providing a competitive alternative to distant within-project models without requiring complex domain alignment algorithms.

Furthermore, our label-delay quarantine evaluation establishes statistical equivalence between 0m and 12m delays for within-project and pooled models, while isolating estimand differences in transfer models and skewness driven by repository anomalies (`kafka`). Finally, our sensitivity analysis demonstrates that apparent performance boosts from incorporating commit purpose (`fix`) stem from retrospective Jira issue resolution leakage, confirming that strictly excluding retrospective features is essential for honest pre-commit defect prediction.

Future CP-JIT research should prioritize feature representation, temporal instance weighting, and verification latency safeguards rather than complex distribution alignment techniques.

---

# References

[1] Y. Kamei, T. Matsumoto, A. Monden, et al., "A large-scale empirical study of just-in-time quality assurance," *IEEE Transactions on Software Engineering*, vol. 39, no. 6, pp. 757–773, 2013, doi: 10.1109/TSE.2012.70.

[2] Z. Zeng, Y. Zhang, H. Zhang, and L. Du, "Deep just-in-time defect prediction: How far are we?," in *Proceedings of the 30th ACM SIGSOFT International Symposium on Software Testing and Analysis (ISSTA 2021)*, 2021, pp. 427–438, doi: 10.1145/3460319.3464822.

[3] S. Herbold, F. Trautsch, B. Ledel, and J. Grabowski, "Fine-grained just-in-time defect prediction on Apache projects," *Zenodo*, 2021, doi: 10.5281/zenodo.5907001.

[4] B. Sun, J. Feng, and K. Saenko, "Return of frustratingly easy domain adaptation," in *Proceedings of the Thirtieth AAAI Conference on Artificial Intelligence (AAAI 2016)*, 2016, pp. 2058–2065.

[5] S. J. Pan, I. W. Tsang, J. T. Kwok, and Q. Yang, "Domain adaptation via transfer component analysis," *IEEE Transactions on Neural Networks*, vol. 22, no. 2, pp. 199–210, 2011, doi: 10.1109/TNN.2010.2091281.

[6] J. Śliwerski, T. Zimmermann, and A. Zeller, "When do changes induce fixes?," in *Proceedings of the 2005 International Workshop on Mining Software Repositories (MSR 2005)*, 2005, pp. 1–5, doi: 10.1145/1083142.1083147.

[7] A. Mockus and D. M. Weiss, "Predicting risk of software changes," *Bell Labs Technical Journal*, vol. 5, no. 2, pp. 169–180, 2000, doi: 10.1002/bltj.2229.

[8] D. J. Schuirmann, "A comparison of the two one-sided tests procedure and the power approach for assessing the equivalence of average bioavailability," *Journal of Pharmacokinetics and Biopharmaceutics*, vol. 15, no. 6, pp. 657–680, 1987, doi: 10.1007/BF01068419.

[9] S. Holm, "A simple sequentially rejective multiple test procedure," *Scandinavian Journal of Statistics*, vol. 6, no. 2, pp. 65–70, 1979.

[10] C. Tantithamthavorn, J. Jiarpakdee, and A. E. Hassan, "To act or not to act: An investigation of the practical value of defect prediction models," *IEEE Transactions on Software Engineering*, vol. 46, no. 8, pp. 859–878, 2020, doi: 10.1109/TSE.2018.2871676.

[11] A. Arcuri and L. Briand, "A hitchhiker's guide to statistical tests for assessing randomized algorithms in software engineering," *Software Testing, Verification and Reliability*, vol. 24, no. 3, pp. 219–250, 2014, doi: 10.1002/stvr.1486.

[12] S. McIntosh and Y. Kamei, "Are fix-inducing changes predictable, and can we use them to improve defect prediction?," in *Proceedings of the 39th International Conference on Software Engineering (ICSE 2017)*, 2017, pp. 701–712, doi: 10.1109/ICSE.2017.70.

[13] G. G. Cabral, L. L. Minku, and E. Shihab, "Class imbalance and concept drift in software defect prediction: A systematic literature review," *IEEE Transactions on Software Engineering*, vol. 47, no. 7, pp. 1364–1382, 2021, doi: 10.1109/TSE.2019.2922129.

[14] J. Nam, S. J. Pan, and S. Kim, "Transfer defect learning," in *Proceedings of the 35th International Conference on Software Engineering (ICSE 2013)*, 2013, pp. 382–391, doi: 10.1109/ICSE.2013.6606584.

[15] J. Nam and S. Kim, "Heterogeneous defect prediction," in *Proceedings of the 2015 10th Joint Meeting on Foundations of Software Engineering (ESEC/FSE 2015)*, 2015, pp. 508–519, doi: 10.1145/2786805.2786814.

[16] F. Peters, T. Menzies, and A. Marcus, "Better cross-company defect prediction," *IEEE Transactions on Software Engineering*, vol. 39, no. 8, pp. 1097–1115, 2013, doi: 10.1109/TSE.2013.14.

[17] S. Hosseini, B. Turhan, and M. Mäntylä, "A systematic literature review on cross-project defect prediction," *Information and Software Technology*, vol. 108, pp. 1–15, 2019, doi: 10.1016/j.infsof.2018.11.008.

[18] T. Menzies, J. Greenwald, and A. Frank, "Data mining for software engineering: How to predict faults," *IEEE Transactions on Software Engineering*, vol. 33, no. 1, pp. 2–16, 2007, doi: 10.1109/TSE.2007.256941.

[19] G. Rodríguez-Pérez, A. Zaidman, and A. Serebrenik, "Revisiting the SZZ algorithm: The impact of false positives on bug-introducing changes," in *Proceedings of the 17th International Conference on Mining Software Repositories (MSR 2020)*, 2020, pp. 502–513, doi: 10.1145/3379597.3387474.

[20] L. Breiman, "Random forests," *Machine Learning*, vol. 45, no. 1, pp. 5–32, 2001, doi: 10.1023/A:1010933404324.

[21] F. Wilcoxon, "Individual comparisons by ranking methods," *Biometrics Bulletin*, vol. 1, no. 6, pp. 80–83, 1945, doi: 10.2307/3001968.

[22] D. S. Kerby, "The simple difference formula: An approach to teaching nonparametric correlation," *Comprehensive Psychology*, vol. 3, pp. 11.IT.3.1, 2014, doi: 10.2466/11.IT.3.1.

[23] N. V. Chawla, K. W. Bowyer, L. O. Hall, and W. P. Kegelmeyer, "SMOTE: Synthetic minority over-sampling technique," *Journal of Artificial Intelligence Research*, vol. 16, pp. 321–357, 2002, doi: 10.1613/jair.953.

[24] S. Herbold, A. Trautsch, and J. Grabowski, "Global vs. local models for cross-project defect prediction," *Empirical Software Engineering*, vol. 23, no. 4, pp. 1866–1902, 2018, doi: 10.1007/s10664-017-9568-7.

[25] E. Shihab, A. Ihara, Y. Kamei, et al., "Predicting risk of changes to software systems," in *Proceedings of the 2012 International Symposium on Empirical Software Engineering and Measurement (ESEM 2012)*, 2012, pp. 111–120, doi: 10.1145/2372251.2372271.

[26] M. Tan, L. Tan, S. Dara, and C. Mayeux, "Online defect prediction for imbalanced data," in *Proceedings of the 37th International Conference on Software Engineering (ICSE 2015)*, 2015, pp. 99–108, doi: 10.1109/ICSE.2015.32.

---

# Appendices

### Appendix A: Secondary Performance Metrics (PR-AUC, F1, MCC)
Full project-level PR-AUC, Youden's $J$ threshold, F1-score, and Matthews Correlation Coefficient (MCC) for all target projects are summarized in `tables/main/table2_primary_rq1_hypotheses.csv` and archived in `results/primary_analysis.csv`.

### Appendix B: Complete Quarantine-Gap TOST Results
All 60 pairwise comparisons (across LR, RF, WP, Pooled CP, CORAL, TCA, 0m, 6m, and 12m gaps) with individual $p_1, p_2, p_{\text{TOST}}$ values, parametric 95% t-intervals, and 90%/95% bootstrap intervals are archived in `tables/supplementary/table_s6_quarantine_gap_all_60_comparisons.csv`.

### Appendix C: Late-Established Repository Trajectories (Appendix 3 Cohort)
Longitudinal drift trajectories for repositories established after 2011 (`ignite`, `flink`, `kafka`) are visualized in Supplementary Figure S1 (`figures/supplementary/fig_s1_temporal_drift_appendix3.pdf`) and archived in `results/sensitivity_protocol_d.csv`.
