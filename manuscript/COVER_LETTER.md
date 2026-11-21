# Submission Cover Letter

**Date:** November 21, 2026  
**To:** The Editor-in-Chief and Editorial Board  
**Target Journal:** *[Target Journal Name - e.g., IEEE Transactions on Software Engineering (TSE) / ACM Transactions on Software Engineering and Methodology (TOSEM) / Empirical Software Engineering (EMSE) - AUTHOR INPUT REQUIRED]*  
**Subject:** Submission of Original Research Article: *"Does Transfer Learning Really Help Cross-Project Just-In-Time Defect Prediction? A Time-Aware Re-Evaluation on ApacheJIT"*  

Dear Editor-in-Chief,

We are pleased to submit our original empirical research article, **"Does Transfer Learning Really Help Cross-Project Just-In-Time Defect Prediction? A Time-Aware Re-Evaluation on ApacheJIT"**, for publication consideration in your journal.

### Research Motivation and Problem Context
Just-In-Time (JIT) defect prediction allows development teams to detect high-risk commits at the moment of submission during code review. However, newly created projects or projects lacking historical defect data face the "cold-start" problem, necessitating Cross-Project Just-In-Time (CP-JIT) defect prediction. Over the past decade, a substantial literature has advocated domain adaptation and transfer learning—specifically Correlation Alignment (CORAL) and Transfer Component Analysis (TCA)—to reconcile distribution shift across donor and recipient repositories.

Despite widespread enthusiasm, recent methodological audits in empirical software engineering indicate that reported transfer learning advantages may be artifacts of future-to-past data leakage, unconstrained cross-validation, or the absence of simple unadapted pooled baselines.

### Primary Empirical Contributions and Findings
In this manuscript, we present a large-scale, time-aware empirical re-evaluation of transfer learning for CP-JIT defect prediction across 106,674 deduplicated commits from 10 mature open-source software systems in the ApacheJIT benchmark. Our evaluation strictly enforces chronological separation, guarantees zero future-to-past leakage, and applies family-wide multiple testing corrections:

1. **Transfer Learning Does Not Outperform Simple Pooling (RQ1):** Across all 10 target projects and multiple classifiers (Logistic Regression and Random Forest), transfer learning through CORAL and TCA fails to demonstrate a statistically significant improvement upon simple unadapted pooled cross-project training after family-wide Holm correction ($p_{\text{adj}} \ge 0.2734$). For linear models, observed mean paired differences are small (within $\pm 0.004$ ROC-AUC) with 95% bootstrap confidence intervals spanning zero. For non-linear tree ensembles, transfer learning shows negative numerical differences relative to unadapted pooling (nominal $p < 0.05$, though not surviving family-wide correction).
2. **Disentangling Historical Drift from Sample Size (RQ2):** Through our Protocol D longitudinal evaluation, we fix training sample sizes across 5 historical years ($\le 2010$ to $\le 2015$) to isolate chronological distance. We show that within-project performance degradation persists under sample-size control (losing $-0.024$ to $-0.033$ ROC-AUC), indicating that training volume alone does not account for longitudinal decay. Conversely, unadapted pooled cross-project models show comparatively small performance changes across historical cutoffs ($\Delta = +0.0001$ on LR, $+0.0063$ on RF over 5 years), exhibiting a numerical advantage over 5-year-old within-project models ($+0.017$ to $+0.019$ ROC-AUC, $p = 0.2188$). Furthermore, feature alignment algorithms exhibit substantial observed declines over wide temporal horizons (RF TCA dropping by $-0.080$ ROC-AUC vs. pooled CP at Gap 5, nominal $p = 0.0312$).
3. **Comprehensive Robustness and Quarantine Latency (RQ3):** We verify the stability of these findings across label-delay quarantine gaps (0m, 6m, 12m) using Two-One-Sided Tests (TOST) for equivalence against $\pm 0.02$ ROC-AUC. Within-project and unadapted pooled models demonstrate equivalence between 0m and 12m delay conditions ($p \le 0.0322$), with percentile bootstrap confidence interval divergence in within-project models traced to extreme negative skewness driven by a single project (`kafka`). For Random Forest CORAL, location-shift equivalence is established via Wilcoxon TOST ($p = 0.0420$), but mean difference equivalence is not established ($p = 0.0901$). We also conduct an audit sensitivity analysis on commit purpose (`fix`), demonstrating that apparent discrimination gains (+0.049 to +0.066 ROC-AUC) reflect retrospective Jira issue resolution leakage, and evaluate class imbalance treatments and alternative chronological splitting designs.

### Practical and Methodological Significance
Our study provides an essential empirical correction for software quality research: when evaluated under leak-free chronological conditions, complex domain adaptation algorithms provide no measurable benefit over simple unadapted pooling across historical donor repositories. We formulate concrete recommendations for future CP-JIT studies, including the mandatory inclusion of unadapted pooled baselines, out-of-sample inductive evaluation, strict chronological validation, and transparent reporting of target base rates.

### Manuscript & Author Declarations
- This manuscript represents original empirical research that has not been published previously and is not currently under consideration elsewhere.
- All authors have reviewed and approved the submission of this manuscript.
- A fully reproducible, self-contained replication package—complete with execution scripts, conda/pip environment specifications, random seed logs, and cryptographic SHA-256 manifests—is provided to facilitate seamless peer review.

Thank you very much for your time and editorial consideration.

Sincerely,  
**Kazi Mazharul Islam Mridul**  
East West University  
*[Contact Email - PENDING AUTHOR INPUT]*
