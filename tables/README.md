# Table Index and Data Mapping

This directory contains tabular data files formatted as CSV for immediate replication and secondary analysis.

### Main Manuscript Tables (`tables/main/`)
- **`table1_target_cohort.csv`:** Target project scale, training ($< 2016$) and testing ($2016$–$2017$) commit and defect counts.
- **`table2_primary_rq1_hypotheses.csv`:** Hypothesis testing across 20 primary confirmatory comparisons with bootstrap CIs, rank-biserial correlations, nominal p, and Holm-adjusted p.
- **`table3_temporal_drift_mature7.csv`:** Longitudinal performance across 6 historical gaps for Natural Growing and Fixed Training Size variants.
- **`table4_quarantine_gap_tost.csv`:** Quarantine-gap equivalence testing (0m, 6m, 12m) with TOST p-values.

### Supplementary Sensitivity Tables (`tables/supplementary/`)
- **`table_s1_sensitivity_13_features.csv`:** Retrospective information leakage sensitivity analysis (12 pre-commit vs. 13 metrics including post-commit Jira resolution feature `fix`).
- **`table_s2_sensitivity_imbalance.csv`:** Comparison of balanced weights, unweighted, and SMOTE.
- **`table_s3_sensitivity_70pct.csv`:** 70% chronological split per-project results.
- **`table_s4_sensitivity_protocol_a.csv`:** Official Zenodo split results.
- **`table_s5_sensitivity_hadoop_full_exclusion.csv`:** Post-hoc exploratory analysis with Hadoop excluded.
- **`table_s6_quarantine_gap_all_60_comparisons.csv`:** Complete 60-comparison TOST recalculation audit file.
