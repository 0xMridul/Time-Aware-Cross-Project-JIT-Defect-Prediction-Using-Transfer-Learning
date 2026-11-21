# Figure Index and Data Provenance

This directory archives all vector (PDF) and high-resolution raster (PNG, 300 DPI) figures for the manuscript.

### Main Manuscript Figures (`figures/main/`)
1. **`fig1_equivalence_forest.pdf` / `.png`:**
   - **Title:** Equivalence Forest Plot for Cross-Project JIT Transfer vs. Unadapted Pooled Training.
   - **Description:** Forest plot displaying paired mean differences in ROC-AUC with 90% (inner bar) and 95% (outer error bar) bootstrap confidence intervals relative to the pre-registered equivalence zone of $\pm 0.02$ ROC-AUC across Logistic Regression and Random Forest.
   - **Source Data:** `results/primary_family_statistics.csv` and `results/quarantine_gap_recalculated_all.csv`.
   - **Generation Script:** `pipeline/generate_plots.py` (or `reproducibility/scripts/generate_plots.py`).
2. **`fig2_temporal_drift_mature7.pdf` / `.png`:**
   - **Title:** Temporal Drift Trajectories across 5 Historical Years (Mature 7 Cohort).
   - **Description:** Line plots showing mean test ROC-AUC across 6 chronological training endpoints ($\le 2015$ down to $\le 2010$) for Natural Growing (left) and Fixed Training Size (right) variants, comparing WP, Pooled CP, CORAL, and TCA.
   - **Source Data:** `results/sensitivity_protocol_d.csv` and `results/sensitivity_protocol_d_summary.csv`.
   - **Generation Script:** `pipeline/generate_plots.py`.

### Supplementary Figures (`figures/supplementary/`)
1. **`fig_s1_temporal_drift_appendix3.pdf` / `.png`:**
   - **Title:** Temporal Drift Trajectories for Late-Established Projects (Appendix 3 Cohort: `ignite`, `flink`, `kafka`).
   - **Description:** Trajectory comparison for repositories with limited or zero commit history prior to 2011.
   - **Source Data:** `results/sensitivity_protocol_d.csv`.
   - **Generation Script:** `pipeline/generate_plots.py`.
