# Step-by-Step Reproduction Instructions

### 1. Environment Setup
```bash
# Clone or navigate to the repository
cd apacheJIT

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r reproducibility/requirements.txt
```

### 2. Verify Dataset Integrity
```bash
python -c "
import pandas as pd
df = pd.read_csv('repo_check/dataset/apachejit_total.csv')
assert len(df) == 106674, f'Dataset count mismatch: {len(df)}'
assert df['commit_id'].nunique() == 106674, 'Duplicate commits found!'
print('Dataset integrity verified: 106,674 deduplicated commits.')
"
```

### 3. Reproduce Primary Confirmatory Experiments (RQ1)
```bash
# Run Design B primary analysis (LR + RF, 10 projects, 10 seeds)
python reproducibility/scripts/run_primary_analysis.py

# Summarize primary results and compute family-wide Holm-Bonferroni correction
python reproducibility/scripts/summarize_primary_results.py
```
Output files will be written to `results/primary_analysis.csv` and `results/primary_family_statistics.csv`.

### 4. Reproduce Longitudinal Temporal Drift Experiments (RQ2)
```bash
# Run Protocol D analysis (6 historical cutoffs, Mature 7 + Appendix 3, 2 variants)
python reproducibility/scripts/run_sensitivity_protocol_d.py

# Summarize temporal drift trajectories and compute 5-year changes
python reproducibility/scripts/summarize_protocol_d.py
```
Output files written to `results/sensitivity_protocol_d.csv` and `results/sensitivity_protocol_d_summary.csv`.

### 5. Reproduce Quarantine-Gap Equivalence Experiments (RQ3)
```bash
# Run 0m, 6m, and 12m quarantine gap evaluations
python reproducibility/scripts/run_sensitivity_gaps.py

# Compute TOST equivalence p-values, 90% and 95% bootstrap intervals
python reproducibility/scripts/summarize_gaps_sensitivity.py
```
Output files written to `results/quarantine_gap_recalculated_all.csv` and `results/sensitivity_gaps_summary.csv`.

### 6. Regenerate Publication Figures
```bash
python reproducibility/scripts/generate_plots.py
```
Generates all publication plots in `figures/main/` and `figures/supplementary/`.
