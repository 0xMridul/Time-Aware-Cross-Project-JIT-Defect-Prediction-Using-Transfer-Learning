import os
import sys
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))
"""
summarize_70pct_sensitivity.py - Summarizes Sensitivity Analysis 4: 70% Chronological Split.
Includes 10,000 bootstrap resamples with fixed seed (seed=42) for project-level paired difference CIs.
"""

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

def summarize():
    df = pd.read_csv('results/sensitivity_70pct.csv')
    std_methods = ['WP', 'Pooled CP', 'Source + Target-Train', 'CORAL', 'TCA with standardization']
    
    agg = df.groupby(['project', 'model', 'method'])[['auc', 'pr_auc', 'f1_youden', 'mcc_youden']].mean().reset_index()
    
    print("=================================================================")
    print("   SENSITIVITY ANALYSIS 4: 70% CHRONOLOGICAL SPLIT SUMMARY       ")
    print("=================================================================")
    
    summary_rows = []
    rng = np.random.RandomState(42)
    
    for model in ['lr', 'rf']:
        print(f"\n==================== Model: {model.upper()} ====================")
        sub = agg[agg['model'] == model]
        piv_auc = sub.pivot(index='project', columns='method', values='auc')
        piv_prauc = sub.pivot(index='project', columns='method', values='pr_auc')
        piv_f1 = sub.pivot(index='project', columns='method', values='f1_youden')
        
        for m in std_methods:
            print(f"{m:25s} | Mean AUC: {piv_auc[m].mean():.4f} | Mean PR-AUC: {piv_prauc[m].mean():.4f} | F1 (Youden): {piv_f1[m].mean():.4f}")
            
        print(f"\nPairwise Transfer vs Baselines (70% Split):")
        for comp_name, m1, m2 in [
            ('Pooled vs WP', 'Pooled CP', 'WP'),
            ('CORAL vs WP', 'CORAL', 'WP'),
            ('TCA vs WP', 'TCA with standardization', 'WP'),
            ('CORAL vs Pooled', 'CORAL', 'Pooled CP'),
            ('TCA vs Pooled', 'TCA with standardization', 'Pooled CP')
        ]:
            diff_auc = piv_auc[m1] - piv_auc[m2]
            diff_prauc = piv_prauc[m1] - piv_prauc[m2]
            w = int(np.sum(diff_auc > 0))
            l = int(np.sum(diff_auc < 0))
            try:
                _, p_auc = wilcoxon(diff_auc, alternative='two-sided')
            except Exception:
                p_auc = 1.0
            try:
                _, p_prauc = wilcoxon(diff_prauc, alternative='two-sided')
            except Exception:
                p_prauc = 1.0
                
            boot_diffs = [np.mean(rng.choice(diff_auc.values, size=len(diff_auc), replace=True)) for _ in range(10000)]
            ci_low, ci_high = np.percentile(boot_diffs, [2.5, 97.5])
            
            summary_rows.append({
                'model': model.upper(),
                'comparison': comp_name,
                'win_loss_auc': f"{w}W / {l}L",
                'mean_diff_auc': diff_auc.mean(),
                'ci_low_10k': ci_low,
                'ci_high_10k': ci_high,
                'nominal_p_auc': p_auc,
                'mean_diff_prauc': diff_prauc.mean(),
                'nominal_p_prauc': p_prauc
            })
            print(f"  {comp_name:20s} | {w}W / {l}L | Diff: {diff_auc.mean():+.4f} (95% CI: [{ci_low:+.4f}, {ci_high:+.4f}], nom p={p_auc:.4f})")
            
    df_summary = pd.DataFrame(summary_rows)
    df_summary.to_csv('results/sensitivity_70pct_summary.csv', index=False)
    print("\nSaved 70% split summary to results/sensitivity_70pct_summary.csv")

if __name__ == '__main__':
    summarize()
