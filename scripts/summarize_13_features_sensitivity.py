import os
import sys
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))
"""
summarize_13_features_sensitivity.py - Compares 13-feature sensitivity results against
the 12-feature primary analysis baseline.
"""

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

def summarize():
    df_12 = pd.read_csv('results/primary_analysis.csv')
    df_13 = pd.read_csv('results/sensitivity_13_features.csv')
    
    # Filter 12-features to standard methods
    std_methods = ['WP', 'Pooled CP', 'Source + Target-Train', 'CORAL', 'TCA with standardization']
    df_12 = df_12[df_12['method'].isin(std_methods)]
    
    # Aggregate across seeds
    agg_12 = df_12.groupby(['project', 'model', 'method'])[['auc', 'pr_auc']].mean().reset_index()
    agg_13 = df_13.groupby(['project', 'model', 'method'])[['auc', 'pr_auc']].mean().reset_index()
    
    merged = pd.merge(agg_12, agg_13, on=['project', 'model', 'method'], suffixes=('_12', '_13'))
    merged['auc_diff'] = merged['auc_13'] - merged['auc_12']
    merged['prauc_diff'] = merged['pr_auc_13'] - merged['pr_auc_12']
    
    print("=================================================================")
    print("   SENSITIVITY ANALYSIS: 13 FEATURES (WITH 'FIX') vs 12 FEATURES  ")
    print("=================================================================")
    
    summary_rows = []
    for model in ['lr', 'rf']:
        print(f"\n--- Model: {model.upper()} ---")
        for method in std_methods:
            sub = merged[(merged['model'] == model) & (merged['method'] == method)]
            diffs_auc = sub['auc_diff'].values
            diffs_prauc = sub['prauc_diff'].values
            
            wins_auc = np.sum(diffs_auc > 0)
            losses_auc = np.sum(diffs_auc < 0)
            
            wins_prauc = np.sum(diffs_prauc > 0)
            losses_prauc = np.sum(diffs_prauc < 0)
            
            mean_diff_auc = np.mean(diffs_auc)
            mean_diff_prauc = np.mean(diffs_prauc)
            
            # Wilcoxon tests (two-sided)
            try:
                stat_auc, p_auc = wilcoxon(diffs_auc, alternative='two-sided')
            except Exception:
                stat_auc, p_auc = np.nan, 1.0
                
            try:
                stat_prauc, p_prauc = wilcoxon(diffs_prauc, alternative='two-sided')
            except Exception:
                stat_prauc, p_prauc = np.nan, 1.0
                
            summary_rows.append({
                'model': model.upper(),
                'method': method,
                'mean_auc_12': sub['auc_12'].mean(),
                'mean_auc_13': sub['auc_13'].mean(),
                'auc_diff': mean_diff_auc,
                'auc_wl': f"{wins_auc}W / {losses_auc}L",
                'auc_p': p_auc,
                'mean_prauc_12': sub['pr_auc_12'].mean(),
                'mean_prauc_13': sub['pr_auc_13'].mean(),
                'prauc_diff': mean_diff_prauc,
                'prauc_wl': f"{wins_prauc}W / {losses_prauc}L",
                'prauc_p': p_prauc
            })
            
            print(f"{method:25s} | AUC 12: {sub['auc_12'].mean():.4f} -> 13: {sub['auc_13'].mean():.4f} (Diff: {mean_diff_auc:+.4f}, {wins_auc}W/{losses_auc}L, p={p_auc:.4f}) | PR-AUC: {sub['pr_auc_12'].mean():.4f} -> {sub['pr_auc_13'].mean():.4f} (Diff: {mean_diff_prauc:+.4f}, p={p_prauc:.4f})")
            
    df_sum = pd.DataFrame(summary_rows)
    df_sum.to_csv('results/sensitivity_13_features_summary.csv', index=False)
    
    # Check if transfer vs WP/Pooled rankings change under 13 features
    print("\n=================================================================")
    print("   TRANSFER vs BASELINES UNDER 13 FEATURES (PAIRWISE COMPARISONS)")
    print("=================================================================")
    for model in ['lr', 'rf']:
        print(f"\n--- Model: {model.upper()} (13 Features) ---")
        piv = agg_13[agg_13['model'] == model].pivot(index='project', columns='method', values='auc')
        
        for comp_name, m1, m2 in [
            ('Pooled vs WP', 'Pooled CP', 'WP'),
            ('CORAL vs WP', 'CORAL', 'WP'),
            ('TCA vs WP', 'TCA with standardization', 'WP'),
            ('CORAL vs Pooled', 'CORAL', 'Pooled CP'),
            ('TCA vs Pooled', 'TCA with standardization', 'Pooled CP')
        ]:
            d = piv[m1] - piv[m2]
            w = np.sum(d > 0)
            l = np.sum(d < 0)
            try:
                _, p = wilcoxon(d, alternative='two-sided')
            except Exception:
                p = 1.0
            print(f"{comp_name:25s} | Win/Loss: {w}W / {l}L | Mean Diff: {d.mean():+.4f} | Wilcoxon p: {p:.4f}")

if __name__ == '__main__':
    summarize()
