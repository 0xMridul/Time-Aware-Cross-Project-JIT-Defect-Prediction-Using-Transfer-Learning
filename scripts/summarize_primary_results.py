import os
import sys
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))
"""
summarize_primary_results.py - Post-processing script for the primary analysis results.
Computes:
1. Per-project tables for Logistic Regression and Random Forest with 95% Bootstrap CIs.
2. Win / Loss / Tie counts vs Within-Project across the 10 target projects.
3. Paired two-tailed Wilcoxon signed-rank tests and Holm-Bonferroni corrected p-values.
4. Cliff's delta effect sizes and magnitude classifications.
"""

import pandas as pd
import numpy as np
from metrics import paired_wilcoxon_test, holm_bonferroni_correction, cliffs_delta

def summarize():
    df = pd.read_csv('results/primary_analysis.csv')
    
    methods_order = ['WP', 'Pooled CP', 'Source + Target-Train', 'CORAL', 'TCA', 'TCA with standardization']
    
    # 1. Base rate per target project
    print("=================================================================")
    print("                     TARGET PROJECT BASE RATES                   ")
    print("=================================================================")
    base_rates = df[['project', 'n_test', 'n_test_bug', 'base_rate']].drop_duplicates().sort_values('project')
    print(base_rates.to_string(index=False))
    
    for model_name in ['lr', 'rf']:
        m_label = "LOGISTIC REGRESSION" if model_name == 'lr' else "RANDOM FOREST (Mean over 10 Seeds)"
        print(f"\n=================================================================")
        print(f"               PRIMARY ANALYSIS: {m_label}               ")
        print(f"=================================================================")
        
        sub = df[df['model'] == model_name]
        
        # Aggregate across seeds if rf
        agg = sub.groupby(['project', 'method']).agg({
            'auc': 'mean',
            'auc_ci_lower': 'mean',
            'auc_ci_upper': 'mean',
            'pr_auc': 'mean',
            'prauc_ci_lower': 'mean',
            'prauc_ci_upper': 'mean',
            'th_chosen': 'mean',
            'f1_0.5': 'mean',
            'f1_youden': 'mean',
            'mcc_0.5': 'mean',
            'mcc_youden': 'mean'
        }).reset_index()
        
        # Per project tables
        projects = sorted(agg['project'].unique())
        
        # Table: AUC across methods
        pivot_auc = agg.pivot(index='project', columns='method', values='auc')[methods_order]
        print("\n--- ROC-AUC by Project and Method ---")
        print(pivot_auc.round(4).to_string())
        
        # Table: PR-AUC across methods
        pivot_prauc = agg.pivot(index='project', columns='method', values='pr_auc')[methods_order]
        print("\n--- PR-AUC by Project and Method ---")
        print(pivot_prauc.round(4).to_string())
        
        # Table: Chosen Youden Threshold by Project and Method
        pivot_th = agg.pivot(index='project', columns='method', values='th_chosen')[methods_order]
        print("\n--- Youden Threshold Chosen by Project and Method ---")
        print(pivot_th.round(4).to_string())
        
        # Table: F1 at Chosen Youden Threshold
        pivot_f1 = agg.pivot(index='project', columns='method', values='f1_youden')[methods_order]
        print("\n--- F1 (at Youden Threshold) by Project and Method ---")
        print(pivot_f1.round(4).to_string())
        
        # Table: MCC at Chosen Youden Threshold
        pivot_mcc = agg.pivot(index='project', columns='method', values='mcc_youden')[methods_order]
        print("\n--- MCC (at Youden Threshold) by Project and Method ---")
        print(pivot_mcc.round(4).to_string())
        
        # 2. Win / Loss Counts vs Within-Project
        print("\n--- Win / Loss / Tie Counts vs Within-Project (WP) across 10 Targets ---")
        wp_auc = pivot_auc['WP']
        wp_prauc = pivot_prauc['WP']
        
        comparison_methods = ['Pooled CP', 'Source + Target-Train', 'CORAL', 'TCA', 'TCA with standardization']
        
        win_loss_records = []
        raw_pvals_auc = []
        raw_pvals_prauc = []
        deltas_auc = []
        deltas_prauc = []
        
        for comp in comparison_methods:
            comp_auc = pivot_auc[comp]
            comp_prauc = pivot_prauc[comp]
            
            # AUC Wins/Losses
            diff_auc = comp_auc - wp_auc
            wins_auc = int(np.sum(diff_auc > 0.001))
            losses_auc = int(np.sum(diff_auc < -0.001))
            ties_auc = 10 - wins_auc - losses_auc
            
            # PR-AUC Wins/Losses
            diff_prauc = comp_prauc - wp_prauc
            wins_prauc = int(np.sum(diff_prauc > 0.001))
            losses_prauc = int(np.sum(diff_prauc < -0.001))
            ties_prauc = 10 - wins_prauc - losses_prauc
            
            # Wilcoxon test across 10 projects
            stat_a, p_a = paired_wilcoxon_test(comp_auc.values, wp_auc.values)
            stat_p, p_p = paired_wilcoxon_test(comp_prauc.values, wp_prauc.values)
            
            raw_pvals_auc.append(p_a)
            raw_pvals_prauc.append(p_p)
            
            d_a, mag_a = cliffs_delta(comp_auc.values, wp_auc.values)
            d_p, mag_p = cliffs_delta(comp_prauc.values, wp_prauc.values)
            deltas_auc.append((d_a, mag_a))
            deltas_prauc.append((d_p, mag_p))
            
            win_loss_records.append({
                'Comparison vs WP': comp,
                'AUC Win/Loss/Tie': f"{wins_auc}W / {losses_auc}L / {ties_auc}T",
                'Mean Δ AUC': float(np.mean(diff_auc)),
                'Wilcoxon p (AUC)': p_a,
                'Cliff δ (AUC)': f"{d_a:.3f} ({mag_a})",
                'PR-AUC Win/Loss/Tie': f"{wins_prauc}W / {losses_prauc}L / {ties_prauc}T",
                'Mean Δ PR-AUC': float(np.mean(diff_prauc)),
                'Wilcoxon p (PR-AUC)': p_p,
                'Cliff δ (PR-AUC)': f"{d_p:.3f} ({mag_p})"
            })
            
        adj_pvals_auc = holm_bonferroni_correction(raw_pvals_auc)
        adj_pvals_prauc = holm_bonferroni_correction(raw_pvals_prauc)
        
        for i in range(len(win_loss_records)):
            win_loss_records[i]['Holm p (AUC)'] = adj_pvals_auc[i]
            win_loss_records[i]['Holm p (PR-AUC)'] = adj_pvals_prauc[i]
            
        df_wl = pd.DataFrame(win_loss_records)
        cols_show = [
            'Comparison vs WP', 'AUC Win/Loss/Tie', 'Mean Δ AUC', 'Wilcoxon p (AUC)', 'Holm p (AUC)', 'Cliff δ (AUC)',
            'PR-AUC Win/Loss/Tie', 'Mean Δ PR-AUC', 'Wilcoxon p (PR-AUC)', 'Holm p (PR-AUC)', 'Cliff δ (PR-AUC)'
        ]
        print(df_wl[cols_show].to_string(index=False))

if __name__ == '__main__':
    summarize()
