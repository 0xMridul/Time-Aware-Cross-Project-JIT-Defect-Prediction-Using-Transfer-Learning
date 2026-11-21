import os
import sys
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))
"""
summarize_gaps_sensitivity.py - Rigorous audit and statistical summary of Sensitivity Analysis 2:
Quarantine Gaps (0m, 6m, and 12m label-delay gaps).
Computes:
1. Direct paired cross-gap differences: 6m - 0m, 12m - 0m, 12m - 6m per method and classifier.
2. Within-gap relative transfer differences: Pooled vs WP, CORAL vs Pooled, TCA vs Pooled.
3. 10,000 bootstrap CIs (95% and 90%), paired Wilcoxon tests, paired t-tests,
   and TOST equivalence tests (+/-0.02) reporting p1, p2, and p_TOST for both Wilcoxon and t-test.
"""

import os
import numpy as np
import pandas as pd
from scipy import stats

def compute_tost_wilcoxon(diffs, margin=0.02):
    """
    Two-One-Sided Equivalence Test (TOST) using non-parametric Wilcoxon signed-rank tests
    against the bounds -margin and +margin.
    H01: diff <= -margin  vs  H11: diff > -margin  (p1)
    H02: diff >= +margin  vs  H12: diff < +margin  (p2)
    p_TOST = max(p1, p2)
    """
    diff_lower = diffs - (-margin)
    diff_upper = diffs - margin
    try:
        _, p1 = stats.wilcoxon(diff_lower, alternative='greater')
    except Exception:
        p1 = 1.0
    try:
        _, p2 = stats.wilcoxon(diff_upper, alternative='less')
    except Exception:
        p2 = 1.0
    return max(p1, p2), p1, p2

def compute_tost_ttest(diffs, margin=0.02):
    """
    Two-One-Sided Equivalence Test (TOST) using paired Student's t-test
    against the bounds -margin and +margin.
    H01: mean <= -margin  vs  H11: mean > -margin  (p1)
    H02: mean >= +margin  vs  H12: mean < +margin  (p2)
    p_TOST = max(p1, p2)
    """
    try:
        _, p1 = stats.ttest_1samp(diffs - (-margin), 0, alternative='greater')
    except Exception:
        p1 = 1.0
    try:
        _, p2 = stats.ttest_1samp(diffs - margin, 0, alternative='less')
    except Exception:
        p2 = 1.0
    return max(p1, p2), p1, p2

def run_summary():
    df_pri = pd.read_csv('results/primary_analysis.csv')
    df_gap = pd.read_csv('results/sensitivity_gaps.csv')
    
    std_methods = ['WP', 'Pooled CP', 'Source + Target-Train', 'CORAL', 'TCA with standardization']
    
    # 0m from primary analysis
    df_0m = df_pri[df_pri['method'].isin(std_methods)].copy()
    df_0m['gap'] = '0m'
    
    common_cols = ['gap', 'project', 'model', 'method', 'seed', 'auc', 'pr_auc']
    df_all = pd.concat([df_0m[common_cols], df_gap[common_cols]], ignore_index=True)
    
    # Save combined raw data
    df_all.to_csv('results/sensitivity_gaps_all.csv', index=False)
    print("Saved combined gap dataset to results/sensitivity_gaps_all.csv")
    
    # Project-level aggregation: for RF, average across the 10 seeds first
    agg = df_all.groupby(['gap', 'project', 'model', 'method'])[['auc', 'pr_auc']].mean().reset_index()
    
    rng = np.random.RandomState(42)
    
    paired_records = []
    recalc_all_records = []
    
    comparisons = [
        ('6m - 0m', '6m', '0m'),
        ('12m - 0m', '12m', '0m'),
        ('12m - 6m', '12m', '6m')
    ]
    
    print("\n=================================================================")
    print("   PART 1: DIRECT PAIRED CROSS-GAP COMPARISONS PER METHOD        ")
    print("=================================================================")
    
    for model in ['lr', 'rf']:
        print(f"\n==================== Model: {model.upper()} ====================")
        for comp_label, g_new, g_old in comparisons:
            print(f"\n--- {comp_label} ---")
            for m in std_methods:
                sub_new = agg[(agg['model'] == model) & (agg['gap'] == g_new) & (agg['method'] == m)].sort_values('project')
                sub_old = agg[(agg['model'] == model) & (agg['gap'] == g_old) & (agg['method'] == m)].sort_values('project')
                
                diff_auc = (sub_new.set_index('project')['auc'] - sub_old.set_index('project')['auc']).values
                diff_prauc = (sub_new.set_index('project')['pr_auc'] - sub_old.set_index('project')['pr_auc']).values
                n_proj = len(diff_auc)
                
                # --- AUC CALCULATIONS ---
                mean_auc = float(np.mean(diff_auc))
                std_auc = float(np.std(diff_auc, ddof=1))
                se_auc = stats.sem(diff_auc)
                
                # Bootstrap CIs (10,000 resamples)
                boot_auc = [np.mean(rng.choice(diff_auc, size=n_proj, replace=True)) for _ in range(10000)]
                ci_low_auc_95, ci_high_auc_95 = [float(x) for x in np.percentile(boot_auc, [2.5, 97.5])]
                ci_low_auc_90, ci_high_auc_90 = [float(x) for x in np.percentile(boot_auc, [5.0, 95.0])]
                
                # Parametric t-CIs
                t_crit_90 = stats.t.ppf(0.95, df=n_proj-1)
                t_crit_95 = stats.t.ppf(0.975, df=n_proj-1)
                ci_low_auc_t90 = float(mean_auc - t_crit_90 * se_auc)
                ci_high_auc_t90 = float(mean_auc + t_crit_90 * se_auc)
                ci_low_auc_t95 = float(mean_auc - t_crit_95 * se_auc)
                ci_high_auc_t95 = float(mean_auc + t_crit_95 * se_auc)
                
                pos_auc = int(np.sum(diff_auc > 0))
                neg_auc = int(np.sum(diff_auc < 0))
                tie_auc = int(np.sum(diff_auc == 0))
                
                # Two-sided tests
                try:
                    _, p_wilc_auc = stats.wilcoxon(diff_auc, alternative='two-sided')
                except Exception:
                    p_wilc_auc = 1.0
                try:
                    _, p_t_auc = stats.ttest_1samp(diff_auc, 0)
                except Exception:
                    p_t_auc = 1.0
                
                # TOST Equivalence Tests (+/- 0.02)
                p_tost_w_auc, p1_w_auc, p2_w_auc = compute_tost_wilcoxon(diff_auc, margin=0.02)
                p_tost_t_auc, p1_t_auc, p2_t_auc = compute_tost_ttest(diff_auc, margin=0.02)
                
                equiv_auc_90 = (ci_low_auc_90 >= -0.02 and ci_high_auc_90 <= 0.02)
                equiv_auc_95 = (ci_low_auc_95 >= -0.02 and ci_high_auc_95 <= 0.02)
                
                # --- PR-AUC CALCULATIONS ---
                mean_prauc = float(np.mean(diff_prauc))
                std_prauc = float(np.std(diff_prauc, ddof=1))
                se_prauc = stats.sem(diff_prauc)
                
                boot_prauc = [np.mean(rng.choice(diff_prauc, size=n_proj, replace=True)) for _ in range(10000)]
                ci_low_prauc_95, ci_high_prauc_95 = [float(x) for x in np.percentile(boot_prauc, [2.5, 97.5])]
                ci_low_prauc_90, ci_high_prauc_90 = [float(x) for x in np.percentile(boot_prauc, [5.0, 95.0])]
                
                ci_low_prauc_t90 = float(mean_prauc - t_crit_90 * se_prauc)
                ci_high_prauc_t90 = float(mean_prauc + t_crit_90 * se_prauc)
                ci_low_prauc_t95 = float(mean_prauc - t_crit_95 * se_prauc)
                ci_high_prauc_t95 = float(mean_prauc + t_crit_95 * se_prauc)
                
                pos_prauc = int(np.sum(diff_prauc > 0))
                neg_prauc = int(np.sum(diff_prauc < 0))
                tie_prauc = int(np.sum(diff_prauc == 0))
                
                try:
                    _, p_wilc_prauc = stats.wilcoxon(diff_prauc, alternative='two-sided')
                except Exception:
                    p_wilc_prauc = 1.0
                try:
                    _, p_t_prauc = stats.ttest_1samp(diff_prauc, 0)
                except Exception:
                    p_t_prauc = 1.0
                    
                p_tost_w_prauc, p1_w_prauc, p2_w_prauc = compute_tost_wilcoxon(diff_prauc, margin=0.02)
                p_tost_t_prauc, p1_t_prauc, p2_t_prauc = compute_tost_ttest(diff_prauc, margin=0.02)
                
                equiv_prauc_90 = (ci_low_prauc_90 >= -0.02 and ci_high_prauc_90 <= 0.02)
                equiv_prauc_95 = (ci_low_prauc_95 >= -0.02 and ci_high_prauc_95 <= 0.02)
                
                # Append to paired_records (for manuscript table generation)
                paired_records.append({
                    'model': model.upper(),
                    'method': m,
                    'comparison': comp_label,
                    'gap_new': g_new,
                    'gap_old': g_old,
                    'n_projects': n_proj,
                    'pos_auc': pos_auc,
                    'neg_auc': neg_auc,
                    'tie_auc': tie_auc,
                    'mean_diff_auc': mean_auc,
                    'ci_low_auc_95': ci_low_auc_95,
                    'ci_high_auc_95': ci_high_auc_95,
                    'ci_low_auc_90': ci_low_auc_90,
                    'ci_high_auc_90': ci_high_auc_90,
                    'p_wilc_auc': p_wilc_auc,
                    'p_t_auc': p_t_auc,
                    'p1_w_auc': p1_w_auc,
                    'p2_w_auc': p2_w_auc,
                    'p_tost_w_auc': p_tost_w_auc,
                    'p1_t_auc': p1_t_auc,
                    'p2_t_auc': p2_t_auc,
                    'p_tost_t_auc': p_tost_t_auc,
                    'equivalent_90_auc': equiv_auc_90,
                    'equivalent_95_auc': equiv_auc_95,
                    'mean_diff_prauc': mean_prauc,
                    'ci_low_prauc_95': ci_low_prauc_95,
                    'ci_high_prauc_95': ci_high_prauc_95,
                    'ci_low_prauc_90': ci_low_prauc_90,
                    'ci_high_prauc_90': ci_high_prauc_90,
                    'p_wilc_prauc': p_wilc_prauc,
                    'p_t_prauc': p_t_prauc,
                    'p1_w_prauc': p1_w_prauc,
                    'p2_w_prauc': p2_w_prauc,
                    'p_tost_w_prauc': p_tost_w_prauc,
                    'p1_t_prauc': p1_t_prauc,
                    'p2_t_prauc': p2_t_prauc,
                    'p_tost_t_prauc': p_tost_t_prauc,
                    'equivalent_90_prauc': equiv_prauc_90,
                    'equivalent_95_prauc': equiv_prauc_95
                })
                
                # Append to recalc_all_records (60 rows total)
                recalc_all_records.append({
                    'model': model.upper(),
                    'metric': 'AUC',
                    'comparison': comp_label,
                    'method': m,
                    'pos': pos_auc,
                    'neg': neg_auc,
                    'mean': mean_auc,
                    'std': std_auc,
                    'ci95_low': ci_low_auc_95,
                    'ci95_high': ci_high_auc_95,
                    'ci90_low': ci_low_auc_90,
                    'ci90_high': ci_high_auc_90,
                    'p_wilc': p_wilc_auc,
                    'p_t': p_t_auc,
                    'p1_w': p1_w_auc,
                    'p2_w': p2_w_auc,
                    'p_tost_w': p_tost_w_auc,
                    'p1_t': p1_t_auc,
                    'p2_t': p2_t_auc,
                    'p_tost_t': p_tost_t_auc
                })
                recalc_all_records.append({
                    'model': model.upper(),
                    'metric': 'PR_AUC',
                    'comparison': comp_label,
                    'method': m,
                    'pos': pos_prauc,
                    'neg': neg_prauc,
                    'mean': mean_prauc,
                    'std': std_prauc,
                    'ci95_low': ci_low_prauc_95,
                    'ci95_high': ci_high_prauc_95,
                    'ci90_low': ci_low_prauc_90,
                    'ci90_high': ci_high_prauc_90,
                    'p_wilc': p_wilc_prauc,
                    'p_t': p_t_prauc,
                    'p1_w': p1_w_prauc,
                    'p2_w': p2_w_prauc,
                    'p_tost_w': p_tost_w_prauc,
                    'p1_t': p1_t_prauc,
                    'p2_t': p2_t_prauc,
                    'p_tost_t': p_tost_t_prauc
                })
                
                print(f"{m:25s} | {comp_label:8s} | +:{pos_auc} -:{neg_auc} | Diff AUC: {mean_auc:+.4f} (90% CI: [{ci_low_auc_90:+.4f}, {ci_high_auc_90:+.4f}]) | p_wilc={p_wilc_auc:.4f} | W-TOST(p1={p1_w_auc:.4f}, p2={p2_w_auc:.4f}, p={p_tost_w_auc:.4f}) | t-TOST(p1={p1_t_auc:.4f}, p2={p2_t_auc:.4f}, p={p_tost_t_auc:.4f})")
                
    df_paired = pd.DataFrame(paired_records)
    df_paired.to_csv('results/sensitivity_gaps_paired_diffs.csv', index=False)
    print("\nSaved paired gap differences to results/sensitivity_gaps_paired_diffs.csv")
    
    df_recalc = pd.DataFrame(recalc_all_records)
    df_recalc.to_csv('results/quarantine_gap_recalculated_all.csv', index=False)
    print("Saved complete 60-row recalculation dataset to results/quarantine_gap_recalculated_all.csv")
    
    # Part 2: Within-gap relative transfer comparisons
    print("\n=================================================================")
    print("   PART 2: WITHIN-GAP TRANSFER COMPARISONS (CORAL/TCA vs POOLED) ")
    print("=================================================================")
    within_records = []
    for model in ['lr', 'rf']:
        for gap in ['0m', '6m', '12m']:
            sub = agg[(agg['model'] == model) & (agg['gap'] == gap)]
            piv_auc = sub.pivot(index='project', columns='method', values='auc')
            piv_prauc = sub.pivot(index='project', columns='method', values='pr_auc')
            
            for comp_name, m1, m2 in [
                ('Pooled vs WP', 'Pooled CP', 'WP'),
                ('CORAL vs WP', 'CORAL', 'WP'),
                ('TCA vs WP', 'TCA with standardization', 'WP'),
                ('CORAL vs Pooled', 'CORAL', 'Pooled CP'),
                ('TCA vs Pooled', 'TCA with standardization', 'Pooled CP')
            ]:
                diff_auc = (piv_auc[m1] - piv_auc[m2]).values
                diff_prauc = (piv_prauc[m1] - piv_prauc[m2]).values
                n_proj = len(diff_auc)
                
                boot_auc = [np.mean(rng.choice(diff_auc, size=n_proj, replace=True)) for _ in range(10000)]
                ci_low_auc, ci_high_auc = np.percentile(boot_auc, [2.5, 97.5])
                
                pos = int(np.sum(diff_auc > 0))
                neg = int(np.sum(diff_auc < 0))
                
                try:
                    _, p_auc = stats.wilcoxon(diff_auc, alternative='two-sided')
                except Exception:
                    p_auc = 1.0
                    
                within_records.append({
                    'model': model.upper(),
                    'gap': gap,
                    'comparison': comp_name,
                    'win_loss_auc': f"{pos}W / {neg}L",
                    'mean_diff_auc': np.mean(diff_auc),
                    'ci_low_auc_95': ci_low_auc,
                    'ci_high_auc_95': ci_high_auc,
                    'nominal_p_auc': p_auc,
                    'mean_diff_prauc': np.mean(diff_prauc)
                })
                print(f"[{model.upper()}][{gap:3s}] {comp_name:18s} | {pos}W / {neg}L | Diff: {np.mean(diff_auc):+.4f} (95% CI: [{ci_low_auc:+.4f}, {ci_high_auc:+.4f}], nom p={p_auc:.4f})")
                
    df_within = pd.DataFrame(within_records)
    df_within.to_csv('results/sensitivity_gaps_summary.csv', index=False)
    print("\nSaved updated gap sensitivity summary to results/sensitivity_gaps_summary.csv")

if __name__ == '__main__':
    run_summary()


if __name__ == '__main__':
    run_summary()
