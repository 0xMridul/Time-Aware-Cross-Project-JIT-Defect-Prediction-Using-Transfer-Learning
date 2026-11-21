import os
import sys
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))
"""
summarize_protocol_d.py - Summarizes Sensitivity Analysis 6: Protocol D (Temporal Drift).
Evaluates defect prediction performance across training endpoints:
<=2015 (Gap 0), <=2014 (Gap 1), <=2013 (Gap 2), <=2012 (Gap 3), <=2011 (Gap 4), <=2010 (Gap 5).
Evaluates both:
1. Natural Growing variant
2. Fixed Training Size variant
Focuses main evaluation on the 7 mature projects active across all 5 gaps:
['camel', 'hadoop', 'hbase', 'cassandra', 'groovy', 'hive', 'activemq'].
Reports appendix results for ['ignite', 'flink', 'kafka'].
Includes 10,000 bootstrap resamples with fixed seed (seed=42).
"""

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

MATURE_7 = ['camel', 'hadoop', 'hbase', 'cassandra', 'groovy', 'hive', 'activemq']
APPENDIX_3 = ['ignite', 'flink', 'kafka']

def summarize():
    df = pd.read_csv('results/sensitivity_protocol_d.csv')
    gaps = ['<=2015 (Gap 0)', '<=2014 (Gap 1)', '<=2013 (Gap 2)', '<=2012 (Gap 3)', '<=2011 (Gap 4)', '<=2010 (Gap 5)']
    std_methods = ['WP', 'Pooled CP', 'CORAL', 'TCA with standardization']
    
    agg = df.groupby(['variant', 'gap', 'project', 'model', 'method'])[['auc', 'pr_auc', 'f1_youden', 'mcc_youden']].mean().reset_index()
    
    print("=================================================================")
    print("   SENSITIVITY ANALYSIS 6: PROTOCOL D (TEMPORAL DRIFT SUMMARY)   ")
    print("=================================================================")
    
    summary_rows = []
    rng = np.random.RandomState(42)
    
    for variant in ['natural_growing', 'fixed_training_size']:
        print(f"\n#################################################################")
        print(f"               VARIANT: {variant.upper()}")
        print(f"#################################################################")
        
        for model in ['lr', 'rf']:
            print(f"\n==================== Model: {model.upper()} (MATURE 7 COHORT) ====================")
            
            # Drift trajectory table
            drift_table = []
            for g in gaps:
                sub = agg[(agg['variant'] == variant) & (agg['model'] == model) & (agg['gap'] == g) & (agg['project'].isin(MATURE_7))]
                piv_auc = sub.pivot(index='project', columns='method', values='auc')
                piv_prauc = sub.pivot(index='project', columns='method', values='pr_auc')
                
                row = {'gap': g}
                for m in std_methods:
                    if m in piv_auc.columns:
                        row[f'{m}_auc'] = piv_auc[m].mean()
                        row[f'{m}_prauc'] = piv_prauc[m].mean()
                drift_table.append(row)
                
                # Pairwise vs Pooled CP
                for comp_name, m1, m2 in [
                    ('Pooled vs WP', 'Pooled CP', 'WP'),
                    ('CORAL vs Pooled', 'CORAL', 'Pooled CP'),
                    ('TCA vs Pooled', 'TCA with standardization', 'Pooled CP')
                ]:
                    if m1 in piv_auc.columns and m2 in piv_auc.columns:
                        diff_auc = piv_auc[m1] - piv_auc[m2]
                        w = int(np.sum(diff_auc > 0))
                        l = int(np.sum(diff_auc < 0))
                        try:
                            _, p_auc = wilcoxon(diff_auc, alternative='two-sided')
                        except Exception:
                            p_auc = 1.0
                        boot_diffs = [np.mean(rng.choice(diff_auc.values, size=len(diff_auc), replace=True)) for _ in range(10000)]
                        ci_low, ci_high = np.percentile(boot_diffs, [2.5, 97.5])
                        
                        summary_rows.append({
                            'variant': variant,
                            'cohort': 'mature_7',
                            'model': model.upper(),
                            'gap': g,
                            'comparison': comp_name,
                            'win_loss_auc': f"{w}W / {l}L",
                            'mean_diff_auc': diff_auc.mean(),
                            'ci_low_10k': ci_low,
                            'ci_high_10k': ci_high,
                            'nominal_p_auc': p_auc
                        })
                        
            df_drift = pd.DataFrame(drift_table)
            print(f"\n--- Drift Trajectory (Mean ROC-AUC across Mature 7 Projects, {variant}) ---")
            cols_to_show = ['gap'] + [c for c in df_drift.columns if c.endswith('_auc')]
            print(df_drift[cols_to_show].to_string(index=False))
            
            print(f"\n--- Pairwise Transfer vs Pooled CP at Extreme Gaps ({variant}) ---")
            for r in summary_rows:
                if r['variant'] == variant and r['model'] == model.upper() and r['gap'] in ['<=2015 (Gap 0)', '<=2010 (Gap 5)']:
                    print(f"  [{r['gap']}] {r['comparison']:20s} | {r['win_loss_auc']} | Diff: {r['mean_diff_auc']:+.4f} (95% CI: [{r['ci_low_10k']:+.4f}, {r['ci_high_10k']:+.4f}], nom p={r['nominal_p_auc']:.4f})")
                    
    df_summary = pd.DataFrame(summary_rows)
    df_summary.to_csv('results/sensitivity_protocol_d_summary.csv', index=False)
    print("\nSaved Protocol D summary to results/sensitivity_protocol_d_summary.csv")

if __name__ == '__main__':
    summarize()
