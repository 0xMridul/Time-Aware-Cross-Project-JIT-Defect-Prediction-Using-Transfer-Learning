import os
import sys
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))
"""
generate_plots.py - Generates publication-ready figures for the empirical paper:
1. figures/fig_temporal_drift_mature7.png / .pdf: Main temporal drift trajectory across 6 gaps for the 7 mature projects.
2. figures/fig_temporal_drift_appendix3.png / .pdf: Appendix plot for the 3 younger projects.
3. figures/fig_equivalence_forest.png / .pdf: Forest plot showing mean paired differences and 90%/95% bootstrap CIs
   against the +/-0.02 equivalence margin.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

os.makedirs('figures', exist_ok=True)

# Publication styling
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 11
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9
plt.rcParams['figure.titlesize'] = 12

def plot_temporal_drift():
    if not os.path.exists('results/sensitivity_protocol_d.csv'):
        print("results/sensitivity_protocol_d.csv not found yet.")
        return
        
    df = pd.read_csv('results/sensitivity_protocol_d.csv')
    mature_7 = ['camel', 'hadoop', 'hbase', 'cassandra', 'groovy', 'hive', 'activemq']
    gaps_ordered = ['<=2015 (Gap 0)', '<=2014 (Gap 1)', '<=2013 (Gap 2)', '<=2012 (Gap 3)', '<=2011 (Gap 4)', '<=2010 (Gap 5)']
    gap_x = [0, 1, 2, 3, 4, 5]
    gap_labels = ['Gap 0\n(<=2015)', 'Gap 1\n(<=2014)', 'Gap 2\n(<=2013)', 'Gap 3\n(<=2012)', 'Gap 4\n(<=2011)', 'Gap 5\n(<=2010)']
    
    # 1. Main Plot: Mature 7 Projects (Natural Growing vs Fixed Size)
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), sharey=True)
    
    colors = {
        'WP': '#1f77b4',
        'Pooled CP': '#ff7f0e',
        'CORAL': '#2ca02c',
        'TCA with standardization': '#d62728'
    }
    markers = {
        'WP': 'o',
        'Pooled CP': 's',
        'CORAL': '^',
        'TCA with standardization': 'D'
    }
    
    for row_idx, variant in enumerate(['natural_growing', 'fixed_training_size']):
        var_label = "Natural Growing (All Historical Data)" if variant == 'natural_growing' else "Fixed Training Size (N = Size at <=2010)"
        for col_idx, model in enumerate(['lr', 'rf']):
            ax = axes[row_idx, col_idx]
            sub = df[(df['variant'] == variant) & (df['model'] == model) & (df['project'].isin(mature_7))]
            agg = sub.groupby(['gap', 'method'])['auc'].mean().reset_index()
            
            for m in ['WP', 'Pooled CP', 'CORAL', 'TCA with standardization']:
                vals = []
                for g in gaps_ordered:
                    v = agg[(agg['gap'] == g) & (agg['method'] == m)]['auc'].values
                    vals.append(v[0] if len(v) > 0 else np.nan)
                ax.plot(gap_x, vals, label=m, color=colors[m], marker=markers[m], linewidth=2, markersize=6)
                
            ax.set_title(f"{'Logistic Regression' if model == 'lr' else 'Random Forest'} — {var_label}")
            ax.set_xticks(gap_x)
            ax.set_xticklabels(gap_labels)
            ax.set_ylim(0.68, 0.82)
            ax.grid(True, linestyle='--', alpha=0.5)
            if col_idx == 0:
                ax.set_ylabel("Mean Test ROC-AUC (2016–2017)")
            if row_idx == 0 and col_idx == 1:
                ax.legend(loc='lower left', framealpha=0.9)
                
    fig.suptitle("Temporal Drift Trajectory across 5 Years (7 Mature Projects Cohort)", fontsize=13, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    plt.savefig('figures/fig_temporal_drift_mature7.png', dpi=300)
    plt.savefig('figures/fig_temporal_drift_mature7.pdf')
    plt.close()
    print("Saved figures/fig_temporal_drift_mature7.png and .pdf")
    
    # 2. Appendix Plot: Younger Projects (Ignite, Flink, Kafka)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    app_3 = ['ignite', 'flink', 'kafka']
    for col_idx, model in enumerate(['lr', 'rf']):
        ax = axes[col_idx]
        sub = df[(df['variant'] == 'natural_growing') & (df['model'] == model) & (df['project'].isin(app_3))]
        agg = sub.groupby(['gap', 'method'])['auc'].mean().reset_index()
        for m in ['WP', 'Pooled CP', 'CORAL', 'TCA with standardization']:
            vals = []
            for g in gaps_ordered:
                v = agg[(agg['gap'] == g) & (agg['method'] == m)]['auc'].values
                vals.append(v[0] if len(v) > 0 else np.nan)
            ax.plot(gap_x, vals, label=m, color=colors[m], marker=markers[m], linewidth=2, markersize=6)
        ax.set_title(f"{'Logistic Regression' if model == 'lr' else 'Random Forest'} (Appendix Projects)")
        ax.set_xticks(gap_x)
        ax.set_xticklabels(gap_labels)
        ax.set_ylim(0.60, 0.85)
        ax.grid(True, linestyle='--', alpha=0.5)
        if col_idx == 0:
            ax.set_ylabel("Mean Test ROC-AUC (2016–2017)")
            ax.legend(loc='lower left')
            
    fig.suptitle("Temporal Drift for Appendix Cohort (Ignite, Flink, Kafka)", fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig('figures/fig_temporal_drift_appendix3.png', dpi=300)
    plt.savefig('figures/fig_temporal_drift_appendix3.pdf')
    plt.close()
    print("Saved figures/fig_temporal_drift_appendix3.png and .pdf")

def plot_equivalence_forest():
    # Equivalence Forest Plot comparing transfer methods to Pooled CP
    # Margin: [-0.02, +0.02]
    # Datasets / analyses to compile:
    # 1. Primary Analysis (Design B, 0m gap, 12 feat)
    # 2. Sensitivity 1 (13 feat)
    # 3. Sensitivity 2 (6m gap)
    # 4. Sensitivity 2 (12m gap)
    # 5. Sensitivity 3 (Balanced)
    # 6. Sensitivity 3 (None)
    # 7. Sensitivity 3 (SMOTE)
    # 8. Sensitivity 4 (70% split)
    # 9. Sensitivity 5 (Protocol A)
    
    comparisons = [
        # (Label, Mean_Diff_LR, CI_low_LR, CI_high_LR, Mean_Diff_RF, CI_low_RF, CI_high_RF)
        ("Primary (12 feat, 0m gap)", 
         -0.0040, -0.0101, 0.0019,  # LR CORAL
         -0.0220, -0.0335, -0.0108), # RF CORAL
        ("Sensitivity 1 (13 feat)", 
         -0.0020, -0.0050, 0.0008, 
         -0.0169, -0.0315, -0.0024),
        ("Sensitivity 2 (6m gap)", 
         -0.0040, -0.0098, 0.0018, 
         -0.0226, -0.0350, -0.0101),
        ("Sensitivity 2 (12m gap)", 
         -0.0038, -0.0091, 0.0014, 
         -0.0273, -0.0477, -0.0095),
        ("Sensitivity 3 (None)", 
         -0.0004, -0.0032, 0.0023, 
         -0.0336, -0.0543, -0.0129),
        ("Sensitivity 3 (SMOTE)", 
         -0.0007, -0.0035, 0.0020, 
         -0.0386, -0.0655, -0.0116),
        ("Sensitivity 4 (70% split)", 
         -0.0017, -0.0048, 0.0015, 
         -0.0100, -0.0210, 0.0011),
        ("Sensitivity 5 (Protocol A)", 
         -0.0037, -0.0081, 0.0000, 
         -0.0070, -0.0154, 0.0022),
    ]
    
    labels = [c[0] for c in comparisons]
    y_pos = np.arange(len(labels))
    
    fig, axes = plt.subplots(1, 2, figsize=(13, 6), sharey=True)
    
    for ax_idx, (model_name, mean_idx, low_idx, high_idx) in enumerate([
        ('Logistic Regression (CORAL vs. Pooled CP)', 1, 2, 3),
        ('Random Forest (CORAL vs. Pooled CP)', 4, 5, 6)
    ]):
        ax = axes[ax_idx]
        means = [c[mean_idx] for c in comparisons]
        lows = [c[low_idx] for c in comparisons]
        highs = [c[high_idx] for c in comparisons]
        err_low = [m - l for m, l in zip(means, lows)]
        err_high = [h - m for m, h in zip(means, highs)]
        
        # Equivalence zone shading
        ax.axvspan(-0.02, 0.02, color='#2ca02c', alpha=0.15, label='Equivalence Zone (±0.02 AUC)')
        ax.axvline(0.0, color='black', linestyle='--', linewidth=1, alpha=0.7)
        ax.axvline(-0.02, color='#d62728', linestyle=':', linewidth=1.2)
        ax.axvline(0.02, color='#d62728', linestyle=':', linewidth=1.2)
        
        # Plot points and error bars
        ax.errorbar(means, y_pos, xerr=[err_low, err_high], fmt='o', color='#1f77b4', ecolor='#1f77b4',
                    elinewidth=2, capsize=4, markersize=6, label='Mean Paired Diff (95% CI)')
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels)
        ax.invert_yaxis()
        ax.set_xlabel("Mean Paired Difference in ROC-AUC")
        ax.set_title(model_name, fontweight='bold')
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.set_xlim(-0.08, 0.04)
        if ax_idx == 0:
            ax.legend(loc='lower left', framealpha=0.9)
            
    fig.suptitle("Equivalence Forest Plot: Transfer (CORAL) vs. Unadapted Pooled CP across All Settings", fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.subplots_adjust(top=0.90)
    plt.savefig('figures/fig_equivalence_forest.png', dpi=300)
    plt.savefig('figures/fig_equivalence_forest.pdf')
    plt.close()
    print("Saved figures/fig_equivalence_forest.png and .pdf")

if __name__ == '__main__':
    plot_temporal_drift()
    plot_equivalence_forest()
