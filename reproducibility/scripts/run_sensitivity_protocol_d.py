import os
import sys
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))
"""
run_sensitivity_protocol_d.py - Sensitivity Analysis 6: Protocol D (Temporal Drift).
Fixed test window: 2016-2017 (author_date in [1451606400, 1514764799]).
Varying training endpoints: <=2015 (Gap 0), <=2014 (Gap 1), <=2013 (Gap 2), <=2012 (Gap 3), <=2011 (Gap 4), <=2010 (Gap 5).
Evaluates both:
1. Natural Growing variant (train on all available commits prior to cutoff).
2. Fixed Training Size variant (train on the most recent N commits prior to cutoff, where N = size at <=2010).
Saves results to results/sensitivity_protocol_d.csv.
"""

import os
import time
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

from data_loader import load_and_preprocess_total, TARGET_PROJECTS_10, KAMEI_12_FEATURES, assert_canonical_test_split
from transfer import CORALAdapter, TCAWithStandardization, tune_tca_hyperparameters
from metrics import compute_all_metrics, optimize_threshold

T_TEST_START = 1451606400  # 2016-01-01 00:00:00 UTC
T_TEST_END   = 1514764799  # 2017-12-31 23:59:59 UTC

DRIFT_GAPS = {
    '<=2015 (Gap 0)': 1451606400,
    '<=2014 (Gap 1)': 1420070400,
    '<=2013 (Gap 2)': 1388534400,
    '<=2012 (Gap 3)': 1356998400,
    '<=2011 (Gap 4)': 1325376000,
    '<=2010 (Gap 5)': 1293840000
}

# 7 mature projects active across all gaps with >= 270 bugs <= 2010
MATURE_7_PROJECTS = [
    'apache/camel', 'apache/hadoop', 'apache/hbase', 'apache/cassandra',
    'apache/groovy', 'apache/hive', 'apache/activemq'
]

def run_protocol_d():
    print("=================================================================")
    print("   RUNNING SENSITIVITY ANALYSIS 6: PROTOCOL D (TEMPORAL DRIFT)   ")
    print("=================================================================")
    df = load_and_preprocess_total()
    os.makedirs('results', exist_ok=True)
    
    # Precompute fixed sizes for mature 7 projects at <=2010
    t_gap5 = DRIFT_GAPS['<=2010 (Gap 5)']
    fixed_sizes_wp = {}
    fixed_sizes_src = {}
    for target in MATURE_7_PROJECTS:
        fixed_sizes_wp[target] = len(df[(df['project'] == target) & (df['author_date'] < t_gap5)])
        fixed_sizes_src[target] = len(df[(df['project'] != target) & (df['author_date'] < t_gap5)])
        
    records = []
    
    for variant in ['natural_growing', 'fixed_training_size']:
        print(f"\n=================================================================")
        print(f"                   VARIANT: {variant.upper()}")
        print(f"=================================================================")
        
        for gap_label, t_cutoff in DRIFT_GAPS.items():
            print(f"\n#################################################################")
            print(f"       [{variant}] TRAINING ENDPOINT: {gap_label}")
            print(f"#################################################################")
            
            for target in TARGET_PROJECTS_10:
                p_name = target.replace('apache/', '')
                
                # Fixed size variant is only evaluated for mature 7 projects
                if variant == 'fixed_training_size' and target not in MATURE_7_PROJECTS:
                    continue
                    
                p_df = df[df['project'] == target].sort_values('author_date')
                target_train = p_df[p_df['author_date'] < t_cutoff].copy()
                target_test = p_df[(p_df['author_date'] >= T_TEST_START) & (p_df['author_date'] <= T_TEST_END)].copy()
                source_train = df[(df['project'] != target) & (df['author_date'] < t_cutoff)].sort_values('author_date').copy()
                
                # Verify strict canonical test split identity
                assert_canonical_test_split(target, target_test)
                
                # Apply fixed size truncation if requested
                if variant == 'fixed_training_size':
                    n_wp_fix = fixed_sizes_wp[target]
                    n_src_fix = fixed_sizes_src[target]
                    target_train = target_train.tail(n_wp_fix).copy()
                    source_train = source_train.tail(n_src_fix).copy()
                
                n_wp_train = len(target_train)
                n_wp_bug = int(target_train['buggy'].sum())
                n_src_train = len(source_train)
                n_test = len(target_test)
                n_test_bug = int(target_test['buggy'].sum())
                base_rate = float(n_test_bug / n_test) if n_test > 0 else 0.0
                
                # Check if project has sufficient data at this gap (require >= 50 train commits and >= 5 bugs)
                has_sufficient_wp = (n_wp_train >= 50 and n_wp_bug >= 5)
                has_sufficient_src = (n_src_train >= 500)
                
                if not has_sufficient_src:
                    print(f"Skipping [{variant}][{gap_label}] {p_name}: insufficient source data ({n_src_train} commits)")
                    continue
                    
                t0_proj = time.time()
                print(f">>> [{variant}][{gap_label}] Target: {p_name} (WP Train: {n_wp_train}, Bugs: {n_wp_bug}, Src: {n_src_train}) ...")
                
                # Feature extraction
                X_src_raw = np.log1p(np.maximum(0, source_train[KAMEI_12_FEATURES].values))
                y_src = source_train['buggy'].values
                
                X_te_raw = np.log1p(np.maximum(0, target_test[KAMEI_12_FEATURES].values))
                y_te = target_test['buggy'].values
                
                s_src = StandardScaler().fit(X_src_raw)
                X_src = s_src.transform(X_src_raw)
                X_te_src = s_src.transform(X_te_raw)
                
                if has_sufficient_wp:
                    X_wp_raw = np.log1p(np.maximum(0, target_train[KAMEI_12_FEATURES].values))
                    y_wp = target_train['buggy'].values
                    s_wp = StandardScaler().fit(X_wp_raw)
                    X_wp = s_wp.transform(X_wp_raw)
                    X_te_wp = s_wp.transform(X_te_raw)
                    
                    # Validation split inside WP training
                    v_idx = max(int(n_wp_train * 0.8), n_wp_train - 1000)
                    X_wp_sub_raw, y_wp_sub = X_wp_raw[:v_idx], y_wp[:v_idx]
                    X_wp_val_raw, y_wp_val = X_wp_raw[v_idx:], y_wp[v_idx:]
                    
                    s_sub = StandardScaler().fit(X_wp_sub_raw)
                    X_wp_sub = s_sub.transform(X_wp_sub_raw)
                    X_wp_val = s_sub.transform(X_wp_val_raw)
                    
                    # CORAL
                    coral = CORALAdapter()
                    X_src_coral = coral.fit_transform(X_src, X_wp)
                    
                    # TCA with standardization
                    best_k, best_mu, _ = tune_tca_hyperparameters(
                        X_src_raw, y_src, X_wp_sub_raw, y_wp_sub, X_wp_val_raw, y_wp_val, with_standardization=True
                    )
                    tcap = TCAWithStandardization(n_components=best_k, mu=best_mu)
                    tcap.fit(X_src_raw, X_wp_raw)
                    X_src_tcap = tcap.transform_source(X_src_raw)
                    X_te_tcap = tcap.transform_target(X_te_raw)
                    X_val_tcap = tcap.transform_target(X_wp_val_raw)
                    
                    methods = {
                        'WP': (X_wp, y_wp, X_te_wp, X_wp_val, y_wp_val),
                        'Pooled CP': (X_src, y_src, X_te_src, X_wp_val, y_wp_val),
                        'CORAL': (X_src_coral, y_src, X_te_wp, X_wp_val, y_wp_val),
                        'TCA with standardization': (X_src_tcap, y_src, X_te_tcap, X_val_tcap, y_wp_val)
                    }
                else:
                    # Target has insufficient training data: evaluate Pooled CP only
                    methods = {
                        'Pooled CP': (X_src, y_src, X_te_src, X_src[:1000], y_src[:1000])
                    }
                    
                # Evaluate LR and RF
                for m_name, (X_tr, y_tr, X_eval, X_v, y_v) in methods.items():
                    clf_lr = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
                    clf_lr.fit(X_tr, y_tr)
                    p_val_lr = clf_lr.predict_proba(X_v)[:, 1]
                    opt_th_lr = optimize_threshold(y_v, p_val_lr)
                    p_te_lr = clf_lr.predict_proba(X_eval)[:, 1]
                    m_def_lr = compute_all_metrics(y_te, p_te_lr, threshold=0.5)
                    m_opt_lr = compute_all_metrics(y_te, p_te_lr, threshold=opt_th_lr)
                    
                    records.append({
                        'gap': gap_label,
                        'variant': variant,
                        'project': p_name,
                        'cohort': 'mature_7' if target in MATURE_7_PROJECTS else 'appendix_3',
                        'model': 'lr',
                        'method': m_name,
                        'seed': 42,
                        'base_rate': base_rate,
                        'n_wp_train': n_wp_train,
                        'n_wp_bug': n_wp_bug,
                        'n_src_train': n_src_train,
                        'n_test': n_test,
                        'n_test_bug': n_test_bug,
                        'auc': m_def_lr['auc'],
                        'pr_auc': m_def_lr['pr_auc'],
                        'th_chosen': opt_th_lr,
                        'f1_0.5': m_def_lr['f1'],
                        'f1_youden': m_opt_lr['f1'],
                        'mcc_0.5': m_def_lr['mcc'],
                        'mcc_youden': m_opt_lr['mcc']
                    })
                    
                    for s in range(42, 52):
                        clf_rf = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=s, n_jobs=-1)
                        clf_rf.fit(X_tr, y_tr)
                        p_val_rf = clf_rf.predict_proba(X_v)[:, 1]
                        opt_th_rf = optimize_threshold(y_v, p_val_rf)
                        p_te_rf = clf_rf.predict_proba(X_eval)[:, 1]
                        m_def_rf = compute_all_metrics(y_te, p_te_rf, threshold=0.5)
                        m_opt_rf = compute_all_metrics(y_te, p_te_rf, threshold=opt_th_rf)
                        
                        records.append({
                            'gap': gap_label,
                            'variant': variant,
                            'project': p_name,
                            'cohort': 'mature_7' if target in MATURE_7_PROJECTS else 'appendix_3',
                            'model': 'rf',
                            'method': m_name,
                            'seed': s,
                            'base_rate': base_rate,
                            'n_wp_train': n_wp_train,
                            'n_wp_bug': n_wp_bug,
                            'n_src_train': n_src_train,
                            'n_test': n_test,
                            'n_test_bug': n_test_bug,
                            'auc': m_def_rf['auc'],
                            'pr_auc': m_def_rf['pr_auc'],
                            'th_chosen': opt_th_rf,
                            'f1_0.5': m_def_rf['f1'],
                            'f1_youden': m_opt_rf['f1'],
                            'mcc_0.5': m_def_rf['mcc'],
                            'mcc_youden': m_opt_rf['mcc']
                        })
                print(f"Finished [{variant}][{gap_label}] {p_name} in {time.time()-t0_proj:.1f} s")
                
    df_res = pd.DataFrame(records)
    df_res.to_csv('results/sensitivity_protocol_d.csv', index=False)
    print("\nSaved all Protocol D sensitivity results to results/sensitivity_protocol_d.csv")
    return df_res

if __name__ == '__main__':
    run_protocol_d()
