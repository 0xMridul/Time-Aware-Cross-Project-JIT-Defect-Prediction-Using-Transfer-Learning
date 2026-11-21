import os
import sys
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))
"""
run_sensitivity_gaps.py - Sensitivity Analysis 2: Label-Delay Quarantine Gaps (6m and 12m).
Applies 6-month and 12-month label-delay gaps symmetrically to both Within-Project
and Cross-Project source training data under the primary 12-feature setup.
Saves results to results/sensitivity_gaps.csv.
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

T_TEST_START_B = 1451606400  # 2016-01-01 00:00:00 UTC
T_TEST_END_B   = 1514764799  # 2017-12-31 23:59:59 UTC

GAPS = {
    '6m': 1435708800,   # 2015-07-01 00:00:00 UTC (6-month quarantine before 2016-01-01)
    '12m': 1420070400   # 2015-01-01 00:00:00 UTC (12-month quarantine before 2016-01-01)
}

def run_gap_sensitivity():
    print("=================================================================")
    print("   RUNNING SENSITIVITY ANALYSIS 2: SYMMETRIC LABEL-DELAY GAPS    ")
    print("=================================================================")
    df = load_and_preprocess_total()
    os.makedirs('results', exist_ok=True)
    
    records = []
    
    for gap_label, t_cutoff in GAPS.items():
        print(f"\n#################################################################")
        print(f"            EVALUATING QUARANTINE GAP: {gap_label} (Cutoff < {t_cutoff})")
        print(f"#################################################################")
        
        for target in TARGET_PROJECTS_10:
            t0_proj = time.time()
            p_name = target.replace('apache/', '')
            print(f"\n>>> [{gap_label}] Target: {p_name} ...")
            
            p_df = df[df['project'] == target].sort_values('author_date')
            target_train = p_df[p_df['author_date'] < t_cutoff].copy()
            target_test = p_df[(p_df['author_date'] >= T_TEST_START_B) & (p_df['author_date'] <= T_TEST_END_B)].copy()
            source_train = df[(df['project'] != target) & (df['author_date'] < t_cutoff)].copy()
            
            # Verify strict canonical test split identity
            assert_canonical_test_split(target, target_test)
            
            n_test = len(target_test)
            n_test_bug = int(target_test['buggy'].sum())
            base_rate = float(n_test_bug / n_test) if n_test > 0 else 0.0
            
            # Extract 12 Kamei features with log1p
            X_wp_raw = np.log1p(np.maximum(0, target_train[KAMEI_12_FEATURES].values))
            y_wp = target_train['buggy'].values
            
            X_src_raw = np.log1p(np.maximum(0, source_train[KAMEI_12_FEATURES].values))
            y_src = source_train['buggy'].values
            
            X_te_raw = np.log1p(np.maximum(0, target_test[KAMEI_12_FEATURES].values))
            y_te = target_test['buggy'].values
            
            X_union_raw = np.vstack([X_src_raw, X_wp_raw])
            y_union = np.concatenate([y_src, y_wp])
            
            # Standardize strictly on training distributions
            s_wp = StandardScaler().fit(X_wp_raw)
            X_wp = s_wp.transform(X_wp_raw)
            X_te_wp = s_wp.transform(X_te_raw)
            
            s_src = StandardScaler().fit(X_src_raw)
            X_src = s_src.transform(X_src_raw)
            X_te_src = s_src.transform(X_te_raw)
            
            s_u = StandardScaler().fit(X_union_raw)
            X_union = s_u.transform(X_union_raw)
            X_te_u = s_u.transform(X_te_raw)
            
            # Validation split inside WP training for threshold & TCA tuning
            n_wp = len(X_wp)
            v_idx = max(int(n_wp * 0.8), n_wp - 1000)
            X_wp_sub_raw, y_wp_sub = X_wp_raw[:v_idx], y_wp[:v_idx]
            X_wp_val_raw, y_wp_val = X_wp_raw[v_idx:], y_wp[v_idx:]
            
            s_sub = StandardScaler().fit(X_wp_sub_raw)
            X_wp_sub = s_sub.transform(X_wp_sub_raw)
            X_wp_val = s_sub.transform(X_wp_val_raw)
            
            # Domain Adaptation
            # 1. CORAL
            coral = CORALAdapter()
            X_src_coral = coral.fit_transform(X_src, X_wp)
            
            # 2. TCA with standardization
            best_k, best_mu, val_auc = tune_tca_hyperparameters(
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
                'Source + Target-Train': (X_union, y_union, X_te_u, X_wp_val, y_wp_val),
                'CORAL': (X_src_coral, y_src, X_te_wp, X_wp_val, y_wp_val),
                'TCA with standardization': (X_src_tcap, y_src, X_te_tcap, X_val_tcap, y_wp_val)
            }
            
            # 1. Logistic Regression
            for m_name, (X_tr, y_tr, X_eval, X_v, y_v) in methods.items():
                clf = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
                clf.fit(X_tr, y_tr)
                p_val = clf.predict_proba(X_v)[:, 1]
                opt_th = optimize_threshold(y_v, p_val)
                p_te = clf.predict_proba(X_eval)[:, 1]
                m_def = compute_all_metrics(y_te, p_te, threshold=0.5)
                m_opt = compute_all_metrics(y_te, p_te, threshold=opt_th)
                
                records.append({
                    'gap': gap_label,
                    'project': p_name,
                    'model': 'lr',
                    'method': m_name,
                    'seed': 42,
                    'base_rate': base_rate,
                    'n_train': len(X_tr),
                    'n_test': n_test,
                    'n_test_bug': n_test_bug,
                    'auc': m_def['auc'],
                    'pr_auc': m_def['pr_auc'],
                    'th_chosen': opt_th,
                    'f1_0.5': m_def['f1'],
                    'f1_youden': m_opt['f1'],
                    'mcc_0.5': m_def['mcc'],
                    'mcc_youden': m_opt['mcc'],
                    'precision': m_opt['precision'],
                    'recall': m_opt['recall']
                })
                
            # 2. Random Forest (10 Seeds)
            for m_name, (X_tr, y_tr, X_eval, X_v, y_v) in methods.items():
                for s in range(42, 52):
                    clf = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=s, n_jobs=-1)
                    clf.fit(X_tr, y_tr)
                    p_val = clf.predict_proba(X_v)[:, 1]
                    opt_th = optimize_threshold(y_v, p_val)
                    p_te = clf.predict_proba(X_eval)[:, 1]
                    m_def = compute_all_metrics(y_te, p_te, threshold=0.5)
                    m_opt = compute_all_metrics(y_te, p_te, threshold=opt_th)
                    
                    records.append({
                        'gap': gap_label,
                        'project': p_name,
                        'model': 'rf',
                        'method': m_name,
                        'seed': s,
                        'base_rate': base_rate,
                        'n_train': len(X_tr),
                        'n_test': n_test,
                        'n_test_bug': n_test_bug,
                        'auc': m_def['auc'],
                        'pr_auc': m_def['pr_auc'],
                        'th_chosen': opt_th,
                        'f1_0.5': m_def['f1'],
                        'f1_youden': m_opt['f1'],
                        'mcc_0.5': m_def['mcc'],
                        'mcc_youden': m_opt['mcc'],
                        'precision': m_opt['precision'],
                        'recall': m_opt['recall']
                    })
            print(f"Finished [{gap_label}] {p_name} in {time.time()-t0_proj:.1f} s (WP Train: {len(X_wp_raw)}, Source Train: {len(X_src_raw)})")
            
    df_res = pd.DataFrame(records)
    df_res.to_csv('results/sensitivity_gaps.csv', index=False)
    print("\nSaved all gap sensitivity results to results/sensitivity_gaps.csv")
    return df_res

if __name__ == '__main__':
    run_gap_sensitivity()
