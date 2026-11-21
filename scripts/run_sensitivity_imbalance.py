import os
import sys
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))
"""
run_sensitivity_imbalance.py - Sensitivity Analysis 3: Class Imbalance Handling.
Evaluates 3 imbalance treatments under primary Design B (12 features, 0-month gap, test 2016-2017):
1. 'balanced': class_weight='balanced' (Primary baseline)
2. 'none': class_weight=None
3. 'smote': SMOTE applied strictly to training folds
Saves results to results/sensitivity_imbalance.csv.
"""

import os
import time
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

from data_loader import load_and_preprocess_total, TARGET_PROJECTS_10, KAMEI_12_FEATURES, assert_canonical_test_split
from transfer import CORALAdapter, TCAWithStandardization, tune_tca_hyperparameters
from metrics import compute_all_metrics, optimize_threshold

T_CUTOFF_B = 1451606400  # 2016-01-01 00:00:00 UTC
T_END_B    = 1514764799  # 2017-12-31 23:59:59 UTC

TREATMENTS = ['balanced', 'none', 'smote']

def run_imbalance_sensitivity():
    print("=================================================================")
    print("   RUNNING SENSITIVITY ANALYSIS 3: CLASS IMBALANCE HANDLING      ")
    print("=================================================================")
    df = load_and_preprocess_total()
    os.makedirs('results', exist_ok=True)
    
    records = []
    
    for target in TARGET_PROJECTS_10:
        t0_proj = time.time()
        p_name = target.replace('apache/', '')
        print(f"\n>>> Processing Target: {p_name} ...")
        
        p_df = df[df['project'] == target].sort_values('author_date')
        target_train = p_df[p_df['author_date'] < T_CUTOFF_B].copy()
        target_test = p_df[(p_df['author_date'] >= T_CUTOFF_B) & (p_df['author_date'] <= T_END_B)].copy()
        source_train = df[(df['project'] != target) & (df['author_date'] < T_CUTOFF_B)].copy()
        
        # Verify strict test identity against locked canonical split
        assert_canonical_test_split(target, target_test)
        
        n_test = len(target_test)
        n_test_bug = int(target_test['buggy'].sum())
        base_rate = float(n_test_bug / n_test) if n_test > 0 else 0.0
        
        # Raw features with log1p
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
        best_k, best_mu, _ = tune_tca_hyperparameters(
            X_src_raw, y_src, X_wp_sub_raw, y_wp_sub, X_wp_val_raw, y_wp_val, with_standardization=True
        )
        tcap = TCAWithStandardization(n_components=best_k, mu=best_mu)
        tcap.fit(X_src_raw, X_wp_raw)
        X_src_tcap = tcap.transform_source(X_src_raw)
        X_te_tcap = tcap.transform_target(X_te_raw)
        X_val_tcap = tcap.transform_target(X_wp_val_raw)
        
        base_methods = {
            'WP': (X_wp, y_wp, X_te_wp, X_wp_val, y_wp_val),
            'Pooled CP': (X_src, y_src, X_te_src, X_wp_val, y_wp_val),
            'Source + Target-Train': (X_union, y_union, X_te_u, X_wp_val, y_wp_val),
            'CORAL': (X_src_coral, y_src, X_te_wp, X_wp_val, y_wp_val),
            'TCA with standardization': (X_src_tcap, y_src, X_te_tcap, X_val_tcap, y_wp_val)
        }
        
        for treatment in TREATMENTS:
            cw = 'balanced' if treatment == 'balanced' else None
            
            # Logistic Regression
            for m_name, (X_tr, y_tr, X_eval, X_v, y_v) in base_methods.items():
                if treatment == 'smote':
                    n_min = int(np.sum(y_tr == 1))
                    if n_min > 5:
                        sm = SMOTE(random_state=42, k_neighbors=min(5, n_min - 1))
                        X_tr_fit, y_tr_fit = sm.fit_resample(X_tr, y_tr)
                    else:
                        X_tr_fit, y_tr_fit = X_tr, y_tr
                else:
                    X_tr_fit, y_tr_fit = X_tr, y_tr
                    
                clf = LogisticRegression(class_weight=cw, max_iter=1000, random_state=42)
                clf.fit(X_tr_fit, y_tr_fit)
                
                p_val = clf.predict_proba(X_v)[:, 1]
                opt_th = optimize_threshold(y_v, p_val)
                p_te = clf.predict_proba(X_eval)[:, 1]
                
                m_def = compute_all_metrics(y_te, p_te, threshold=0.5)
                m_opt = compute_all_metrics(y_te, p_te, threshold=opt_th)
                
                records.append({
                    'project': p_name,
                    'treatment': treatment,
                    'model': 'lr',
                    'method': m_name,
                    'seed': 42,
                    'base_rate': base_rate,
                    'n_train': len(X_tr_fit),
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
                
            # Random Forest (10 Seeds)
            for m_name, (X_tr, y_tr, X_eval, X_v, y_v) in base_methods.items():
                for s in range(42, 52):
                    if treatment == 'smote':
                        n_min = int(np.sum(y_tr == 1))
                        if n_min > 5:
                            sm = SMOTE(random_state=s, k_neighbors=min(5, n_min - 1))
                            X_tr_fit, y_tr_fit = sm.fit_resample(X_tr, y_tr)
                        else:
                            X_tr_fit, y_tr_fit = X_tr, y_tr
                    else:
                        X_tr_fit, y_tr_fit = X_tr, y_tr
                        
                    clf = RandomForestClassifier(n_estimators=100, class_weight=cw, random_state=s, n_jobs=-1)
                    clf.fit(X_tr_fit, y_tr_fit)
                    
                    p_val = clf.predict_proba(X_v)[:, 1]
                    opt_th = optimize_threshold(y_v, p_val)
                    p_te = clf.predict_proba(X_eval)[:, 1]
                    
                    m_def = compute_all_metrics(y_te, p_te, threshold=0.5)
                    m_opt = compute_all_metrics(y_te, p_te, threshold=opt_th)
                    
                    records.append({
                        'project': p_name,
                        'treatment': treatment,
                        'model': 'rf',
                        'method': m_name,
                        'seed': s,
                        'base_rate': base_rate,
                        'n_train': len(X_tr_fit),
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
        print(f"Finished {p_name} in {time.time()-t0_proj:.1f} s")
        
    df_res = pd.DataFrame(records)
    df_res.to_csv('results/sensitivity_imbalance.csv', index=False)
    print("\nSaved all imbalance sensitivity results to results/sensitivity_imbalance.csv")
    return df_res

if __name__ == '__main__':
    run_imbalance_sensitivity()
