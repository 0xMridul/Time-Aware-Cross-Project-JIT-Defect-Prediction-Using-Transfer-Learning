import os
import sys
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))
"""
run_sensitivity_13_features.py - Sensitivity Analysis 1: 13 features (with 'fix' purpose metric).
Evaluates Design B, 0-month gap, class weights, 10 targets across all methods.
Saves results to results/sensitivity_13_features.csv.
"""

import os
import time
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

from data_loader import load_and_preprocess_total, TARGET_PROJECTS_10, KAMEI_13_FEATURES, assert_canonical_test_split
from transfer import CORALAdapter, TCAWithStandardization, tune_tca_hyperparameters
from metrics import compute_all_metrics, optimize_threshold, bootstrap_ci

T_CUTOFF_B = 1451606400  # 2016-01-01 00:00:00 UTC
T_END_B = 1514764799     # 2017-12-31 23:59:59 UTC

def run_13_features_analysis():
    print("=================================================================")
    print("   RUNNING SENSITIVITY ANALYSIS 1: 13 FEATURES (WITH 'FIX')      ")
    print("=================================================================")
    df = load_and_preprocess_total()
    os.makedirs('results', exist_ok=True)
    
    records = []
    
    for target in TARGET_PROJECTS_10:
        t0_proj = time.time()
        p_name = target.replace('apache/', '')
        print(f"\n>>> Processing Target: {p_name} (13 Features) ...")
        
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
        X_wp_raw = np.log1p(np.maximum(0, target_train[KAMEI_13_FEATURES].values))
        y_wp = target_train['buggy'].values
        
        X_src_raw = np.log1p(np.maximum(0, source_train[KAMEI_13_FEATURES].values))
        y_src = source_train['buggy'].values
        
        X_te_raw = np.log1p(np.maximum(0, target_test[KAMEI_13_FEATURES].values))
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
        
        # Validation split inside WP training for threshold & TCA tuning (last 20% or max 1000)
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
        
        # 2. TCA with standardization (extended grid k in [4, 6, 8, 10, 12], mu in [0.001, 0.01, 0.1, 1, 10])
        # Tune on sub-validation split
        best_k, best_mu, best_auc = 10, 0.001, -1.0
        for k in [4, 6, 8, 10, 12]:
            for mu in [0.001, 0.01, 0.1, 1.0, 10.0]:
                tcap_tune = TCAWithStandardization(n_components=k, mu=mu)
                tcap_tune.fit(X_src_raw, X_wp_sub_raw)
                Xs_proj = tcap_tune.transform_source(X_src_raw)
                Xv_proj = tcap_tune.transform_target(X_wp_val_raw)
                clf_t = LogisticRegression(class_weight='balanced', max_iter=500, random_state=42)
                clf_t.fit(Xs_proj, y_src)
                from sklearn.metrics import roc_auc_score
                auc_t = roc_auc_score(y_wp_val, clf_t.predict_proba(Xv_proj)[:, 1])
                if auc_t > best_auc:
                    best_auc = auc_t
                    best_k, best_mu = k, mu
                    
        tcap = TCAWithStandardization(n_components=best_k, mu=best_mu)
        tcap.fit(X_src_raw, X_wp_raw)
        X_src_tcap = tcap.transform_source(X_src_raw)
        X_te_tcap = tcap.transform_target(X_te_raw)
        X_val_tcap = tcap.transform_target(X_wp_val_raw)
        
        # Define methods
        methods = {
            'WP': (X_wp, y_wp, X_te_wp, X_wp_val, y_wp_val),
            'Pooled CP': (X_src, y_src, X_te_src, X_wp_val, y_wp_val),
            'Source + Target-Train': (X_union, y_union, X_te_u, X_wp_val, y_wp_val),
            'CORAL': (X_src_coral, y_src, X_te_wp, X_wp_val, y_wp_val),
            'TCA with standardization': (X_src_tcap, y_src, X_te_tcap, X_val_tcap, y_wp_val)
        }
        
        # Logistic Regression (Deterministic)
        for m_name, (X_tr, y_tr, X_eval, X_v, y_v) in methods.items():
            clf = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42, solver='lbfgs')
            clf.fit(X_tr, y_tr)
            
            p_val = clf.predict_proba(X_v)[:, 1]
            opt_th = optimize_threshold(y_v, p_val)
            
            p_te = clf.predict_proba(X_eval)[:, 1]
            m_def = compute_all_metrics(y_te, p_te, threshold=0.5)
            m_opt = compute_all_metrics(y_te, p_te, threshold=opt_th)
            ci = bootstrap_ci(y_te, p_te, n_bootstraps=1000, seed=42)
            
            records.append({
                'project': p_name,
                'model': 'lr',
                'method': m_name,
                'features': '13_features',
                'seed': 42,
                'tca_k': best_k if 'TCA' in m_name else np.nan,
                'tca_mu': best_mu if 'TCA' in m_name else np.nan,
                'base_rate': base_rate,
                'n_train': len(X_tr),
                'n_test': n_test,
                'n_test_bug': n_test_bug,
                'auc': m_def['auc'],
                'auc_ci_lower': ci['auc_ci_lower'],
                'auc_ci_upper': ci['auc_ci_upper'],
                'pr_auc': m_def['pr_auc'],
                'prauc_ci_lower': ci['prauc_ci_lower'],
                'prauc_ci_upper': ci['prauc_ci_upper'],
                'th_chosen': opt_th,
                'f1_0.5': m_def['f1'],
                'f1_youden': m_opt['f1'],
                'mcc_0.5': m_def['mcc'],
                'mcc_youden': m_opt['mcc'],
                'precision': m_opt['precision'],
                'recall': m_opt['recall']
            })
            
        # Random Forest (10 Seeds)
        for m_name, (X_tr, y_tr, X_eval, X_v, y_v) in methods.items():
            for s in range(42, 52):
                clf = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=s, n_jobs=-1)
                clf.fit(X_tr, y_tr)
                
                p_val = clf.predict_proba(X_v)[:, 1]
                opt_th = optimize_threshold(y_v, p_val)
                
                p_te = clf.predict_proba(X_eval)[:, 1]
                m_def = compute_all_metrics(y_te, p_te, threshold=0.5)
                m_opt = compute_all_metrics(y_te, p_te, threshold=opt_th)
                ci = bootstrap_ci(y_te, p_te, n_bootstraps=1000, seed=s)
                
                records.append({
                    'project': p_name,
                    'model': 'rf',
                    'method': m_name,
                    'features': '13_features',
                    'seed': s,
                    'tca_k': best_k if 'TCA' in m_name else np.nan,
                    'tca_mu': best_mu if 'TCA' in m_name else np.nan,
                    'base_rate': base_rate,
                    'n_train': len(X_tr),
                    'n_test': n_test,
                    'n_test_bug': n_test_bug,
                    'auc': m_def['auc'],
                    'auc_ci_lower': ci['auc_ci_lower'],
                    'auc_ci_upper': ci['auc_ci_upper'],
                    'pr_auc': m_def['pr_auc'],
                    'prauc_ci_lower': ci['prauc_ci_lower'],
                    'prauc_ci_upper': ci['prauc_ci_upper'],
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
    df_res.to_csv('results/sensitivity_13_features.csv', index=False)
    print("\nSaved all 13-feature sensitivity results to results/sensitivity_13_features.csv")
    return df_res

if __name__ == '__main__':
    run_13_features_analysis()
