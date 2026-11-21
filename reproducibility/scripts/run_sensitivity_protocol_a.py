import os
import sys
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))
"""
run_sensitivity_protocol_a.py - Sensitivity Analysis 5: Protocol A (Official Zenodo Split).
Evaluates the official benchmark split:
- Train: repo_check/dataset/apachejit_train.csv (44,834 balanced commits)
- Test Large: repo_check/dataset/apachejit_test_large.csv (30,111 commits)
- Test Small: repo_check/dataset/apachejit_test_small.csv (7,526 commits)
Evaluates across the 10 target projects under primary 12-feature setup.
Saves results to results/sensitivity_protocol_a.csv.
"""

import os
import time
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

from data_loader import TARGET_PROJECTS_10, KAMEI_12_FEATURES
from transfer import CORALAdapter, TCAWithStandardization, tune_tca_hyperparameters
from metrics import compute_all_metrics, optimize_threshold

def run_protocol_a():
    print("=================================================================")
    print("   RUNNING SENSITIVITY ANALYSIS 5: PROTOCOL A (OFFICIAL SPLIT)   ")
    print("=================================================================")
    df_train = pd.read_csv('repo_check/dataset/apachejit_train.csv')
    df_test_large = pd.read_csv('repo_check/dataset/apachejit_test_large.csv')
    os.makedirs('results', exist_ok=True)
    
    records = []
    
    for target in TARGET_PROJECTS_10:
        t0_proj = time.time()
        p_name = target.replace('apache/', '')
        print(f"\n>>> Processing Target: {p_name} (Protocol A) ...")
        
        target_train = df_train[df_train['project'] == target].copy()
        target_test = df_test_large[df_test_large['project'] == target].copy()
        source_train = df_train[df_train['project'] != target].copy()
        
        n_test = len(target_test)
        n_test_bug = int(target_test['buggy'].sum())
        base_rate = float(n_test_bug / n_test) if n_test > 0 else 0.0
        
        if n_test == 0 or n_test_bug == 0:
            print(f"Skipping {p_name}: no test bugs in official split (test={n_test}, bugs={n_test_bug})")
            continue
            
        n_wp_train = len(target_train)
        n_wp_bug = int(target_train['buggy'].sum())
        
        # Raw features with log1p
        X_src_raw = np.log1p(np.maximum(0, source_train[KAMEI_12_FEATURES].values))
        y_src = source_train['buggy'].values
        
        X_te_raw = np.log1p(np.maximum(0, target_test[KAMEI_12_FEATURES].values))
        y_te = target_test['buggy'].values
        
        s_src = StandardScaler().fit(X_src_raw)
        X_src = s_src.transform(X_src_raw)
        X_te_src = s_src.transform(X_te_raw)
        
        if n_wp_train >= 50 and n_wp_bug >= 5:
            X_wp_raw = np.log1p(np.maximum(0, target_train[KAMEI_12_FEATURES].values))
            y_wp = target_train['buggy'].values
            
            s_wp = StandardScaler().fit(X_wp_raw)
            X_wp = s_wp.transform(X_wp_raw)
            X_te_wp = s_wp.transform(X_te_raw)
            
            X_union_raw = np.vstack([X_src_raw, X_wp_raw])
            y_union = np.concatenate([y_src, y_wp])
            s_u = StandardScaler().fit(X_union_raw)
            X_union = s_u.transform(X_union_raw)
            X_te_u = s_u.transform(X_te_raw)
            
            # Validation split inside WP train
            v_idx = max(int(n_wp_train * 0.8), n_wp_train - 1000)
            X_wp_sub_raw, y_wp_sub = X_wp_raw[:v_idx], y_wp[:v_idx]
            X_wp_val_raw, y_wp_val = X_wp_raw[v_idx:], y_wp[v_idx:]
            
            s_sub = StandardScaler().fit(X_wp_sub_raw)
            X_wp_sub = s_sub.transform(X_wp_sub_raw)
            X_wp_val = s_sub.transform(X_wp_val_raw)
            
            # Domain Adaptation
            coral = CORALAdapter()
            X_src_coral = coral.fit_transform(X_src, X_wp)
            
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
                'Source + Target-Train': (X_union, y_union, X_te_u, X_wp_val, y_wp_val),
                'CORAL': (X_src_coral, y_src, X_te_wp, X_wp_val, y_wp_val),
                'TCA with standardization': (X_src_tcap, y_src, X_te_tcap, X_val_tcap, y_wp_val)
            }
        else:
            methods = {
                'Pooled CP': (X_src, y_src, X_te_src, X_src[:1000], y_src[:1000])
            }
            
        for m_name, (X_tr, y_tr, X_eval, X_v, y_v) in methods.items():
            clf_lr = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
            clf_lr.fit(X_tr, y_tr)
            p_val_lr = clf_lr.predict_proba(X_v)[:, 1]
            opt_th_lr = optimize_threshold(y_v, p_val_lr)
            p_te_lr = clf_lr.predict_proba(X_eval)[:, 1]
            m_def_lr = compute_all_metrics(y_te, p_te_lr, threshold=0.5)
            m_opt_lr = compute_all_metrics(y_te, p_te_lr, threshold=opt_th_lr)
            
            records.append({
                'project': p_name,
                'protocol': 'Protocol A (Official Split)',
                'model': 'lr',
                'method': m_name,
                'seed': 42,
                'base_rate': base_rate,
                'n_train': len(X_tr),
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
                    'project': p_name,
                    'protocol': 'Protocol A (Official Split)',
                    'model': 'rf',
                    'method': m_name,
                    'seed': s,
                    'base_rate': base_rate,
                    'n_train': len(X_tr),
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
        print(f"Finished {p_name} in {time.time()-t0_proj:.1f} s")
        
    df_res = pd.DataFrame(records)
    df_res.to_csv('results/sensitivity_protocol_a.csv', index=False)
    print("\nSaved all Protocol A sensitivity results to results/sensitivity_protocol_a.csv")
    return df_res

if __name__ == '__main__':
    run_protocol_a()
