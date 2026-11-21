import os
import sys
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))
"""
run_sensitivity_70pct.py - Sensitivity Analysis 4: 70% Per-Project Chronological Split.
Evaluates 70% train / 30% test chronological splits per target project under primary 12-feature setup.
Saves results to results/sensitivity_70pct.csv.
"""

import os
import time
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

from data_loader import load_and_preprocess_total, TARGET_PROJECTS_10, KAMEI_12_FEATURES, get_protocol_b_and_c_splits
from transfer import CORALAdapter, TCAWithStandardization, tune_tca_hyperparameters
from metrics import compute_all_metrics, optimize_threshold

def run_70pct_analysis():
    print("=================================================================")
    print("   RUNNING SENSITIVITY ANALYSIS 4: 70% CHRONOLOGICAL SPLITS      ")
    print("=================================================================")
    df = load_and_preprocess_total()
    os.makedirs('results', exist_ok=True)
    
    records = []
    
    for target in TARGET_PROJECTS_10:
        t0_proj = time.time()
        p_name = target.replace('apache/', '')
        print(f"\n>>> Processing Target: {p_name} (70% Split) ...")
        
        splits = get_protocol_b_and_c_splits(df, target, KAMEI_12_FEATURES, split_ratio=0.7, label_delay_gap_months=0)
        
        n_test = splits['n_target_test']
        n_test_bug = int(np.sum(splits['y_target_test'] == 1))
        base_rate = float(splits['target_test_defect_rate'])
        
        X_wp_raw = np.log1p(np.maximum(0, splits['X_target_train']))
        y_wp = splits['y_target_train']
        
        X_src_raw = np.log1p(np.maximum(0, splits['X_source_train']))
        y_src = splits['y_source_train']
        
        X_te_raw = np.log1p(np.maximum(0, splits['X_target_test']))
        y_te = splits['y_target_test']
        
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
        print(f"Finished {p_name} in {time.time()-t0_proj:.1f} s")
        
    df_res = pd.DataFrame(records)
    df_res.to_csv('results/sensitivity_70pct.csv', index=False)
    print("\nSaved all 70% split sensitivity results to results/sensitivity_70pct.csv")
    return df_res

if __name__ == '__main__':
    run_70pct_analysis()
