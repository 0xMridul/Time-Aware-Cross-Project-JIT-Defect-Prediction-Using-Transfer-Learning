import os
import sys
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))
"""
run_primary_analysis.py - Execution engine for the locked primary analysis:
Design B, 0-month gap, 12 features, class weights, 10 targets, all methods.
"""

import os
import time
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, roc_auc_score

from data_loader import load_and_preprocess_total, TARGET_PROJECTS_10, KAMEI_12_FEATURES
from transfer import CORALAdapter, LinearTCA, TCAWithStandardization, tune_tca_hyperparameters
from metrics import compute_all_metrics, optimize_threshold, bootstrap_ci, paired_wilcoxon_test, holm_bonferroni_correction, cliffs_delta

T_CUTOFF_B = 1451606400  # 2016-01-01 00:00:00 UTC
T_END_B = 1514764799     # 2017-12-31 23:59:59 UTC

def run_pilot_camel():
    print("=================================================================")
    print("                 RE-RUNNING PILOT: APACHE/CAMEL                  ")
    print("=================================================================")
    df = load_and_preprocess_total()
    target_project = 'apache/camel'
    
    p_df = df[df['project'] == target_project].sort_values('author_date')
    target_train = p_df[p_df['author_date'] < T_CUTOFF_B].copy()
    target_test = p_df[(p_df['author_date'] >= T_CUTOFF_B) & (p_df['author_date'] <= T_END_B)].copy()
    source_train = df[(df['project'] != target_project) & (df['author_date'] < T_CUTOFF_B)].copy()
    
    # Feature extraction with log1p
    X_wp_raw = target_train[KAMEI_12_FEATURES].values
    y_wp = target_train['buggy'].values
    X_src_raw = source_train[KAMEI_12_FEATURES].values
    y_src = source_train['buggy'].values
    X_te_raw = target_test[KAMEI_12_FEATURES].values
    y_te = target_test['buggy'].values
    
    # Combined Source + Target-Train
    X_union_raw = np.vstack([X_src_raw, X_wp_raw])
    y_union = np.concatenate([y_src, y_wp])
    
    X_wp = np.log1p(np.maximum(0, X_wp_raw))
    X_src = np.log1p(np.maximum(0, X_src_raw))
    X_te = np.log1p(np.maximum(0, X_te_raw))
    X_union = np.log1p(np.maximum(0, X_union_raw))
    
    # Scalers
    scaler_wp = StandardScaler().fit(X_wp)
    X_wp_s = scaler_wp.transform(X_wp)
    X_te_s_wp = scaler_wp.transform(X_te)
    
    scaler_src = StandardScaler().fit(X_src)
    X_src_s = scaler_src.transform(X_src)
    X_te_s_src = scaler_src.transform(X_te)
    
    scaler_union = StandardScaler().fit(X_union)
    X_union_s = scaler_union.transform(X_union)
    X_te_s_union = scaler_union.transform(X_te)
    
    # CORAL
    coral = CORALAdapter()
    X_src_coral = coral.fit_transform(X_src_s, X_wp_s)
    
    print(f"Target Test Instances: {len(y_te)} (Buggy: {np.sum(y_te)}, Base Rate: {np.mean(y_te):.4f})")
    print(f"Target Train Instances: {len(y_wp)} (Buggy: {np.sum(y_wp)}, Base Rate: {np.mean(y_wp):.4f})")
    print(f"Source Train Instances: {len(y_src)} (Buggy: {np.sum(y_src)}, Base Rate: {np.mean(y_src):.4f})")
    
    # Validation split inside WP training for threshold tuning (last 20%)
    n_wp = len(X_wp)
    v_idx = int(n_wp * 0.8)
    X_wp_sub, y_wp_sub = X_wp[:v_idx], y_wp[:v_idx]
    X_wp_val, y_wp_val = X_wp[v_idx:], y_wp[v_idx:]
    s_sub = StandardScaler().fit(X_wp_sub)
    X_wp_sub_s = s_sub.transform(X_wp_sub)
    X_wp_val_s = s_sub.transform(X_wp_val)
    
    pilot_results = []
    
    # LR Models
    lr_configs = [
        ('WP', X_wp_s, y_wp, X_te_s_wp, X_wp_val_s, y_wp_val),
        ('Pooled CP', X_src_s, y_src, X_te_s_src, X_wp_val_s, y_wp_val),
        ('Source + Target-Train', X_union_s, y_union, X_te_s_union, X_wp_val_s, y_wp_val),
        ('CORAL', X_src_coral, y_src, X_te_s_wp, X_wp_val_s, y_wp_val),
    ]
    
    print("\n--- Logistic Regression Pilot ---")
    for name, X_tr, y_tr, X_eval, X_v, y_v in lr_configs:
        clf = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42, solver='lbfgs')
        clf.fit(X_tr, y_tr)
        
        # Threshold tuned on time-ordered validation split
        p_val = clf.predict_proba(X_v)[:, 1]
        opt_th = optimize_threshold(y_v, p_val)
        
        p_te = clf.predict_proba(X_eval)[:, 1]
        m_def = compute_all_metrics(y_te, p_te, threshold=0.5)
        m_opt = compute_all_metrics(y_te, p_te, threshold=opt_th)
        ci = bootstrap_ci(y_te, p_te, n_bootstraps=1000, seed=42)
        
        pilot_results.append({
            'Model': 'LR', 'Method': name,
            'AUC': m_def['auc'], 'AUC_CI': f"[{ci['auc_ci_lower']:.4f}, {ci['auc_ci_upper']:.4f}]",
            'PR_AUC': m_def['pr_auc'], 'PR_AUC_CI': f"[{ci['prauc_ci_lower']:.4f}, {ci['prauc_ci_upper']:.4f}]",
            'Th_Chosen': opt_th,
            'F1_0.5': m_def['f1'], 'F1_Youden': m_opt['f1'],
            'MCC_0.5': m_def['mcc'], 'MCC_Youden': m_opt['mcc']
        })
        print(f"LR {name:20s} | AUC: {m_def['auc']:.4f} {pilot_results[-1]['AUC_CI']} | PR-AUC: {m_def['pr_auc']:.4f} {pilot_results[-1]['PR_AUC_CI']} | Th_Chosen: {opt_th:.4f} | F1@0.5: {m_def['f1']:.4f} | F1@Y: {m_opt['f1']:.4f} | MCC@0.5: {m_def['mcc']:.4f} | MCC@Y: {m_opt['mcc']:.4f}")
        
    print("\n--- Random Forest Pilot (10 Seeds) ---")
    for name, X_tr, y_tr, X_eval, X_v, y_v in lr_configs:
        aucs, praucs, f1_defs, f1_opts, mcc_defs, mcc_opts, ths = [], [], [], [], [], [], []
        ci_lowers_auc, ci_uppers_auc = [], []
        ci_lowers_prauc, ci_uppers_prauc = [], []
        
        for s in range(42, 52):
            clf = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=s, n_jobs=-1)
            clf.fit(X_tr, y_tr)
            
            p_val = clf.predict_proba(X_v)[:, 1]
            opt_th = optimize_threshold(y_v, p_val)
            ths.append(opt_th)
            
            p_te = clf.predict_proba(X_eval)[:, 1]
            m_def = compute_all_metrics(y_te, p_te, threshold=0.5)
            m_opt = compute_all_metrics(y_te, p_te, threshold=opt_th)
            
            aucs.append(m_def['auc'])
            praucs.append(m_def['pr_auc'])
            f1_defs.append(m_def['f1'])
            f1_opts.append(m_opt['f1'])
            mcc_defs.append(m_def['mcc'])
            mcc_opts.append(m_opt['mcc'])
            
            ci = bootstrap_ci(y_te, p_te, n_bootstraps=1000, seed=s)
            ci_lowers_auc.append(ci['auc_ci_lower'])
            ci_uppers_auc.append(ci['auc_ci_upper'])
            ci_lowers_prauc.append(ci['prauc_ci_lower'])
            ci_uppers_prauc.append(ci['prauc_ci_upper'])
            
        pilot_results.append({
            'Model': 'RF', 'Method': name,
            'AUC': np.mean(aucs), 'AUC_CI': f"[{np.mean(ci_lowers_auc):.4f}, {np.mean(ci_uppers_auc):.4f}]",
            'PR_AUC': np.mean(praucs), 'PR_AUC_CI': f"[{np.mean(ci_lowers_prauc):.4f}, {np.mean(ci_uppers_prauc):.4f}]",
            'Th_Chosen': np.mean(ths),
            'F1_0.5': np.mean(f1_defs), 'F1_Youden': np.mean(f1_opts),
            'MCC_0.5': np.mean(mcc_defs), 'MCC_Youden': np.mean(mcc_opts)
        })
        print(f"RF {name:20s} | AUC: {np.mean(aucs):.4f} +/- {np.std(aucs):.4f} {pilot_results[-1]['AUC_CI']} | PR-AUC: {np.mean(praucs):.4f} +/- {np.std(praucs):.4f} {pilot_results[-1]['PR_AUC_CI']} | Th_Chosen: {np.mean(ths):.4f} | F1@0.5: {np.mean(f1_defs):.4f} | F1@Y: {np.mean(f1_opts):.4f} | MCC@0.5: {np.mean(mcc_defs):.4f} | MCC@Y: {np.mean(mcc_opts):.4f}")

    # Leakage check with 100 label shuffles
    print("\n--- Leakage Sanity Check (100 Label Shuffles) ---")
    shuff_aucs = []
    for s in range(100):
        rng = np.random.RandomState(s)
        y_shuff = rng.permutation(y_wp)
        clf = LogisticRegression(class_weight='balanced', max_iter=500, random_state=s, solver='lbfgs')
        clf.fit(X_wp_s, y_shuff)
        p = clf.predict_proba(X_te_s_wp)[:, 1]
        shuff_aucs.append(roc_auc_score(y_te, p))
    print(f"100 Label Shuffles LR AUC: Mean={np.mean(shuff_aucs):.4f}, Std={np.std(shuff_aucs):.4f}, Min={np.min(shuff_aucs):.4f}, Max={np.max(shuff_aucs):.4f}")
    
    return pd.DataFrame(pilot_results)

def run_primary_analysis_all_targets():
    print("\n=================================================================")
    print("      RUNNING FULL PRIMARY ANALYSIS ACROSS ALL 10 TARGETS        ")
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
        
        # 1. Validation split inside WP training for threshold & TCA tuning (last 20%)
        n_wp = len(X_wp)
        v_idx = max(int(n_wp * 0.8), n_wp - 1000)
        X_wp_sub_raw, y_wp_sub = X_wp_raw[:v_idx], y_wp[:v_idx]
        X_wp_val_raw, y_wp_val = X_wp_raw[v_idx:], y_wp[v_idx:]
        
        s_sub = StandardScaler().fit(X_wp_sub_raw)
        X_wp_sub = s_sub.transform(X_wp_sub_raw)
        X_wp_val = s_sub.transform(X_wp_val_raw)
        
        # 2. Transfer representations
        # A) CORAL
        coral = CORALAdapter()
        X_src_coral = coral.fit_transform(X_src, X_wp)
        
        # B) LinearTCA (tuned k and mu)
        k_tca, mu_tca, val_auc_tca = tune_tca_hyperparameters(
            X_src, y_src, X_wp_sub, y_wp_sub, X_wp_val, y_wp_val, with_standardization=False
        )
        tca = LinearTCA(n_components=k_tca, mu=mu_tca)
        tca.fit(X_src, X_wp)
        X_src_tca = tca.transform(X_src)
        X_te_tca = tca.transform(X_te_wp)
        X_val_tca = tca.transform(X_wp_val)
        
        # C) TCA with standardization (tuned k and mu)
        k_tcap, mu_tcap, val_auc_tcap = tune_tca_hyperparameters(
            X_src_raw, y_src, X_wp_sub_raw, y_wp_sub, X_wp_val_raw, y_wp_val, with_standardization=True
        )
        tcap = TCAWithStandardization(n_components=k_tcap, mu=mu_tcap)
        tcap.fit(X_src_raw, X_wp_raw)
        X_src_tcap = tcap.transform_source(X_src_raw)
        X_te_tcap = tcap.transform_target(X_te_raw)
        X_val_tcap = tcap.transform_target(X_wp_val_raw)
        
        # Define all methods
        methods = {
            'WP': (X_wp, y_wp, X_te_wp, X_wp_val, y_wp_val),
            'Pooled CP': (X_src, y_src, X_te_src, X_wp_val, y_wp_val),
            'Source + Target-Train': (X_union, y_union, X_te_u, X_wp_val, y_wp_val),
            'CORAL': (X_src_coral, y_src, X_te_wp, X_wp_val, y_wp_val),
            'TCA': (X_src_tca, y_src, X_te_tca, X_val_tca, y_wp_val),
            'TCA with standardization': (X_src_tcap, y_src, X_te_tcap, X_val_tcap, y_wp_val)
        }
        
        # 1. Logistic Regression (Deterministic)
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
                'seed': 42,
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
                ci = bootstrap_ci(y_te, p_te, n_bootstraps=1000, seed=s)
                
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
        
    df_results = pd.DataFrame(records)
    df_results.to_csv('results/primary_analysis.csv', index=False)
    print("\nSaved all primary results to results/primary_analysis.csv")
    return df_results

if __name__ == '__main__':
    run_pilot_camel()
    run_primary_analysis_all_targets()
