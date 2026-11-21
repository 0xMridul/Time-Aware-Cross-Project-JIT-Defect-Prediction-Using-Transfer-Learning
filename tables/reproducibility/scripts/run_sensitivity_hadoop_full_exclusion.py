import os
import sys
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))
"""
run_sensitivity_hadoop_full_exclusion.py - Evaluates the 9 target projects
when Hadoop is excluded from BOTH the target cohort AND the cross-project source pool.
"""

import os
import time
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score
from scipy.stats import wilcoxon

from data_loader import load_and_preprocess_total, TARGET_PROJECTS_10, KAMEI_12_FEATURES
from transfer import CORALAdapter, TCAWithStandardization, tune_tca_hyperparameters
from metrics import compute_all_metrics, optimize_threshold

T_CUTOFF_B = 1451606400  # 2016-01-01 00:00:00 UTC
T_END_B = 1514764799     # 2017-12-31 23:59:59 UTC

def run_hadoop_full_exclusion():
    print("=================================================================")
    print("   EVALUATING FULL HADOOP EXCLUSION (FROM TARGETS & SOURCE POOL) ")
    print("=================================================================")
    df = load_and_preprocess_total()
    
    # 9 targets (all targets except hadoop)
    targets_9 = [t for t in TARGET_PROJECTS_10 if t != 'apache/hadoop']
    
    # Source pool is strictly filtered to exclude apache/hadoop everywhere
    df_no_hadoop = df[df['project'] != 'apache/hadoop'].copy()
    
    records = []
    
    for target in targets_9:
        p_name = target.replace('apache/', '')
        t0_proj = time.time()
        
        p_df = df_no_hadoop[df_no_hadoop['project'] == target].sort_values('author_date')
        target_train = p_df[p_df['author_date'] < T_CUTOFF_B].copy()
        target_test = p_df[(p_df['author_date'] >= T_CUTOFF_B) & (p_df['author_date'] <= T_END_B)].copy()
        
        # Source pool excludes target AND excludes hadoop
        source_train = df_no_hadoop[(df_no_hadoop['project'] != target) & (df_no_hadoop['author_date'] < T_CUTOFF_B)].copy()
        
        n_test = len(target_test)
        n_test_bug = int(target_test['buggy'].sum())
        base_rate = float(n_test_bug / n_test) if n_test > 0 else 0.0
        
        X_wp_raw = np.log1p(np.maximum(0, target_train[KAMEI_12_FEATURES].values))
        y_wp = target_train['buggy'].values
        
        X_src_raw = np.log1p(np.maximum(0, source_train[KAMEI_12_FEATURES].values))
        y_src = source_train['buggy'].values
        
        X_te_raw = np.log1p(np.maximum(0, target_test[KAMEI_12_FEATURES].values))
        y_te = target_test['buggy'].values
        
        # Scalers
        s_wp = StandardScaler().fit(X_wp_raw)
        X_wp = s_wp.transform(X_wp_raw)
        X_te_wp = s_wp.transform(X_te_raw)
        
        s_src = StandardScaler().fit(X_src_raw)
        X_src = s_src.transform(X_src_raw)
        X_te_src = s_src.transform(X_te_raw)
        
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
            'CORAL': (X_src_coral, y_src, X_te_wp, X_wp_val, y_wp_val),
            'TCA with standardization': (X_src_tcap, y_src, X_te_tcap, X_val_tcap, y_wp_val)
        }
        
        # Logistic Regression
        for m_name, (X_tr, y_tr, X_eval, X_v, y_v) in methods.items():
            clf = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
            clf.fit(X_tr, y_tr)
            p_val = clf.predict_proba(X_v)[:, 1]
            opt_th = optimize_threshold(y_v, p_val)
            p_te = clf.predict_proba(X_eval)[:, 1]
            m_def = compute_all_metrics(y_te, p_te, threshold=0.5)
            m_opt = compute_all_metrics(y_te, p_te, threshold=opt_th)
            
            records.append({
                'project': p_name, 'model': 'lr', 'method': m_name, 'seed': 42,
                'n_train': len(X_tr), 'n_test': n_test, 'n_test_bug': n_test_bug,
                'auc': m_def['auc'], 'pr_auc': m_def['pr_auc'],
                'f1_youden': m_opt['f1'], 'mcc_youden': m_opt['mcc']
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
                
                records.append({
                    'project': p_name, 'model': 'rf', 'method': m_name, 'seed': s,
                    'n_train': len(X_tr), 'n_test': n_test, 'n_test_bug': n_test_bug,
                    'auc': m_def['auc'], 'pr_auc': m_def['pr_auc'],
                    'f1_youden': m_opt['f1'], 'mcc_youden': m_opt['mcc']
                })
        print(f"Finished {p_name} in {time.time()-t0_proj:.1f} s (Source pool: {len(X_src_raw)} commits)")
        
    df_res = pd.DataFrame(records)
    df_res.to_csv('results/sensitivity_hadoop_full_exclusion.csv', index=False)
    print("\nSaved full Hadoop exclusion results to results/sensitivity_hadoop_full_exclusion.csv")
    
    # Statistical Summary
    print("\n=================================================================")
    print("      STATISTICAL SUMMARY: HADOOP EXCLUDED FROM TARGETS & SOURCE ")
    print("=================================================================")
    agg = df_res.groupby(['project', 'model', 'method'])[['auc', 'pr_auc']].mean().reset_index()
    
    for model in ['lr', 'rf']:
        print(f"\n--- Model: {model.upper()} (9 Projects, No Hadoop Anywhere) ---")
        piv = agg[agg['model'] == model].pivot(index='project', columns='method', values='auc')
        
        for comp_name, m1, m2 in [
            ('Pooled vs WP', 'Pooled CP', 'WP'),
            ('CORAL vs WP', 'CORAL', 'WP'),
            ('TCA vs WP', 'TCA with standardization', 'WP'),
            ('CORAL vs Pooled', 'CORAL', 'Pooled CP'),
            ('TCA vs Pooled', 'TCA with standardization', 'Pooled CP')
        ]:
            d = piv[m1] - piv[m2]
            w = np.sum(d > 0)
            l = np.sum(d < 0)
            try:
                _, p = wilcoxon(d, alternative='two-sided')
            except Exception:
                p = 1.0
            print(f"{comp_name:25s} | Win/Loss: {w}W / {l}L | Mean Diff: {d.mean():+.4f} | Nominal Wilcoxon p: {p:.4f}")

if __name__ == '__main__':
    run_hadoop_full_exclusion()
