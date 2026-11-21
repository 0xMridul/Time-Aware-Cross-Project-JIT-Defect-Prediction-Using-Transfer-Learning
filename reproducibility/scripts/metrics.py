import os
import sys
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))
"""
metrics.py - Evaluation metrics, bootstrap confidence intervals, and statistical testing for defect prediction.
"""

import numpy as np
from sklearn.metrics import (
    roc_auc_score, average_precision_score, matthews_corrcoef,
    f1_score, precision_score, recall_score, roc_curve
)
from scipy.stats import wilcoxon

def compute_all_metrics(y_true, y_prob, threshold=0.5):
    """
    Computes all standard defect prediction metrics.
    Guarantees no crashes on edge cases (e.g., zero positive predictions).
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    y_pred = (y_prob >= threshold).astype(int)
    
    # Base-rate insensitive ranking
    try:
        auc = float(roc_auc_score(y_true, y_prob))
    except ValueError:
        auc = np.nan
        
    # Base-rate sensitive metrics
    try:
        pr_auc = float(average_precision_score(y_true, y_prob))
    except ValueError:
        pr_auc = np.nan
        
    mcc = float(matthews_corrcoef(y_true, y_pred))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    
    return {
        'auc': auc,
        'pr_auc': pr_auc,
        'mcc': mcc,
        'f1': f1,
        'precision': prec,
        'recall': rec,
        'threshold': float(threshold)
    }

def optimize_threshold(y_true, y_prob):
    """
    Finds optimal decision threshold maximizing Youden's J statistic (Sensitivity + Specificity - 1 = TPR - FPR)
    using exact ROC curve thresholds.
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    fpr, tpr, ths = roc_curve(y_true, y_prob)
    if len(ths) > 1:
        # Drop the first threshold which is ths[0] = max(y_prob) + 1
        j_scores = tpr[1:] - fpr[1:]
        best_idx = int(np.argmax(j_scores) + 1)
        best_th = float(ths[best_idx])
        return np.clip(best_th, 0.01, 0.99)
    return 0.5

def bootstrap_ci(y_true, y_prob, n_bootstraps=1000, seed=42):
    """
    Computes 95% bootstrap confidence intervals for ROC-AUC and PR-AUC.
    Resamples instances with replacement.
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    N = len(y_true)
    rng = np.random.RandomState(seed)
    
    aucs = []
    praucs = []
    
    for _ in range(n_bootstraps):
        idx = rng.randint(0, N, N)
        y_b = y_true[idx]
        p_b = y_prob[idx]
        # Require at least one positive and one negative sample
        if len(np.unique(y_b)) < 2:
            continue
        try:
            aucs.append(roc_auc_score(y_b, p_b))
        except ValueError:
            pass
        try:
            praucs.append(average_precision_score(y_b, p_b))
        except ValueError:
            pass
            
    auc_ci = (float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))) if len(aucs) > 0 else (np.nan, np.nan)
    prauc_ci = (float(np.percentile(praucs, 2.5)), float(np.percentile(praucs, 97.5))) if len(praucs) > 0 else (np.nan, np.nan)
    
    return {
        'auc_ci_lower': auc_ci[0],
        'auc_ci_upper': auc_ci[1],
        'prauc_ci_lower': prauc_ci[0],
        'prauc_ci_upper': prauc_ci[1]
    }

def cliffs_delta(lst1, lst2):
    """
    Computes Cliff's delta non-parametric effect size.
    """
    x1 = np.asarray(lst1)
    x2 = np.asarray(lst2)
    n1 = len(x1)
    n2 = len(x2)
    if n1 == 0 or n2 == 0:
        return 0.0, 'Negligible'
    
    greater = 0
    less = 0
    for v in x1:
        greater += np.sum(v > x2)
        less += np.sum(v < x2)
    
    delta = float((greater - less) / (n1 * n2))
    abs_d = abs(delta)
    if abs_d < 0.147:
        magnitude = 'Negligible'
    elif abs_d < 0.330:
        magnitude = 'Small'
    elif abs_d < 0.474:
        magnitude = 'Medium'
    else:
        magnitude = 'Large'
    return delta, magnitude

def paired_wilcoxon_test(x, y):
    """
    Paired two-sided Wilcoxon signed-rank test.
    """
    diffs = np.asarray(x) - np.asarray(y)
    if np.all(diffs == 0):
        return 0.0, 1.0
    try:
        stat, pval = wilcoxon(x, y, zero_method='wilcox', alternative='two-sided')
        return float(stat), float(pval)
    except Exception:
        return 0.0, 1.0

def holm_bonferroni_correction(p_values):
    """
    Adjusts p-values using the Holm-Bonferroni step-down method.
    """
    p_vals = np.asarray(p_values)
    n = len(p_vals)
    sorted_indices = np.argsort(p_vals)
    sorted_p = p_vals[sorted_indices]
    adjusted_p = np.zeros(n)
    
    cum_max = 0.0
    for i in range(n):
        val = sorted_p[i] * (n - i)
        val = min(1.0, val)
        cum_max = max(cum_max, val)
        adjusted_p[sorted_indices[i]] = cum_max
        
    return adjusted_p
