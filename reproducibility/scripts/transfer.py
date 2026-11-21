import os
import sys
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))
"""
transfer.py - Domain Adaptation algorithms for tabular cross-project JIT defect prediction.
Implements:
1. CORAL (Correlation Alignment, Sun et al. AAAI 2016)
2. LinearTCA (Transfer Component Analysis, Pan et al. IEEE TNN 2011)
3. TCAWithStandardization (TCA+; TCA preceded by independent domain standardization, Nam et al. ICSE 2013)
"""

import numpy as np
from scipy.linalg import fractional_matrix_power, eigh
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

class CORALAdapter:
    """
    Correlation Alignment for Unsupervised Domain Adaptation (Sun et al. AAAI 2016).
    Aligns the second-order statistics (covariance matrices) of source features to target features.
    """
    def __init__(self, reg=1.0):
        self.reg = reg
        self.scaler_s = StandardScaler(with_std=False)  # centering only
        self.scaler_t = StandardScaler(with_std=False)
        self.mean_t_ = None
        self.whiten_ = None
        self.recolor_ = None

    def fit_transform(self, Xs, Xt):
        Xs_c = self.scaler_s.fit_transform(Xs)
        Xt_c = self.scaler_t.fit_transform(Xt)
        self.mean_t_ = self.scaler_t.mean_
        
        d = Xs.shape[1]
        eye = np.eye(d)
        
        Cs = np.cov(Xs_c, rowvar=False) + self.reg * eye
        Ct = np.cov(Xt_c, rowvar=False) + self.reg * eye
        
        self.whiten_ = fractional_matrix_power(Cs, -0.5)
        self.recolor_ = fractional_matrix_power(Ct, 0.5)
        
        Xs_coral = np.real(np.dot(np.dot(Xs_c, self.whiten_), self.recolor_)) + self.mean_t_
        return Xs_coral

class LinearTCA:
    """
    Transfer Component Analysis (TCA) for Tabular Features (Pan et al. IEEE TNN 2011).
    Minimizes Maximum Mean Discrepancy (MMD) between source and target domains
    while preserving data scatter.
    """
    def __init__(self, n_components=6, mu=1.0):
        self.n_components = n_components
        self.mu = mu
        self.W_ = None
        self.mean_all_ = None

    def fit(self, Xs, Xt):
        d = Xs.shape[1]
        dim = min(self.n_components, d)
        
        ms = np.mean(Xs, axis=0, keepdims=True)
        mt = np.mean(Xt, axis=0, keepdims=True)
        diff_m = ms - mt
        
        M = np.dot(diff_m.T, diff_m)
        
        X_all = np.vstack([Xs, Xt])
        self.mean_all_ = np.mean(X_all, axis=0, keepdims=True)
        X_c = X_all - self.mean_all_
        S_total = np.dot(X_c.T, X_c) + self.mu * np.eye(d)
        
        eps_eye = 1e-4 * np.eye(d)
        evals, evecs = eigh(S_total, M + eps_eye)
        
        idx = np.argsort(evals)[::-1][:dim]
        self.W_ = np.real(evecs[:, idx])
        return self

    def transform(self, X):
        return np.dot(X, self.W_)

    def fit_transform(self, Xs, Xt):
        self.fit(Xs, Xt)
        return self.transform(Xs), self.transform(Xt)

class TCAWithStandardization:
    """
    TCA with standardization (Nam et al. ICSE 2013).
    Applies independent domain-specific Z-score standardization followed by LinearTCA.
    """
    def __init__(self, n_components=6, mu=1.0):
        self.n_components = n_components
        self.mu = mu
        self.tca = LinearTCA(n_components=n_components, mu=mu)
        self.scaler_s = StandardScaler()
        self.scaler_t = StandardScaler()

    def fit(self, Xs, Xt):
        Xs_norm = self.scaler_s.fit_transform(Xs)
        Xt_norm = self.scaler_t.fit_transform(Xt)
        self.tca.fit(Xs_norm, Xt_norm)
        return self

    def transform_source(self, Xs):
        Xs_norm = self.scaler_s.transform(Xs)
        return self.tca.transform(Xs_norm)

    def transform_target(self, Xt):
        Xt_norm = self.scaler_t.transform(Xt)
        return self.tca.transform(Xt_norm)

    def fit_transform(self, Xs, Xt):
        self.fit(Xs, Xt)
        return self.transform_source(Xs), self.transform_target(Xt)

# Alias for backward compatibility
TCAPlus = TCAWithStandardization

def tune_tca_hyperparameters(Xs, ys, Xt_tr, yt_tr, Xt_val, yt_val, with_standardization=False):
    """
    Tunes TCA hyperparameters (k and mu) using a time-ordered validation split
    strictly inside the training period.
    """
    k_candidates = [4, 6, 8, 10]
    mu_candidates = [0.1, 1.0, 10.0]
    best_auc = -1.0
    best_params = (6, 1.0)
    
    for k in k_candidates:
        for mu in mu_candidates:
            if with_standardization:
                adapter = TCAWithStandardization(n_components=k, mu=mu)
                adapter.fit(Xs, Xt_tr)
                Xs_proj = adapter.transform_source(Xs)
                Xt_val_proj = adapter.transform_target(Xt_val)
            else:
                adapter = LinearTCA(n_components=k, mu=mu)
                adapter.fit(Xs, Xt_tr)
                Xs_proj = adapter.transform(Xs)
                Xt_val_proj = adapter.transform(Xt_val)
                
            clf = LogisticRegression(class_weight='balanced', max_iter=500, random_state=42)
            clf.fit(Xs_proj, ys)
            p_val = clf.predict_proba(Xt_val_proj)[:, 1]
            try:
                auc = roc_auc_score(yt_val, p_val)
            except ValueError:
                auc = 0.5
                
            if auc > best_auc:
                best_auc = auc
                best_params = (k, mu)
                
    return best_params[0], best_params[1], best_auc
