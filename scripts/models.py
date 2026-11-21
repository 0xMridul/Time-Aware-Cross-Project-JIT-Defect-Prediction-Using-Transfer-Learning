import os
import sys
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))
"""
models.py - Classifier wrappers, imbalance handlers, and prediction pipelines.
"""

import numpy as np
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

def get_base_classifier(model_name, seed=42, class_weight='balanced'):
    """
    Instantiates the requested classifier with pinned random seed.
    """
    if model_name == 'lr':
        return LogisticRegression(
            max_iter=1000,
            class_weight=class_weight,
            random_state=seed,
            solver='lbfgs'
        )
    elif model_name == 'rf':
        return RandomForestClassifier(
            n_estimators=100,
            class_weight=class_weight,
            random_state=seed,
            n_jobs=-1
        )
    elif model_name == 'lgbm':
        cw = 'balanced' if class_weight == 'balanced' else None
        return LGBMClassifier(
            n_estimators=100,
            class_weight=cw,
            random_state=seed,
            verbose=-1,
            n_jobs=-1
        )
    elif model_name == 'online_sgd':
        return SGDClassifier(
            loss='log_loss',
            class_weight=class_weight,
            random_state=seed
        )
    else:
        raise ValueError(f"Unknown model_name: {model_name}")

def preprocess_features(X_train, X_test=None, apply_log1p=True):
    """
    Applies log1p transformation followed by StandardScaler fitted strictly on X_train.
    """
    if apply_log1p:
        X_tr = np.log1p(np.maximum(0, np.asarray(X_train, dtype=float)))
        X_te = np.log1p(np.maximum(0, np.asarray(X_test, dtype=float))) if X_test is not None else None
    else:
        X_tr = np.asarray(X_train, dtype=float)
        X_te = np.asarray(X_test, dtype=float) if X_test is not None else None
        
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te) if X_te is not None else None
    
    return X_tr_s, X_te_s, scaler

def train_and_predict(X_train, y_train, X_test, model_name='lr',
                      imbalance='class_weight', seed=42, apply_log1p=True):
    """
    Trains model strictly on X_train, y_train and outputs predicted probabilities on X_test.
    Fits scalers and resamplers strictly on X_train.
    """
    X_tr, X_te, _ = preprocess_features(X_train, X_test, apply_log1p=apply_log1p)
    y_tr = np.copy(y_train)
    
    # Imbalance handling applied ONLY on training data
    cw = 'balanced' if imbalance == 'class_weight' else None
    
    if imbalance == 'smote':
        minority_count = int(np.sum(y_tr == 1))
        if minority_count >= 6:
            k_neighbors = min(5, minority_count - 1)
            sm = SMOTE(random_state=seed, k_neighbors=k_neighbors)
            X_tr, y_tr = sm.fit_resample(X_tr, y_tr)
        elif minority_count > 1:
            sm = SMOTE(random_state=seed, k_neighbors=1)
            X_tr, y_tr = sm.fit_resample(X_tr, y_tr)
            
    clf = get_base_classifier(model_name, seed=seed, class_weight=cw)
    clf.fit(X_tr, y_tr)
    
    if hasattr(clf, 'predict_proba'):
        y_prob = clf.predict_proba(X_te)[:, 1]
    else:
        dfn = clf.decision_function(X_te)
        y_prob = 1.0 / (1.0 + np.exp(-dfn))
        
    return y_prob, clf
