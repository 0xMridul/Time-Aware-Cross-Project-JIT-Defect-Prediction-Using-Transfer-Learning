import os
import sys
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))
"""
data_loader.py - Data ingestion, filtering, and split generation for ApacheJIT empirical evaluation.
"""

import os
import pandas as pd
import numpy as np

KAMEI_12_FEATURES = [
    'la', 'ld', 'nf', 'nd', 'ns', 'ent',
    'ndev', 'age', 'nuc', 'aexp', 'arexp', 'asexp'
]

KAMEI_13_FEATURES = KAMEI_12_FEATURES + ['fix']

TARGET_PROJECTS_10 = [
    'apache/camel', 'apache/hadoop', 'apache/ignite', 'apache/flink',
    'apache/hbase', 'apache/cassandra', 'apache/groovy', 'apache/hive',
    'apache/activemq', 'apache/kafka'
]

def load_and_preprocess_total(csv_path='repo_check/dataset/apachejit_total.csv'):
    df = pd.read_csv(csv_path)
    # Merge Hadoop subprojects
    df['project'] = df['project'].replace({
        'apache/hadoop-hdfs': 'apache/hadoop',
        'apache/hadoop-mapreduce': 'apache/hadoop'
    })
    # Convert types
    df['author_date'] = df['author_date'].astype(np.int64)
    df['buggy'] = df['buggy'].astype(str).str.strip().str.lower().map({'true': 1, 'false': 0}).astype(int)
    df['fix'] = df['fix'].astype(str).str.strip().str.lower().map({'true': 1.0, 'false': 0.0}).astype(float)
    for col in KAMEI_12_FEATURES:
        df[col] = df[col].astype(float)
    return df

def load_official_splits(train_path='repo_check/dataset/apachejit_train.csv',
                         test_path='repo_check/dataset/apachejit_test_large.csv'):
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    for d in [train_df, test_df]:
        d['project'] = d['project'].replace({
            'apache/hadoop-hdfs': 'apache/hadoop',
            'apache/hadoop-mapreduce': 'apache/hadoop'
        })
        d['author_date'] = d['author_date'].astype(np.int64)
        d['buggy'] = d['buggy'].astype(str).str.strip().str.lower().map({'true': 1, 'false': 0}).astype(int)
        d['fix'] = d['fix'].astype(str).str.strip().str.lower().map({'true': 1.0, 'false': 0.0}).astype(float)
        for col in KAMEI_12_FEATURES:
            d[col] = d[col].astype(float)
    return train_df, test_df

def get_protocol_b_and_c_splits(df, target_project, feature_cols, split_ratio=0.7, label_delay_gap_months=0):
    """
    Protocol B (Cross-Project) & Protocol C (Within-Project) for target project P.
    Time-ordered strictly by author_date.
    """
    target_commits = df[df['project'] == target_project].sort_values('author_date').copy()
    n_target = len(target_commits)
    split_idx = int(n_target * split_ratio)
    
    target_train = target_commits.iloc[:split_idx + 1]
    target_test = target_commits.iloc[split_idx + 1:]
    
    cutoff_ts = target_train['author_date'].max()
    gap_seconds = int(label_delay_gap_months * (365.25 / 12) * 86400)
    source_cutoff_ts = cutoff_ts - gap_seconds
    
    source_pool = df[(df['project'] != target_project) & (df['author_date'] <= source_cutoff_ts)].copy()
    
    return {
        'target_project': target_project,
        'cutoff_ts': cutoff_ts,
        'source_cutoff_ts': source_cutoff_ts,
        'gap_months': label_delay_gap_months,
        'X_target_train': target_train[feature_cols].values,
        'y_target_train': target_train['buggy'].values,
        'X_target_test': target_test[feature_cols].values,
        'y_target_test': target_test['buggy'].values,
        'X_source_train': source_pool[feature_cols].values,
        'y_source_train': source_pool['buggy'].values,
        'n_target_train': len(target_train),
        'n_target_test': len(target_test),
        'n_source_train': len(source_pool),
        'target_test_defect_rate': float(np.mean(target_test['buggy'].values)),
        'source_train_defect_rate': float(np.mean(source_pool['buggy'].values)) if len(source_pool) > 0 else 0.0
    }

def get_mature_calendar_splits(df, target_project, feature_cols, window_type='2016-2017'):
    """
    Redesigned sensitivity analysis with mature labels (ending <= 2017).
    """
    p_commits = df[df['project'] == target_project].sort_values('author_date').copy()
    
    if window_type == '2016-2017':
        # Train <= 2015-12-31, Test = 2016-01-01 to 2017-12-31
        ts_train_cutoff = 1451606400  # 2016-01-01 00:00:00 UTC
        ts_test_end = 1514764799      # 2017-12-31 23:59:59 UTC
        
        target_train = p_commits[p_commits['author_date'] < ts_train_cutoff]
        target_test = p_commits[(p_commits['author_date'] >= ts_train_cutoff) & (p_commits['author_date'] <= ts_test_end)]
        source_train = df[(df['project'] != target_project) & (df['author_date'] < ts_train_cutoff)]
    elif window_type == '2017':
        # Train <= 2016-12-31, Test = 2017-01-01 to 2017-12-31
        ts_train_cutoff = 1483228800  # 2017-01-01 00:00:00 UTC
        ts_test_end = 1514764799      # 2017-12-31 23:59:59 UTC
        
        target_train = p_commits[p_commits['author_date'] < ts_train_cutoff]
        target_test = p_commits[(p_commits['author_date'] >= ts_train_cutoff) & (p_commits['author_date'] <= ts_test_end)]
        source_train = df[(df['project'] != target_project) & (df['author_date'] < ts_train_cutoff)]
    else:
        raise ValueError(f"Unknown window_type: {window_type}")
        
    return {
        'target_project': target_project,
        'window_type': window_type,
        'X_target_train': target_train[feature_cols].values,
        'y_target_train': target_train['buggy'].values,
        'X_target_test': target_test[feature_cols].values,
        'y_target_test': target_test['buggy'].values,
        'X_source_train': source_train[feature_cols].values,
        'y_source_train': source_train['buggy'].values,
        'n_target_train': len(target_train),
        'n_target_test': len(target_test),
        'n_source_train': len(source_train)
    }

def assert_canonical_test_split(target_name, test_df):
    """
    Asserts that test_df matches the canonical Design B test set exactly:
    author_date in [1451606400, 1514764799] for target_name.
    Verifies row counts, commit ID sequence, and bug counts.
    """
    df = load_and_preprocess_total()
    p_df = df[df['project'] == target_name].sort_values('author_date')
    canonical_test = p_df[(p_df['author_date'] >= 1451606400) & (p_df['author_date'] <= 1514764799)]
    
    canonical_ids = list(canonical_test['commit_id'])
    test_ids = list(test_df['commit_id'])
    
    assert len(test_ids) == len(canonical_ids), f"Test N mismatch for {target_name}: {len(test_ids)} vs {len(canonical_ids)}"
    assert test_ids == canonical_ids, f"Test commit ID ordering or content mismatch for {target_name}!"
    assert int(test_df['buggy'].sum()) == int(canonical_test['buggy'].sum()), f"Buggy count mismatch for {target_name}!"
    return True

