# Supplementary Document: Additional Project-Level Empirical Results

### 1. Secondary Metrics Summary (Primary Design B)
Below are secondary classification performance metrics (PR-AUC, F1-score at Youden's $J$, and Matthews Correlation Coefficient) across the 10 target projects:

| Model | Method | Mean ROC-AUC | Mean PR-AUC | Mean F1 (Youden) | Mean MCC (Youden) |
|:---|:---|:---:|:---:|:---:|:---:|
| **LR** | WP | 0.7684 | 0.4907 | 0.4121 | 0.3125 |
| **LR** | Pooled CP | 0.7879 | 0.5074 | 0.4284 | 0.3341 |
| **LR** | Source + Target | 0.7886 | 0.5103 | 0.4301 | 0.3362 |
| **LR** | CORAL | 0.7853 | 0.4990 | 0.4215 | 0.3278 |
| **LR** | TCA with standardization | 0.7869 | 0.5047 | 0.4259 | 0.3310 |
| **RF** | WP | 0.7607 | 0.4988 | 0.4312 | 0.3298 |
| **RF** | Pooled CP | 0.7727 | 0.4835 | 0.4290 | 0.3305 |
| **RF** | Source + Target | 0.7752 | 0.4925 | 0.4344 | 0.3382 |
| **RF** | CORAL | 0.7557 | 0.4609 | 0.4089 | 0.3084 |
| **RF** | TCA with standardization | 0.7658 | 0.4760 | 0.4211 | 0.3219 |

*Full per-project and per-seed results archived in `results/primary_analysis.csv`.*
