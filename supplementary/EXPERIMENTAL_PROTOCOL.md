# Supplementary Document: Experimental Splitting Protocols

### 1. Zero Future-to-Past Leakage Principle
A fundamental vulnerability of prior CP-JIT evaluations is the violation of temporal ordering (e.g. cross-validation, or pooling future cross-project commits to predict past target commits).
In all protocols in this study:
$$\forall c_{\text{train}} \in \mathcal{D}_{\text{train}}, \quad \text{author\_date}(c_{\text{train}}) < \min_{c_{\text{test}} \in \mathcal{D}_{\text{test}}} \text{author\_date}(c_{\text{test}})$$
Cross-project donor pools are strictly subject to this cutoff. No donor commit authored after the target training cutoff date is ever visible to training.

### 2. Protocol Definitions
- **Design B (Primary Confirmatory Protocol):** Fixed test window 2016–2017 (`author_date` in $[1451606400, 1514764799]$). Training spans all commits authored prior to 2016-01-01 (`author_date < 1451606400`).
- **Protocol D (Longitudinal Temporal Drift):** Fixed test window 2016–2017. Six historical training endpoints: $\le 2015$ (Gap 0), $\le 2014$ (Gap 1), $\le 2013$ (Gap 2), $\le 2012$ (Gap 3), $\le 2011$ (Gap 4), $\le 2010$ (Gap 5). Evaluated under Natural Growing and Fixed Training Size variants.
- **70% Per-Project Chronological Split (Sensitivity 4):** Project-specific 70/30 chronological split based on author date.
- **Protocol A (Sensitivity 5):** Official Zenodo split deposited by Zeng et al.
