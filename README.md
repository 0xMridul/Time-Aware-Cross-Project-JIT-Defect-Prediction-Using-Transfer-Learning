# Does Transfer Learning Really Help Cross-Project Just-In-Time Defect Prediction?
# AISLOP(IGNORE THIS REPO)

### A Time-Aware Re-Evaluation on ApacheJIT

This repository contains the research materials, experimental results, analysis scripts, statistical evaluations, and manuscript for an empirical study investigating whether transfer learning improves cross-project Just-In-Time (JIT) software defect prediction under time-aware evaluation.

## Research Overview

Just-In-Time defect prediction aims to identify potentially defective software changes before they are integrated into a software project. Cross-project approaches attempt to leverage historical data from other projects when training data from the target project is limited.

However, software projects evolve over time, and differences between historical and future data may affect model performance. This study re-evaluates cross-project JIT defect prediction using chronological evaluation protocols and investigates whether transfer learning and domain adaptation provide measurable improvements over conventional approaches.

## Research Objectives

The study investigates:

- Whether transfer learning improves cross-project JIT defect prediction.
- How domain adaptation methods perform against unadapted pooled cross-project learning.
- How prediction performance changes across historical training and testing periods.
- Whether quarantine gaps influence model performance and statistical equivalence.
- How feature availability and retrospective information affect prediction validity.
- Whether experimental conclusions remain consistent under alternative evaluation protocols and sensitivity analyses.

## Dataset

The experiments use the ApacheJIT benchmark dataset.

- **Dataset:** ApacheJIT
- **Source:** Zenodo
- **DOI:** [10.5281/zenodo.5907001](https://doi.org/10.5281/zenodo.5907001)
- **Projects:** 13 Apache Java projects
- **Deduplicated commits:** 106,674

Please consult the dataset source and the data availability documentation before reproducing the experiments.

## Experimental Design

The study evaluates the following approaches:

1. **Within-Project (WP):** Models trained using historical data from the target project.
2. **Pooled Cross-Project (CP):** Models trained using data pooled from other projects.
3. **CORAL:** Correlation Alignment for domain adaptation.
4. **TCA:** Transfer Component Analysis for domain adaptation.
5. **Source + Target:** A combined training approach using source-project and target-project data.

### Classification Algorithms

- Logistic Regression (LR)
- Random Forest (RF)

### Evaluation Metrics

- ROC-AUC
- PR-AUC
- F1-score
- Matthews Correlation Coefficient (MCC)

The evaluation includes chronological splits, historical training gaps, fixed training-size comparisons, quarantine-gap analyses, and sensitivity experiments.

## Main Findings

The primary statistical analysis does not establish a statistically significant improvement from CORAL or TCA over unadapted pooled cross-project learning after family-wide Holm multiple-testing correction.

Additional temporal analyses show numerical performance differences between within-project and cross-project approaches. These differences should not be interpreted as statistically established superiority.

The study also examines retrospective feature leakage. In particular, the `fix` feature is investigated separately because its information may not be available at the time a commit-level prediction would be made.

The findings are presented with attention to statistical uncertainty, evaluation assumptions, and prediction-time validity.

## Repository Structure

```text
.
├── data_availability/
│   ├── DATA_AVAILABILITY.md
│   ├── DATA_DICTIONARY.md
│   └── DATA_SOURCE_REFERENCES.md
│
├── figures/
│   ├── main/
│   └── supplementary/
│
├── manuscript/
│   ├── ABSTRACT.md
│   ├── COVER_LETTER.md
│   ├── KEYWORDS.md
│   ├── MANUSCRIPT.md
│   ├── MANUSCRIPT.pdf
│   ├── MANUSCRIPT.docx
│   └── references.bib
│
├── reproducibility/
│   ├── CONFIGURATION.md
│   ├── DATA_PROVENANCE.md
│   ├── RANDOM_SEEDS.md
│   ├── RUN_INSTRUCTIONS.md
│   ├── environment.yml
│   ├── requirements.txt
│   └── scripts/
│
├── results/
│   ├── primary_analysis.csv
│   ├── primary_family_statistics.csv
│   └── sensitivity_*.csv
│
├── scripts/
│
├── supplementary/
│
├── tables/
│   ├── main/
│   └── supplementary/
│
└── README.md
```

## Reproducibility

The repository includes experiment scripts, environment specifications, result files, and supporting documentation.

For reproduction instructions, see:

- `reproducibility/README.md`
- `reproducibility/RUN_INSTRUCTIONS.md`
- `reproducibility/CONFIGURATION.md`
- `reproducibility/DATA_PROVENANCE.md`

Install the required dependencies using the provided environment specification or requirements file.

```bash
pip install -r reproducibility/requirements.txt
```

Refer to the run instructions for dataset preparation, execution order, expected input paths, and individual experiment commands.

**Note:** The ApacheJIT dataset must be obtained separately from its original source unless a permitted local copy has been provided. Check the data availability documentation for details.

## Statistical Analysis

The research includes:

- Paired statistical comparisons.
- Family-wide Holm multiple-testing correction.
- Confidence interval estimation.
- Equivalence testing using Two One-Sided Tests (TOST).
- Quarantine-gap sensitivity analysis.
- Temporal and training-size sensitivity experiments.
- Prediction-time feature validity assessment.

Interpretation of results considers the distinction between statistical significance, practical effect size, equivalence, and insufficient evidence.

## Research Manuscript

The manuscript is available in:

- PDF: `manuscript/MANUSCRIPT.pdf`
- DOCX: `manuscript/MANUSCRIPT.docx`
- Markdown: `manuscript/MANUSCRIPT.md`

## Author

**Kazi Mazharul Islam Mridul**  
East West University

## Citation

If you use this repository or build upon this research, please cite the manuscript after its publication details become available.

A formal citation and publication identifier will be added when applicable.

## Research Status

This repository contains research materials and experimental artifacts associated with an empirical software engineering study.

The reported findings should be interpreted within the dataset, experimental protocols, and limitations described in the manuscript.

## License

A repository license should be added after the author determines the appropriate terms for the source code, documentation, and research materials. The ApacheJIT dataset remains subject to its own licensing and usage conditions.
