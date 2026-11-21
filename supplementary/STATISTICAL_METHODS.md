# Supplementary Document: Inferential Statistics, Hypothesis Testing, and TOST

### 1. Paired Wilcoxon Signed-Rank Test
For project-level paired metrics ($x_i, y_i$), differences $d_i = x_i - y_i$ are computed across the $n = 10$ target projects. The two-tailed Wilcoxon signed-rank test evaluates the null hypothesis that differences are symmetrically distributed around zero. Effect sizes are measured via the rank-biserial correlation:
$$r_{rb} = \frac{W^+ - W^-}{W^+ + W^-}$$

### 2. Family-Wide Multiple Testing Correction
We correct for family-wise error rate (FWER) across the primary confirmatory family of 20 comparisons (2 classifiers $\times$ 2 metrics $\times$ 5 comparisons) using the sequential rejective Holm-Bonferroni procedure at $\alpha = 0.05$.

### 3. Bootstrap Confidence Intervals
Confidence intervals for mean paired differences are constructed using $\ge 10,000$ bias-corrected or percentile bootstrap resamples over projects.

### 4. Two-One-Sided Equivalence Testing (TOST)
To establish that transfer learning is practically indistinguishable from unadapted pooling, we test equivalence against a pre-registered margin of $\delta = \pm 0.02$ ROC-AUC.
We evaluate:
$$H_{01}: \theta \le -0.02 \quad \text{vs.} \quad H_{11}: \theta > -0.02$$
$$H_{02}: \theta \ge +0.02 \quad \text{vs.} \quad H_{12}: \theta < +0.02$$
$$p_{\text{TOST}} = \max(p_1, p_2)$$
Conventional $\alpha = 0.05$ TOST corresponds to the $90\%$ two-sided confidence interval $(1 - 2\alpha)$.
