# Pseudo-Random Number Generator Seeds

To guarantee exact reproducibility:
1. **Random Forest Seed Suite:**
   All Random Forest models are evaluated across 10 deterministic integer seeds:
   $$S = \{42, 43, 44, 45, 46, 47, 48, 49, 50, 51\}$$
   In reports and tables, Random Forest performance represents the exact arithmetic mean across these 10 seeds.
2. **SMOTE Resampling Seed:**
   When SMOTE is evaluated in Sensitivity Analysis 3, the random state is set to `random_state=42`.
3. **Bootstrap Confidence Interval Seed:**
   Percentile bootstrap confidence intervals use NumPy's default RNG initialized with seed `seed=42` across $\ge 10,000$ resamples.
