"""
Reviewer 1, Comment 5: GMM component-count selection and stability.

Analysis:
- Restrict copper occurrences to the Chagai Belt window:
  60.0–67.5 degrees E and 26.0–32.5 degrees N.
- Standardize longitude and latitude.
- Evaluate K = 1–10 with full-covariance Gaussian mixture models.
- Use 10 random seeds and 20 internal initializations per seed.
- Report AIC, BIC, pairwise adjusted Rand index, component sizes,
  and maximum posterior membership probabilities.
- Exclude solutions containing singleton components as degenerate
  point-specific mixtures.
- Select K = 3 as the lowest-BIC non-degenerate model.
- Fit the final K = 3 model with 100 initializations and label
  components A–C from west to east.

The maximum posterior membership probability measures only separation
within the fitted coordinate model. It is not geological confidence,
prospectivity, deposit type, or mineralization genesis.
"""
