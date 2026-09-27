# Cleaning and transformation

These plots document what was removed, flagged or transformed on the way from raw frames to the model table, so every dropped row has a reason.

## before_after_transforms.png

**Technical** — 2x2 histograms comparing raw blur_var vs log1p(blur_var) and raw glare_pct vs log1p(glare_pct), 60 bins each.

**Layman** — Before/after views showing how skewed measurements were straightened out.

**ELI5** — Like squashing a stretched-out spring back into an even spring.

**What to look for** — The raw plots are long right tails; the log1p plots should look much closer to a bell shape, which is what linear models and distance metrics prefer.

**What it gives you** — Justification for the log columns recorded in feature_spec.json.

**Observed in the current run** — blur_var skew raw=1.92 vs log1p=-1.72.

## reconciliation.png

**Technical** — Bar chart of cleaning counts from reconciliation_dataset.json: input_rows, dropped_missing_metrics, dropped_out_of_range, outlier_flagged, dropped_near_duplicates and kept_rows.

**Layman** — How many images went in, how many were kept, and why the rest were dropped.

**ELI5** — How many crayons you kept, threw away because they were broken, or removed because they were duplicates of ones you already had.

**What to look for** — Drops should be small and explainable. Outliers are flagged and kept for analysis (not silently deleted); near-duplicate drops should be modest because frames were already subsampled every N-th frame.

**What it gives you** — The audit trail that makes the dataset trustworthy and reproducible.

**Observed in the current run** — input=1980, kept=1228, missing=0, out_of_range=41, outliers_flagged=0, near_duplicates=711.
