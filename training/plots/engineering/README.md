# Feature engineering and reduction

These plots justify the final feature set: which features carry signal, which are redundant, and how separable the classes look in feature space.

## correlation_heatmap.png

**Technical** — Spearman correlation heatmap (coolwarm, centered at 0) among the top 15 MI-ranked features.

**Layman** — Which measurements move together, meaning they carry the same information.

**ELI5** — Finding out which crayons always break at the same time, so you don't need to pack both.

**What to look for** — Off-diagonal blocks above |0.95| indicate redundant features that reduction removed. A mostly pale matrix means the kept features add distinct information.

**What it gives you** — Evidence that the correlation-pruned feature set is not wasteful.

**Observed in the current run** — 31 features survived pruning; correlation threshold=0.95; low-variance dropped=0.

## mi_ranking.png

**Technical** — Horizontal bars of mutual information between each candidate feature and severity_index (top 20), computed on the training split.

**Layman** — Which measurements tell you the most about overall image badness.

**ELI5** — Ranking detective clues by how much each one helps predict the answer.

**What to look for** — Blur and edge-energy features usually rank first because severity_index is blur-dominated. Dimension-specific features (glare, brightness) appear lower but must stay non-zero, otherwise that dimension needs new features.

**What it gives you** — The ranking used to build feature_spec.json and to choose the compact top-k list for the TypeScript port.

**Observed in the current run** — Top MI features: overexposed_pct=0.798, glare_blob_pct=0.767, glare_ratio=0.763, glare_pct=0.754, log_glare_blob_pct=0.733.

## pca_scatter.png

**Technical** — 2-D PCA projection of standardized selected features (8,000-row sample), points colored by readable_overall_learned.

**Layman** — A flat map of all images, colored by whether they are readable.

**ELI5** — Squashing a 3-D toy box onto paper to see whether red and yellow balls separate into different areas.

**What to look for** — Partial color separation is good; total overlap means the features cannot easily separate classes. Isolated clusters often point to a specific document type or capture condition.

**What it gives you** — Holistic view of separability, cluster structure and potential outliers.

**Observed in the current run** — Explained variance PC1+PC2 = 38.8% over 31 features.

## quality_prior_by_class.png

**Technical** — KDE of the hand-weighted quality_prior score (mean of fixed-range normalized blur, brightness, glare, resolution terms) split by readable_overall_learned.

**Layman** — Does a simple homemade quality score already separate good from bad images?

**ELI5** — Comparing a quick-and-dirty grade against the official pass/fail stamp.

**What to look for** — Shifted densities with limited overlap mean the simple prior works; heavy overlap means the learned model must (and does) add value.

**What it gives you** — A transparent baseline the learned classifier should beat.

**Observed in the current run** — Mean quality_prior by class: {0: 0.476, 1: 0.664}.
