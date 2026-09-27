# Classification labels

These plots describe the readability pass/fail labels (overall and per dimension) and how well the engineered features separate the classes.

## class_balance.png

**Technical** — Bar charts of 0/1 counts for readable_overall (4-dimension physical label), readable_overall_learned (blur AND brightness AND glare, the label the learned overall classifier is trained on) and the four per-dimension labels. resolution is deterministic via the pixel-size rule.

**Layman** — How many images pass and fail the readability check, overall and for each individual flaw.

**ELI5** — A tally of green check marks and red crosses for the whole set and for each type of problem.

**What to look for** — Severe imbalance (e.g. 95/5) needs class weights or threshold tuning. Per-dimension rates differ a lot: glare failures are rare because the glare radius grid is mostly mild.

**What it gives you** — Class weighting strategy and the baseline positive rate each F1 score must beat.

**Observed in the current run** — Positive rates: {'readable_overall': '0.107', 'readable_overall_learned': '0.164', 'readable_blur': '0.451', 'readable_brightness': '0.476', 'readable_glare': '0.544', 'readable_resolution': '0.343'}.

## class_balance_by_type.png

**Technical** — Grouped bar chart of readable_overall counts split by document_type.

**Layman** — Pass/fail rates for ID cards versus passports.

**ELI5** — Pass rate for each classroom, to check one classroom is not treated unfairly.

**What to look for** — Similar pass rates across types are healthy. A large gap means the model (or the label rule) behaves differently per document design, so always report per-type metrics too.

**What it gives you** — The single-model vs per-type decision and fairness checks for evaluation.

**Observed in the current run** — readable_overall rate by type: {'id_card': 0.101, 'passport': 0.111}.

## feature_hist_by_class.png

**Technical** — Kernel density estimates of the top-k selected features, split by readable_overall_learned (0 red, 1 blue), on a 20,000-row sample. This is the label the learned overall classifier predicts.

**Layman** — Do the numbers look different for readable versus unreadable images?

**ELI5** — Two bell curves: if they sit far apart, the clue is strong; if they overlap heavily, the clue is weak.

**What to look for** — Well-separated peaks (blur, glare features) versus heavy overlap (some brightness/resolution features). Overlap on a top feature predicts a weak classifier for that dimension.

**What it gives you** — A quick separability read before training and a sanity check on the top-k feature ranking.

**Observed in the current run** — Top-k features shown: ['overexposed_pct', 'glare_ratio', 'blur_var_half', 'blur_var_big', 'blur_var', 'brightness_p95'].
