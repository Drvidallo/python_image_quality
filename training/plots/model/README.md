# Learned model evaluation

These plots evaluate the trained classifiers and regressors on validation and held-out test data: where they are right, where they fail, and which features they rely on.

## confusion_matrices.png

**Technical** — One confusion matrix per learned classifier on the validation split (rows actual, columns predicted), computed at the auto-selected operating threshold from val precision-recall targeting the per-class recall in config (training.target_recall); thresholds are saved to models/thresholds.json. Resolution has no matrix because it uses the deterministic pixel-size rule.

**Layman** — A right/wrong scorecard: how many pass/fail calls were correct and what kind of mistakes were made.

**ELI5** — Marking a quiz: how many answers were right, and did the mistakes mark good answers wrong or bad answers right?

**What to look for** — False negatives (bad image called readable) are the costly mistakes for OCR gating. resolution and brightness show the most errors; if false negatives concentrate there, raise that threshold or add features.

**What it gives you** — Per-dimension error profile and threshold-tuning direction.

**Observed in the current run** — Thresholds: {'readable_overall': 0.38610519713585323, 'readable_blur': 0.5474257577392726, 'readable_brightness': 0.3704752623367095, 'readable_glare': 0.34061558354865723}; validation F1: {'readable_overall': '0.737', 'readable_blur': '0.882', 'readable_brightness': '0.796', 'readable_glare': '0.893'}.

## roc_curves.png

**Technical** — ROC curves with AUC for each learned classifier on validation, one curve per label (overall is trained on readable_overall_learned). Resolution is excluded: it is deterministic via the document pixel-size rule and is reported separately in model_report.json as resolution_rule and combined_system.

**Layman** — How well each model ranks bad images above good ones across all possible thresholds.

**ELI5** — A graph of how many bad ones you catch versus how many false alarms you raise as you tighten the rule.

**What to look for** — Curves hugging the top-left corner are strong (glare, blur). A curve close to the diagonal (resolution) means the model barely separates classes.

**What it gives you** — Model comparison independent of threshold and the operating-point choice for production.

**Observed in the current run** — Validation AUC per classifier: {'readable_overall': '0.952', 'readable_blur': '0.951', 'readable_brightness': '0.837', 'readable_glare': '0.960'}. Resolution rule val F1 = 1.000. Combined system val F1 = 0.837.

## regression_pred_vs_actual.png

**Technical** — Scatter of predicted versus actual severity for blur, brightness, glare and severity_index_learned (mean of the three learned dimensions) on validation, with a dashed identity line. Resolution severity is not regressed because resolution is deterministic in production.

**Layman** — How close the predicted severity is to the true severity.

**ELI5** — Guessing someone's age and plotting the guess against the real age: perfect guesses sit on the diagonal.

**What to look for** — Points should cluster on the diagonal. Systematic flattening at the extremes is regression to the mean; a wide vertical spread at a given actual value means that dimension is under-determined by the features.

**What it gives you** — Reliability of each severity score the Angular app would display.

**Observed in the current run** — Validation R2 per regressor: {'blur_severity': '0.706', 'brightness_severity': '0.614', 'glare_severity': '0.697', 'severity_index': '0.741'}.

## permutation_importance.png

**Technical** — Permutation importance on validation: ROC-AUC drop for readable_overall and MAE increase for severity_index when each feature is shuffled (5 repeats).

**Layman** — Which measurements the model actually relies on to make decisions.

**ELI5** — Figuring out which clues the detective really used by hiding each clue and seeing if the case falls apart.

**What to look for** — Blur and edge-energy features (blur_var, tenengrad, fft_hf_ratio, quality_prior) should dominate. Near-zero bars are candidates for removal and shrink the TypeScript port.

**What it gives you** — Final feature pruning evidence and the shortlist of features to implement in pure TypeScript.

**Observed in the current run** — readable_overall: overexposed_pct=0.043, blur_scale_ratio=0.026, tenengrad=0.018; severity_index: tenengrad=0.043, blur_var=0.036, overexposed_pct=0.035
