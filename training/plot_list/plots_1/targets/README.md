# Regression targets

These plots describe the synthetic degradation severity targets used for the regression head: how strongly each flaw (blur, brightness, glare, resolution) was applied and whether features track them.

## hist_targets.png

**Technical** — Histograms (50 bins) of the continuous regression targets on degraded rows: target_blur_sigma, target_brightness_gamma, target_glare_radius_frac, target_downscale_factor and the composite severity_index (0-1).

**Layman** — How strongly each simulated flaw was applied across the training images.

**ELI5** — How much rain, sun, blur and dirt we added to each photo on purpose, and how many photos got each amount.

**What to look for** — Coverage across the whole grid: spikes only at the neutral values (sigma=0, gamma=1, factor=1) mean too few degraded examples; severity_index should be spread between 0 and 1 rather than piled at the extremes.

**What it gives you** — Regression target coverage; under-sampled severities will make the model weak exactly where it matters (moderate degradation).

**Observed in the current run** — 5731 degraded rows; severity_index min/median/max = 0.00/0.56/1.00.

## hist_severities.png

**Technical** — Histograms of the per-dimension severities (blur, brightness, glare, resolution) normalized to 0-1 plus severity_index. Severity is derived from degradation parameters, never from measured metrics.

**Layman** — Each flaw rescaled so 0 means perfect and 1 means worst, so they can be compared on one scale.

**ELI5** — Giving each problem a grade from 0 (none) to 1 (very bad) so blur and glare can be compared fairly.

**What to look for** — Mass should exist near the label cutoffs, not only at 0 and 1. If all mass sits at 0, the classifier sees only easy negatives and will disappoint in the real moderate-degradation range.

**What it gives you** — Realism of the class boundary and difficulty of the learning problem.

**Observed in the current run** — Mean severity per dimension: {'blur_severity': '0.66', 'brightness_severity': '0.53', 'glare_severity': '0.50', 'resolution_severity': '0.54'}.

## feature_vs_target.png

**Technical** — Hexbin density plots (40 bins, 5,000-row sample) of log_blur_var vs blur_severity, brightness_L vs brightness_severity, log_glare_pct vs glare_severity and doc_short_side_px vs resolution_severity.

**Layman** — Do the measured numbers actually move together with how degraded the image was made?

**ELI5** — Does the thermometer reading go up when the soup gets hotter? If not, the thermometer is not useful.

**What to look for** — Monotone bands or clouds mean the feature is learnable. A diffuse blob (often resolution and brightness) means the degradation is hard to see from that feature, and the model will score lower on that dimension.

**What it gives you** — Which dimensions are learnable from image evidence and where to invest in better feature engineering.

**Observed in the current run** — Spearman correlations: {'log_blur_var~blur_severity': '-0.74', 'brightness_L~brightness_severity': '-0.02', 'log_glare_pct~glare_severity': '0.55', 'doc_short_side_px~resolution_severity': '-0.00'}.
