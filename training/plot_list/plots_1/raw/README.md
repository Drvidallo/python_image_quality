# Raw metric distributions

These plots describe the ingested frames BEFORE cleaning. They answer: do the four quality measurements vary enough to learn from, and are the document types and capture setups represented fairly?

## hist_raw_metrics.png

**Technical** — 2x3 grid of 50-bin histograms over all ingested frames: blur_var, brightness_L, glare_pct, glare_blob_pct and doc_short_side_px. blur_var, glare_pct and glare_blob_pct are shown as log1p because they are strongly right-skewed; brightness is Lab L-channel mean; short side is the minimum warped document dimension in pixels.

**Layman** — A bird's-eye view of how sharp, bright, glary and large the documents are across the whole dataset, before any filtering.

**ELI5** — Imagine measuring every photo's sharpness, brightness and size, then drawing bars showing how many photos fell into each range.

**What to look for** — blur_var should span from very low (blurry) to high (sharp) with a long right tail; brightness of warped document crops should sit in a broad mid-to-high band (roughly 120-220 Lab L) and not pile up at 0 or 255; glare should be mostly near zero with a few high-glare outliers; short side should cluster in the 400-800 px range for warped documents.

**What it gives you** — Baseline sanity: if a metric is constant or has no spread, no model can learn it. This is the first go/no-go check for each dimension.

**Observed in the current run** — 1980 frames; median blur_var=66.887, brightness_L=185.1, glare_pct=0.715; doc short side p10/p50/p90 = 504/611/734 px.

## hist_by_doc_type.png

**Technical** — Boxplots of blur_var (log scale), brightness_L and glare_pct split by document_type (id_card vs passport), one panel per metric.

**Layman** — Compares the quality ranges of ID cards versus passports so you can see whether one document style is systematically different.

**ELI5** — Like comparing test scores between two classrooms to see if one class always scores higher, regardless of the students.

**What to look for** — Overlapping medians and ranges mean the two types behave similarly and one model can serve both. One type consistently darker, glarier or sharper means the feature reflects the design, not the capture quality.

**What it gives you** — Evidence for the single-model vs per-type decision and for keeping document_type as an input feature.

**Observed in the current run** — Frames per type: {'id_card': 1140, 'passport': 840}.

## capture_condition_counts.png

**Technical** — Bar chart of frame counts per capture_condition, which is the raw MIDV-500 clip prefix (CA/CS/HA/HS/KA/KS/PA/PS/TA/TS).

**Layman** — How many photos came from each capture setup or device condition.

**ELI5** — Counting how many kids arrived on each school bus.

**What to look for** — Roughly equal bars are healthy. A missing prefix means that capture setup was not ingested; a dominant prefix can bias splits and evaluation.

**What it gives you** — Sampling balance across capture setups, which feeds stratified split choices.

**Observed in the current run** — 10 capture conditions; largest: CA=198, CS=198, HA=198, HS=198.

## missingness.png

**Technical** — Bar chart of NaN counts per column of raw_metrics.parquet. Empty chart with a 'No missing values' label means every metric computed for every frame.

**Layman** — Which measurements failed to compute for some photos.

**ELI5** — A checklist showing which fields of each photo's report card were left blank.

**What to look for** — Ideally no bars at all. Bars over quad or metric columns mean broken annotations or decode failures upstream.

**What it gives you** — Ingest integrity confirmation before cleaning even runs.

**Observed in the current run** — Columns with any missing value: 0 of 46.
