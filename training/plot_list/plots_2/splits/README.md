# Train / validation / test splits

These plots verify that the splits are leak-free (groups never shared), balanced, and that the held-out test documents are genuinely different.

## split_counts.png

**Technical** — Three bars: rows per split, unique groups (doc_folder/clip) per split, and readable_overall rate per split. Test is a leave-doc-types-out set.

**Layman** — Are the train, validation and test buckets fair and are test documents completely unseen?

**ELI5** — Three buckets of candy: do they have similar amounts and kinds, and is the test bucket made of candies the model never tasted?

**What to look for** — Class rates should be roughly similar across splits. The test bucket contains whole document designs never seen in training, so some metric drop is expected and is the honest measure of generalization.

**What it gives you** — Trust in the evaluation: no frame, clip or document design leaks across splits.

**Observed in the current run** — Rows: {'train': 4356, 'test': 1200, 'val': 696}; groups: {'train': 153, 'val': 27, 'test': 46}.

## target_dist_by_split.png

**Technical** — KDE of severity_index, blur_severity and resolution_severity per split.

**Layman** — Do the splits contain a similar mix of easy and hard images?

**ELI5** — Checking each bucket got the same recipe portions.

**What to look for** — Overlapping curves mean consistent difficulty. A shifted test curve warns that test scores are not directly comparable to validation scores.

**What it gives you** — Fairness check for comparing validation and held-out test metrics.

**Observed in the current run** — Median severity_index per split: {'test': 0.581, 'train': 0.579, 'val': 0.566}.
