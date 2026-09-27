# Project Plan: Philippine ID/Passport OCR-Readability Classifier

**Audience for this document:** an AI coding agent executing this project.
**Human role:** provides approvals at checkpoints, supplies real Philippine
ID/passport samples if requested, and makes the judgment calls flagged in
"Decisions requiring human input" throughout.

Read this entire document before starting any work. Execute phases in
order. Do not skip a phase's acceptance criteria to move faster — later
phases depend on earlier ones producing correct, verified artifacts.

---

## 0. Context you must understand before starting

### 0.1 The primary goal (read this carefully — it has changed the priority
order of this project versus earlier drafts)

**The main deliverable is a quality classifier that determines whether a
photo/scan of a Philippine ID or Philippine passport is readable enough for
OCR, based on blur, glare, resolution, and brightness.** Everything else in
this plan — document type classification, corner/boundary detection — exists
to *support* that goal by (a) knowing which physical document format you're
looking at, so resolution/DPI is estimated against the right known size, and
(b) cropping out background before the quality metrics are computed, so a
messy background doesn't corrupt a blur/brightness/glare reading that's
supposed to describe the document itself.

Do not let the corner-detection or type-classification work become the
project's center of gravity. They are Phase 3 and 4 below, in service of
Phase 5's quality classifier, which is where "done" is actually measured.

### 0.2 Existing codebase this project extends

- **File:** `ocr-image-quality-detector.js` — a browser/Node module (uses
  OpenCV.js / `@techstark/opencv-js`) that:
  1. Detects a document's boundary via classical CV (`detectDocumentCorners`
     — Canny + `findContours` + `approxPolyDP`, requires a closed 4-point
     contour), falling back to `detectPartialEdgeCorners` (Hough-line
     detection of left/right/bottom edges + known-aspect-ratio
     reconstruction of the top edge) for documents — passports, specifically
     — whose top edge is often undetectable (spine curl, low contrast,
     shadow).
  2. Warps the detected quadrilateral flat via `warpDocument`, cropping out
     background.
  3. Computes four quality metrics on the cropped result: `computeBlurScore`
     (Laplacian variance), `computeGlareScore` (HSV-based blown-highlight
     detection), `computeBrightnessScore` (Lab L-channel mean),
     `computeResolutionScore` (estimated DPI from known physical document
     size — currently a lookup table of standard sizes including `a4`,
     `id_card`, and `passport`).
  4. Compares each metric against fixed constants in `DEFAULT_THRESHOLDS`
     and returns a pass/fail report with warnings.
- **File:** `demo.html` — a browser test harness for the above.

### 0.3 What's new in this revision: MIDV-500 as reference data

[MIDV-500](https://arxiv.org/abs/1807.05786) is a public research dataset of
50 identity document types (a mix of ID cards, passports, and driving
licenses from various countries), each captured as short video clips under
five conditions (varying background, occlusion, tilt), with **ground-truth
quadrangle corner annotations for every frame**. It does **not** contain any
Philippine documents. Its role in this project is narrowly scoped:

- **Pretraining/reference data for corner/boundary detection (Phase 4)** —
  real-world capture diversity (real cameras, real lighting, real motion
  blur, real occlusion) that synthetic compositing can't fully replicate.
  A corner detector pretrained on MIDV-500 and then fine-tuned on
  Philippine-specific synthetic data should generalize better than one
  trained on synthetic data alone.
- **Pretraining/reference data for the ID-vs-passport type classifier
  (Phase 3)** — MIDV-500 includes several passport-type documents (which
  have a two-line MRZ block and ID-3 proportions) and many ID-card-type
  documents (ID-1 proportions, no MRZ), giving a visually diverse basis for
  learning the *general* passport-vs-card distinction before specializing
  to Philippine document designs specifically.
- **Explicitly not used for:** the quality classifier in Phase 5. Blur/
  glare/brightness/resolution readability is about image *degradation*, not
  document *identity or nationality* — MIDV-500's foreign document designs
  don't need to appear in that training set. Phase 5 trains on Philippine-
  specific (synthetic + real) data only, per Section 0.1.

**Decision requiring human input, before Phase 1 begins:** confirm the
current MIDV-500 license/usage terms (check the dataset's official page —
terms may have changed since this plan was written) permit your intended
use, including whether commercial deployment requires anything beyond
research use. Do not proceed with download/use until this is confirmed.

### 0.4 Why Philippine-specific documents don't need a new geometric model

The Philippine national ID (PhilSys ID) is ID-1 format (85.6mm × 53.98mm,
the same international standard as a credit card), and the Philippine
passport is ID-3 format (125mm × 88mm) — identical physical dimensions to
the generic `id_card` and `passport` entries already in
`PHYSICAL_SIZES_INCHES` in the existing JS file. This means the geometry
(aspect ratio assumptions, corner-count expectations) this project relies
on does not need to be reinvented per-country. What's actually
Philippine-specific and does need dedicated data:

- The visual design (colors, field layout, security patterns, photo
  placement) of the PhilSys ID front/back and the passport data page —
  needed for realistic synthetic generation (Phase 1) and for validating
  the type classifier isn't confused by design elements MIDV-500 never
  showed it.
- Real capture conditions specific to your actual use case (e.g., phone
  camera app scanning flow) — needed for the real holdout set (Phase 1.4).

---

## 1. Definition of done

Ranked by how much each matters to the stated goal (Section 0.1):

- [ ] **(Primary)** A quality classifier exists, trained on Philippine ID +
      passport data, that predicts OCR-readability from blur/glare/
      brightness/resolution features, evaluated against real Philippine
      document photos, exported to a JS-runnable format, and wired into
      `analyzeImageQuality()` as an opt-in replacement for
      `DEFAULT_THRESHOLDS`.
- [ ] **(Supporting)** A document-type classifier (ID vs. passport) exists,
      pretrained using MIDV-500 and fine-tuned on Philippine-specific data,
      exported and wired in so `analyzeImageQuality()` can auto-select the
      correct physical size / aspect ratio instead of requiring
      `documentType` to be passed in manually.
- [ ] **(Supporting)** A corner/boundary detector exists, pretrained using
      MIDV-500 and fine-tuned on Philippine-specific synthetic data, that
      matches or beats the classical pipeline's accuracy — measured
      specifically on the passport top-edge-degraded case that motivated
      `detectPartialEdgeCorners` in the first place — and is wired in as a
      leading detection tier ahead of the classical fallbacks.
- [ ] EDA findings for all three models are written up in
      `training/EDA_FINDINGS.md` in prose, not left as raw notebook output.
- [ ] All models are evaluated on a **real Philippine document holdout
      set**, not synthetic data alone, and `training/RESULTS.md` reports
      real-holdout metrics separately from synthetic-test metrics for each
      model, with an honest go/no-go recommendation per model.
- [ ] If the quality classifier (the primary deliverable) cannot be properly
      evaluated because a real Philippine holdout set isn't available, this
      is flagged prominently — do not present synthetic-only readability
      metrics as production-ready.

---

## 2. Required project layout

```
project-root/
  ocr-image-quality-detector.js      # existing file — modify in Phase 6 only
  demo.html                          # existing file — modify in Phase 6 only
  training/
    requirements.txt
    download_midv500.py
    preprocess_midv500.py
    generate_synthetic_ph_documents.py
    degradations.py
    eda_document_types.py            # MIDV-500 + synthetic PH type-distinguishing features
    eda_corner_failures.py           # classical pipeline baseline
    eda_quality_thresholds.py        # readability feature EDA — THE most important EDA here
    EDA_FINDINGS.md
    train_type_classifier.py
    train_corner_model.py
    train_quality_classifier.py
    export_onnx.py
    evaluate.py
    RESULTS.md
  data/
    midv500/
      raw/                          # downloaded dataset, unmodified
      processed/
        frames/                     # extracted frames, subsampled
        annotations.csv             # parsed ground-truth quads + doc-type label per frame
        DOCUMENT_TYPE_MAPPING.md    # explicit, human-verified mapping of MIDV-500 folder -> {id_card, passport}
    backgrounds/
    synthetic/
      ph_id/
      ph_passport/
    real_holdout/
      ph_id/
      ph_passport/
      README.md                    # documents provenance/consent of any real images used
  models/
    type_classifier.onnx
    corner_regressor.onnx
    quality_classifier.onnx         # THE primary deliverable
    MODEL_CARD.md
```

---

## 3. Environment setup

1. Create `training/requirements.txt` with, at minimum: `opencv-python`,
   `numpy`, `pandas`, `scikit-learn`, `matplotlib`, `seaborn`, `torch`,
   `torchvision`, `onnx`, `onnxruntime`, `skl2onnx`, `Pillow`, `requests`
   (for MIDV-500 download).
2. Verify all imports work before writing pipeline code.
3. Confirm `ocr-image-quality-detector.js` runs correctly as-is (open
   `demo.html`, test with at least one sample image) before making any
   changes — this is your known-good baseline for later comparison.

**Decision requiring human input:** if GPU compute is unavailable and
projected CPU training time for Phase 4 (corner model, the heaviest of the
three) exceeds a few hours, pause and propose to the human: (a) proceed on
CPU with a reduced MIDV-500 subsample, (b) request GPU access, or (c) defer
Phase 4 and ship Phase 3 + Phase 5 only, using the existing classical
`detectPartialEdgeCorners` for cropping in the meantime (Phase 5's quality
classifier does not require Phase 4 to be complete — it only requires
*some* cropping step, classical or learned).

---

## 4. Phase 1 — Data acquisition

### 4.1 MIDV-500 download and preprocessing

1. `download_midv500.py`: download the dataset per its official
   distribution instructions into `data/midv500/raw/`. Confirm the license
   check from Section 0.3 before running this.
2. `preprocess_midv500.py`:
   - Extract frames from each video clip, subsampled (e.g., every 5th
     frame) to avoid near-duplicate frames dominating the dataset.
   - Parse each frame's ground-truth quadrangle annotation (MIDV-500
     provides these per-frame; check the dataset's documentation for the
     exact annotation format/location, which may vary by release version).
   - **Build `data/midv500/processed/DOCUMENT_TYPE_MAPPING.md`**: an
     explicit table mapping each of the 50 MIDV-500 document folders to
     `id_card` or `passport`. Do this by inspecting the actual document
     images/documentation (folder names alone may not be self-explanatory)
     — do not guess. **Flag this table for human review before using it to
     generate labels** — a wrong mapping here silently corrupts every
     downstream type-classifier label.
   - Write `data/midv500/processed/annotations.csv` with: `filepath`,
     `document_type` (id_card/passport), `corner_tl_x` ... `corner_bl_y`,
     `capture_condition` (MIDV-500's own condition label, e.g. table/
     occlusion/tilt — useful for slicing evaluation later).

### 4.2 Synthetic Philippine ID/passport generation

Adapt the compositing pipeline from `ML_MODEL_GUIDE.md` (Section 1.2) —
random-homography compositing onto varied backgrounds, ground truth corners
computed from the homography, no manual labeling needed — but render
Philippine-specific templates instead of generic ones:

- **Synthetic PhilSys National ID:** ID-1 proportions (85.6mm × 53.98mm),
  matching the real card's general layout at a non-infringing level of
  detail (placeholder photo box, randomized synthetic name/number fields,
  approximate color scheme/security-pattern texture) — the goal is visual
  realism sufficient for a detector to learn on, not a precise facsimile.
- **Synthetic Philippine passport:** ID-3 proportions (125mm × 88mm data
  page), with a photo placeholder, randomized synthetic body text fields, a
  two-line MRZ block (values do not need to be valid check-digit MRZ), and
  a background texture approximating the passport's visual security
  pattern.
- **Required, as in the prior plan:** simulate spine curl / top-edge
  degradation on the passport template — mild local warp plus a
  contrast-reduction/shadow gradient on the top ~15% of the page, applied
  to a randomized ~50% of generated passport samples. This is still the
  specific real-world failure mode motivating the corner-detection work,
  and it must be present in the Philippine-specific training data too, not
  just the MIDV-500 pretraining data (MIDV-500's passports won't
  necessarily exhibit this exact degradation pattern).
- Apply the same `degradations.py` blur/brightness/glare/JPEG-compression
  functions from the prior plan, post-compositing — this is what generates
  Phase 5's quality-classifier training labels.
- Record the same metadata schema as the prior plan (corners, capture
  angle, applied degradations and their parameters, document type,
  template ID) to `data/synthetic/metadata.csv`.

**Target volume:** 8,000–15,000 samples per document type, as before.

### 4.3 Real Philippine holdout set

Ask the human for 30–100 real photos each of Philippine IDs and Philippine
passports (the human's own documents, consenting colleagues, or specimen/
sample documents — not scraped or third-party PII). Hand-label corners
(CVAT, Label Studio, or a simple corner-click script) and, separately, have
a human reviewer assign a readability pass/fail label per image based on
whether they could actually read all fields clearly — **this second label
set is the ground truth Phase 5's quality classifier is ultimately judged
against**, since it's the actual target the classifier is meant to predict,
and synthetic degradation labels are only a proxy for it.

Document provenance and consent in `data/real_holdout/README.md`.

**Decision requiring human input:** this real holdout set — specifically
the human-reviewed readability labels — is the most important data in this
entire project, since it's the only ground truth that isn't a proxy
(synthetic degradation parameters are a reasonable proxy for "how blurry is
this," but "would a human successfully read this passport" is what you
actually care about). If the human cannot supply this at project start,
flag clearly that Phase 5's final evaluation will be proxy-label-only until
it's available, and treat any pre-holdout results as provisional.

### Phase 1 acceptance criteria

- [ ] `data/midv500/processed/DOCUMENT_TYPE_MAPPING.md` exists and has been
      reviewed/approved by the human before any labels derived from it are
      used in training.
- [ ] `data/midv500/processed/annotations.csv` is populated and spot-checked
      (overlay 20 random ground-truth quads on their frames, visually
      confirm alignment).
- [ ] `data/synthetic/ph_id/` and `data/synthetic/ph_passport/` each
      contain 8,000+ images with complete metadata, spot-checked the same
      way as the MIDV-500 data plus a visual check that the passport
      top-edge degradation is present on the expected subset.
- [ ] `data/real_holdout/` contains labeled real Philippine document images
      with both corner labels and human-reviewed readability labels, with
      documented provenance — or the gap is explicitly flagged per Section
      4.3's decision point.

---

## 5. Phase 2 — Exploratory data analysis

Complete before any training. Findings here set hyperparameters and
priorities for Phases 3–5 — do not skip ahead and guess.

### 5.1 Document-type-distinguishing EDA (informs Phase 3)

Using MIDV-500's `annotations.csv` plus the synthetic Philippine data,
explore what visually/geometrically distinguishes ID cards from passports
that a model could learn: aspect ratio distribution per type (should
cluster around ID-1's ~1.586 and ID-3's ~1.42 respectively — verify this
holds even after perspective distortion in the training crops), presence/
absence of an MRZ-like dense-text band in a fixed region, and whether
MIDV-500's document designs and the synthetic Philippine designs occupy
similar or noticeably different feature space (a large gap here predicts
poor transfer from MIDV-500 pretraining and should adjust your fine-tuning
strategy in Phase 3).

### 5.2 Corner-detection failure EDA (informs Phase 4)

Same procedure as the prior plan: run the *existing* classical pipeline
(`detectDocumentCorners` → `detectPartialEdgeCorners`) over both the
MIDV-500 frames and the synthetic Philippine data, compute IoU and
per-corner error against ground truth, slice by document type, capture
angle/condition, and `has_top_edge_degradation`. This produces the
baseline Phase 4's model must beat, with particular attention to whether
MIDV-500's real-world passport frames show the same top-edge failure
pattern the synthetic Philippine passports were built to simulate — if they
don't, that's useful evidence about how representative the synthetic
degradation is of real conditions.

### 5.3 Quality-threshold EDA (informs Phase 5 — the most important EDA here)

This is the EDA most directly tied to the project's actual goal. Using
**only Philippine synthetic + real holdout data** (per Section 0.3, do not
mix in MIDV-500 here):

1. Compute the four existing quality features (blur variance, glare %,
   glare blob %, brightness, estimated DPI) over all Philippine synthetic
   samples and any labeled real holdout images.
2. If the real holdout set's human-reviewed readability labels (Section
   4.3) are available, use those as the primary label. Otherwise, derive a
   provisional label from synthetic degradation parameters as in the prior
   plan (propose specific numeric cutoffs, flag for human review) —
   **but clearly mark any resulting analysis as provisional** until real
   labels exist, per Section 4.3's decision point.
3. Produce and save (to `training/plots/`): per-feature histograms split
   by readability label, a pairplot across all four features, and
   false-positive/false-negative rates of the current `DEFAULT_THRESHOLDS`
   against this labeled set — separately for ID cards and passports, since
   the two document types may have different failure characteristics
   (e.g., passport security-pattern textures might trigger the glare
   detector differently than a card's simpler surface).
4. Write findings in `EDA_FINDINGS.md`, including an explicit
   recommendation on whether a single quality model should serve both
   document types or whether separate models (or a model that takes
   document type as an input feature) perform better — check this
   empirically rather than assuming.

### Phase 2 acceptance criteria

- [ ] `EDA_FINDINGS.md` covers all three sub-sections above in prose, with
      plots saved under `training/plots/` and referenced by relative path.
- [ ] The quality-threshold EDA (5.3) explicitly states whether it used
      real human-reviewed labels or provisional synthetic-proxy labels —
      this distinction must be carried forward into how Phase 5's results
      are reported and trusted.
- [ ] A recommendation on single-model-vs-per-type for Phase 5 is stated
      and justified.

**Decision requiring human input:** review and approve the provisional
readability label definition (5.3, step 2) before it's used for training,
same as the prior plan — this is a judgment call about what counts as
"readable" and should not be finalized unilaterally.

---

## 6. Phase 3 — Document type classifier (ID vs. passport)

**Role:** supporting infrastructure for Phase 5 (correct physical-size
selection) and Phase 6 (automatic `documentType` detection instead of
requiring it as a manual parameter).

1. Build `train_type_classifier.py`: a small CNN classifier (a MobileNetV3-
   small backbone is sufficient — this is a two-class problem on a
   well-framed single object, not a task needing a large model).
2. **Pretrain on MIDV-500** (`data/midv500/processed/`, using the
   human-approved `DOCUMENT_TYPE_MAPPING.md` labels), then **fine-tune on
   the Philippine synthetic set** (`data/synthetic/ph_id/` +
   `data/synthetic/ph_passport/`). This order matters: MIDV-500 teaches
   general passport-vs-card visual cues (MRZ presence, aspect ratio) across
   many designs; fine-tuning specializes the model to Philippine document
   appearance specifically.
3. Evaluate on a held-out split of Philippine synthetic data, AND on the
   real Philippine holdout set (Section 4.3) if available. Report both.
4. Export to ONNX, verify exported predictions match the source PyTorch
   model on a held-out batch.

### Phase 3 acceptance criteria

- [ ] `models/type_classifier.onnx` exists and is verified against the
      source model.
- [ ] Accuracy on Philippine synthetic test data and (if available) real
      holdout data are both reported in `RESULTS.md`, separately.
- [ ] Accuracy exceeds a naive baseline (e.g., aspect-ratio-only
      thresholding) — if it doesn't, a learned classifier isn't earning its
      complexity here and this should be reported honestly rather than
      shipped anyway.

---

## 7. Phase 4 — Corner/boundary detector

Follow the same architecture, training, and evaluation approach as the
prior plan's Track B (MobileNetV3-small + 8-value corner regression head,
MSE loss, template/background-disjoint train/test split), with these
changes:

1. **Training data is now MIDV-500 (pretrain) + Philippine synthetic data
   (fine-tune)** — same rationale as Phase 3: real-world capture diversity
   from MIDV-500, then specialization to Philippine document appearance and
   the specific top-edge-degradation case.
2. Evaluate on: Philippine synthetic test split, MIDV-500's own held-out
   split (sanity check that fine-tuning didn't catastrophically forget the
   general capability), and the real Philippine holdout set.
3. **The specific comparison that matters most** (same as the prior plan):
   does this model outperform the classical `detectPartialEdgeCorners`
   fallback specifically on top-edge-degraded Philippine passports and on
   steep capture angles? Report this explicitly, sliced exactly this way,
   in `RESULTS.md`.
4. Export to ONNX, verify against the source model, and define
   `isPlausibleQuadrilateral()` bounds as in the prior plan (convexity,
   area range, aspect-ratio tolerance per document type — now informed by
   Phase 3's type classification, since the correct aspect-ratio tolerance
   depends on knowing which document type you're checking against).

### Phase 4 acceptance criteria

Same as the prior plan's Track B criteria: exported and verified model,
real-holdout metrics reported separately from synthetic, the top-edge/
steep-angle comparison reported explicitly, and an honest report (not
silent shipping) if the model underperforms the classical baseline on real
data.

**Decision requiring human input:** same as the prior plan — if real-holdout
performance is meaningfully worse than synthetic-test performance, pause
before Phase 6 and propose options (more real training data, better
synthetic realism, or deferring this phase and shipping the classical
`detectPartialEdgeCorners` as the production cropping method while Phase 5
proceeds independently).

---

## 8. Phase 5 — Quality (readability) classifier — THE PRIMARY DELIVERABLE

This is what the project is actually for. Treat its evaluation rigor
accordingly — do not let it inherit a rushed pace from earlier phases.

1. Build `train_quality_classifier.py` per the model choice justified in
   Phase 2's EDA (5.3) — likely gradient-boosted trees over the four
   existing features, per the reasoning in `ML_MODEL_GUIDE.md` Section 5,
   unless EDA showed a reason to choose otherwise.
2. **Input images must be cropped/background-removed before features are
   computed** — use Phase 4's learned corner detector if it passed its
   acceptance criteria, otherwise fall back to the existing classical
   pipeline (`extractDocument`). Either way, the quality features must be
   computed on the *document*, not the raw uncropped frame, exactly as the
   existing production code does today.
3. Train on Philippine synthetic data (both document types, combined or
   separate per Phase 2's 5.3 recommendation). Evaluate on a held-out
   synthetic split.
4. **Primary evaluation: the real Philippine holdout set with human-
   reviewed readability labels** (Section 4.3). This is the number that
   actually answers the project's stated goal — report it prominently and
   separately from synthetic-test numbers, and do not let a good synthetic
   score substitute for this if the real evaluation is weak or unavailable.
5. Report precision/recall/F1 and a precision-recall curve, with a
   recommended operating threshold and justification (asymmetric cost of
   false negatives — a bad scan getting through — vs. false positives — an
   unnecessary retake — same reasoning as the prior plan).
6. Export to ONNX (or hand-translate a small tree ensemble to plain JS, per
   `ML_MODEL_GUIDE.md` Section 4's guidance), verify exported predictions
   match the source model.

### Phase 5 acceptance criteria

- [ ] `models/quality_classifier.onnx` exists and is verified against the
      source model.
- [ ] Real-holdout F1 (using human-reviewed labels, if available) is
      reported explicitly in `RESULTS.md`, separate from synthetic-test F1.
- [ ] F1 exceeds the current `DEFAULT_THRESHOLDS` baseline (computed the
      same way, over the same real holdout set) by a non-trivial margin —
      if evaluated only on synthetic-proxy labels because real labels
      weren't available, this must be stated as a limitation, not presented
      as equivalent evidence.
- [ ] Results are reported separately for ID cards and passports (per
      Phase 2's 5.3 recommendation on single-vs-per-type modeling), so a
      strong aggregate score can't hide weak performance on one document
      type.

---

## 9. Phase 6 — Integration into the JS pipeline

Only integrate a given model if its phase's acceptance criteria passed.
All integrations remain additive and opt-in, defaulting to the existing
classical behavior, exactly as in the prior plan.

1. **Type classifier (Phase 3):** load as an optional ONNX model; when
   enabled, run it before corner detection to auto-select
   `documentType`/`physicalSize` instead of requiring the caller to pass it
   in. Fall back to requiring the manual parameter if the model is
   unavailable.
2. **Corner detector (Phase 4):** wire in as a new leading tier ahead of
   `detectDocumentCorners`/`detectPartialEdgeCorners`, gated by
   `isPlausibleQuadrilateral()`, exactly as the prior plan's Section 8.2
   describes. Add `'ml-corners'` as a new `detectionMethod` value.
3. **Quality classifier (Phase 5):** load as an optional ONNX model behind
   a `useLearnedQualityModel` flag (default false); when enabled, replace
   the `DEFAULT_THRESHOLDS` comparison with the model's prediction using
   the operating threshold from Phase 5, step 5.
4. Update `demo.html` with toggles for all three models and display which
   type was detected, which detection method was used, and which quality
   path (learned vs. threshold) produced the report — extending the
   existing `detectionMethod` display rather than restructuring it.

### Phase 6 acceptance criteria

- [ ] All three integrations are behind explicit opt-in flags, defaulted
      off.
- [ ] `demo.html` can toggle each model independently and visibly shows
      which method/path was used for a given test image.
- [ ] Running the full existing test flow with all flags off produces
      identical output to the pre-project baseline.

---

## 10. Phase 7 — Final reporting

Write/finalize `training/RESULTS.md` summarizing, in this order of
emphasis (matching Section 0.1's priority):

1. **Quality classifier (Phase 5) results first and most prominently** —
   real-holdout F1 (or the flagged limitation if unavailable), per-document-
   type breakdown, comparison against the existing threshold baseline, and
   an explicit go/no-go recommendation. This is the section a reviewer
   should be able to read on its own and understand whether the project
   achieved its goal.
2. Type classifier (Phase 3) and corner detector (Phase 4) results,
   framed as supporting infrastructure — did they work well enough to
   productively feed Phase 5, and did the corner detector specifically
   improve on the passport top-edge/steep-angle case.
3. Data summary: MIDV-500 usage and license compliance confirmation,
   Philippine synthetic data volumes, real holdout set size and label
   provenance.
4. Known limitations and next steps — in particular, flag prominently if
   the real Philippine holdout set was small, synthetic-only, or missing
   human-reviewed readability labels, since that directly limits how much
   confidence to place in the primary deliverable's reported metrics.
5. A list of every decision point in this document resolved without
   explicit human sign-off, if any, flagged for retroactive review.

**Do not mark this project complete until `RESULTS.md` has been reviewed by
the human.** This document guides execution; it does not substitute for the
human's final judgment on whether to ship any of these models to
production, particularly the quality classifier given its direct role in
accepting or rejecting real users' document scans.
