# Amazon ML Challenge 2025 — Smart Product Pricing

End-to-end multimodal price prediction pipeline built for the Amazon ML Challenge 2025, covering data setup, rich text feature engineering, GPU-accelerated model training, and submission generation with strict compliance to the competition rules.

## Highlights

- Trained locally on an RTX 3070 Ti with CUDA-enabled XGBoost/LightGBM for fast experimentation and iteration.
- Extensive text feature engineering (IPQ, numeric parsing, brand/category detectors, TF‑IDF, char n‑grams, interactions) optimized for SMAPE.
- Submission packaging and verification scripts aligned with challenge formatting constraints.

## Results

| Split | SMAPE | Notes |
|---|---|---|
| Public Leaderboard | 58.450% | Current deployed submission |
| Public Rank | 2800 | Latest visible rank reported on the board |

The objective metric is SMAPE, bounded in [0%, 200%], defined as $$ \text{SMAPE} = \frac{1}{n} \sum_{t=1}^{n} \frac{|F_t - A_t|}{(|A_t| + |F_t|)/2} $$ and commonly implemented with absolute values in the denominator, as used in this project’s evaluation pipeline.

## Problem

Predict product price from product catalog text and image links for 75k train and 75k test samples while avoiding any external price lookup or augmentation, with strict file formatting for submission.

## Data

- train.csv: sample_id, catalog_content, image_link, price (target).
- test.csv: sample_id, catalog_content, image_link (no price).
- Predictions must be positive floats with exactly one row per test sample_id.

## Metric

- Primary metric: SMAPE (symmetric mean absolute percentage error) with the accepted modern denominator using absolute values, ensuring bounds and symmetry for over/under-estimation, which is well-suited for wide price ranges.
- Internal evaluation functions compute SMAPE consistently with the leaderboard behavior to ensure reproducibility and sanity checks before submission.

## Hardware and Environment

- GPU: NVIDIA GeForce RTX 3070 Ti configured for CUDA acceleration with GPU histogram tree building for XGBoost training and inference.
- Python 3.10 with PyTorch, scikit‑learn, XGBoost, LightGBM, CatBoost, and TorchVision installed via conda/pip for a reproducible environment.

## Repository Structure

- src/baseline_model.py — Text-only baseline with hand-crafted features and GPU XGBoost for quick iterations.
- src/ensemble_simple.py — Two-model ensemble (XGBoost + LightGBM) on engineered + TF‑IDF features for variance reduction.
- src/advanced_text_optimization.py — K‑Fold ensemble with expanded feature space (word/char TF‑IDF, count vectors, SVD/NMF, interactions) and weighted blending, designed for maximal text‑only performance.
- src/final_multimodal_model.py — Text + image feature fusion with PCA compression over image embeddings and tree ensembles for regression.
- src/fast_image_extraction.py — Parallel image downloader + ResNet‑50 feature extractor with GPU batching for 2048‑D embeddings per image.
- src/utils.py — Reusable helpers for submission verification, logging, and safe IO.

## Installation

- Create a fresh conda environment, install GPU‑enabled XGBoost/LightGBM, and core dependencies (see requirements.txt) to ensure consistent GPU acceleration and library versions.
- Verify CUDA availability with torch.cuda.is_available() and confirm GPU histogram is used by XGBoost via the tree_method/device parameters.

## Reproducible Workflow

1) Data check and EDA  
- Validate schema, missingness, extreme values, and distribution skews for price to select stable transformations and interaction candidates.

2) Text feature engineering  
- Structured features: text length, word counts, uppercase/digit/special char ratios for stylistic signals.
- Numeric mining: extract all quantities (max, min, median, mean, sum, std, range) from catalog_content for size/capacity cues.
- IPQ: robust multi‑pattern pack quantity extraction with polynomial and root transforms to capture nonlinearity.
- Brands/categories/units/quality: indicator grids and counts for brand power, product type, measurement presence, and premium cues.
- NLP: word TF‑IDF (1–3 grams), char TF‑IDF (3–6 grams), count vectors with SVD/NMF reductions to compact semantics.
- Interactions: IPQ × category/units, brand × electronics, numeric × IPQ, and text_len × word_count for boosted separability.

3) Modeling and validation  
- GPU XGBoost with hist tree method for efficient training on high‑dimensional feature unions, early stopping, and log‑space targets when beneficial.
- GPU LightGBM ensemble for complementary splits with similar regularization and early stopping schedules.
- Optional CatBoost on GPU with Bernoulli bootstrap when using subsample to avoid bootstrap incompatibilities.
- K‑Fold ensembling with inverse‑SMAPE weighting for robust blending and stable generalization.

4) Multimodal fusion (when images succeed)  
- ResNet‑50 global pooled features (2048‑D), PCA reduction to ~300‑D, concatenated with text blocks, and trained with the same ensemble strategy.
- Network throttling and host blocking can cause high download failure rates; implement retry/backoff, user‑agent rotation, and reduced parallelism to increase success.

5) Submission integrity  
- Ensure exactly 75,000 predictions, positive float prices, and exact column order sample_id,price, with a verification script before upload.

## How to Run

- Baseline (quick): python src/baseline_model.py → outputs/test_out.csv for a fast sanity‑check submission.
- Text ensemble: python src/ensemble_simple.py → outputs/test_out_ensemble.csv for improved stability and leaderboard performance.
- Advanced text: python src/advanced_text_optimization.py → outputs/test_out_ULTIMATE.csv for the strongest text‑only run with K‑Fold ensemble.
- Multimodal: python src/final_multimodal_model.py after extracting embeddings to outputs/*.npy for text+image fusion.

## Submission Verification

- A small utility confirms column names, row count, unique sample_id coverage, positivity, and file size to avoid evaluation rejection on the portal.
- The submission CSV format is strictly sample_id,price with one row per test sample id; any mismatch in count or ordering invalidates evaluation.

## Key Learnings

- IPQ and related interactions contribute disproportionately to price variance capture in text‑only regimes and should be engineered early and validated with importances.
- Word + character n‑grams encode brand/style cues and noisy text, and SVD/NMF compactions keep GPU tree training tractable without severe degradation, especially with log targets.
- GPU histogram tree methods materially reduce wall‑clock times on 75k × 200–500D fused matrices and make iterative leaderboard cycles feasible on a single 3070 Ti.

## Known Issues and Mitigations

- Image throttling and failures can be significant on large parallel downloads; reduce workers, use exponential backoff, rotate user‑agents, and checkpoint embeddings to improve yield.
- Public vs private/test drift can widen SMAPE gaps; prefer K‑Fold OOF tracking and inverse‑SMAPE weighted blends to improve leaderboard stability.

## Roadmap

- Improve image success rate with robust fetching strategy and increase coverage above minimal thresholds for meaningful multimodal gains.
- Add CLIP/VLM embeddings and shallow fusion to capture visual semantics with fewer network requests when feasible under rules, without external price lookups.
- Explore direct SMAPE‑aware objectives or calibrated post‑processing to better match leaderboard metric behavior across price bands.

## License

- MIT or Apache‑2.0 compatible with the competition constraints and typical open ML tooling licenses used here.

## Acknowledgments

- Thanks to open‑source maintainers of XGBoost/LightGBM/CatBoost and the broader GPU ecosystem that made rapid iteration practical on local hardware.

## Appendix — Core Parameters

- XGBoost: device=cuda, tree_method hist/gpu_hist, max_depth 8–12, n_estimators 600–1000, early_stopping_rounds 50–100, regularization tuned for tabular+TF‑IDF blends.
- LightGBM: device gpu, similar depth/learning rate, subsample/colsample aligned with XGBoost to diversify splitting patterns in the ensemble.
- SMAPE reference: denominator uses absolute values, bounded and symmetric, suitable for large relative errors common in price prediction tasks.

This README consolidates the work completed for Amazon ML Challenge 2025, documents the current public SMAPE of 58.450%, and records the latest public rank of 2800 for transparent tracking and further iteration planning.
