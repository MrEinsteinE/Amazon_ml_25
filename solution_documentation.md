# Amazon ML Challenge 2025 - Solution Documentation

**Team Name:** Visioneers

## 1. Methodology

### Approach
Multimodal price prediction using text feature engineering and gradient boosting ensemble.

### Key Insight
Item Pack Quantity (IPQ) and numeric value extraction from product descriptions are the strongest price predictors.

## 2. Model Architecture

### Models Used
- **XGBoost** (GPU-accelerated, tree_method='gpu_hist')
- **LightGBM** (GPU device)
- **CatBoost** (GPU task_type)

### Ensemble Strategy
5-Fold Cross-Validation with weighted averaging based on inverse SMAPE scores.

## 3. Feature Engineering

### Text Features (400+ features)
**Structured Features:**
- Text statistics (length, word count, character ratios)
- Numeric value extraction (max, min, avg, median, sum, std)
- Item Pack Quantity (IPQ) with polynomial transforms
- Brand detection (30+ brands)
- Category classification (6 categories)
- Unit measurements (volume, weight, dimension, power)
- Quality indicators (premium, luxury, etc.)
- Interaction features (IPQ × category, brand × electronics, etc.)

**NLP Features:**
- Word TF-IDF (4000 features, 1-3 grams) → SVD 200D
- Character TF-IDF (2000 features, 3-6 grams) → SVD 100D  
- Count Vectorizer (1000 features) → SVD 50D
- NMF decomposition (50 components)

### Preprocessing
- RobustScaler for outlier resilience
- Log transformation of target variable
- Missing value imputation

## 4. Hyperparameters

**XGBoost:**
n_estimators=1000, max_depth=11, learning_rate=0.02,
subsample=0.75, colsample_bytree=0.6,
min_child_weight=2, gamma=0.15,
reg_alpha=0.2, reg_lambda=2.5,
tree_method='gpu_hist', early_stopping_rounds=100

text

**LightGBM:**
n_estimators=1000, max_depth=11, learning_rate=0.02,
subsample=0.75, colsample_bytree=0.6,
min_child_weight=2, reg_alpha=0.2, reg_lambda=2.5,
device='gpu'

text

**CatBoost:**
iterations=1000, depth=11, learning_rate=0.02,
subsample=0.75, bootstrap_type='Bernoulli',
reg_lambda=2.5, task_type='GPU'

text

## 5. Training Details

- **Cross-Validation:** 5-Fold Stratified KFold
- **Metric:** SMAPE (Symmetric Mean Absolute Percentage Error)
- **Hardware:** NVIDIA RTX 3070 Ti (8GB VRAM)
- **Training Time:** ~2 hours for full pipeline
- **Out-of-Fold SMAPE:** [Your OOF Score]%

## 6. Key Learnings

1. **IPQ is critical:** Item Pack Quantity explains 20-30% of price variance
2. **Polynomial features matter:** IPQ², IPQ³, sqrt(IPQ) improve performance
3. **Interaction features:** IPQ × category, brand × electronics are powerful
4. **K-Fold ensemble:** More stable than single train/val split
5. **GPU acceleration:** Essential for 1000+ tree models

## 7. Future Improvements

- Image feature extraction (attempted, 79% failure rate due to throttling)
- Vision-Language Models (Qwen2-VL, CLIP)
- Stacked ensembling with meta-learner
- Custom SMAPE loss function for direct optimization

## 8. Code Repository

All code available at: [GitHub link if applicable]

**Main Files:**
- `advanced_text_optimization.py` - Final model
- `extract_features()` - Feature engineering
- `outputs/test_out_ULTIMATE.csv` - Final submission

## 9. References

- XGBoost Documentation: https://xgboost.readthedocs.io/
- LightGBM Documentation: https://lightgbm.readthedocs.io/