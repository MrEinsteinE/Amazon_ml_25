import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import RobustScaler
import xgboost as xgb
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

def smape(y_true, y_pred):
    y_pred = np.clip(y_pred, 0.1, None)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    diff = np.abs(y_true - y_pred) / np.maximum(denominator, 1e-10)
    return 100 * np.mean(diff)

print("="*60)
print("MULTIMODAL MODEL - TEXT + IMAGE FEATURES")
print("="*60)

# Load text features (your existing pipeline)
from final_optimized_model import extract_features

train_df = pd.read_csv('../dataset/train.csv')
test_df = pd.read_csv('../dataset/test.csv')
y_train = train_df['price'].values

print("\n[1/3] Loading TEXT features...")
X_train_text_struct = extract_features(train_df).drop('sample_id', axis=1).values
X_test_text_struct = extract_features(test_df).drop('sample_id', axis=1).values

tfidf = TfidfVectorizer(max_features=3000, ngram_range=(1, 2), min_df=3, max_df=0.8)
tfidf_train = tfidf.fit_transform(train_df['catalog_content'].fillna(''))
tfidf_test = tfidf.transform(test_df['catalog_content'].fillna(''))

svd = TruncatedSVD(n_components=150, random_state=42)
tfidf_train_reduced = svd.fit_transform(tfidf_train)
tfidf_test_reduced = svd.transform(tfidf_test)

X_train_text = np.hstack([X_train_text_struct, tfidf_train_reduced])
X_test_text = np.hstack([X_test_text_struct, tfidf_test_reduced])
print(f"✓ Text features: {X_train_text.shape[1]}")

# Load IMAGE features
print("\n[2/3] Loading IMAGE features...")
X_train_image = np.load('../outputs/train_image_features.npy')
X_test_image = np.load('../outputs/test_image_features.npy')
print(f"✓ Image features: {X_train_image.shape[1]}")

# Reduce image dimensions (optional - for speed)
from sklearn.decomposition import PCA
pca = PCA(n_components=300, random_state=42)
X_train_image_reduced = pca.fit_transform(X_train_image)
X_test_image_reduced = pca.transform(X_test_image)
print(f"✓ Image PCA reduced to: {X_train_image_reduced.shape[1]}")

# COMBINE TEXT + IMAGE
print("\n[3/3] Combining TEXT + IMAGE...")
X_train_full = np.hstack([X_train_text, X_train_image_reduced])
X_test_full = np.hstack([X_test_text, X_test_image_reduced])
print(f"✓ TOTAL FEATURES: {X_train_full.shape[1]} (Text:{X_train_text.shape[1]} + Image:{X_train_image_reduced.shape[1]})")

# Scale
scaler = RobustScaler()
X_train_full = scaler.fit_transform(X_train_full)
X_test_full = scaler.transform(X_test_full)

# Log transform target
y_train_log = np.log1p(y_train)

# Split
X_tr, X_val, y_tr, y_val = train_test_split(X_train_full, y_train_log, test_size=0.15, random_state=42)
_, _, _, y_val_orig = train_test_split(X_train_full, y_train, test_size=0.15, random_state=42)

print("\n" + "="*60)
print("TRAINING MULTIMODAL ENSEMBLE")
print("="*60)

# XGBoost
print("\n[1/3] XGBoost...")
xgb_model = xgb.XGBRegressor(
    n_estimators=800, max_depth=10, learning_rate=0.03,
    subsample=0.8, colsample_bytree=0.65,
    tree_method='gpu_hist', gpu_id=0,
    early_stopping_rounds=75, random_state=42
)
xgb_model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
xgb_pred = np.expm1(xgb_model.predict(X_val))
xgb_smape = smape(y_val_orig, xgb_pred)
print(f"✓ XGBoost SMAPE: {xgb_smape:.4f}%")

# LightGBM
print("\n[2/3] LightGBM...")
lgb_model = lgb.LGBMRegressor(
    n_estimators=800, max_depth=10, learning_rate=0.03,
    subsample=0.8, colsample_bytree=0.65,
    device='gpu', random_state=42, verbose=-1
)
lgb_model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)])
lgb_pred = np.expm1(lgb_model.predict(X_val))
lgb_smape = smape(y_val_orig, lgb_pred)
print(f"✓ LightGBM SMAPE: {lgb_smape:.4f}%")

# CatBoost (try again with fixed params)
print("\n[3/3] CatBoost...")
try:
    from catboost import CatBoostRegressor
    cat_model = CatBoostRegressor(
        iterations=800, depth=10, learning_rate=0.03,
        subsample=0.8, bootstrap_type='Bernoulli',
        task_type='GPU', random_state=42, verbose=False
    )
    cat_model.fit(X_tr, y_tr, eval_set=(X_val, y_val))
    cat_pred = np.expm1(cat_model.predict(X_val))
    cat_smape = smape(y_val_orig, cat_pred)
    print(f"✓ CatBoost SMAPE: {cat_smape:.4f}%")
    use_cat = True
except:
    print("✗ CatBoost failed, using 2-model ensemble")
    use_cat = False

# Ensemble
if use_cat:
    weights = np.array([1/xgb_smape, 1/lgb_smape, 1/cat_smape])
    weights = weights / weights.sum()
    ens_pred = weights[0]*xgb_pred + weights[1]*lgb_pred + weights[2]*cat_pred
else:
    weights = np.array([1/xgb_smape, 1/lgb_smape])
    weights = weights / weights.sum()
    ens_pred = weights[0]*xgb_pred + weights[1]*lgb_pred

ens_smape = smape(y_val_orig, ens_pred)

print(f"\n✓ MULTIMODAL ENSEMBLE SMAPE: {ens_smape:.4f}%")
print(f"✓ Improvement from text-only: {58.45 - ens_smape:.2f}%")
print(f"✓ Expected leaderboard score: ~{ens_smape + 3:.1f}% (with val/test gap)")

# Train full models
print("\nTraining on full data...")
xgb_full = xgb.XGBRegressor(n_estimators=700, max_depth=10, learning_rate=0.03,
    tree_method='gpu_hist', gpu_id=0, random_state=42)
xgb_full.fit(X_train_full, y_train_log, verbose=False)

lgb_full = lgb.LGBMRegressor(n_estimators=700, max_depth=10, learning_rate=0.03,
    device='gpu', random_state=42, verbose=-1)
lgb_full.fit(X_train_full, y_train_log)

# Predictions
xgb_test = np.expm1(xgb_full.predict(X_test_full))
lgb_test = np.expm1(lgb_full.predict(X_test_full))

if use_cat:
    cat_full = CatBoostRegressor(iterations=700, depth=10, learning_rate=0.03,
        subsample=0.8, bootstrap_type='Bernoulli',
        task_type='GPU', random_state=42, verbose=False)
    cat_full.fit(X_train_full, y_train_log)
    cat_test = np.expm1(cat_full.predict(X_test_full))
    final_pred = weights[0]*xgb_test + weights[1]*lgb_test + weights[2]*cat_test
else:
    final_pred = weights[0]*xgb_test + weights[1]*lgb_test

# Save
submission = pd.DataFrame({
    'sample_id': test_df['sample_id'],
    'price': final_pred
})
submission.to_csv('../outputs/test_out_multimodal_FINAL.csv', index=False)

print(f"\n✓ SAVED: ../outputs/test_out_multimodal_FINAL.csv")
print(f"✓ Target leaderboard rank: TOP 50-100")
print("="*60)
