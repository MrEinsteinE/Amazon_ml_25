import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import TruncatedSVD, NMF
from sklearn.preprocessing import RobustScaler, QuantileTransformer
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor
import re
import warnings
warnings.filterwarnings('ignore')

def smape(y_true, y_pred):
    y_pred = np.clip(y_pred, 0.1, None)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    diff = np.abs(y_true - y_pred) / np.maximum(denominator, 1e-10)
    return 100 * np.mean(diff)

def extract_mega_features(df):
    """Extract MAXIMUM features from text"""
    features = pd.DataFrame()
    features['sample_id'] = df['sample_id']
    df['catalog_content'] = df['catalog_content'].fillna('').astype(str)
    
    # Basic text
    features['text_length'] = df['catalog_content'].str.len()
    features['word_count'] = df['catalog_content'].str.split().str.len()
    features['avg_word_len'] = df['catalog_content'].apply(lambda x: np.mean([len(w) for w in str(x).split()]) if str(x).split() else 0)
    features['unique_words'] = df['catalog_content'].apply(lambda x: len(set(str(x).lower().split())))
    features['uppercase_ratio'] = df['catalog_content'].apply(lambda x: sum(1 for c in x if c.isupper()) / max(len(x), 1))
    features['digit_ratio'] = df['catalog_content'].apply(lambda x: sum(1 for c in x if c.isdigit()) / max(len(x), 1))
    features['special_ratio'] = df['catalog_content'].apply(lambda x: sum(1 for c in x if not c.isalnum() and not c.isspace()) / max(len(x), 1))
    
    # Numbers (CRITICAL for price)
    def extract_numbers(text):
        numbers = re.findall(r'\d+\.?\d*', str(text))
        return [float(n) for n in numbers if 0 < float(n) < 100000]
    
    df['numbers'] = df['catalog_content'].apply(extract_numbers)
    features['num_count'] = df['numbers'].apply(len)
    features['max_number'] = df['numbers'].apply(lambda x: max(x) if x else 0)
    features['min_number'] = df['numbers'].apply(lambda x: min(x) if x else 0)
    features['avg_number'] = df['numbers'].apply(lambda x: np.mean(x) if x else 0)
    features['median_number'] = df['numbers'].apply(lambda x: np.median(x) if x else 0)
    features['sum_numbers'] = df['numbers'].apply(lambda x: sum(x) if x else 0)
    features['std_numbers'] = df['numbers'].apply(lambda x: np.std(x) if len(x) > 1 else 0)
    features['range_numbers'] = df['numbers'].apply(lambda x: max(x) - min(x) if len(x) > 1 else 0)
    
    # Log transforms
    for col in ['max_number', 'avg_number', 'sum_numbers', 'median_number']:
        features[f'log_{col}'] = np.log1p(features[col])
        features[f'sqrt_{col}'] = np.sqrt(features[col])
    
    # IPQ - MOST IMPORTANT
    def extract_ipq(text):
        patterns = [
            r'pack\s*of\s*(\d+)', r'(\d+)\s*pack', r'quantity[\s:]*(\d+)',
            r'count[\s:]*(\d+)', r'(\d+)\s*pieces?', r'set\s*of\s*(\d+)',
            r'(\d+)\s*units?', r'(\d+)\s*items?', r'box\s*of\s*(\d+)',
            r'(\d+)\s*pcs', r'(\d+)pc'
        ]
        for pattern in patterns:
            match = re.search(pattern, str(text).lower())
            if match:
                return min(int(match.group(1)), 5000)
        return 1
    
    features['ipq'] = df['catalog_content'].apply(extract_ipq)
    features['ipq_log'] = np.log1p(features['ipq'])
    features['ipq_squared'] = features['ipq'] ** 2
    features['ipq_sqrt'] = np.sqrt(features['ipq'])
    features['ipq_cubed'] = features['ipq'] ** 3
    
    # Brands (expanded)
    brands = ['amazon', 'sony', 'samsung', 'apple', 'lg', 'nike', 'adidas', 'puma',
              'hp', 'dell', 'lenovo', 'asus', 'panasonic', 'philips', 'bosch', 'havells',
              'bajaj', 'mi', 'realme', 'vivo', 'oppo', 'oneplus', 'motorola', 'nokia',
              'canon', 'nikon', 'jbl', 'bose', 'logitech', 'tp-link', 'dlink']
    
    for brand in brands:
        features[f'brand_{brand}'] = df['catalog_content'].str.lower().str.contains(brand, regex=False).astype(int)
    
    features['brand_count'] = features[[f'brand_{b}' for b in brands]].sum(axis=1)
    features['has_premium_brand'] = df['catalog_content'].str.lower().str.contains('apple|sony|samsung|bose|canon', regex=True).astype(int)
    
    # Categories (expanded)
    categories = {
        'electronics': ['phone', 'laptop', 'tablet', 'camera', 'tv', 'headphone', 'charger', 'speaker', 'mouse', 'keyboard', 'monitor', 'smartwatch', 'earbuds', 'router', 'modem'],
        'clothing': ['shirt', 'pant', 'dress', 'shoe', 'jacket', 'jeans', 'tshirt', 't-shirt', 'trouser', 'sweater', 'hoodie', 'shorts', 'socks', 'underwear'],
        'home': ['furniture', 'bed', 'table', 'chair', 'lamp', 'kitchen', 'curtain', 'cushion', 'mattress', 'sofa', 'decor', 'rug', 'carpet'],
        'beauty': ['cream', 'shampoo', 'soap', 'perfume', 'makeup', 'cosmetic', 'lotion', 'serum', 'facewash', 'skincare', 'lipstick', 'nail'],
        'food': ['coffee', 'tea', 'snack', 'chocolate', 'protein', 'vitamin', 'supplement', 'oil', 'spice', 'organic', 'juice'],
        'sports': ['gym', 'fitness', 'yoga', 'dumbbell', 'treadmill', 'cycle', 'sports', 'exercise', 'workout', 'athletic', 'running']
    }
    
    for cat, keywords in categories.items():
        features[f'cat_{cat}'] = df['catalog_content'].str.lower().apply(lambda x: int(any(kw in str(x) for kw in keywords)))
        features[f'cat_{cat}_count'] = df['catalog_content'].str.lower().apply(lambda x: sum(kw in str(x) for kw in keywords))
    
    # Units with extraction
    features['has_volume'] = df['catalog_content'].str.lower().str.contains('ml|liter|litre|oz|gallon', regex=True).astype(int)
    features['has_weight'] = df['catalog_content'].str.lower().str.contains('kg|gram|gm|pound|lb|mg', regex=True).astype(int)
    features['has_dimension'] = df['catalog_content'].str.lower().str.contains('cm|inch|mm|meter|ft', regex=True).astype(int)
    features['has_power'] = df['catalog_content'].str.lower().str.contains('watt|volt|amp|mah|kwh', regex=True).astype(int)
    
    # Quality indicators
    quality_words = ['premium', 'luxury', 'professional', 'pro', 'deluxe', 'ultra', 'hd', '4k', '8k',
                     'wireless', 'smart', 'original', 'genuine', 'certified', 'warranty']
    for word in quality_words:
        features[f'quality_{word}'] = df['catalog_content'].str.lower().str.contains(word, regex=False).astype(int)
    
    # Interactions (CRITICAL)
    features['ipq_x_volume'] = features['ipq'] * features['has_volume']
    features['ipq_x_weight'] = features['ipq'] * features['has_weight']
    features['ipq_x_cat_electronics'] = features['ipq'] * features['cat_electronics']
    features['ipq_x_cat_food'] = features['ipq'] * features['cat_food']
    features['brand_x_electronics'] = features['brand_count'] * features['cat_electronics']
    features['text_len_x_word_count'] = features['text_length'] * features['word_count']
    features['max_num_x_ipq'] = features['max_number'] * features['ipq']
    features['avg_num_x_ipq'] = features['avg_number'] * features['ipq']
    
    return features

print("="*60)
print("ADVANCED TEXT OPTIMIZATION - FINAL PUSH")
print("="*60)

train_df = pd.read_csv('../dataset/train.csv')
test_df = pd.read_csv('../dataset/test.csv')
y_train = train_df['price'].values

print("\n[1/4] Extracting MEGA features...")
X_train_struct = extract_mega_features(train_df).drop('sample_id', axis=1).values
X_test_struct = extract_mega_features(test_df).drop('sample_id', axis=1).values
print(f"✓ Structured: {X_train_struct.shape[1]} features")

# Multiple TF-IDF representations
print("\n[2/4] Advanced TF-IDF...")
# Word TF-IDF
tfidf_word = TfidfVectorizer(max_features=4000, ngram_range=(1, 3), min_df=2, max_df=0.85)
tfidf_word_train = tfidf_word.fit_transform(train_df['catalog_content'].fillna(''))
tfidf_word_test = tfidf_word.transform(test_df['catalog_content'].fillna(''))

# Character TF-IDF (captures misspellings)
tfidf_char = TfidfVectorizer(max_features=2000, ngram_range=(3, 6), min_df=5, analyzer='char')
tfidf_char_train = tfidf_char.fit_transform(train_df['catalog_content'].fillna(''))
tfidf_char_test = tfidf_char.transform(test_df['catalog_content'].fillna(''))

# Count Vectorizer
count_vec = CountVectorizer(max_features=1000, ngram_range=(1, 2), min_df=3)
count_train = count_vec.fit_transform(train_df['catalog_content'].fillna(''))
count_test = count_vec.transform(test_df['catalog_content'].fillna(''))

print("\n[3/4] Dimensionality reduction...")
# SVD for TF-IDF
svd_word = TruncatedSVD(n_components=200, random_state=42)
svd_char = TruncatedSVD(n_components=100, random_state=42)
svd_count = TruncatedSVD(n_components=50, random_state=42)

tfidf_word_reduced = svd_word.fit_transform(tfidf_word_train)
tfidf_char_reduced = svd_char.fit_transform(tfidf_char_train)
count_reduced = svd_count.fit_transform(count_train)

tfidf_word_test_reduced = svd_word.transform(tfidf_word_test)
tfidf_char_test_reduced = svd_char.transform(tfidf_char_test)
count_test_reduced = svd_count.transform(count_test)

# NMF (alternative decomposition)
nmf = NMF(n_components=50, random_state=42, max_iter=500)
nmf_train = nmf.fit_transform(np.abs(tfidf_word_train.toarray()))
nmf_test = nmf.transform(np.abs(tfidf_word_test.toarray()))

print("\n[4/4] Combining ALL features...")
X_train_full = np.hstack([
    X_train_struct,
    tfidf_word_reduced,
    tfidf_char_reduced,
    count_reduced,
    nmf_train
])

X_test_full = np.hstack([
    X_test_struct,
    tfidf_word_test_reduced,
    tfidf_char_test_reduced,
    count_test_reduced,
    nmf_test
])

print(f"✓ TOTAL: {X_train_full.shape[1]} features")

# Robust scaling
scaler = RobustScaler()
X_train_full = scaler.fit_transform(X_train_full)
X_test_full = scaler.transform(X_test_full)

# Target
y_train_log = np.log1p(y_train)

# K-Fold Cross-Validation Ensemble
print("\n" + "="*60)
print("TRAINING K-FOLD ENSEMBLE (5 FOLDS)")
print("="*60)

kf = KFold(n_splits=5, shuffle=True, random_state=42)
oof_preds_xgb = np.zeros(len(X_train_full))
oof_preds_lgb = np.zeros(len(X_train_full))
oof_preds_cat = np.zeros(len(X_train_full))

test_preds_xgb = np.zeros(len(X_test_full))
test_preds_lgb = np.zeros(len(X_test_full))
test_preds_cat = np.zeros(len(X_test_full))

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train_full)):
    print(f"\nFold {fold+1}/5")
    
    X_tr, X_val = X_train_full[train_idx], X_train_full[val_idx]
    y_tr, y_val = y_train_log[train_idx], y_train_log[val_idx]
    
    # XGBoost
    xgb_model = xgb.XGBRegressor(
        n_estimators=1000, max_depth=11, learning_rate=0.02,
        subsample=0.75, colsample_bytree=0.6,
        min_child_weight=2, gamma=0.15, reg_alpha=0.2, reg_lambda=2.5,
        tree_method='gpu_hist', gpu_id=0,
        early_stopping_rounds=100, random_state=42+fold
    )
    xgb_model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
    
    oof_preds_xgb[val_idx] = xgb_model.predict(X_val)
    test_preds_xgb += xgb_model.predict(X_test_full) / 5
    
    # LightGBM
    lgb_model = lgb.LGBMRegressor(
        n_estimators=1000, max_depth=11, learning_rate=0.02,
        subsample=0.75, colsample_bytree=0.6,
        min_child_weight=2, reg_alpha=0.2, reg_lambda=2.5,
        device='gpu', random_state=42+fold, verbose=-1
    )
    lgb_model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)])
    
    oof_preds_lgb[val_idx] = lgb_model.predict(X_val)
    test_preds_lgb += lgb_model.predict(X_test_full) / 5
    
    # CatBoost
    cat_model = CatBoostRegressor(
        iterations=1000, depth=11, learning_rate=0.02,
        subsample=0.75, bootstrap_type='Bernoulli',
        reg_lambda=2.5, task_type='GPU',
        random_state=42+fold, verbose=False
    )
    cat_model.fit(X_tr, y_tr, eval_set=(X_val, y_val))
    
    oof_preds_cat[val_idx] = cat_model.predict(X_val)
    test_preds_cat += cat_model.predict(X_test_full) / 5

# Calculate OOF scores
oof_xgb_real = np.expm1(oof_preds_xgb)
oof_lgb_real = np.expm1(oof_preds_lgb)
oof_cat_real = np.expm1(oof_preds_cat)

xgb_smape = smape(y_train, oof_xgb_real)
lgb_smape = smape(y_train, oof_lgb_real)
cat_smape = smape(y_train, oof_cat_real)

print("\n" + "="*60)
print("OUT-OF-FOLD SCORES")
print("="*60)
print(f"XGBoost: {xgb_smape:.4f}%")
print(f"LightGBM: {lgb_smape:.4f}%")
print(f"CatBoost: {cat_smape:.4f}%")

# Weighted ensemble
weights = np.array([1/xgb_smape, 1/lgb_smape, 1/cat_smape])
weights = weights / weights.sum()

oof_ensemble = weights[0]*oof_xgb_real + weights[1]*oof_lgb_real + weights[2]*oof_cat_real
ensemble_smape = smape(y_train, oof_ensemble)

print(f"\n✓ ENSEMBLE OOF: {ensemble_smape:.4f}%")
print(f"✓ Improvement: {58.45 - ensemble_smape:.2f}% from current leaderboard")
print(f"✓ Weights: XGBoost={weights[0]:.3f}, LightGBM={weights[1]:.3f}, CatBoost={weights[2]:.3f}")

# Final predictions
test_preds_real_xgb = np.expm1(test_preds_xgb)
test_preds_real_lgb = np.expm1(test_preds_lgb)
test_preds_real_cat = np.expm1(test_preds_cat)

final_pred = weights[0]*test_preds_real_xgb + weights[1]*test_preds_real_lgb + weights[2]*test_preds_real_cat

submission = pd.DataFrame({
    'sample_id': test_df['sample_id'],
    'price': final_pred
})

submission.to_csv('../outputs/test_out_ULTIMATE.csv', index=False)

print(f"\n✓ SAVED: ../outputs/test_out_ULTIMATE.csv")
print(f"✓ Expected leaderboard: ~{ensemble_smape + 3:.1f}%")

if ensemble_smape + 3 <= 43:
    print(f"✓✓ TARGET ACHIEVED - TOP 50 POSSIBLE!")
else:
    print(f"⚠ Need {(ensemble_smape + 3) - 43:.1f}% more improvement for TOP 50")

print("="*60)
