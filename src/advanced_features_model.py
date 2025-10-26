import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import QuantileTransformer, PolynomialFeatures
import xgboost as xgb
import lightgbm as lgb
import re
import warnings
warnings.filterwarnings('ignore')

def smape(y_true, y_pred):
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    diff = np.abs(y_true - y_pred) / denominator
    diff[denominator == 0] = 0.0
    return 100 * np.mean(diff)

def winsorize_outliers(df, column='price', lower=0.01, upper=0.99):
    """Clip extreme values instead of removing"""
    lower_val = df[column].quantile(lower)
    upper_val = df[column].quantile(upper)
    df[column] = df[column].clip(lower_val, upper_val)
    print(f"  Winsorized to range: ${lower_val:.2f} - ${upper_val:.2f}")
    return df

def extract_advanced_features(df):
    """Extract MORE features with better engineering"""
    features = pd.DataFrame()
    features['sample_id'] = df['sample_id']
    df['catalog_content'] = df['catalog_content'].fillna('').astype(str)
    
    # Basic text features
    features['text_length'] = df['catalog_content'].str.len()
    features['word_count'] = df['catalog_content'].str.split().str.len()
    features['avg_word_length'] = df['catalog_content'].apply(lambda x: np.mean([len(w) for w in str(x).split()]) if len(str(x).split()) > 0 else 0)
    features['uppercase_ratio'] = df['catalog_content'].apply(lambda x: sum(1 for c in x if c.isupper()) / max(len(x), 1))
    features['digit_ratio'] = df['catalog_content'].apply(lambda x: sum(1 for c in x if c.isdigit()) / max(len(x), 1))
    features['special_char_ratio'] = df['catalog_content'].apply(lambda x: sum(1 for c in x if not c.isalnum() and not c.isspace()) / max(len(x), 1))
    
    # Advanced numeric extraction
    def extract_numbers(text):
        numbers = re.findall(r'\d+\.?\d*', str(text))
        return [float(n) for n in numbers if float(n) > 0]
    
    df['numbers'] = df['catalog_content'].apply(extract_numbers)
    features['num_count'] = df['numbers'].apply(len)
    features['max_number'] = df['numbers'].apply(lambda x: max(x) if x else 0)
    features['min_number'] = df['numbers'].apply(lambda x: min(x[1:]) if len(x) > 1 else 0)
    features['avg_number'] = df['numbers'].apply(lambda x: np.mean(x) if x else 0)
    features['median_number'] = df['numbers'].apply(lambda x: np.median(x) if x else 0)
    features['sum_numbers'] = df['numbers'].apply(lambda x: sum(x) if x else 0)
    features['std_numbers'] = df['numbers'].apply(lambda x: np.std(x) if len(x) > 1 else 0)
    features['range_numbers'] = df['numbers'].apply(lambda x: max(x) - min(x) if len(x) > 1 else 0)
    
    # Log transforms of numeric features
    features['log_max_number'] = np.log1p(features['max_number'])
    features['log_sum_numbers'] = np.log1p(features['sum_numbers'])
    features['log_avg_number'] = np.log1p(features['avg_number'])
    
    # IPQ with multiple patterns
    def extract_ipq(text):
        patterns = [
            r'pack\s*of\s*(\d+)', r'(\d+)\s*pack', r'quantity[\s:]*(\d+)',
            r'count[\s:]*(\d+)', r'(\d+)\s*pieces?', r'set\s*of\s*(\d+)',
            r'(\d+)\s*units?', r'(\d+)\s*items?', r'box\s*of\s*(\d+)'
        ]
        for pattern in patterns:
            match = re.search(pattern, str(text).lower())
            if match:
                return int(match.group(1))
        return 1
    
    features['ipq'] = df['catalog_content'].apply(extract_ipq)
    features['ipq_squared'] = features['ipq'] ** 2
    features['ipq_log'] = np.log1p(features['ipq'])
    features['ipq_sqrt'] = np.sqrt(features['ipq'])
    
    # Expanded brand detection
    brands = ['amazon', 'sony', 'samsung', 'apple', 'lg', 'nike', 'adidas', 'puma',
              'hp', 'dell', 'lenovo', 'asus', 'panasonic', 'philips', 'bosch', 'havells',
              'bajaj', 'mi', 'realme', 'vivo', 'oppo', 'oneplus', 'motorola', 'nokia']
    
    for brand in brands:
        features[f'brand_{brand}'] = df['catalog_content'].str.lower().str.contains(brand, regex=False).astype(int)
    features['brand_count'] = features[[f'brand_{b}' for b in brands]].sum(axis=1)
    features['has_premium_brand'] = df['catalog_content'].str.lower().str.contains('apple|sony|samsung', regex=True).astype(int)
    
    # Enhanced categories
    categories = {
        'electronics': ['phone', 'laptop', 'tablet', 'camera', 'tv', 'headphone', 'charger', 'speaker', 'mouse', 'keyboard', 'monitor', 'smartwatch', 'earbuds'],
        'clothing': ['shirt', 'pant', 'dress', 'shoe', 'jacket', 'jeans', 'tshirt', 't-shirt', 'trouser', 'sweater', 'hoodie', 'shorts'],
        'home': ['furniture', 'bed', 'table', 'chair', 'lamp', 'kitchen', 'curtain', 'cushion', 'mattress', 'sofa', 'decor'],
        'beauty': ['cream', 'shampoo', 'soap', 'perfume', 'makeup', 'cosmetic', 'lotion', 'serum', 'facewash', 'skincare'],
        'food': ['coffee', 'tea', 'snack', 'chocolate', 'protein', 'vitamin', 'supplement', 'oil', 'spice', 'organic'],
        'sports': ['gym', 'fitness', 'yoga', 'dumbbell', 'treadmill', 'cycle', 'sports', 'exercise', 'workout', 'athletic']
    }
    
    for category, keywords in categories.items():
        features[f'cat_{category}'] = df['catalog_content'].str.lower().apply(lambda x: int(any(kw in str(x) for kw in keywords)))
    features['category_count'] = features[[f'cat_{c}' for c in categories.keys()]].sum(axis=1)
    
    # Unit detection with values
    units = {
        'volume': ['ml', 'liter', 'litre', 'oz', 'gallon', 'l '],
        'weight': ['kg', 'gram', 'gm', 'g ', 'pound', 'lb', 'mg'],
        'dimension': ['cm', 'inch', 'mm', 'meter', 'ft', 'feet', 'm '],
        'power': ['watt', 'w ', 'volt', 'v ', 'amp', 'mah', 'kwh']
    }
    
    for unit_type, unit_list in units.items():
        features[f'has_{unit_type}'] = df['catalog_content'].str.lower().apply(lambda x: int(any(u in str(x) for u in unit_list)))
    
    # Material indicators (high price correlation)
    materials = ['leather', 'cotton', 'silk', 'wood', 'metal', 'plastic', 'glass', 'steel', 'aluminum', 'gold', 'silver']
    for material in materials:
        features[f'material_{material}'] = df['catalog_content'].str.lower().str.contains(material, regex=False).astype(int)
    
    # Quality indicators
    quality_keywords = ['premium', 'luxury', 'professional', 'pro', 'deluxe', 'ultra', 'hd', '4k', 'wireless', 'smart']
    for keyword in quality_keywords:
        features[f'quality_{keyword}'] = df['catalog_content'].str.lower().str.contains(keyword, regex=False).astype(int)
    
    # Interaction features (CRITICAL for boosting models)
    features['ipq_x_has_volume'] = features['ipq'] * features['has_volume']
    features['ipq_x_has_weight'] = features['ipq'] * features['has_weight']
    features['brand_count_x_cat_electronics'] = features['brand_count'] * features['cat_electronics']
    features['text_length_x_word_count'] = features['text_length'] * features['word_count']
    features['num_count_x_avg_number'] = features['num_count'] * features['avg_number']
    
    return features

print("="*60)
print("ADVANCED FEATURE ENGINEERING MODEL")
print("="*60)

train_df = pd.read_csv('../dataset/train.csv')
test_df = pd.read_csv('../dataset/test.csv')
print(f"✓ Loaded {len(train_df):,} training + {len(test_df):,} test")

# Winsorize instead of remove
print("\nWinsorizing outliers (keeping all data)...")
train_df = winsorize_outliers(train_df, 'price', lower=0.01, upper=0.99)

print("\nExtracting ADVANCED features...")
X_train_structured = extract_advanced_features(train_df)
X_test_structured = extract_advanced_features(test_df)
y_train_full = train_df['price'].values

print(f"✓ Structured features: {X_train_structured.shape[1] - 1}")

# Enhanced TF-IDF with character n-grams
print("\nEnhanced TF-IDF (word + char ngrams)...")
tfidf_word = TfidfVectorizer(max_features=3000, ngram_range=(1, 2), min_df=3, max_df=0.8, analyzer='word')
tfidf_char = TfidfVectorizer(max_features=2000, ngram_range=(3, 5), min_df=5, analyzer='char')

train_text = train_df['catalog_content'].fillna('').astype(str)
test_text = test_df['catalog_content'].fillna('').astype(str)

tfidf_word_train = tfidf_word.fit_transform(train_text)
tfidf_word_test = tfidf_word.transform(test_text)

tfidf_char_train = tfidf_char.fit_transform(train_text)
tfidf_char_test = tfidf_char.transform(test_text)

# SVD reduction
svd_word = TruncatedSVD(n_components=150, random_state=42)
svd_char = TruncatedSVD(n_components=50, random_state=42)

tfidf_word_reduced = svd_word.fit_transform(tfidf_word_train)
tfidf_char_reduced = svd_char.fit_transform(tfidf_char_train)

tfidf_word_test_reduced = svd_word.transform(tfidf_word_test)
tfidf_char_test_reduced = svd_char.transform(tfidf_char_test)

print(f"✓ Word TF-IDF: {tfidf_word_reduced.shape[1]} features")
print(f"✓ Char TF-IDF: {tfidf_char_reduced.shape[1]} features")

# Combine all features
train_ids = X_train_structured['sample_id'].values
test_ids = X_test_structured['sample_id'].values
X_train_struct_array = X_train_structured.drop('sample_id', axis=1).values
X_test_struct_array = X_test_structured.drop('sample_id', axis=1).values

X_train_full = np.hstack([X_train_struct_array, tfidf_word_reduced, tfidf_char_reduced])
X_test_full = np.hstack([X_test_struct_array, tfidf_word_test_reduced, tfidf_char_test_reduced])

print(f"\n✓ TOTAL FEATURES: {X_train_full.shape[1]}")

# Quantile transformation (robust to outliers)
print("\nApplying Quantile Transformation...")
qt = QuantileTransformer(output_distribution='normal', random_state=42)
X_train_full = qt.fit_transform(X_train_full)
X_test_full = qt.transform(X_test_full)

# Target transformation
y_train_log = np.log1p(y_train_full)

# Split
X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_log, test_size=0.15, random_state=42)
y_val_original = np.expm1(y_val)

print(f"Train: {len(X_train):,} | Val: {len(X_val):,}")

print("\n" + "="*60)
print("TRAINING OPTIMIZED ENSEMBLE")
print("="*60)

print("\n[1/2] XGBoost (deeper trees)...")
xgb_model = xgb.XGBRegressor(
    n_estimators=1000, max_depth=12, learning_rate=0.02,
    subsample=0.75, colsample_bytree=0.6, min_child_weight=2,
    gamma=0.2, reg_alpha=0.2, reg_lambda=2.0,
    tree_method='gpu_hist', gpu_id=0, predictor='gpu_predictor',
    early_stopping_rounds=75, random_state=42
)
xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
xgb_pred = np.expm1(xgb_model.predict(X_val))
xgb_smape = smape(y_val_original, xgb_pred)
print(f"✓ XGBoost SMAPE: {xgb_smape:.4f}%")

print("\n[2/2] LightGBM (dart booster)...")
lgb_model = lgb.LGBMRegressor(
    n_estimators=1000, max_depth=12, learning_rate=0.02,
    subsample=0.75, colsample_bytree=0.6, min_child_weight=2,
    reg_alpha=0.2, reg_lambda=2.0, boosting_type='dart',
    device='gpu', random_state=42, verbose=-1
)
lgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)])
lgb_pred = np.expm1(lgb_model.predict(X_val))
lgb_smape = smape(y_val_original, lgb_pred)
print(f"✓ LightGBM SMAPE: {lgb_smape:.4f}%")

# Ensemble
weights = np.array([1/xgb_smape, 1/lgb_smape])
weights = weights / weights.sum()
ensemble_pred = weights[0] * xgb_pred + weights[1] * lgb_pred
ensemble_smape = smape(y_val_original, ensemble_pred)

print(f"\n✓ ENSEMBLE SMAPE: {ensemble_smape:.4f}%")
print(f"✓ Improvement from baseline: {60.93 - ensemble_smape:.2f}%")

# Train on full data
print("\nTraining on full data...")
xgb_full = xgb.XGBRegressor(n_estimators=xgb_model.best_iteration if hasattr(xgb_model, 'best_iteration') else 700,
    max_depth=12, learning_rate=0.02, subsample=0.75, colsample_bytree=0.6,
    tree_method='gpu_hist', gpu_id=0, random_state=42)
xgb_full.fit(X_train_full, y_train_log, verbose=False)

lgb_full = lgb.LGBMRegressor(n_estimators=700, max_depth=12, learning_rate=0.02,
    boosting_type='dart', device='gpu', random_state=42, verbose=-1)
lgb_full.fit(X_train_full, y_train_log)

# Predict
xgb_test = np.expm1(xgb_full.predict(X_test_full))
lgb_test = np.expm1(lgb_full.predict(X_test_full))
ensemble_test = weights[0] * xgb_test + weights[1] * lgb_test

submission = pd.DataFrame({'sample_id': test_ids, 'price': ensemble_test})
submission.to_csv('../outputs/test_out_advanced.csv', index=False)

print(f"\n✓ Saved: ../outputs/test_out_advanced.csv")
print(f"✓ Price range: ${ensemble_test.min():.2f} - ${ensemble_test.max():.2f}")
print(f"\nExpected SMAPE: ~{ensemble_smape:.1f}%")
print("="*60)
