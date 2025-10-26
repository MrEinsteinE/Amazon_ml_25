import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import RobustScaler
import xgboost as xgb
import lightgbm as lgb
import re
import warnings
warnings.filterwarnings('ignore')

def smape(y_true, y_pred):
    y_pred = np.clip(y_pred, 0.1, None)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    diff = np.abs(y_true - y_pred) / np.maximum(denominator, 1e-10)
    return 100 * np.mean(diff)

def extract_features(df):
    features = pd.DataFrame()
    features['sample_id'] = df['sample_id']
    df['catalog_content'] = df['catalog_content'].fillna('').astype(str)
    
    features['text_length'] = df['catalog_content'].str.len()
    features['word_count'] = df['catalog_content'].str.split().str.len()
    features['avg_word_len'] = df['catalog_content'].apply(lambda x: np.mean([len(w) for w in str(x).split()]) if str(x).split() else 0)
    features['uppercase_ratio'] = df['catalog_content'].apply(lambda x: sum(1 for c in x if c.isupper()) / max(len(x), 1))
    features['digit_ratio'] = df['catalog_content'].apply(lambda x: sum(1 for c in x if c.isdigit()) / max(len(x), 1))
    
    def extract_numbers(text):
        numbers = re.findall(r'\d+\.?\d*', str(text))
        return [float(n) for n in numbers if 0 < float(n) < 1000000]
    
    df['numbers'] = df['catalog_content'].apply(extract_numbers)
    features['num_count'] = df['numbers'].apply(len)
    features['max_number'] = df['numbers'].apply(lambda x: max(x) if x else 0)
    features['min_number'] = df['numbers'].apply(lambda x: min(x) if x else 0)
    features['avg_number'] = df['numbers'].apply(lambda x: np.mean(x) if x else 0)
    features['sum_numbers'] = df['numbers'].apply(lambda x: sum(x) if x else 0)
    features['std_numbers'] = df['numbers'].apply(lambda x: np.std(x) if len(x) > 1 else 0)
    features['median_number'] = df['numbers'].apply(lambda x: np.median(x) if x else 0)
    
    features['log_max_num'] = np.log1p(features['max_number'])
    features['log_avg_num'] = np.log1p(features['avg_number'])
    features['log_sum_num'] = np.log1p(features['sum_numbers'])
    
    def extract_ipq(text):
        patterns = [r'pack\s*of\s*(\d+)', r'(\d+)\s*pack', r'quantity[\s:]*(\d+)',
                   r'count[\s:]*(\d+)', r'(\d+)\s*pieces?', r'set\s*of\s*(\d+)',
                   r'(\d+)\s*units?', r'(\d+)\s*items?']
        for pattern in patterns:
            match = re.search(pattern, str(text).lower())
            if match:
                qty = int(match.group(1))
                return min(qty, 1000)
        return 1
    
    features['ipq'] = df['catalog_content'].apply(extract_ipq)
    features['ipq_log'] = np.log1p(features['ipq'])
    features['ipq_squared'] = features['ipq'] ** 2
    features['ipq_sqrt'] = np.sqrt(features['ipq'])
    
    brands = ['amazon', 'sony', 'samsung', 'apple', 'lg', 'nike', 'adidas',
              'hp', 'dell', 'lenovo', 'asus', 'panasonic', 'philips', 'bosch']
    for brand in brands:
        features[f'brand_{brand}'] = df['catalog_content'].str.lower().str.contains(brand, regex=False).astype(int)
    features['has_brand'] = (features[[f'brand_{b}' for b in brands]].sum(axis=1) > 0).astype(int)
    
    categories = {
        'electronics': ['phone', 'laptop', 'tablet', 'camera', 'tv', 'headphone', 'charger', 'speaker'],
        'clothing': ['shirt', 'pant', 'dress', 'shoe', 'jacket', 'jeans'],
        'home': ['furniture', 'bed', 'table', 'chair', 'lamp', 'kitchen'],
        'beauty': ['cream', 'shampoo', 'soap', 'perfume', 'makeup'],
        'food': ['coffee', 'tea', 'snack', 'chocolate', 'protein', 'vitamin'],
        'sports': ['gym', 'fitness', 'yoga', 'dumbbell', 'cycle']
    }
    for cat, keywords in categories.items():
        features[f'cat_{cat}'] = df['catalog_content'].str.lower().apply(lambda x: int(any(kw in str(x) for kw in keywords)))
    
    features['has_volume'] = df['catalog_content'].str.lower().str.contains('ml|liter|litre|oz', regex=True).astype(int)
    features['has_weight'] = df['catalog_content'].str.lower().str.contains('kg|gram|gm|pound', regex=True).astype(int)
    features['has_dimension'] = df['catalog_content'].str.lower().str.contains('cm|inch|mm|meter', regex=True).astype(int)
    features['has_power'] = df['catalog_content'].str.lower().str.contains('watt|volt|amp|mah', regex=True).astype(int)
    
    features['ipq_x_volume'] = features['ipq'] * features['has_volume']
    features['ipq_x_weight'] = features['ipq'] * features['has_weight']
    
    return features

print("="*60)
print("FINAL OPTIMIZED MODEL v2 - BUG FIXED")
print("="*60)

train_df = pd.read_csv('../dataset/train.csv')
test_df = pd.read_csv('../dataset/test.csv')
print(f"✓ Training: {len(train_df):,} | Test: {len(test_df):,}")
print(f"Price range: ${train_df['price'].min():.2f} - ${train_df['price'].max():.2f}")

X_train_struct = extract_features(train_df)
X_test_struct = extract_features(test_df)
y_train_full = train_df['price'].values

tfidf = TfidfVectorizer(max_features=3000, ngram_range=(1, 2), min_df=3, max_df=0.8)
tfidf_train = tfidf.fit_transform(train_df['catalog_content'].fillna('').astype(str))
tfidf_test = tfidf.transform(test_df['catalog_content'].fillna('').astype(str))

svd = TruncatedSVD(n_components=150, random_state=42)
tfidf_train_reduced = svd.fit_transform(tfidf_train)
tfidf_test_reduced = svd.transform(tfidf_test)

train_ids = X_train_struct['sample_id'].values
test_ids = X_test_struct['sample_id'].values
X_train_full = np.hstack([X_train_struct.drop('sample_id', axis=1).values, tfidf_train_reduced])
X_test_full = np.hstack([X_test_struct.drop('sample_id', axis=1).values, tfidf_test_reduced])

print(f"✓ TOTAL: {X_train_full.shape[1]} features")

scaler = RobustScaler()
X_train_full = scaler.fit_transform(X_train_full)
X_test_full = scaler.transform(X_test_full)

# Test transforms
y_train_log = np.log1p(y_train_full)
X_tr, X_vl, y_tr_log, y_vl_log = train_test_split(X_train_full, y_train_log, test_size=0.15, random_state=42)
_, _, y_tr_orig, y_vl_orig = train_test_split(X_train_full, y_train_full, test_size=0.15, random_state=42)

print("\nTesting LOG transform...")
xgb_test = xgb.XGBRegressor(n_estimators=600, max_depth=8, learning_rate=0.05,
    tree_method='gpu_hist', gpu_id=0, random_state=42)
xgb_test.fit(X_tr, y_tr_log, verbose=False)
pred_log = np.expm1(xgb_test.predict(X_vl))
smape_log = smape(y_vl_orig, pred_log)
print(f"✓ LOG SMAPE: {smape_log:.4f}%")

print("\nUsing LOG transform for final model...")

# Final ensemble
X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_log, test_size=0.15, random_state=42)
_, _, _, y_val_prices = train_test_split(X_train_full, y_train_full, test_size=0.15, random_state=42)

xgb_final = xgb.XGBRegressor(n_estimators=700, max_depth=9, learning_rate=0.04,
    tree_method='gpu_hist', gpu_id=0, early_stopping_rounds=50, random_state=42)
xgb_final.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

lgb_final = lgb.LGBMRegressor(n_estimators=700, max_depth=9, learning_rate=0.04,
    device='gpu', random_state=42, verbose=-1)
lgb_final.fit(X_train, y_train)

xgb_pred = np.expm1(xgb_final.predict(X_val))
lgb_pred = np.expm1(lgb_final.predict(X_val))

xgb_sm = smape(y_val_prices, xgb_pred)
lgb_sm = smape(y_val_prices, lgb_pred)

print(f"✓ XGBoost: {xgb_sm:.4f}%")
print(f"✓ LightGBM: {lgb_sm:.4f}%")

w = np.array([1/xgb_sm, 1/lgb_sm])
w = w / w.sum()
ens_pred = w[0] * xgb_pred + w[1] * lgb_pred
ens_sm = smape(y_val_prices, ens_pred)

print(f"\n✓ ENSEMBLE: {ens_sm:.4f}%")

# Full training
xgb_full = xgb.XGBRegressor(n_estimators=600, max_depth=9, learning_rate=0.04,
    tree_method='gpu_hist', gpu_id=0, random_state=42)
xgb_full.fit(X_train_full, y_train_log, verbose=False)

lgb_full = lgb.LGBMRegressor(n_estimators=600, max_depth=9, learning_rate=0.04,
    device='gpu', random_state=42, verbose=-1)
lgb_full.fit(X_train_full, y_train_log)

xgb_test_pred = np.expm1(xgb_full.predict(X_test_full))
lgb_test_pred = np.expm1(lgb_full.predict(X_test_full))
final_pred = w[0] * xgb_test_pred + w[1] * lgb_test_pred

submission = pd.DataFrame({'sample_id': test_ids, 'price': final_pred})
submission.to_csv('../outputs/test_out_final.csv', index=False)

print(f"\n✓ Saved: ../outputs/test_out_final.csv")
print(f"Expected SMAPE: ~{ens_sm:.1f}%")
print("="*60)
