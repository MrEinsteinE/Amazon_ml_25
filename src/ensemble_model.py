import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
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

def remove_outliers(df, column='price', factor=2.5):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - factor * IQR
    upper = Q3 + factor * IQR
    mask = (df[column] >= lower) & (df[column] <= upper)
    print(f"  Removing {len(df) - mask.sum():,} outliers ({(len(df) - mask.sum())/len(df)*100:.2f}%)")
    return df[mask].copy()

def extract_quick_features(df):
    features = pd.DataFrame()
    features['sample_id'] = df['sample_id']
    df['catalog_content'] = df['catalog_content'].fillna('').astype(str)
    
    features['text_length'] = df['catalog_content'].str.len()
    features['word_count'] = df['catalog_content'].str.split().str.len()
    features['uppercase_ratio'] = df['catalog_content'].apply(lambda x: sum(1 for c in x if c.isupper()) / max(len(x), 1))
    features['digit_count'] = df['catalog_content'].str.count(r'\d')
    features['special_char_count'] = df['catalog_content'].str.count(r'[^a-zA-Z0-9\s]')
    
    def extract_numbers(text):
        numbers = re.findall(r'\d+\.?\d*', str(text))
        return [float(n) for n in numbers if float(n) > 0]
    
    df['numbers'] = df['catalog_content'].apply(extract_numbers)
    features['num_count'] = df['numbers'].apply(len)
    features['max_number'] = df['numbers'].apply(lambda x: max(x) if x else 0)
    features['min_number'] = df['numbers'].apply(lambda x: min(x[1:]) if len(x) > 1 else 0)
    features['avg_number'] = df['numbers'].apply(lambda x: np.mean(x) if x else 0)
    features['sum_numbers'] = df['numbers'].apply(lambda x: sum(x) if x else 0)
    features['std_numbers'] = df['numbers'].apply(lambda x: np.std(x) if len(x) > 1 else 0)
    
    def extract_ipq(text):
        patterns = [r'pack\s*of\s*(\d+)', r'(\d+)\s*pack', r'quantity[\s:]*(\d+)', 
                   r'count[\s:]*(\d+)', r'(\d+)\s*pieces?', r'set\s*of\s*(\d+)', r'(\d+)\s*units?']
        for pattern in patterns:
            match = re.search(pattern, str(text).lower())
            if match:
                return int(match.group(1))
        return 1
    
    features['ipq'] = df['catalog_content'].apply(extract_ipq)
    features['ipq_log'] = np.log1p(features['ipq'])
    
    brands = ['amazon', 'sony', 'samsung', 'apple', 'lg', 'nike', 'adidas', 'puma', 
              'hp', 'dell', 'lenovo', 'asus', 'panasonic', 'philips', 'bosch', 'havells', 'bajaj']
    for brand in brands:
        features[f'brand_{brand}'] = df['catalog_content'].str.lower().str.contains(brand, regex=False).astype(int)
    features['has_any_brand'] = (features[[f'brand_{b}' for b in brands]].sum(axis=1) > 0).astype(int)
    
    categories = {
        'electronics': ['phone', 'laptop', 'tablet', 'camera', 'tv', 'headphone', 'charger', 'speaker', 'mouse', 'keyboard', 'monitor'],
        'clothing': ['shirt', 'pant', 'dress', 'shoe', 'jacket', 'jeans', 'tshirt', 't-shirt', 'trouser', 'sweater'],
        'home': ['furniture', 'bed', 'table', 'chair', 'lamp', 'kitchen', 'curtain', 'cushion', 'mattress', 'sofa'],
        'beauty': ['cream', 'shampoo', 'soap', 'perfume', 'makeup', 'cosmetic', 'lotion', 'serum', 'facewash'],
        'food': ['coffee', 'tea', 'snack', 'chocolate', 'protein', 'vitamin', 'supplement', 'oil', 'spice'],
        'sports': ['gym', 'fitness', 'yoga', 'dumbbell', 'treadmill', 'cycle', 'sports', 'exercise', 'workout']
    }
    for category, keywords in categories.items():
        features[f'cat_{category}'] = df['catalog_content'].str.lower().apply(lambda x: int(any(kw in str(x) for kw in keywords)))
    
    units = {
        'volume': ['ml', 'liter', 'litre', 'oz', 'gallon'],
        'weight': ['kg', 'gram', 'gm', 'g ', 'pound', 'lb'],
        'dimension': ['cm', 'inch', 'mm', 'meter', 'ft', 'feet'],
        'power': ['watt', 'w ', 'volt', 'v ', 'amp', 'mah']
    }
    for unit_type, unit_list in units.items():
        features[f'has_{unit_type}'] = df['catalog_content'].str.lower().apply(lambda x: int(any(u in str(x) for u in unit_list)))
    
    return features

print("="*60)
print("SIMPLE ENSEMBLE - XGBoost + LightGBM")
print("="*60)

train_df = pd.read_csv('../dataset/train.csv')
test_df = pd.read_csv('../dataset/test.csv')
print(f"✓ Loaded {len(train_df):,} training + {len(test_df):,} test samples")

train_df_clean = remove_outliers(train_df, 'price', factor=2.5)
print(f"✓ Clean training: {len(train_df_clean):,} samples")

X_train_structured = extract_quick_features(train_df_clean)
X_test_structured = extract_quick_features(test_df)
y_train_full = train_df_clean['price'].values

print("\nTF-IDF extraction...")
tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 3), min_df=3, max_df=0.8)
tfidf_train = tfidf.fit_transform(train_df_clean['catalog_content'].fillna('').astype(str))
tfidf_test = tfidf.transform(test_df['catalog_content'].fillna('').astype(str))

svd = TruncatedSVD(n_components=200, random_state=42)
tfidf_train_reduced = svd.fit_transform(tfidf_train)
tfidf_test_reduced = svd.transform(tfidf_test)

train_ids = X_train_structured['sample_id'].values
test_ids = X_test_structured['sample_id'].values
X_train_full = np.hstack([X_train_structured.drop('sample_id', axis=1).values, tfidf_train_reduced])
X_test_full = np.hstack([X_test_structured.drop('sample_id', axis=1).values, tfidf_test_reduced])

print(f"✓ Total features: {X_train_full.shape[1]}")

y_train_log = np.log1p(y_train_full)
X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_log, test_size=0.15, random_state=42)
y_val_original = np.expm1(y_val)

print(f"Train: {len(X_train):,} | Val: {len(X_val):,}")

print("\n" + "="*60)
print("TRAINING 2-MODEL ENSEMBLE")
print("="*60)

print("\n[1/2] XGBoost...")
xgb_model = xgb.XGBRegressor(
    n_estimators=800, max_depth=10, learning_rate=0.03,
    subsample=0.8, colsample_bytree=0.7, min_child_weight=3, gamma=0.1,
    reg_alpha=0.1, reg_lambda=1.5, tree_method='gpu_hist', gpu_id=0,
    predictor='gpu_predictor', early_stopping_rounds=50, random_state=42
)
xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
xgb_pred = np.expm1(xgb_model.predict(X_val))
xgb_smape = smape(y_val_original, xgb_pred)
print(f"✓ XGBoost SMAPE: {xgb_smape:.4f}%")

print("\n[2/2] LightGBM...")
lgb_model = lgb.LGBMRegressor(
    n_estimators=800, max_depth=10, learning_rate=0.03,
    subsample=0.8, colsample_bytree=0.7, min_child_weight=3,
    reg_alpha=0.1, reg_lambda=1.5, device='gpu', random_state=42, verbose=-1
)
lgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)])
lgb_pred = np.expm1(lgb_model.predict(X_val))
lgb_smape = smape(y_val_original, lgb_pred)
print(f"✓ LightGBM SMAPE: {lgb_smape:.4f}%")

# Weighted ensemble
weights = np.array([1/xgb_smape, 1/lgb_smape])
weights = weights / weights.sum()
print(f"\nWeights: XGBoost={weights[0]:.3f}, LightGBM={weights[1]:.3f}")

ensemble_pred = weights[0] * xgb_pred + weights[1] * lgb_pred
ensemble_smape = smape(y_val_original, ensemble_pred)
print(f"\n✓ ENSEMBLE SMAPE: {ensemble_smape:.4f}%")
print(f"✓ Improvement: {54.54 - ensemble_smape:.2f}% from TF-IDF baseline!")

print("\nTraining on full data...")
xgb_full = xgb.XGBRegressor(n_estimators=xgb_model.best_iteration if hasattr(xgb_model, 'best_iteration') else 600,
    max_depth=10, learning_rate=0.03, subsample=0.8, colsample_bytree=0.7,
    tree_method='gpu_hist', gpu_id=0, random_state=42)
xgb_full.fit(X_train_full, y_train_log, verbose=False)

lgb_full = lgb.LGBMRegressor(n_estimators=600, max_depth=10, learning_rate=0.03,
    device='gpu', random_state=42, verbose=-1)
lgb_full.fit(X_train_full, y_train_log)

xgb_test = np.expm1(xgb_full.predict(X_test_full))
lgb_test = np.expm1(lgb_full.predict(X_test_full))
ensemble_test = weights[0] * xgb_test + weights[1] * lgb_test

submission = pd.DataFrame({'sample_id': test_ids, 'price': ensemble_test})
submission.to_csv('../outputs/test_out_ensemble.csv', index=False)

print(f"\n✓ Saved: ../outputs/test_out_ensemble.csv")
print(f"✓ Price range: ${ensemble_test.min():.2f} - ${ensemble_test.max():.2f}")
print(f"\nExpected SMAPE: ~{ensemble_smape:.1f}%")
print("="*60)
