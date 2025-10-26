import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import xgboost as xgb
import re
import warnings
import scipy.sparse as sp
warnings.filterwarnings('ignore')

def smape(y_true, y_pred):
    """Calculate SMAPE metric"""
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    diff = np.abs(y_true - y_pred) / denominator
    diff[denominator == 0] = 0.0
    return 100 * np.mean(diff)

def extract_quick_features(df):
    """Extract text-based features (same as before)"""
    features = pd.DataFrame()
    features['sample_id'] = df['sample_id']
    
    df['catalog_content'] = df['catalog_content'].fillna('').astype(str)
    
    print("  - Extracting text statistics...")
    features['text_length'] = df['catalog_content'].str.len()
    features['word_count'] = df['catalog_content'].str.split().str.len()
    features['uppercase_ratio'] = df['catalog_content'].apply(
        lambda x: sum(1 for c in x if c.isupper()) / max(len(x), 1)
    )
    features['digit_count'] = df['catalog_content'].str.count(r'\d')
    features['special_char_count'] = df['catalog_content'].str.count(r'[^a-zA-Z0-9\s]')
    
    print("  - Extracting numeric values...")
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
    
    print("  - Extracting Item Pack Quantity (IPQ)...")
    def extract_ipq(text):
        patterns = [
            r'pack\s*of\s*(\d+)',
            r'(\d+)\s*pack',
            r'quantity[\s:]*(\d+)',
            r'count[\s:]*(\d+)',
            r'(\d+)\s*pieces?',
            r'set\s*of\s*(\d+)',
            r'(\d+)\s*units?'
        ]
        for pattern in patterns:
            match = re.search(pattern, str(text).lower())
            if match:
                return int(match.group(1))
        return 1
    
    features['ipq'] = df['catalog_content'].apply(extract_ipq)
    features['ipq_log'] = np.log1p(features['ipq'])
    
    print("  - Extracting brand indicators...")
    brands = ['amazon', 'sony', 'samsung', 'apple', 'lg', 'nike', 
              'adidas', 'puma', 'hp', 'dell', 'lenovo', 'asus',
              'panasonic', 'philips', 'bosch', 'havells', 'bajaj']
    
    for brand in brands:
        features[f'brand_{brand}'] = df['catalog_content'].str.lower().str.contains(
            brand, regex=False
        ).astype(int)
    
    features['has_any_brand'] = features[[f'brand_{b}' for b in brands]].sum(axis=1) > 0
    features['has_any_brand'] = features['has_any_brand'].astype(int)
    
    print("  - Extracting category keywords...")
    categories = {
        'electronics': ['phone', 'laptop', 'tablet', 'camera', 'tv', 'headphone', 
                       'charger', 'speaker', 'mouse', 'keyboard', 'monitor'],
        'clothing': ['shirt', 'pant', 'dress', 'shoe', 'jacket', 'jeans', 
                    'tshirt', 't-shirt', 'trouser', 'sweater'],
        'home': ['furniture', 'bed', 'table', 'chair', 'lamp', 'kitchen',
                'curtain', 'cushion', 'mattress', 'sofa'],
        'beauty': ['cream', 'shampoo', 'soap', 'perfume', 'makeup', 'cosmetic',
                  'lotion', 'serum', 'facewash'],
        'food': ['coffee', 'tea', 'snack', 'chocolate', 'protein', 'vitamin',
                'supplement', 'oil', 'spice'],
        'sports': ['gym', 'fitness', 'yoga', 'dumbbell', 'treadmill', 'cycle',
                  'sports', 'exercise', 'workout']
    }
    
    for category, keywords in categories.items():
        features[f'cat_{category}'] = df['catalog_content'].str.lower().apply(
            lambda x: int(any(kw in str(x) for kw in keywords))
        )
    
    print("  - Extracting unit measurements...")
    units = {
        'volume': ['ml', 'liter', 'litre', 'oz', 'gallon'],
        'weight': ['kg', 'gram', 'gm', 'g ', 'pound', 'lb'],
        'dimension': ['cm', 'inch', 'mm', 'meter', 'ft', 'feet'],
        'power': ['watt', 'w ', 'volt', 'v ', 'amp', 'mah']
    }
    
    for unit_type, unit_list in units.items():
        features[f'has_{unit_type}'] = df['catalog_content'].str.lower().apply(
            lambda x: int(any(u in str(x) for u in unit_list))
        )
    
    return features

print("="*60)
print("AMAZON ML CHALLENGE 2025 - IMPROVED MODEL WITH TF-IDF")
print("="*60)

print("\nLoading data...")
train_df = pd.read_csv('../dataset/train.csv')
test_df = pd.read_csv('../dataset/test.csv')
print(f"✓ Loaded {len(train_df):,} training samples")
print(f"✓ Loaded {len(test_df):,} test samples")

print("\nExtracting training features...")
X_train_structured = extract_quick_features(train_df)
y_train_full = train_df['price'].values

print("\nExtracting test features...")
X_test_structured = extract_quick_features(test_df)

# Extract TF-IDF features
print("\n" + "="*60)
print("EXTRACTING TF-IDF TEXT FEATURES (CRITICAL IMPROVEMENT)")
print("="*60)

print("Training TF-IDF vectorizer...")
tfidf = TfidfVectorizer(
    max_features=5000,  # Top 5000 words
    ngram_range=(1, 3),  # Unigrams, bigrams, trigrams
    min_df=3,  # Minimum document frequency
    max_df=0.8,  # Maximum document frequency
    strip_accents='unicode',
    lowercase=True,
    analyzer='word',
    token_pattern=r'\w{1,}'
)

# Fit on training data
train_text = train_df['catalog_content'].fillna('').astype(str)
test_text = test_df['catalog_content'].fillna('').astype(str)

tfidf_train = tfidf.fit_transform(train_text)
tfidf_test = tfidf.transform(test_text)

print(f"✓ TF-IDF shape: {tfidf_train.shape}")

# Apply dimensionality reduction for efficiency
print("Applying SVD dimensionality reduction...")
svd = TruncatedSVD(n_components=200, random_state=42)
tfidf_train_reduced = svd.fit_transform(tfidf_train)
tfidf_test_reduced = svd.transform(tfidf_test)

print(f"✓ Reduced to {tfidf_train_reduced.shape[1]} components")
print(f"✓ Explained variance: {svd.explained_variance_ratio_.sum():.2%}")

# Combine structured features with TF-IDF
print("\nCombining structured and TF-IDF features...")
train_ids = X_train_structured['sample_id'].values
test_ids = X_test_structured['sample_id'].values

X_train_structured = X_train_structured.drop('sample_id', axis=1).values
X_test_structured = X_test_structured.drop('sample_id', axis=1).values

# Concatenate features
X_train_full = np.hstack([X_train_structured, tfidf_train_reduced])
X_test_full = np.hstack([X_test_structured, tfidf_test_reduced])

print(f"✓ Total features: {X_train_full.shape[1]} (41 structured + 200 TF-IDF)")

# Log transform target
print("\nApplying log transformation to prices...")
y_train_log = np.log1p(y_train_full)

# Split for validation
print("\nCreating train/validation split...")
X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_train_log, 
    test_size=0.15, 
    random_state=42
)

y_val_original = np.expm1(y_val)

print(f"Training set: {len(X_train):,}")
print(f"Validation set: {len(X_val):,}")

print("\n" + "="*60)
print("TRAINING XGBOOST WITH ENHANCED FEATURES")
print("="*60)

model = xgb.XGBRegressor(
    n_estimators=1000,
    max_depth=10,  # Increased depth for more features
    learning_rate=0.03,  # Lower LR for stability
    subsample=0.8,
    colsample_bytree=0.7,
    min_child_weight=3,
    gamma=0.1,
    reg_alpha=0.1,
    reg_lambda=1.5,
    random_state=42,
    tree_method='gpu_hist',
    gpu_id=0,
    predictor='gpu_predictor',
    early_stopping_rounds=50
)

print("\nTraining started...")
model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    verbose=50
)

print("\n" + "="*60)
print("VALIDATION RESULTS")
print("="*60)

y_pred_log = model.predict(X_val)
y_pred = np.expm1(y_pred_log)

val_smape = smape(y_val_original, y_pred)
print(f"\nValidation SMAPE: {val_smape:.4f}%")
print(f"Improvement: {60.93 - val_smape:.2f}% better than baseline!")

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
mae = mean_absolute_error(y_val_original, y_pred)
rmse = np.sqrt(mean_squared_error(y_val_original, y_pred))
r2 = r2_score(y_val_original, y_pred)

print(f"MAE: ${mae:.2f}")
print(f"RMSE: ${rmse:.2f}")
print(f"R²: {r2:.4f}")

print("\n" + "="*60)
print("TRAINING ON FULL DATA & GENERATING PREDICTIONS")
print("="*60)

print("\nRetraining on full training set...")
model_full = xgb.XGBRegressor(
    n_estimators=model.best_iteration if hasattr(model, 'best_iteration') else 600,
    max_depth=10,
    learning_rate=0.03,
    subsample=0.8,
    colsample_bytree=0.7,
    min_child_weight=3,
    gamma=0.1,
    reg_alpha=0.1,
    reg_lambda=1.5,
    random_state=42,
    tree_method='gpu_hist',
    gpu_id=0,
    predictor='gpu_predictor'
)

model_full.fit(X_train_full, y_train_log, verbose=False)

print("Generating test predictions...")
test_pred_log = model_full.predict(X_test_full)
test_pred = np.expm1(test_pred_log)

submission = pd.DataFrame({
    'sample_id': test_ids,
    'price': test_pred
})

output_path = '../outputs/test_out_tfidf.csv'
submission.to_csv(output_path, index=False)

print(f"\n✓ Predictions saved to: {output_path}")
print(f"✓ Total predictions: {len(submission):,}")
print(f"✓ Price range: ${test_pred.min():.2f} - ${test_pred.max():.2f}")

import pickle
model_path = '../models/xgb_with_tfidf.pkl'
with open(model_path, 'wb') as f:
    pickle.dump({'model': model_full, 'tfidf': tfidf, 'svd': svd}, f)
print(f"✓ Model saved to: {model_path}")

print("\n" + "="*60)
print("IMPROVED MODEL COMPLETE!")
print("="*60)
print(f"\nExpected SMAPE: ~{val_smape:.1f}% (Target: <25%)")
print("\nNext improvements:")
print("1. Add image features → Expected -8 to -12% SMAPE")
print("2. Ensemble with LightGBM → Expected -2 to -4% SMAPE")
print("3. Hyperparameter tuning → Expected -1 to -2% SMAPE")
print("="*60)
