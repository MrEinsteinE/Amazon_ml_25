import pandas as pd
import numpy as np
from text_features import TextFeatureExtractor
from image_features import ImageFeatureExtractor
import pickle
import os

def extract_and_save_features(mode='train'):
    """
    Extract features from train or test dataset
    mode: 'train' or 'test'
    """
    
    print(f"="*60)
    print(f"FEATURE EXTRACTION - {mode.upper()} SET")
    print(f"="*60)
    
    # Load data
    df = pd.read_csv(f'../dataset/{mode}.csv')
    print(f"Loaded {len(df)} samples")
    
    # Initialize extractors
    text_extractor = TextFeatureExtractor(bert_model='bert-base-uncased')
    image_extractor = ImageFeatureExtractor(model_name='resnet50')
    
    # Extract text features
    print("\n" + "="*60)
    print("TEXT FEATURE EXTRACTION")
    print("="*60)
    text_features = text_extractor.extract_all_features(df)
    print(f"Extracted {text_features.shape[1]} text features")
    
    # Extract image features
    print("\n" + "="*60)
    print("IMAGE FEATURE EXTRACTION")
    print("="*60)
    image_features = image_extractor.extract_all_features(df)
    print(f"Extracted {image_features.shape[1]} image features")
    
    # Combine all features
    print("\n" + "="*60)
    print("COMBINING FEATURES")
    print("="*60)
    
    # Keep sample_id
    combined_features = pd.DataFrame()
    combined_features['sample_id'] = df['sample_id']
    
    # Add text features
    for col in text_features.columns:
        combined_features[col] = text_features[col].values
    
    # Add image features
    for col in image_features.columns:
        combined_features[col] = image_features[col].values
    
    # Add target for train
    if mode == 'train':
        combined_features['price'] = df['price']
    
    print(f"Total features: {combined_features.shape[1]}")
    
    # Save features
    output_path = f'../outputs/{mode}_features.csv'
    combined_features.to_csv(output_path, index=False)
    print(f"\n✓ Features saved to: {output_path}")
    
    return combined_features

if __name__ == "__main__":
    # Extract training features
    train_features = extract_and_save_features(mode='train')
    
    # Extract test features
    test_features = extract_and_save_features(mode='test')
    
    print("\n" + "="*60)
    print("FEATURE EXTRACTION COMPLETE!")
    print("="*60)
