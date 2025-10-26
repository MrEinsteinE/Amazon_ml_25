import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import requests
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import time
import warnings
warnings.filterwarnings('ignore')

class RobustImageExtractor:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        self.model = models.resnet50(weights='IMAGENET1K_V1')
        self.model = nn.Sequential(*list(self.model.children())[:-1])
        self.model.to(self.device)
        self.model.eval()
        
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        # Multiple user agents to avoid blocking
        self.user_agents = [
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36',
            'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36'
        ]
    
    def download_image(self, url, timeout=10, retries=3):
        """More aggressive retry with longer timeout"""
        for attempt in range(retries):
            try:
                headers = {'User-Agent': self.user_agents[attempt % len(self.user_agents)]}
                response = requests.get(url, timeout=timeout, headers=headers, verify=False)
                
                if response.status_code == 200:
                    return Image.open(BytesIO(response.content)).convert('RGB')
                
                # Wait longer between retries
                if attempt < retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff: 1s, 2s, 4s
                    
            except:
                if attempt < retries - 1:
                    time.sleep(2 ** attempt)
        
        return None
    
    def download_batch_parallel(self, urls, max_workers=10):
        """REDUCED workers to 10 (vs 50) to avoid throttling"""
        images = [None] * len(urls)
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_idx = {executor.submit(self.download_image, url): i 
                           for i, url in enumerate(urls)}
            
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    images[idx] = future.result()
                except:
                    images[idx] = None
        
        return images
    
    def extract_features_batch(self, images):
        valid_images = []
        valid_indices = []
        
        for i, img in enumerate(images):
            if img is not None:
                try:
                    valid_images.append(self.transform(img))
                    valid_indices.append(i)
                except:
                    pass
        
        if not valid_images:
            return np.zeros((len(images), 2048))
        
        batch_tensor = torch.stack(valid_images).to(self.device)
        with torch.no_grad():
            features = self.model(batch_tensor).squeeze().cpu().numpy()
        
        result = np.zeros((len(images), 2048))
        if len(valid_indices) == 1:
            result[valid_indices[0]] = features
        else:
            for i, idx in enumerate(valid_indices):
                result[idx] = features[i]
        
        return result
    
    def extract_all(self, urls, download_batch=100, process_batch=32):
        """REDUCED batch sizes and workers"""
        all_features = []
        failed = 0
        
        print(f"Processing {len(urls)} images...")
        print(f"Download batch: {download_batch} (slower but more reliable)")
        print(f"Using 10 parallel workers (vs 50 - avoids throttling)")
        
        for i in tqdm(range(0, len(urls), download_batch)):
            batch_urls = urls[i:i+download_batch]
            
            # Slower parallel download (more reliable)
            images = self.download_batch_parallel(batch_urls, max_workers=10)
            failed += sum(1 for img in images if img is None)
            
            # Process in smaller batches
            for j in range(0, len(images), process_batch):
                sub_batch = images[j:j+process_batch]
                features = self.extract_features_batch(sub_batch)
                all_features.append(features)
            
            # Rate limiting between batches (critical!)
            time.sleep(5)  # 5 second pause between batches
        
        all_features = np.vstack(all_features)
        success_rate = (1 - failed/len(urls)) * 100
        print(f"\n✓ Complete! Success: {len(urls)-failed}/{len(urls)} ({success_rate:.1f}%)")
        
        return all_features

print("="*60)
print("ROBUST IMAGE EXTRACTION - FIXED")
print("="*60)

import urllib3
urllib3.disable_warnings()  # Disable SSL warnings

train_df = pd.read_csv('../dataset/train.csv')
test_df = pd.read_csv('../dataset/test.csv')

print(f"\nDataset: {len(train_df)} train + {len(test_df)} test")
print("Expected time: 5-7 hours (slower but reliable)")
print("Target success rate: >80%")

extractor = RobustImageExtractor()

# TRAINING
print("\n" + "="*60)
print("[1/2] TRAINING IMAGES (slower, more reliable)")
print("="*60)
train_features = extractor.extract_all(
    train_df['image_link'].tolist(),
    download_batch=100,
    process_batch=32
)

np.save('../outputs/train_image_features_v2.npy', train_features)
print(f"✓ Saved: {train_features.shape}")

# TEST
print("\n" + "="*60)
print("[2/2] TEST IMAGES")
print("="*60)
test_features = extractor.extract_all(
    test_df['image_link'].tolist(),
    download_batch=100,
    process_batch=32
)

np.save('../outputs/test_image_features_v2.npy', test_features)
print(f"✓ Saved: {test_features.shape}")

print("\n" + "="*60)
print("EXTRACTION COMPLETE!")
print("="*60)
