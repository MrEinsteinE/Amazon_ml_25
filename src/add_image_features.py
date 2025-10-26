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

class FastImageExtractor:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # ResNet-50
        self.model = models.resnet50(weights='IMAGENET1K_V1')  # Fixed deprecation warning
        self.model = nn.Sequential(*list(self.model.children())[:-1])
        self.model.to(self.device)
        self.model.eval()
        
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        self.session = requests.Session()
        self.session.headers.update({'User-Agent': 'Mozilla/5.0'})
    
    def download_image(self, url, timeout=3, retries=2):
        """Fast download with timeout"""
        for attempt in range(retries):
            try:
                response = self.session.get(url, timeout=timeout)
                if response.status_code == 200:
                    return Image.open(BytesIO(response.content)).convert('RGB')
            except:
                if attempt < retries - 1:
                    time.sleep(0.2)
        return None
    
    def download_batch_parallel(self, urls, max_workers=50):
        """Download images in parallel - KEY SPEEDUP"""
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
        """Process batch on GPU"""
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
        
        # Process on GPU in batch
        batch_tensor = torch.stack(valid_images).to(self.device)
        with torch.no_grad():
            features = self.model(batch_tensor).squeeze().cpu().numpy()
        
        # Fill results
        result = np.zeros((len(images), 2048))
        if len(valid_indices) == 1:
            result[valid_indices[0]] = features
        else:
            for i, idx in enumerate(valid_indices):
                result[idx] = features[i]
        
        return result
    
    def extract_all(self, urls, download_batch=200, process_batch=64):
        """Extract features with parallel download + GPU batch processing"""
        all_features = []
        failed = 0
        
        print(f"Processing {len(urls)} images...")
        print(f"Download batch: {download_batch}, Process batch: {process_batch}")
        print(f"Using {50} parallel downloads (FAST!)")
        
        for i in tqdm(range(0, len(urls), download_batch)):
            batch_urls = urls[i:i+download_batch]
            
            # PARALLEL DOWNLOAD (FAST!)
            images = self.download_batch_parallel(batch_urls, max_workers=50)
            
            # Count failures
            failed += sum(1 for img in images if img is None)
            
            # Process in smaller batches on GPU
            for j in range(0, len(images), process_batch):
                sub_batch = images[j:j+process_batch]
                features = self.extract_features_batch(sub_batch)
                all_features.append(features)
        
        all_features = np.vstack(all_features)
        print(f"\n✓ Complete! Failed: {failed}/{len(urls)} ({failed/len(urls)*100:.1f}%)")
        
        return all_features

print("="*60)
print("FAST PARALLEL IMAGE EXTRACTION")
print("="*60)

train_df = pd.read_csv('../dataset/train.csv')
test_df = pd.read_csv('../dataset/test.csv')

print(f"\nDataset: {len(train_df)} train + {len(test_df)} test")
print("Expected time: 2-3 hours (vs 10+ hours sequential)")

extractor = FastImageExtractor()

# TRAINING IMAGES
print("\n" + "="*60)
print("[1/2] TRAINING IMAGE FEATURES")
print("="*60)
train_features = extractor.extract_all(
    train_df['image_link'].tolist(),
    download_batch=200,  # Download 200 at once
    process_batch=64     # GPU processes 64 at once
)

np.save('../outputs/train_image_features.npy', train_features)
print(f"✓ Saved: {train_features.shape}")

# TEST IMAGES
print("\n" + "="*60)
print("[2/2] TEST IMAGE FEATURES")
print("="*60)
test_features = extractor.extract_all(
    test_df['image_link'].tolist(),
    download_batch=200,
    process_batch=64
)

np.save('../outputs/test_image_features.npy', test_features)
print(f"✓ Saved: {test_features.shape}")

print("\n" + "="*60)
print("EXTRACTION COMPLETE!")
print("="*60)
print("Next: Run 'combined_text_image_model.py'")
print("="*60)
