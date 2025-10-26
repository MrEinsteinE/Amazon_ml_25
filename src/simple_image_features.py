import pandas as pd
import numpy as np
import torch
from torchvision import models, transforms
from PIL import Image
import requests
from io import BytesIO
from tqdm import tqdm
import time

class QuickImageExtractor:
    def __init__(self):
        """Use lightweight EfficientNet for speed"""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Use EfficientNet-B0 (faster than ResNet50)
        self.model = models.efficientnet_b0(pretrained=True)
        self.model.classifier = torch.nn.Identity()
        self.model.to(self.device)
        self.model.eval()
        
        self.transform = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    
    def get_image(self, url, timeout=5, max_retries=2):
        """Download with aggressive timeout"""
        for _ in range(max_retries):
            try:
                response = requests.get(url, timeout=timeout)
                return Image.open(BytesIO(response.content)).convert('RGB')
            except:
                time.sleep(0.5)
        return None
    
    def extract_batch(self, urls, batch_size=32):
        """Extract features in batches"""
        all_features = []
        
        for i in tqdm(range(0, len(urls), batch_size)):
            batch_urls = urls[i:i+batch_size]
            batch_images = []
            
            # Download batch
            for url in batch_urls:
                img = self.get_image(url)
                if img:
                    batch_images.append(self.transform(img))
                else:
                    # Zero vector for failed downloads
                    batch_images.append(torch.zeros(3, 224, 224))
            
            # Process batch on GPU
            batch_tensor = torch.stack(batch_images).to(self.device)
            with torch.no_grad():
                features = self.model(batch_tensor).cpu().numpy()
            
            all_features.append(features)
            
            # Throttle to avoid blocking
            if i % 500 == 0:
                time.sleep(2)
        
        return np.vstack(all_features)

# Usage
extractor = QuickImageExtractor()
train_df = pd.read_csv('../dataset/train.csv')

# Extract for subset first to test
print("Testing on 1000 samples...")
sample = train_df.sample(1000, random_state=42)
features = extractor.extract_batch(sample['image_link'].tolist())
print(f"Extracted {features.shape[1]} features per image")
