import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import requests
from io import BytesIO
from tqdm import tqdm
import time

class ImageFeatureExtractor:
    def __init__(self, model_name='resnet50'):
        """Initialize image feature extractor"""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load pre-trained model
        if model_name == 'resnet50':
            self.model = models.resnet50(pretrained=True)
            # Remove final classification layer
            self.model = nn.Sequential(*list(self.model.children())[:-1])
            self.feature_dim = 2048
        elif model_name == 'efficientnet':
            self.model = models.efficientnet_b0(pretrained=True)
            self.model.classifier = nn.Identity()
            self.feature_dim = 1280
        
        self.model.to(self.device)
        self.model.eval()
        
        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    def download_image(self, url, max_retries=3):
        """Download image from URL with retry logic"""
        for attempt in range(max_retries):
            try:
                response = requests.get(url, timeout=10)
                if response.status_code == 200:
                    img = Image.open(BytesIO(response.content)).convert('RGB')
                    return img
            except Exception as e:
                if attempt == max_retries - 1:
                    return None
                time.sleep(1)  # Wait before retry
        return None
    
    def extract_features_from_image(self, img):
        """Extract features from a single image"""
        if img is None:
            return np.zeros(self.feature_dim)
        
        try:
            # Preprocess image
            img_tensor = self.transform(img).unsqueeze(0).to(self.device)
            
            # Extract features
            with torch.no_grad():
                features = self.model(img_tensor)
                features = features.squeeze().cpu().numpy()
            
            return features
        except Exception as e:
            print(f"Error extracting features: {e}")
            return np.zeros(self.feature_dim)
    
    def extract_features_from_urls(self, urls, batch_size=32):
        """Extract features from list of image URLs"""
        all_features = []
        
        print(f"Processing {len(urls)} images...")
        for i in tqdm(range(0, len(urls), batch_size)):
            batch_urls = urls[i:i+batch_size]
            batch_features = []
            
            for url in batch_urls:
                img = self.download_image(url)
                features = self.extract_features_from_image(img)
                batch_features.append(features)
            
            all_features.extend(batch_features)
            
            # Sleep to avoid throttling
            if i % 100 == 0:
                time.sleep(2)
        
        return np.array(all_features)
    
    def extract_all_features(self, df):
        """Extract all image features from dataframe"""
        print("Extracting image features...")
        urls = df['image_link'].fillna('').tolist()
        
        image_features = self.extract_features_from_urls(urls)
        
        # Create feature columns
        image_df = pd.DataFrame(
            image_features,
            columns=[f'img_{i}' for i in range(image_features.shape[1])]
        )
        
        return image_df
