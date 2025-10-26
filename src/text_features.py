import pandas as pd
import numpy as np
import re
from transformers import AutoTokenizer, AutoModel
import torch
from sklearn.feature_extraction.text import TfidfVectorizer

class TextFeatureExtractor:
    def __init__(self, bert_model='bert-base-uncased'):
        """Initialize text feature extractor with BERT"""
        self.tokenizer = AutoTokenizer.from_pretrained(bert_model)
        self.model = AutoModel.from_pretrained(bert_model)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model.eval()
        
    def extract_structured_features(self, text):
        """Extract structured features from catalog content"""
        features = {}
        
        # Text length features
        features['text_length'] = len(str(text))
        features['word_count'] = len(str(text).split())
        
        # Extract Item Pack Quantity (IPQ)
        ipq_match = re.search(r'(?:pack|quantity|count|qty)[\s:]*(\d+)', str(text).lower())
        features['item_pack_qty'] = int(ipq_match.group(1)) if ipq_match else 1
        
        # Extract numeric values (dimensions, weights, capacities)
        numbers = re.findall(r'\d+\.?\d*', str(text))
        features['num_numeric_values'] = len(numbers)
        features['max_numeric_value'] = max([float(n) for n in numbers]) if numbers else 0
        features['avg_numeric_value'] = np.mean([float(n) for n in numbers]) if numbers else 0
        
        # Brand indicators (common patterns)
        text_lower = str(text).lower()
        brand_keywords = ['amazon', 'sony', 'samsung', 'apple', 'lg', 'nike', 'adidas']
        features['has_brand'] = int(any(brand in text_lower for brand in brand_keywords))
        
        # Product category keywords
        categories = {
            'electronics': ['phone', 'laptop', 'tablet', 'camera', 'tv', 'headphone'],
            'clothing': ['shirt', 't-shirt', 'pant', 'dress', 'shoe', 'jacket'],
            'home': ['furniture', 'bed', 'table', 'chair', 'lamp'],
            'beauty': ['cream', 'shampoo', 'soap', 'perfume', 'makeup']
        }
        
        for category, keywords in categories.items():
            features[f'is_{category}'] = int(any(kw in text_lower for kw in keywords))
        
        # Special characters and formatting
        features['has_uppercase'] = int(any(c.isupper() for c in str(text)))
        features['special_char_count'] = len(re.findall(r'[^a-zA-Z0-9\s]', str(text)))
        
        return features
    
    def get_bert_embeddings(self, texts, batch_size=16):
        """Extract BERT embeddings for text"""
        embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            
            # Tokenize and encode
            encoded = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors='pt'
            )
            
            # Move to GPU
            encoded = {k: v.to(self.device) for k, v in encoded.items()}
            
            # Get embeddings
            with torch.no_grad():
                outputs = self.model(**encoded)
                # Use [CLS] token embedding
                batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            
            embeddings.append(batch_embeddings)
            
            if i % 100 == 0:
                print(f"Processed {i}/{len(texts)} texts")
        
        return np.vstack(embeddings)
    
    def extract_all_features(self, df):
        """Extract all text features from dataframe"""
        print("Extracting structured features...")
        structured_features = df['catalog_content'].apply(self.extract_structured_features)
        structured_df = pd.DataFrame(structured_features.tolist())
        
        print("Extracting BERT embeddings...")
        texts = df['catalog_content'].fillna('').astype(str).tolist()
        bert_embeddings = self.get_bert_embeddings(texts)
        
        # Create embedding columns
        bert_df = pd.DataFrame(
            bert_embeddings,
            columns=[f'bert_{i}' for i in range(bert_embeddings.shape[1])]
        )
        
        # Combine all features
        text_features = pd.concat([structured_df, bert_df], axis=1)
        
        return text_features
