import sys
sys.path.append('../src')
from utils import download_images
import pandas as pd

# Load sample data
train_df = pd.read_csv('../dataset/train.csv')

# Test download first 5 images
print("Testing image download...")
sample_links = train_df['image_link'].head(5).tolist()
sample_ids = train_df['sample_id'].head(5).tolist()

download_images(sample_links, sample_ids, '../images/')
print("✓ Image download test complete!")
