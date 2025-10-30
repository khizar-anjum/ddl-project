import os
import json
import zipfile
import requests
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from transformers import BlipProcessor
from typing import Optional
from tqdm import tqdm


def prepare_data(data_dir):
    """Download and extract COCO dataset"""
    # URLs for COCO 2014
    urls = {
        'train_images': 'http://images.cocodataset.org/zips/train2014.zip',
        'val_images': 'http://images.cocodataset.org/zips/val2014.zip',
        'annotations': 'http://images.cocodataset.org/annotations/annotations_trainval2014.zip'
    }

    data_dir = Path(data_dir)
    
    data_dir.mkdir(parents=True, exist_ok=True)
    
    for name, url in urls.items():
        zip_path = data_dir / f"{name}.zip"
        
        # Download if not exists
        if not zip_path.exists():
            print(f"Downloading {name} from {url}...")
            download_file(url, zip_path)
        
        # Extract if not already extracted
        if name == 'train_images':
            extract_dir = data_dir / 'train2014'
        elif name == 'val_images':
            extract_dir = data_dir / 'val2014'
        else:
            extract_dir = data_dir / 'annotations'
        
        if not extract_dir.exists():
            print(f"Extracting {name}...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(data_dir)
            print(f"Extracted to {extract_dir}")

def download_file(url, destination):
    """Download file with progress bar"""
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(destination, 'wb') as f, tqdm(
        desc=destination.name,
        total=total_size,
        unit='B',
        unit_scale=True,
        unit_divisor=1024,
    ) as pbar:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
            pbar.update(len(chunk))


if __name__ == "__main__":
    # define the data directory path
    data_dir='../data'

    # Download and prepare data (only runs once)
    print("Downloading and Preparing dataset...")
    prepare_data(data_dir)
    print("\nDataset preparation complete!")
    