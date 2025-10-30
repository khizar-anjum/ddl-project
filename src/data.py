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


class COCODataset(Dataset):
    """COCO Caption Dataset for BLIP model training"""
    
    def __init__(self, img_dir, ann_file, processor, max_length=128):
        """
        Args:
            img_dir: Directory containing images
            ann_file: Path to annotation JSON file
            processor: BLIP processor for image and text preprocessing
            max_length: Maximum caption length
        """
        self.img_dir = Path(img_dir)
        self.processor = processor
        self.max_length = max_length
        
        # Load annotations
        with open(ann_file, 'r') as f:
            coco = json.load(f)
        
        # Create image_id to filename mapping
        self.id_to_filename = {img['id']: img['file_name'] for img in coco['images']}
        
        # Store all image-caption pairs
        self.samples = []
        for ann in coco['annotations']:
            img_id = ann['image_id']
            if img_id in self.id_to_filename:
                self.samples.append({
                    'image_id': img_id,
                    'filename': self.id_to_filename[img_id],
                    'caption': ann['caption']
                })
        
        print(f"Loaded {len(self.samples)} image-caption pairs")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load and process image
        img_path = self.img_dir / sample['filename']
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a blank image if loading fails
            image = Image.new('RGB', (224, 224), (0, 0, 0))
        
        # Process image and caption
        encoding = self.processor(
            images=image,
            text=sample['caption'],
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt"
        )
        
        # Remove batch dimension
        encoding = {k: v.squeeze(0) for k, v in encoding.items()}
        
        return encoding


class COCODataModule(pl.LightningDataModule):
    """PyTorch Lightning DataModule for COCO dataset"""
    
    def __init__(
        self,
        data_dir='../data',
        batch_size=32,
        num_workers=4,
        max_length=128,
        processor=None
    ):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.max_length = max_length
        
        # Detect device and adjust settings
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
        
        # Adjust num_workers based on device
        # GPUs benefit from more workers, CPUs may want fewer
        if self.device == 'cuda':
            self.num_workers = num_workers
            self.pin_memory = True
            self.persistent_workers = True if num_workers > 0 else False
        else:
            # Reduce workers for CPU to avoid overhead
            self.num_workers = min(num_workers, 2)
            self.pin_memory = False
            self.persistent_workers = False
        
        print(f"Device: {self.device}")
        print(f"Number of GPUs: {self.num_gpus}")
        print(f"Number of workers: {self.num_workers}")
        print(f"Pin memory: {self.pin_memory}")
        
        # Initialize BLIP processor if not provided
        if processor is None:
            from transformers import BlipProcessor
            self.processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        else:
            self.processor = processor
    
    def setup(self, stage: Optional[str] = None):
        """Setup train and validation datasets from separate directories"""
        
        if stage == 'fit' or stage is None:
            # Load training dataset
            train_img_dir = self.data_dir / 'train2014'
            train_ann_file = self.data_dir / 'annotations' / 'captions_train2014.json'
            
            self.train_dataset = COCODataset(
                train_img_dir,
                train_ann_file,
                self.processor,
                self.max_length
            )
            
            # Load validation dataset
            val_img_dir = self.data_dir / 'val2014'
            val_ann_file = self.data_dir / 'annotations' / 'captions_val2014.json'
            
            self.val_dataset = COCODataset(
                val_img_dir,
                val_ann_file,
                self.processor,
                self.max_length
            )
            
            print(f"Train samples: {len(self.train_dataset)}")
            print(f"Validation samples: {len(self.val_dataset)}")
    
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,   
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers
        )
