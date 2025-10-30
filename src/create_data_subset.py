# Usage: python create_data_subset.py --data_dir=../data --subset_size=1000  --subset_dir=../data/subsets

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
import random
import shutil


class COCODataset(Dataset):
    """COCO Caption Dataset for BLIP model training"""
    
    def __init__(self, img_dir, ann_file, processor, max_length=128, use_subset=False):
        """
        Args:
            img_dir: Directory containing images
            ann_file: Path to annotation JSON file (or subset JSON)
            processor: BLIP processor for image and text preprocessing
            max_length: Maximum caption length
            use_subset: If True, expect ann_file to be a subset JSON
        """
        self.img_dir = Path(img_dir)
        self.processor = processor
        self.max_length = max_length
        
        # Load annotations
        with open(ann_file, 'r') as f:
            data = json.load(f)
        
        if use_subset:
            # Subset format: list of samples directly
            self.samples = data
            print(f"Loaded {len(self.samples)} samples from subset file")
        else:
            # Original COCO format
            # Create image_id to filename mapping
            self.id_to_filename = {img['id']: img['file_name'] for img in data['images']}
            
            # Store all image-caption pairs
            self.samples = []
            for ann in data['annotations']:
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
        processor=None,
        # NEW: Subset configuration
        use_subset=False,
        subset_size=None,
        subset_dir='../data/subsets',
        force_recreate_subset=False
    ):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.max_length = max_length
        
        # NEW: Subset parameters
        self.use_subset = use_subset
        self.subset_size = subset_size
        self.subset_dir = Path(subset_dir)
        self.force_recreate_subset = force_recreate_subset
        
        # Detect device and adjust settings
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
        
        # Adjust num_workers based on device
        if self.device == 'cuda':
            self.num_workers = num_workers
            self.pin_memory = True
            self.persistent_workers = True if num_workers > 0 else False
        else:
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
    
    def create_subsets(self):
        """
        Create subset of COCO data and copy images to subset directory.
        Saves subset metadata to JSON for quick loading.
        """
        self.subset_dir.mkdir(parents=True, exist_ok=True)
        
        # Define subset paths
        train_subset_json = self.subset_dir / f'train_subset_{self.subset_size}.json'
        val_subset_json = self.subset_dir / f'val_subset_{self.subset_size}.json'
        train_subset_img_dir = self.subset_dir / f'train_images_{self.subset_size}'
        val_subset_img_dir = self.subset_dir / f'val_images_{self.subset_size}'
        
        # Check if subsets already exist
        if (train_subset_json.exists() and val_subset_json.exists() and
            train_subset_img_dir.exists() and val_subset_img_dir.exists() and
            not self.force_recreate_subset):
            print(f"Subsets already exist in {self.subset_dir}. Skipping creation.")
            print(f"Use force_recreate_subset=True to recreate.")
            return
        
        print(f"\nCreating subsets of size {self.subset_size}...")
        
        # Process training subset
        print("\n--- Creating Training Subset ---")
        self._create_split_subset(
            original_img_dir=self.data_dir / 'train2014',
            original_ann_file=self.data_dir / 'annotations' / 'captions_train2014.json',
            subset_json_path=train_subset_json,
            subset_img_dir=train_subset_img_dir,
            split='train'
        )
        
        # Process validation subset
        print("\n--- Creating Validation Subset ---")
        # Use smaller size for validation (e.g., 10% of training subset size)
        val_subset_size = max(100, self.subset_size // 10)
        self._create_split_subset(
            original_img_dir=self.data_dir / 'val2014',
            original_ann_file=self.data_dir / 'annotations' / 'captions_val2014.json',
            subset_json_path=val_subset_json,
            subset_img_dir=val_subset_img_dir,
            split='val',
            size=val_subset_size
        )
        
        print(f"\n✓ Subsets created successfully!")
        print(f"  Train subset: {train_subset_json}")
        print(f"  Val subset: {val_subset_json}")
        print(f"  Train images: {train_subset_img_dir}")
        print(f"  Val images: {val_subset_img_dir}")
    
    def _create_split_subset(self, original_img_dir, original_ann_file, 
                           subset_json_path, subset_img_dir, split, size=None):
        """Create subset for a specific split (train/val)"""
        
        if size is None:
            size = self.subset_size
        
        # Load original annotations
        print(f"Loading annotations from {original_ann_file}...")
        with open(original_ann_file, 'r') as f:
            coco_data = json.load(f)
        
        # Create image_id to filename mapping
        id_to_filename = {img['id']: img['file_name'] for img in coco_data['images']}
        
        # Collect all samples
        all_samples = []
        for ann in tqdm(coco_data['annotations'], desc="Processing annotations"):
            img_id = ann['image_id']
            if img_id in id_to_filename:
                all_samples.append({
                    'image_id': img_id,
                    'filename': id_to_filename[img_id],
                    'caption': ann['caption']
                })
        
        print(f"Total samples available: {len(all_samples)}")
        
        # Randomly sample subset
        if len(all_samples) > size:
            print(f"Sampling {size} samples...")
            random.seed(42)  # For reproducibility
            subset_samples = random.sample(all_samples, size)
        else:
            print(f"Using all {len(all_samples)} samples (requested size {size} is larger)")
            subset_samples = all_samples
        
        # Create subset image directory
        subset_img_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy images and update paths
        print(f"Copying images to {subset_img_dir}...")
        final_samples = []
        failed_copies = []
        
        # Get unique images (multiple captions per image)
        unique_images = {}
        for sample in subset_samples:
            img_id = sample['image_id']
            if img_id not in unique_images:
                unique_images[img_id] = sample['filename']
        
        # Copy unique images
        for img_id, filename in tqdm(unique_images.items(), desc="Copying images"):
            src_path = original_img_dir / filename
            dst_path = subset_img_dir / filename
            
            try:
                if not dst_path.exists():
                    shutil.copy2(src_path, dst_path)
            except Exception as e:
                print(f"Failed to copy {src_path}: {e}")
                failed_copies.append(img_id)
        
        # Filter out samples with failed image copies
        for sample in subset_samples:
            if sample['image_id'] not in failed_copies:
                final_samples.append(sample)
        
        # Save subset metadata to JSON
        print(f"Saving subset metadata to {subset_json_path}...")
        with open(subset_json_path, 'w') as f:
            json.dump(final_samples, f, indent=2)
        
        # Save summary statistics
        summary = {
            'split': split,
            'total_samples': len(final_samples),
            'unique_images': len(unique_images) - len(failed_copies),
            'requested_size': size,
            'failed_copies': len(failed_copies),
            'image_directory': str(subset_img_dir),
            'annotation_file': str(subset_json_path)
        }
        
        summary_path = subset_json_path.parent / f'{split}_subset_{size}_summary.json'
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"✓ {split.capitalize()} subset created:")
        print(f"  Samples: {len(final_samples)}")
        print(f"  Unique images: {len(unique_images) - len(failed_copies)}")
        print(f"  Failed copies: {len(failed_copies)}")
    
    def setup(self, stage: Optional[str] = None):
        """Setup train and validation datasets"""
        
        if stage == 'fit' or stage is None:
            if self.use_subset and self.subset_size:
                # Load from subset files
                print("\n--- Loading from Subset Files ---")
                
                train_img_dir = self.subset_dir / f'train_images_{self.subset_size}'
                train_ann_file = self.subset_dir / f'train_subset_{self.subset_size}.json'
                
                val_subset_size = max(100, self.subset_size // 10)
                val_img_dir = self.subset_dir / f'val_images_{val_subset_size}'
                val_ann_file = self.subset_dir / f'val_subset_{val_subset_size}.json'
                
                # Check if subset files exist
                if not all([train_ann_file.exists(), val_ann_file.exists(),
                           train_img_dir.exists(), val_img_dir.exists()]):
                    raise FileNotFoundError(
                        f"Subset files not found. Please run prepare_data() first or "
                        f"set use_subset=False to use full dataset."
                    )
                
                self.train_dataset = COCODataset(
                    train_img_dir,
                    train_ann_file,
                    self.processor,
                    self.max_length,
                    use_subset=True
                )
                
                self.val_dataset = COCODataset(
                    val_img_dir,
                    val_ann_file,
                    self.processor,
                    self.max_length,
                    use_subset=True
                )
            else:
                # Load from full dataset
                print("\n--- Loading from Full Dataset ---")
                
                train_img_dir = self.data_dir / 'train2014'
                train_ann_file = self.data_dir / 'annotations' / 'captions_train2014.json'
                
                val_img_dir = self.data_dir / 'val2014'
                val_ann_file = self.data_dir / 'annotations' / 'captions_val2014.json'
                
                self.train_dataset = COCODataset(
                    train_img_dir,
                    train_ann_file,
                    self.processor,
                    self.max_length,
                    use_subset=False
                )
                
                self.val_dataset = COCODataset(
                    val_img_dir,
                    val_ann_file,
                    self.processor,
                    self.max_length,
                    use_subset=False
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


# Utility function to create subsets without full training setup
def create_coco_subset(
    data_dir='../data',
    subset_size=1000,
    subset_dir='../data/subsets',
    force_recreate=False
):
    """
    Standalone function to create COCO subsets without initializing the full datamodule.
    
    Args:
        data_dir: Directory containing full COCO dataset
        subset_size: Number of training samples to include
        subset_dir: Directory to save subset files
        force_recreate: Whether to recreate if subset already exists
    
    Example:
        >>> from data import create_coco_subset
        >>> create_coco_subset(subset_size=5000, force_recreate=True)
    """
    from transformers import BlipProcessor
    
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    
    datamodule = COCODataModule(
        data_dir=data_dir,
        batch_size=32,
        processor=processor,
        use_subset=True,
        subset_size=subset_size,
        subset_dir=subset_dir,
        force_recreate_subset=force_recreate
    )
    # This will create the subsets
    datamodule.create_subsets()
    
    print(f"\n{'='*60}")
    print(f"Subset creation complete!")
    print(f"{'='*60}")
    print(f"To use the subset in training:")
    print(f"")
    print(f"datamodule = COCODataModule(")
    print(f"    data_dir='{data_dir}',")
    print(f"    use_subset=True,")
    print(f"    subset_size={subset_size},")
    print(f"    subset_dir='{subset_dir}'")
    print(f")")
    print(f"{'='*60}")


if __name__ == "__main__":
    """
    Example usage:
    
    # Create a subset of 5000 training samples
    python create_data_subset.py --data_dir=../data --subset_size=1000  --subset_dir=../data/subsets
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Create COCO dataset subset')
    parser.add_argument('--data_dir', type=str, default='../data',
                       help='Directory containing full COCO dataset')
    parser.add_argument('--subset_size', type=int, default=5000,
                       help='Number of training samples in subset')
    parser.add_argument('--subset_dir', type=str, default='../data/subsets',
                       help='Directory to save subset files')
    parser.add_argument('--force_recreate', action='store_true',
                       help='Force recreation of subset if it exists')
    
    args = parser.parse_args()
    
    create_coco_subset(
        data_dir=args.data_dir,
        subset_size=args.subset_size,
        subset_dir=args.subset_dir,
        force_recreate=args.force_recreate
    )