"""
BLIP Model Training Script for COCO Captions
Integrates with COCODataModule from data.py
"""

import argparse
import os
import ruamel.yaml as yaml
import numpy as np
import random
import time
import datetime
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger
from transformers import BlipProcessor, BlipForConditionalGeneration
from tqdm import tqdm
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.utils.data import DataLoader

from blip import blip_decoder
import utils
from utils import cosine_lr_schedule
# from src.coco_data import create_dataset, create_sampler, create_loader
from utils_data import save_result, coco_caption_eval
# Import the COCODataModule from data.py
from code.src.data import COCODataModule


class BLIPCaptioningModel(pl.LightningModule):
    """PyTorch Lightning module for BLIP image captioning"""
    
    def __init__(
        self,
        model_name="Salesforce/blip-image-captioning-base",
        learning_rate=1e-5,
        weight_decay=0.05,
        warmup_steps=1000,
        max_epochs=10,
        num_beams=3,
        max_length=30,
        min_length=5,
    ):
        super().__init__()
        self.save_hyperparameters()
        
        # Load BLIP model and processor
        self.model = BlipForConditionalGeneration.from_pretrained(model_name)
        self.processor = BlipProcessor.from_pretrained(model_name)
        
        # Training hyperparameters
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.max_epochs = max_epochs
        
        # Generation parameters
        self.num_beams = num_beams
        self.max_length = max_length
        self.min_length = min_length
        
        # Metrics storage
        self.validation_outputs = []
        
    def forward(self, pixel_values, input_ids, attention_mask):
        """Forward pass"""
        outputs = self.model(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=input_ids,
            return_dict=True
        )
        return outputs.loss
    
    def training_step(self, batch, batch_idx):
        """Training step"""
        # Extract data from batch
        pixel_values = batch['pixel_values']
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        
        # Forward pass
        loss = self(pixel_values, input_ids, attention_mask)
        
        # Log metrics
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        """Validation step"""
        pixel_values = batch['pixel_values']
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        
        # Calculate loss
        loss = self(pixel_values, input_ids, attention_mask)
        
        # Generate captions for evaluation
        generated_ids = self.model.generate(
            pixel_values=pixel_values,
            num_beams=self.num_beams,
            max_length=self.max_length,
            min_length=self.min_length,
        )
        
        # Decode generated and reference captions
        generated_captions = self.processor.batch_decode(generated_ids, skip_special_tokens=True)
        reference_captions = self.processor.batch_decode(input_ids, skip_special_tokens=True)
        
        # Store for epoch-end processing
        self.validation_outputs.append({
            'loss': loss,
            'generated': generated_captions,
            'reference': reference_captions
        })
        
        # Log metrics
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        
        return loss
    
    def on_validation_epoch_end(self):
        """Process validation results at epoch end"""
        # Calculate average loss
        avg_loss = torch.stack([x['loss'] for x in self.validation_outputs]).mean()
        
        # Log sample predictions
        if self.global_rank == 0:  # Only on main process
            print("\n" + "="*80)
            print("Sample Predictions:")
            print("="*80)
            for i in range(min(5, len(self.validation_outputs[0]['generated']))):
                print(f"\nSample {i+1}:")
                print(f"  Generated: {self.validation_outputs[0]['generated'][i]}")
                print(f"  Reference: {self.validation_outputs[0]['reference'][i]}")
            print("="*80 + "\n")
        
        # Clear outputs for next epoch
        self.validation_outputs.clear()
    
    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler"""
        # Separate parameters for different learning rates
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {
                'params': [p for n, p in self.model.named_parameters() 
                          if not any(nd in n for nd in no_decay)],
                'weight_decay': self.weight_decay,
            },
            {
                'params': [p for n, p in self.model.named_parameters() 
                          if any(nd in n for nd in no_decay)],
                'weight_decay': 0.0,
            },
        ]
        
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.learning_rate)
        
        # Cosine annealing scheduler
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=self.trainer.max_epochs,
            eta_min=self.learning_rate * 0.1
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch',
                'frequency': 1
            }
        }
    
    def generate_caption(self, image):
        """Generate caption for a single image"""
        self.eval()
        with torch.no_grad():
            # Process image
            inputs = self.processor(images=image, return_tensors="pt")
            pixel_values = inputs.pixel_values.to(self.device)
            
            # Generate caption
            generated_ids = self.model.generate(
                pixel_values=pixel_values,
                num_beams=self.num_beams,
                max_length=self.max_length,
                min_length=self.min_length,
            )
            
            # Decode caption
            caption = self.processor.decode(generated_ids[0], skip_special_tokens=True)
            
        return caption


def train_blip(args):
    """Main training function"""
    
    # Set seed for reproducibility
    pl.seed_everything(args.seed)
    
    # Create output directories
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output_dir / 'checkpoints'
    checkpoint_dir.mkdir(exist_ok=True)
    
    print("="*80)
    print("BLIP Image Captioning Training")
    print("="*80)
    print(f"Output directory: {output_dir}")
    print(f"Using device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    print(f"Number of GPUs: {torch.cuda.device_count()}")
    print("="*80 + "\n")
    
    # Initialize data module
    print("Initializing data module...")
    datamodule = COCODataModule(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_length=args.max_length,
        use_subset=args.use_subset,
        subset_size=args.subset_size if args.use_subset else None,
        subset_dir=args.subset_dir,
        force_recreate_subset=args.force_recreate_subset
    )
    
    # Prepare data (download/extract if needed, create subset if requested)
    if not args.skip_prepare_data:
        print("Preparing data...")
        datamodule.prepare_data()
    
    # Setup datasets
    print("Setting up datasets...")
    datamodule.setup('fit')
    
    print(f"Training samples: {len(datamodule.train_dataset)}")
    print(f"Validation samples: {len(datamodule.val_dataset)}")
    print()

    device = torch.device(args.device)
     #### Model #### 
    print("Creating model")
    model = blip_decoder(pretrained=config['pretrained'], image_size=config['image_size'], vit=config['vit'], 
                           vit_grad_ckpt=config['vit_grad_ckpt'], vit_ckpt_layer=config['vit_ckpt_layer'], 
                           prompt=config['prompt'])

    model = model.to(device)   
    
    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module    

    # Initialize model
    """
    print("Initializing BLIP model...")
    model = BLIPCaptioningModel(
        model_name=args.model_name,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        max_epochs=args.max_epochs,
        num_beams=args.num_beams,
        max_length=args.max_length,
        min_length=args.min_length,
    )
    """
    print(f"Model: {args.model_name}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    print()
    
    # Callbacks
    callbacks = []
    
    # Model checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename='blip-{epoch:02d}-{val_loss:.4f}',
        save_top_k=3,
        monitor='val_loss',
        mode='min',
        save_last=True,
        verbose=True
    )
    callbacks.append(checkpoint_callback)
    
    # Learning rate monitor
    lr_monitor = LearningRateMonitor(logging_interval='step')
    callbacks.append(lr_monitor)
    
    # Early stopping
    if args.early_stopping:
        early_stop_callback = EarlyStopping(
            monitor='val_loss',
            patience=args.patience,
            mode='min',
            verbose=True
        )
        callbacks.append(early_stop_callback)
    
    # Loggers
    loggers = []
    
    # TensorBoard logger
    tb_logger = TensorBoardLogger(
        save_dir=output_dir,
        name='tensorboard_logs',
        version=args.experiment_name
    )
    loggers.append(tb_logger)
    
    # CSV logger
    csv_logger = CSVLogger(
        save_dir=output_dir,
        name='csv_logs',
        version=args.experiment_name
    )
    loggers.append(csv_logger)
    
    # Initialize trainer
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator='auto',
        devices='auto',
        strategy='ddp' if torch.cuda.device_count() > 1 else 'auto',
        precision=args.precision,
        gradient_clip_val=args.grad_clip,
        accumulate_grad_batches=args.accumulate_grad_batches,
        log_every_n_steps=args.log_interval,
        val_check_interval=args.val_check_interval,
        callbacks=callbacks,
        logger=loggers,
        enable_progress_bar=True,
        enable_model_summary=True,
    )
    
    # Save configuration
    config = vars(args)
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    # Train model
    print("Starting training...")
    start_time = time.time()
    
    if args.resume_from_checkpoint:
        print(f"Resuming from checkpoint: {args.resume_from_checkpoint}")
        trainer.fit(model, datamodule, ckpt_path=args.resume_from_checkpoint)
    else:
        trainer.fit(model, datamodule)
    
    # Training completed
    training_time = time.time() - start_time
    print(f"\nTraining completed in {str(datetime.timedelta(seconds=int(training_time)))}")
    print(f"Best model checkpoint: {checkpoint_callback.best_model_path}")
    print(f"Best validation loss: {checkpoint_callback.best_model_score:.4f}")
    
    # Save training summary
    summary = {
        'training_time': str(datetime.timedelta(seconds=int(training_time))),
        'best_checkpoint': str(checkpoint_callback.best_model_path),
        'best_val_loss': float(checkpoint_callback.best_model_score),
        'total_epochs': trainer.current_epoch + 1,
        'config': config
    }
    
    with open(output_dir / 'training_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nTraining summary saved to {output_dir / 'training_summary.json'}")
    
    return model, trainer


def main():
    parser = argparse.ArgumentParser(description='Train BLIP model on COCO captions')
    
    # Data arguments
    parser.add_argument('--data_dir', type=str, default='../data',
                       help='Directory containing COCO dataset')
    parser.add_argument('--use_subset', action='store_true',
                       help='Use subset of data for faster training')
    parser.add_argument('--subset_size', type=int, default=5000,
                       help='Size of training subset (if use_subset=True)')
    parser.add_argument('--subset_dir', type=str, default='../data/subsets',
                       help='Directory to save/load subsets')
    parser.add_argument('--force_recreate_subset', action='store_true',
                       help='Force recreation of subset')
    parser.add_argument('--skip_prepare_data', action='store_true',
                       help='Skip data preparation (assume data is ready)')
    
    # Model arguments
    parser.add_argument('--model_name', type=str, 
                       default='Salesforce/blip-image-captioning-base',
                       help='BLIP model name from HuggingFace')
    parser.add_argument('--max_length', type=int, default=30,
                       help='Maximum caption length')
    parser.add_argument('--min_length', type=int, default=5,
                       help='Minimum caption length')
    parser.add_argument('--num_beams', type=int, default=3,
                       help='Number of beams for generation')
    
    # Training arguments
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size per GPU')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loading workers')
    parser.add_argument('--max_epochs', type=int, default=10,
                       help='Maximum number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-5,
                       help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.05,
                       help='Weight decay')
    parser.add_argument('--warmup_steps', type=int, default=1000,
                       help='Number of warmup steps')
    parser.add_argument('--grad_clip', type=float, default=1.0,
                       help='Gradient clipping value')
    parser.add_argument('--accumulate_grad_batches', type=int, default=1,
                       help='Accumulate gradients over N batches')
    
    # Logging and checkpointing
    parser.add_argument('--output_dir', type=str, default='./output/blip_training',
                       help='Output directory for checkpoints and logs')
    parser.add_argument('--experiment_name', type=str, default='blip_coco',
                       help='Experiment name for logging')
    parser.add_argument('--log_interval', type=int, default=50,
                       help='Log every N steps')
    parser.add_argument('--val_check_interval', type=float, default=1.0,
                       help='Validation check interval (float for fraction of epoch)')
    
    # Other arguments
    parser.add_argument('--precision', type=str, default='16-mixed',
                       choices=['32', '16-mixed', 'bf16-mixed'],
                       help='Training precision')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--early_stopping', action='store_true',
                       help='Enable early stopping')
    parser.add_argument('--patience', type=int, default=3,
                       help='Early stopping patience')
    parser.add_argument('--resume_from_checkpoint', type=str, default=None,
                       help='Path to checkpoint to resume from')
    
    args = parser.parse_args()
    
    # Train model
    model, trainer = train_blip(args)
    
    print("\n" + "="*80)
    print("Training completed successfully!")
    print("="*80)


if __name__ == '__main__':
    main()