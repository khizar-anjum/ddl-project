import torch
import torch.nn as nn
import pytorch_lightning as pl
from transformers import BlipForConditionalGeneration, BlipProcessor
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from typing import Optional, Dict, Any
import evaluate

class ImageEncoder(nn.module):
    pass
class textEncoder(nn.module):
    pass


class BlipModel():
    def __init__():
        pass
    def forward():
        pass
    

class BLIPCaptioningModel(pl.LightningModule):
    """
    PyTorch Lightning module for training BLIP model on image captioning task
    """
    
    def __init__(
        self,
        model_name: str = "Salesforce/blip-image-captioning-base",
        learning_rate: float = 5e-5,
        weight_decay: float = 0.01,
        warmup_steps: int = 500,
        max_epochs: int = 10,
        gradient_clip_val: float = 1.0,
        freeze_vision_encoder: bool = False,
        freeze_text_encoder: bool = False,
    ):
        """
        Args:
            model_name: Pretrained BLIP model name from HuggingFace
            learning_rate: Learning rate for optimizer
            weight_decay: Weight decay for optimizer
            warmup_steps: Number of warmup steps for learning rate scheduler
            max_epochs: Maximum number of training epochs
            gradient_clip_val: Gradient clipping value
            freeze_vision_encoder: Whether to freeze vision encoder weights
            freeze_text_encoder: Whether to freeze text encoder weights
        """
        super().__init__()
        self.save_hyperparameters()
        
        # Load pretrained BLIP model
        self.model = BlipForConditionalGeneration.from_pretrained(model_name)
        self.processor = BlipProcessor.from_pretrained(model_name)
        
        # Freeze encoders if specified
        if freeze_vision_encoder:
            print("Freezing vision encoder...")
            for param in self.model.vision_model.parameters():
                param.requires_grad = False
        
        if freeze_text_encoder:
            print("Freezing text encoder...")
            for param in self.model.text_encoder.parameters():
                param.requires_grad = False
        
        # Initialize metrics
        self.train_losses = []
        self.val_losses = []
        
        # Load BLEU metric for evaluation
        self.bleu_metric = evaluate.load("bleu")
        
        print(f"Model initialized: {model_name}")
        print(f"Total parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"Trainable parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}")
    
    def forward(self, pixel_values, input_ids, attention_mask=None):
        """Forward pass through the model"""
        outputs = self.model(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        return outputs
    
    def training_step(self, batch, batch_idx):
        """Training step"""
        outputs = self(
            pixel_values=batch['pixel_values'],
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask']
        )
        
        loss = outputs.loss
        
        # Log metrics
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.train_losses.append(loss.item())
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        """Validation step"""
        outputs = self(
            pixel_values=batch['pixel_values'],
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask']
        )
        
        loss = outputs.loss
        
        # Generate captions for BLEU score calculation
        with torch.no_grad():
            generated_ids = self.model.generate(
                pixel_values=batch['pixel_values'],
                max_length=50,
                num_beams=3
            )
        
        # Decode generated and reference captions
        generated_captions = self.processor.batch_decode(generated_ids, skip_special_tokens=True)
        reference_captions = self.processor.batch_decode(batch['input_ids'], skip_special_tokens=True)
        
        # Log metrics
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.val_losses.append(loss.item())
        
        return {
            'val_loss': loss,
            'generated_captions': generated_captions,
            'reference_captions': reference_captions
        }
    
    def on_validation_epoch_end(self):
        """Calculate and log BLEU score at the end of validation epoch"""
        # This is called automatically by PyTorch Lightning
        pass
    
    def predict_step(self, batch, batch_idx):
        """Prediction step for generating captions"""
        with torch.no_grad():
            generated_ids = self.model.generate(
                pixel_values=batch['pixel_values'],
                max_length=50,
                num_beams=5,
                early_stopping=True
            )
        
        captions = self.processor.batch_decode(generated_ids, skip_special_tokens=True)
        return captions
    
    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler"""
        # Separate parameters for different learning rates (optional)
        optimizer = AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        # Cosine annealing scheduler
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=self.hparams.max_epochs,
            eta_min=1e-6
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch',
                'frequency': 1
            }
        }
    
    def generate_caption(self, image, max_length=50, num_beams=5):
        """
        Generate caption for a single image
        
        Args:
            image: PIL Image or tensor
            max_length: Maximum caption length
            num_beams: Number of beams for beam search
            
        Returns:
            Generated caption string
        """
        self.eval()
        
        # Process image
        if not isinstance(image, torch.Tensor):
            inputs = self.processor(images=image, return_tensors="pt")
            pixel_values = inputs.pixel_values.to(self.device)
        else:
            pixel_values = image.unsqueeze(0).to(self.device) if image.dim() == 3 else image.to(self.device)
        
        # Generate caption
        with torch.no_grad():
            generated_ids = self.model.generate(
                pixel_values=pixel_values,
                max_length=max_length,
                num_beams=num_beams,
                early_stopping=True
            )
        
        caption = self.processor.decode(generated_ids[0], skip_special_tokens=True)
        return caption


def train_blip_model(
    data_module,
    model_name: str = "Salesforce/blip-image-captioning-base",
    max_epochs: int = 10,
    learning_rate: float = 5e-5,
    accumulate_grad_batches: int = 1,
    precision: str = '16-mixed',
    checkpoint_dir: str = './checkpoints',
    freeze_vision_encoder: bool = False,
    freeze_text_encoder: bool = False,
):
    """
    Train BLIP model with the given data module
    
    Args:
        data_module: PyTorch Lightning DataModule
        model_name: Pretrained model name
        max_epochs: Number of training epochs
        learning_rate: Learning rate
        accumulate_grad_batches: Gradient accumulation steps
        precision: Training precision ('32', '16-mixed', 'bf16-mixed')
        checkpoint_dir: Directory to save checkpoints
        freeze_vision_encoder: Whether to freeze vision encoder
        freeze_text_encoder: Whether to freeze text encoder
    
    Returns:
        Trained model
    """
    # Initialize model
    model = BLIPCaptioningModel(
        model_name=model_name,
        learning_rate=learning_rate,
        max_epochs=max_epochs,
        freeze_vision_encoder=freeze_vision_encoder,
        freeze_text_encoder=freeze_text_encoder,
    )
    
    # Callbacks
    from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
    
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename='blip-{epoch:02d}-{val_loss:.4f}',
        save_top_k=3,
        monitor='val_loss',
        mode='min',
        save_last=True,
        verbose=True
    )
    
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=3,
        mode='min',
        verbose=True
    )
    
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    
    # Determine accelerator
    accelerator = 'gpu' if torch.cuda.is_available() else 'cpu'
    devices = torch.cuda.device_count() if torch.cuda.is_available() else 1
    
    # Initialize trainer
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator=accelerator,
        devices=devices,
        precision=precision if accelerator == 'gpu' else '32',
        accumulate_grad_batches=accumulate_grad_batches,
        gradient_clip_val=1.0,
        callbacks=[checkpoint_callback, early_stopping, lr_monitor],
        log_every_n_steps=50,
        val_check_interval=0.5,  # Validate twice per epoch
        enable_progress_bar=True,
        enable_model_summary=True,
    )
    
    # Train model
    print("\n" + "="*60)
    print("Starting Training")
    print("="*60)
    print(f"Accelerator: {accelerator}")
    print(f"Devices: {devices}")
    print(f"Precision: {precision if accelerator == 'gpu' else '32'}")
    print(f"Max Epochs: {max_epochs}")
    print(f"Learning Rate: {learning_rate}")
    print(f"Batch Size: {data_module.batch_size}")
    print(f"Accumulate Grad Batches: {accumulate_grad_batches}")
    print("="*60 + "\n")
    
    trainer.fit(model, data_module)
    
    print("\n" + "="*60)
    print("Training Completed!")
    print(f"Best model checkpoint: {checkpoint_callback.best_model_path}")
    print("="*60 + "\n")
    
    return model, trainer
