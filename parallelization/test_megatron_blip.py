# test_blip_megatron.py

import torch
import torch.distributed as dist
from megatron.core import parallel_state
from megatron.core import tensor_parallel
from PIL import Image
import requests
from io import BytesIO
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

import sys
sys.path.append('/pscratch/sd/t/te137/BLIP')
from models.blip import blip_decoder

def initialize_megatron(tensor_model_parallel_size=2):
    """Initialize Megatron distributed environment"""
    if not dist.is_initialized():
        dist.init_process_group(backend='nccl')
    
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    
    # Only initialize model parallel after process group exists
    parallel_state.initialize_model_parallel(
        tensor_model_parallel_size=tensor_model_parallel_size,
        pipeline_model_parallel_size=1
    )
    
    return rank, world_size


def parallelize_blip_vision_encoder(vision_encoder):
    """Apply tensor parallelism to vision encoder (ViT)"""
    # Parallelize attention layers
    for block in vision_encoder.blocks:
        # Split attention QKV projections
        block.attn.qkv = tensor_parallel.ColumnParallelLinear(
            block.attn.qkv.in_features,
            block.attn.qkv.out_features,
            bias=True,
            gather_output=False,
            init_method=lambda x: x
        )
        
        # Split attention output projection
        block.attn.proj = tensor_parallel.RowParallelLinear(
            block.attn.proj.in_features,
            block.attn.proj.out_features,
            bias=True,
            input_is_parallel=True,
            init_method=lambda x: x
        )
        
        # Split MLP layers
        block.mlp.fc1 = tensor_parallel.ColumnParallelLinear(
            block.mlp.fc1.in_features,
            block.mlp.fc1.out_features,
            bias=True,
            gather_output=False,
            init_method=lambda x: x
        )
        
        block.mlp.fc2 = tensor_parallel.RowParallelLinear(
            block.mlp.fc2.in_features,
            block.mlp.fc2.out_features,
            bias=True,
            input_is_parallel=True,
            init_method=lambda x: x
        )
    
    return vision_encoder


def parallelize_blip_text_decoder(text_decoder):
    """Apply tensor parallelism to text decoder (BERT-based)"""
    for layer in text_decoder.bert.encoder.layer:
        # Split attention Q, K, V
        layer.attention.self.query = tensor_parallel.ColumnParallelLinear(
            layer.attention.self.query.in_features,
            layer.attention.self.query.out_features,
            bias=True,
            gather_output=False,
            init_method=lambda x: x
        )
        
        layer.attention.self.key = tensor_parallel.ColumnParallelLinear(
            layer.attention.self.key.in_features,
            layer.attention.self.key.out_features,
            bias=True,
            gather_output=False,
            init_method=lambda x: x
        )
        
        layer.attention.self.value = tensor_parallel.ColumnParallelLinear(
            layer.attention.self.value.in_features,
            layer.attention.self.value.out_features,
            bias=True,
            gather_output=False,
            init_method=lambda x: x
        )
        
        # Split attention output
        layer.attention.output.dense = tensor_parallel.RowParallelLinear(
            layer.attention.output.dense.in_features,
            layer.attention.output.dense.out_features,
            bias=True,
            input_is_parallel=True,
            init_method=lambda x: x
        )
        
        # Split FFN layers
        layer.intermediate.dense = tensor_parallel.ColumnParallelLinear(
            layer.intermediate.dense.in_features,
            layer.intermediate.dense.out_features,
            bias=True,
            gather_output=False,
            init_method=lambda x: x
        )
        
        layer.output.dense = tensor_parallel.RowParallelLinear(
            layer.output.dense.in_features,
            layer.output.dense.out_features,
            bias=True,
            input_is_parallel=True,
            init_method=lambda x: x
        )
    
    return text_decoder


def test_blip_with_megatron(tensor_parallel_size=2):
    """Test BLIP with Megatron model parallelism"""
    print("Testing BLIP with Megatron model parallelism...")
    
    # Initialize distributed environment
    rank, world_size = initialize_megatron(tensor_parallel_size)
    local_rank = rank % torch.cuda.device_count()
    torch.cuda.set_device(local_rank)
    
    print(f"Rank {rank}/{world_size} initialized on GPU {local_rank}")
    print(f"Tensor parallel group: {parallel_state.get_tensor_model_parallel_group()}")
    print(f"Tensor parallel world size: {parallel_state.get_tensor_model_parallel_world_size()}")
    
    # Load BLIP model
    model = blip_decoder(
        pretrained='/pscratch/sd/t/te137/BLIP/checkpoints/model_base_caption_capfilt_large.pth',
        image_size=384,
        vit='base'
    )
    
    # Apply model parallelism
    print("Applying tensor parallelism to vision encoder...")
    model.visual_encoder = parallelize_blip_vision_encoder(model.visual_encoder)
    
    print("Applying tensor parallelism to text decoder...")
    model.text_decoder = parallelize_blip_text_decoder(model.text_decoder)
    
    model = model.cuda()
    model.eval()
    
    # Load and process image (only on rank 0)
    if rank == 0:
        import torchvision.transforms as transforms
        
        transform = transforms.Compose([
            transforms.Resize((384, 384)),
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), 
                               (0.26862954, 0.26130258, 0.27577711))
        ])
        
        url = "https://storage.googleapis.com/sfr-vision-language-research/BLIP/demo.jpg"
        session = requests.Session()
        response = session.get(url, timeout=5)
        image = Image.open(BytesIO(response.content)).convert('RGB')
        image_tensor = transform(image).unsqueeze(0).cuda()
    else:
        image_tensor = torch.empty(1, 3, 384, 384).cuda()
    
    # Broadcast image to all ranks
    dist.broadcast(image_tensor, src=0)
    
    # Generate caption with model parallelism
    with torch.no_grad():
        # Convert PIL image to tensor for the model
        if rank == 0:
            url = "https://storage.googleapis.com/sfr-vision-language-research/BLIP/demo.jpg"
            session = requests.Session()
            response = session.get(url, timeout=5)
            image = Image.open(BytesIO(response.content)).convert('RGB')
        else:
            # Create dummy image on other ranks
            image = Image.new('RGB', (384, 384))
        
        caption = model.generate(image, sample=False, num_beams=3, 
                                max_length=20, min_length=5)
    
    if rank == 0:
        print(f"\nCaption: {caption[0]}")
        print("✓ BLIP with Megatron model parallel test passed\n")
    
    # Clean up
    parallel_state.destroy_model_parallel()
    
    return model


def main():
    """Main execution function"""
    print("="*60)
    print("BLIP Model Parallel Training with Megatron")
    print("="*60)
    
    # Test with 2-way tensor parallelism
    tensor_parallel_size = 2
    
    print(f"\nInitializing with tensor_parallel_size={tensor_parallel_size}")
    model = test_blip_with_megatron(tensor_parallel_size)
    
    print("\nModel parallelism setup complete!")
    print(f"Vision encoder parallelized: ✓")
    print(f"Text decoder parallelized: ✓")


if __name__ == "__main__":
    main()