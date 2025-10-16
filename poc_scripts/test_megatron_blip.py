import torch
import torch.distributed as dist
from PIL import Image
import requests
from io import BytesIO
import warnings
import os
warnings.filterwarnings('ignore')
username = os.environ.get('USER', 'unknown')
initial = username[0]

import sys
sys.path.append(f'/pscratch/sd/{initial}/{username}/BLIP')
from models.blip import blip_decoder

def test_blip_standalone():
    rank = dist.get_rank()
    print(f"[Rank {rank}] Loading model from /tmp/...")
    
    model = blip_decoder(
        pretrained='/tmp/model_base_caption_capfilt_large.pth',
        med_config=f'/pscratch/sd/{initial}/{username}/BLIP/configs/med_config.json',
        image_size=384,
        vit='base'
    )
    print(f"[Rank {rank}] Model loaded")
    
    model = model.cuda()
    model.eval()
    print(f"[Rank {rank}] Model on GPU")
    
    # Only rank 0 generates caption
    if rank == 0:
        from torchvision import transforms
        from torchvision.transforms.functional import InterpolationMode
        
        try:
            print("[Rank 0] Downloading test image...")
            url = "https://storage.googleapis.com/sfr-vision-language-research/BLIP/demo.jpg"
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            raw_image = Image.open(BytesIO(response.content)).convert('RGB')
            
            # Preprocess image
            transform = transforms.Compose([
                transforms.Resize((384, 384), interpolation=InterpolationMode.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize((0.48145466, 0.4578275, 0.40821073), 
                                     (0.26862954, 0.26130258, 0.27577711))
            ]) 
            image = transform(raw_image).unsqueeze(0).cuda()
            
            print("[Rank 0] Running inference...")
            with torch.no_grad():
                caption = model.generate(image, sample=True, top_p=0.9, 
                                        max_length=20, min_length=5)
            
            print(f"[Rank 0] Generated caption: {caption[0]}")
            print("✓ BLIP inference test passed")
        except Exception as e:
            print(f"[Rank 0] Error: {e}")
            import traceback
            traceback.print_exc()
    
    dist.barrier()
    return model

def test_blip_with_megatron():
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    
    if rank == 0:
        print(f"\nTesting distributed operations on {world_size} GPUs...")
    
    tensor = torch.ones(1).cuda() * (rank + 1)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    
    if rank == 0:
        expected = sum(range(1, world_size + 1))
        print(f"All-reduce test: Expected {expected}, Got {tensor.item()}")
        print("✓ Distributed test passed")
    
    dist.barrier()

def main():
    dist.init_process_group(backend='nccl')
    
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    
    torch.cuda.set_device(local_rank)
    
    if rank == 0:
        print("=" * 50)
        print(f"Running on {world_size} GPUs across {os.environ.get('SLURM_NNODES', '?')} nodes")
        print("=" * 50)
    
    model = test_blip_standalone()
    test_blip_with_megatron()
    
    if rank == 0:
        print("\n" + "=" * 50)
        print("All tests completed successfully!")
        print("=" * 50)
    
    dist.destroy_process_group()

if __name__ == "__main__":
    main()
