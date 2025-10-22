# test_parallel_blip.py

import torch
import torch.distributed as dist
import time
import os
import argparse
from PIL import Image
import requests
from io import BytesIO
import torchvision.transforms as transforms

# Import your parallel BLIP implementation
import sys
sys.path.append('/pscratch/sd/t/te137/BLIP')
from test_megatron_blip import initialize_megatron, parallelize_blip_vision_encoder, parallelize_blip_text_decoder
from models.blip import blip_decoder


def load_test_images():
    """Load multiple test images"""
    urls = [
        "https://storage.googleapis.com/sfr-vision-language-research/BLIP/demo.jpg",
        "https://images.unsplash.com/photo-1583511655857-d19b40a7a54e",  # dog
        "https://images.unsplash.com/photo-1514888286974-6c03e2ca1dba",  # cat
    ]
    
    images = []
    for url in urls:
        try:
            response = requests.get(url, timeout=10)
            img = Image.open(BytesIO(response.content)).convert('RGB')
            images.append(img)
        except:
            # Fallback: create synthetic test image
            images.append(Image.new('RGB', (384, 384), color=(100, 150, 200)))
    
    return images


def test_correctness(rank, world_size, args):
    """Compare parallel output with standalone BLIP"""
    if rank == 0:
        print(f"\n{'='*60}")
        print("TEST 1/4: CORRECTNESS CHECK")
        print(f"{'='*60}")
    
    print(f"[Rank {rank}] Starting correctness test...")
    
    # Load reference model (standalone) on rank 0
    if rank == 0:
        print("[Rank 0] Loading standalone BLIP for reference...")
        ref_model = blip_decoder(
            pretrained='/pscratch/sd/k/kanjum/BLIP/checkpoints/model_base_caption_capfilt_large.pth',
            image_size=384,
            vit='base'
        ).cuda().eval()
        print("[Rank 0] ✓ Standalone model loaded")
    
    # Load parallel model with sequential loading to avoid I/O contention
    print(f"[Rank {rank}] Initializing Megatron with {world_size} GPUs...")
    rank, world_size = initialize_megatron(tensor_model_parallel_size=args.tensor_parallel_size)
    local_rank = rank % torch.cuda.device_count()
    torch.cuda.set_device(local_rank)
    print(f"[Rank {rank}] ✓ Megatron initialized, using GPU {local_rank}")
    
    # Sequential loading: rank 0 loads first, then others
    if rank == 0:
        print("[Rank 0] Loading base BLIP model...")
        parallel_model = blip_decoder(
            pretrained='/pscratch/sd/k/kanjum/BLIP/checkpoints/model_base_caption_capfilt_large.pth',
            image_size=384,
            vit='base'
        )
        print("[Rank 0] ✓ Base model loaded")
    
    dist.barrier()  # Wait for rank 0
    
    if rank != 0:
        print(f"[Rank {rank}] Loading base BLIP model...")
        parallel_model = blip_decoder(
            pretrained='/pscratch/sd/k/kanjum/BLIP/checkpoints/model_base_caption_capfilt_large.pth',
            image_size=384,
            vit='base'
        )
        print(f"[Rank {rank}] ✓ Base model loaded")
    
    dist.barrier()
    
    print(f"[Rank {rank}] Applying tensor parallelism to vision encoder...")
    parallel_model.visual_encoder = parallelize_blip_vision_encoder(parallel_model.visual_encoder)
    print(f"[Rank {rank}] ✓ Vision encoder parallelized")
    
    print(f"[Rank {rank}] Applying tensor parallelism to text decoder...")
    parallel_model.text_decoder = parallelize_blip_text_decoder(parallel_model.text_decoder)
    print(f"[Rank {rank}] ✓ Text decoder parallelized")
    
    parallel_model = parallel_model.cuda().eval()
    print(f"[Rank {rank}] ✓ Model moved to GPU and set to eval mode")
    
    # Test image
    if rank == 0:
        print("[Rank 0] Loading test image...")
    test_img = load_test_images()[0]
    if rank == 0:
        print("[Rank 0] ✓ Test image loaded")
    
    with torch.no_grad():
        # Generate with parallel model
        if rank == 0:
            print("[Rank 0] Generating caption with parallel model...")
        parallel_caption = parallel_model.generate(
            test_img, sample=False, num_beams=3, max_length=20, min_length=5
        )
        if rank == 0:
            print("[Rank 0] ✓ Parallel generation complete")
        
        # Generate with reference model (rank 0 only)
        if rank == 0:
            print("[Rank 0] Generating caption with reference model...")
            ref_caption = ref_model.generate(
                test_img, sample=False, num_beams=3, max_length=20, min_length=5
            )
            print("[Rank 0] ✓ Reference generation complete")
            
            print("\n" + "="*60)
            print("CORRECTNESS TEST RESULTS")
            print("="*60)
            print(f"Reference caption: {ref_caption[0]}")
            print(f"Parallel caption:  {parallel_caption[0]}")
            match = ref_caption[0] == parallel_caption[0]
            print(f"Match: {'✓ PASS' if match else '✗ FAIL'}")
            print("="*60 + "\n")


def test_performance(rank, world_size, args):
    """Measure throughput and speedup"""
    if rank == 0:
        print(f"\n{'='*60}")
        print("TEST 2/4: PERFORMANCE BENCHMARK")
        print(f"{'='*60}")
    
    print(f"[Rank {rank}] Starting performance test...")
    
    print(f"[Rank {rank}] Initializing Megatron...")
    rank, world_size = initialize_megatron(tensor_model_parallel_size=args.tensor_parallel_size)
    local_rank = rank % torch.cuda.device_count()
    torch.cuda.set_device(local_rank)
    print(f"[Rank {rank}] ✓ Using GPU {local_rank}")
    
    # Sequential loading
    if rank == 0:
        print("[Rank 0] Loading parallel BLIP model...")
        model = blip_decoder(
            pretrained='/pscratch/sd/k/kanjum/BLIP/checkpoints/model_base_caption_capfilt_large.pth',
            image_size=384,
            vit='base'
        )
    
    dist.barrier()
    
    if rank != 0:
        print(f"[Rank {rank}] Loading parallel BLIP model...")
        model = blip_decoder(
            pretrained='/pscratch/sd/k/kanjum/BLIP/checkpoints/model_base_caption_capfilt_large.pth',
            image_size=384,
            vit='base'
        )
    
    dist.barrier()
    
    model.visual_encoder = parallelize_blip_vision_encoder(model.visual_encoder)
    model.text_decoder = parallelize_blip_text_decoder(model.text_decoder)
    model = model.cuda().eval()
    print(f"[Rank {rank}] ✓ Model loaded and parallelized")
    
    # Load test images
    if rank == 0:
        print("[Rank 0] Loading test images...")
    images = load_test_images()
    if rank == 0:
        print(f"[Rank 0] ✓ Loaded {len(images)} test images")
    
    # Warmup
    if rank == 0:
        print("[Rank 0] Running warmup (3 iterations)...")
    with torch.no_grad():
        for i in range(3):
            _ = model.generate(images[0], sample=False, num_beams=1, max_length=10)
            if rank == 0:
                print(f"[Rank 0]   Warmup iteration {i+1}/3 complete")
    
    torch.cuda.synchronize()
    dist.barrier()
    if rank == 0:
        print("[Rank 0] ✓ Warmup complete")
    
    # Benchmark
    num_iterations = 10
    if rank == 0:
        print(f"[Rank 0] Running benchmark ({num_iterations} iterations)...")
    
    start_time = time.time()
    
    with torch.no_grad():
        for i in range(num_iterations):
            img = images[i % len(images)]
            _ = model.generate(img, sample=False, num_beams=3, max_length=20, min_length=5)
            if rank == 0 and (i + 1) % 2 == 0:
                print(f"[Rank 0]   Completed {i+1}/{num_iterations} iterations")
    
    torch.cuda.synchronize()
    dist.barrier()
    end_time = time.time()
    
    if rank == 0:
        avg_time = (end_time - start_time) / num_iterations
        throughput = 1.0 / avg_time
        
        print("\n" + "="*60)
        print("PERFORMANCE TEST")
        print("="*60)
        print(f"World size: {world_size} GPUs")
        print(f"Iterations: {num_iterations}")
        print(f"Avg time per image: {avg_time*1000:.2f} ms")
        print(f"Throughput: {throughput:.2f} images/sec")
        print("="*60 + "\n")


def test_memory_usage(rank, world_size, args):
    """Check memory consumption per GPU"""
    if rank == 0:
        print(f"\n{'='*60}")
        print("TEST 3/4: MEMORY USAGE")
        print(f"{'='*60}")
    
    print(f"[Rank {rank}] Starting memory test...")
    
    print(f"[Rank {rank}] Initializing Megatron...")
    rank, world_size = initialize_megatron(tensor_model_parallel_size=args.tensor_parallel_size)
    local_rank = rank % torch.cuda.device_count()
    torch.cuda.set_device(local_rank)
    print(f"[Rank {rank}] ✓ Using GPU {local_rank}")
    
    torch.cuda.reset_peak_memory_stats()
    print(f"[Rank {rank}] Memory stats reset")
    
    # Sequential loading
    if rank == 0:
        print("[Rank 0] Loading model...")
        model = blip_decoder(
            pretrained='/pscratch/sd/k/kanjum/BLIP/checkpoints/model_base_caption_capfilt_large.pth',
            image_size=384,
            vit='base'
        )
    
    dist.barrier()
    
    if rank != 0:
        print(f"[Rank {rank}] Loading model...")
        model = blip_decoder(
            pretrained='/pscratch/sd/k/kanjum/BLIP/checkpoints/model_base_caption_capfilt_large.pth',
            image_size=384,
            vit='base'
        )
    
    dist.barrier()
    
    model.visual_encoder = parallelize_blip_vision_encoder(model.visual_encoder)
    model.text_decoder = parallelize_blip_text_decoder(model.text_decoder)
    model = model.cuda().eval()
    print(f"[Rank {rank}] ✓ Model loaded")
    
    model_memory = torch.cuda.memory_allocated() / 1024**2  # MB
    print(f"[Rank {rank}] Model memory allocated: {model_memory:.2f} MB")
    
    # Run inference
    if rank == 0:
        print("[Rank 0] Running inference to measure peak memory...")
    test_img = load_test_images()[0]
    with torch.no_grad():
        _ = model.generate(test_img, sample=False, num_beams=3, max_length=20)
    print(f"[Rank {rank}] ✓ Inference complete")
    
    peak_memory = torch.cuda.max_memory_allocated() / 1024**2  # MB
    print(f"[Rank {rank}] Peak memory: {peak_memory:.2f} MB")
    
    # Gather from all ranks
    memory_tensor = torch.tensor([model_memory, peak_memory], device='cuda')
    memory_list = [torch.zeros(2, device='cuda') for _ in range(world_size)]
    dist.all_gather(memory_list, memory_tensor)
    
    if rank == 0:
        print("\n" + "="*60)
        print("MEMORY USAGE TEST")
        print("="*60)
        for i, mem in enumerate(memory_list):
            print(f"GPU {i}:")
            print(f"  Model memory: {mem[0].item():.2f} MB")
            print(f"  Peak memory:  {mem[1].item():.2f} MB")
        print(f"\nTotal model memory: {sum(m[0].item() for m in memory_list):.2f} MB")
        print("="*60 + "\n")


def test_batch_processing(rank, world_size, args):
    """Test with different batch sizes"""
    if rank == 0:
        print(f"\n{'='*60}")
        print("TEST 4/4: BATCH PROCESSING")
        print(f"{'='*60}")
    
    print(f"[Rank {rank}] Starting batch processing test...")
    
    print(f"[Rank {rank}] Initializing Megatron...")
    rank, world_size = initialize_megatron(tensor_model_parallel_size=args.tensor_parallel_size)
    local_rank = rank % torch.cuda.device_count()
    torch.cuda.set_device(local_rank)
    print(f"[Rank {rank}] ✓ Using GPU {local_rank}")
    
    # Sequential loading
    if rank == 0:
        print("[Rank 0] Loading model...")
        model = blip_decoder(
            pretrained='/pscratch/sd/k/kanjum/BLIP/checkpoints/model_base_caption_capfilt_large.pth',
            image_size=384,
            vit='base'
        )
    
    dist.barrier()
    
    if rank != 0:
        print(f"[Rank {rank}] Loading model...")
        model = blip_decoder(
            pretrained='/pscratch/sd/k/kanjum/BLIP/checkpoints/model_base_caption_capfilt_large.pth',
            image_size=384,
            vit='base'
        )
    
    dist.barrier()
    
    model.visual_encoder = parallelize_blip_vision_encoder(model.visual_encoder)
    model.text_decoder = parallelize_blip_text_decoder(model.text_decoder)
    model = model.cuda().eval()
    print(f"[Rank {rank}] ✓ Model loaded")
    
    images = load_test_images()
    
    batch_sizes = [1, 2, 4]
    results = []
    
    for bs in batch_sizes:
        if rank == 0:
            print(f"[Rank 0] Testing batch size {bs}...")
        torch.cuda.synchronize()
        dist.barrier()
        start = time.time()
        
        with torch.no_grad():
            for i in range(bs):
                _ = model.generate(images[i % len(images)], sample=False, num_beams=1, max_length=15)
        
        torch.cuda.synchronize()
        dist.barrier()
        elapsed = time.time() - start
        results.append((bs, elapsed))
        if rank == 0:
            print(f"[Rank 0]   ✓ Batch size {bs} complete: {elapsed*1000:.2f} ms")
    
    if rank == 0:
        print("\n" + "="*60)
        print("BATCH PROCESSING TEST")
        print("="*60)
        for bs, elapsed in results:
            print(f"Batch size {bs}: {elapsed*1000:.2f} ms total, {elapsed*1000/bs:.2f} ms per image")
        print("="*60 + "\n")


def main():
    parser = argparse.ArgumentParser(description='BLIP Tensor Parallel Test Suite')
    parser.add_argument('--tensor-parallel-size', type=int, default=2,
                    help='Number of GPUs for tensor parallelism (default: 2)')
    parser.add_argument('--checkpoint', type=str, 
                    default='/pscratch/sd/t/te137/BLIP/checkpoints/model_base_caption_capfilt_large.pth',
                    help='Path to BLIP checkpoint')
    parser.add_argument('--skip-correctness', action='store_true',
                    help='Skip correctness test (faster)')
    args = parser.parse_args()

    # Initialize distributed
    if not dist.is_initialized():
        print("Initializing distributed process group...")
        dist.init_process_group(backend='nccl')
        print("✓ Process group initialized")
    
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    
    if rank == 0:
        print(f"\n{'='*60}")
        print(f"BLIP TENSOR PARALLEL TEST SUITE")
        print(f"{'='*60}")
        print(f"World size: {world_size} GPUs")
        print(f"Backend: NCCL")
        print(f"{'='*60}\n")
    
    print(f"[Rank {rank}] Process initialized (rank {rank}/{world_size})")
    
    try:
        # Run all tests
        if rank == 0:
            print("\n🚀 Starting test suite...\n")
        
        test_correctness(rank, world_size, args)
        test_performance(rank, world_size, args)
        test_memory_usage(rank, world_size, args)
        test_batch_processing(rank, world_size, args)

        if rank == 0:
            print("\n" + "="*60)
            print("✓ ALL TESTS COMPLETED SUCCESSFULLY")
            print("="*60 + "\n")
    
    except Exception as e:
        print(f"\n[Rank {rank}] ✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        if rank == 0:
            print("\n" + "="*60)
            print("✗ TEST SUITE FAILED")
            print("="*60 + "\n")
    
    finally:
        if rank == 0:
            print("Cleaning up distributed resources...")
        if dist.is_initialized():
            dist.destroy_process_group()
        if rank == 0:
            print("✓ Cleanup complete")


if __name__ == "__main__":
    main()