# Testing Guide for Adaptive Pipeline Monitoring

This guide explains how to test the newly implemented monitoring infrastructure.

## What Was Implemented

### 1. Runtime P2P Monitoring
- **File**: `megatron/core/pipeline_parallel/monitored_p2p_communication.py`
- Tracks bandwidth and latency during training
- Uses async CUDA events for minimal overhead (<1%)
- Probabilistic sampling (default: 10% of operations)

### 2. Initial Network Profiling
- **Files**: `topology.py`, `bandwidth_profiler.py`, `initial_profiling.py`
- Discovers network topology (which GPUs on which nodes)
- Measures intra-node bandwidth (NVLink: ~400 GB/s)
- Measures inter-node bandwidth (InfiniBand: ~20 GB/s)
- Runs once at job startup (~30-60 seconds)

### 3. Configuration
- **File**: `megatron/core/model_parallel_config.py`
- `enable_p2p_monitoring`: Enable runtime monitoring
- `p2p_monitoring_sample_rate`: Sample rate (0.0-1.0)
- `enable_initial_profiling`: Enable startup profiling

---

## Test Scripts

### Test 1: Profiling Only (Recommended First)

**Script**: `test_profiling_only.py`

This is the **simplest test** - it only tests the profiling infrastructure without any model training.

**What it does**:
- Discovers topology (which GPUs on which nodes)
- Measures intra-node bandwidth
- Measures inter-node bandwidth (if multi-node)
- Runs ring test validation
- Prints analysis and recommendations

**Expected output**:
```
======================================================================
                     Network Profiling Test
======================================================================
World size: 8 GPUs
Backend: nccl
======================================================================

Starting initial network profiling...
======================================================================

Phase 1: Discovering topology...
  World size: 8 GPUs
  Nodes: 2
    Node 0 (gpu-node-01): GPUs [0, 1, 2, 3]
    Node 1 (gpu-node-02): GPUs [4, 5, 6, 7]

Phase 2: Profiling intra-node bandwidth...
  Node 0: 420.34 GB/s (NVLink/PCIe)
  Node 1: 418.92 GB/s (NVLink/PCIe)

Phase 3: Profiling inter-node bandwidth...
  Node 0 <-> Node 1: 22.45 GB/s (IB/Ethernet)

Phase 4: Running ring validation test...
Ring test: avg=21.34 GB/s, min=17.94, max=22.67

======================================================================
Initial profiling complete! (elapsed: 28.4s)
======================================================================

Bandwidth Summary:
  Intra-node average: 419.63 GB/s
  Inter-node average: 22.45 GB/s
  Intra/Inter ratio: 18.7×

======================================================================
```

### Test 2: Full Training Test (Advanced)

**Script**: `test_adaptive_monitoring.py`

This tests both profiling AND runtime monitoring with a simple model.

**What it does**:
- Runs initial profiling
- Creates a simple model
- Runs training iterations
- Monitors P2P communication during training

---

## How to Run Tests

### Option A: Single Node Test (Start Here!)

**Purpose**: Test on one node first to verify basic functionality

**SLURM Script**: `slurm_test_single_node.sh`

```bash
# 1. Edit the script to load your modules
vi slurm_test_single_node.sh

# Uncomment and modify these lines:
# module load cuda/12.1
# module load pytorch/2.1
# source activate your_env_name

# 2. Submit the job
sbatch slurm_test_single_node.sh

# 3. Check output
tail -f logs/profiling_single_<jobid>.out
```

**Expected behavior**:
- Should complete in ~1 minute
- Will show only intra-node bandwidth (no inter-node)
- Bandwidth should be 200-600 GB/s (NVLink/PCIe)

---

### Option B: Multi-Node Test (Main Test)

**Purpose**: Test across multiple nodes to measure inter-node bandwidth

**SLURM Script**: `slurm_test_profiling.sh`

```bash
# 1. Edit the script
vi slurm_test_profiling.sh

# Modify these settings for your cluster:
#SBATCH --nodes=2              # Number of nodes
#SBATCH --ntasks-per-node=4    # GPUs per node
#SBATCH --gres=gpu:4           # Request 4 GPUs per node
#SBATCH --partition=gpu        # Your GPU partition name

# Uncomment and modify module loads:
# module load cuda/12.1
# module load nccl/2.18
# module load pytorch/2.1
# source activate megatron

# 2. Submit the job
sbatch slurm_test_profiling.sh

# 3. Monitor progress
squeue -u $USER
tail -f logs/profiling_test_<jobid>.out

# 4. When done, check results
cat logs/profiling_test_<jobid>.out
```

**Expected behavior**:
- Should complete in ~1-2 minutes
- Will show both intra-node and inter-node bandwidth
- Intra-node: 200-600 GB/s
- Inter-node: 10-25 GB/s (InfiniBand) or 1-10 GB/s (Ethernet)

---

## Interpreting Results

### Good Results

**Single Node (4 GPUs)**:
```
Phase 2: Profiling intra-node bandwidth...
  Node 0: 420.34 GB/s (NVLink/PCIe)      ✓ Good!
  Node 0: Single GPU (no intra-node comm) ✓ Also fine (only 1 GPU on node)
```

**Multi-Node (2 nodes, 4 GPUs each)**:
```
Phase 2: Profiling intra-node bandwidth...
  Node 0: 420.34 GB/s (NVLink/PCIe)      ✓ Good!
  Node 1: 418.92 GB/s (NVLink/PCIe)      ✓ Good!

Phase 3: Profiling inter-node bandwidth...
  Node 0 <-> Node 1: 22.45 GB/s          ✓ Good! (InfiniBand HDR)
  Node 0 <-> Node 1: 12.34 GB/s          ✓ OK (InfiniBand EDR or 100GbE)
```

### Potential Issues

**Low Intra-Node Bandwidth** (<100 GB/s):
```
Node 0: 35.21 GB/s (NVLink/PCIe)         ⚠️ Unexpected - should be higher
```
- **Cause**: May be using PCIe instead of NVLink, or shared bus
- **Impact**: Pipeline stages on same node will be slower than expected

**Very Low Inter-Node Bandwidth** (<5 GB/s):
```
Node 0 <-> Node 1: 2.34 GB/s             ⚠️ Low - check network
```
- **Cause**: Network congestion, Ethernet instead of InfiniBand, or misconfiguration
- **Impact**: Inter-node pipeline stages will be bottleneck

**Ring Test Mismatch**:
```
Ring test: avg=10.34 GB/s, min=2.34, max=22.67  ⚠️ Large variance
```
- **Cause**: Network heterogeneity or congestion
- **Impact**: Some GPU pairs have poor connectivity

---

## Troubleshooting

### Issue: "torch.distributed not initialized"

**Solution**: Make sure you're using `srun` or `torchrun`:
```bash
# Don't do this:
python test_profiling_only.py

# Do this:
srun python test_profiling_only.py
```

### Issue: "CUDA out of memory"

**Solution**: Reduce tensor size in profiling:

Edit `bandwidth_profiler.py`, change:
```python
tensor_size_mb: int = 100,  # Change to 50 or 25
```

### Issue: Job hangs during profiling

**Cause**: Deadlock or barrier mismatch

**Solution**:
1. Check that all GPUs are healthy: `nvidia-smi`
2. Verify NCCL can initialize: `srun python -c "import torch; torch.distributed.init_process_group('nccl')"`
3. Check SLURM allocation matches script: `echo $SLURM_NTASKS`

### Issue: Permission denied for logs directory

**Solution**:
```bash
mkdir -p logs
chmod 755 logs
```

---

## Next Steps After Testing

### 1. If Single-Node Test Works:
✓ Basic infrastructure is working
→ Try multi-node test

### 2. If Multi-Node Test Works:
✓ Profiling infrastructure fully working
→ Integrate with actual training script
→ Test with real models (GPT, etc.)

### 3. If Both Tests Work:
✓ Ready for production use
→ Enable in your training configs:
```python
config = ModelParallelConfig(
    enable_initial_profiling=True,
    enable_p2p_monitoring=True,
    p2p_monitoring_sample_rate=0.1
)
```

---

## Understanding the Output

### Topology Discovery
```
Node 0 (gpu-node-01): GPUs [0, 1, 2, 3]
```
- Tells you which global ranks are on which physical nodes
- Used to determine if communication is intra-node (fast) or inter-node (slower)

### Bandwidth Measurements
- **Intra-node**: GPU-to-GPU on same node (NVLink or PCIe)
- **Inter-node**: GPU-to-GPU on different nodes (InfiniBand or Ethernet)
- **Ring test**: All GPUs in circle, validates no major bottlenecks

### Analysis Section
```
Recommendations for Pipeline Parallelism:
• Pack 4 pipeline stages per node to minimize inter-node communication
• With 8 GPUs, you have 2 nodes
• Good pipeline parallel sizes: [2, 4, 8]
```
- Suggests how to assign pipeline stages to minimize slow inter-node transfers

---

## Quick Start Commands

**Fastest way to test**:

```bash
# 1. Go to Megatron directory
cd /home/khizar/Archives/Rutgers/Fall_2025/DDL/Megatron-LM

# 2. Create logs directory
mkdir -p logs

# 3. Edit SLURM script for your cluster
vi slurm_test_single_node.sh
# (uncomment module loads, set partition name)

# 4. Submit job
sbatch slurm_test_single_node.sh

# 5. Watch output
watch -n 2 'ls -lrt logs/ | tail -5'
tail -f logs/profiling_single_*.out
```

---

## What to Report

When you run tests, please share:

1. **Job output**: The full output from the test
2. **System info**:
   - Number of nodes
   - GPUs per node
   - Network type (InfiniBand HDR/EDR, Ethernet 100GbE, etc.)
3. **Bandwidth results**:
   - Intra-node bandwidth
   - Inter-node bandwidth (if multi-node)
4. **Any errors or warnings**

This will help debug any issues and validate the implementation!

---

## Summary

**Start with**: `slurm_test_single_node.sh` (easiest, 1 node, 4 GPUs)
**Then try**: `slurm_test_profiling.sh` (full test, 2 nodes, 8 GPUs)
**Expected time**: 1-2 minutes per test
**Expected result**: Bandwidth measurements and topology info

Good luck! 🚀
