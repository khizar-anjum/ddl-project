#!/bin/bash
#SBATCH -A m4431_g
#SBATCH -C gpu
#SBATCH -q debug
#SBATCH -t 00:30:00
#SBATCH -N 2
#SBATCH --ntasks-per-node=1
#SBATCH -c 32
#SBATCH --gpus-per-task=4
#SBATCH --mail-user=te137@echo.rutgers.edu
#SBATCH --mail-type=ALL

echo "=========================================="
echo "Testing BLIP on 8-GPU Setup (2 Nodes)"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Number of nodes: $SLURM_NNODES"
echo "Node list: $SLURM_JOB_NODELIST"
echo ""

# Load modules
module load pytorch

# Get the master node address
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=29500

echo "Master addr: $MASTER_ADDR"
echo "Master port: $MASTER_PORT"
echo ""

# Set environment variables
export WORLD_SIZE=8
export NCCL_DEBUG=WARN
export NCCL_IB_DISABLE=0
export HF_HOME=/tmp/hf_cache

# Copy checkpoint to local storage on each node
echo "Copying checkpoint to local storage on each node..."
srun --ntasks-per-node=1 cp /pscratch/sd/${USER:0:1}/${USER}/BLIP/checkpoints/model_base_caption_capfilt_large.pth /tmp/
echo "Copy complete!"
echo ""

# Download tokenizer to cache on each node
echo "Downloading tokenizer to cache on each node..."
srun --ntasks-per-node=1 bash -c "
    mkdir -p /tmp/hf_cache && 
    python -c 'from transformers import BertTokenizer; BertTokenizer.from_pretrained(\"bert-base-uncased\")'
"
echo "Tokenizer cached!"
echo ""

# Set offline mode to use cached tokenizer
export TRANSFORMERS_OFFLINE=1

echo "Running BLIP test on 8 GPUs..."
echo ""

# Run with torchrun
srun bash -c "cd /pscratch/sd/${USER:0:1}/${USER}/practice_setup && torchrun \
    --nproc_per_node=4 \
    --nnodes=2 \
    --node_rank=\$SLURM_PROCID \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    test_megatron_blip.py"

echo ""
echo "Test completed!"
