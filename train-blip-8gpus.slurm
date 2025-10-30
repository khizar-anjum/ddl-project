#!/bin/bash
##salloc --nodes 1 --ntasks-per-node=2 --gpus-per-node=2 --qos shared_interactive --time 0:30:00 --constraint gpu --account=m4431_g
#SBATCH -A <your_allocation_account>          # <-- replace with your project ID, e.g. m1234
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -t 04:00:00                           # walltime (adjust as needed)
#SBATCH -N 2                                  # number of nodes
#SBATCH --ntasks-per-node=4                   # 4 GPUs per node (A100s)
#SBATCH --gpus-per-node=4
#SBATCH -J blip_train
#SBATCH -o logs/blip_train_%j.out
#SBATCH -e logs/blip_train_%j.err

# --------------------------
# Load your modules
# --------------------------
module load cudnn
module load NCCL
module load craype-accel-nvidia80

# --------------------------
# Environment Setup
# --------------------------
export OMP_NUM_THREADS=8
export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=hsn
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=29500
export WORLD_SIZE=$SLURM_NTASKS
export RANK=$SLURM_PROCID

echo "MASTER_ADDR = $MASTER_ADDR"
echo "WORLD_SIZE= $WORLD_SIZE"
echo "RANK= $RANK"

# --------------------------
# Paths
# --------------------------
#DATA_ROOT=/pscratch/<your_username>/datasets/coco2014
CONFIG=../config/caption_coco.yaml
OUTPUT_DIR=../output/Caption_coco_${SLURM_JOB_ID}

mkdir -p $OUTPUT_DIR
mkdir -p logs

# --------------------------
# Launch with torchrun
# --------------------------
#  srun lsof -i :$MASTER_PORT
#   srun lsof -i :$MASTER_PORT
#   srun lsof -i :$MASTER_PORT
#srun torchrun --nnodes=2 --nproc_per_node=4 --node_rank=$NODE_RANK --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT --rdzv_id=$SLURM_JOB_ID --rdzv_backend=c10d --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT train_caption.py --config $CONFIG --output_dir $OUTPUT_DIR --distributed True --device cuda --seed 42

#srun torchrun --nnodes=2 --nproc_per_node=4 --node_rank=$NODE_RANK --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT train_caption.py
srun torchrun \
    --nnodes=$SLURM_NNODES \
    --nproc_per_node=4 \
    --rdzv_id=$SLURM_JOB_ID \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    train_caption.py \
    --config $CONFIG \
    --output_dir $OUTPUT_DIR \
    --distributed True \
    --device cuda \
    --seed 42

echo "✅ Training completed at $(date)"

