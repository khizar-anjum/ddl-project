#!/bin/bash
#SBATCH -A m4431_g
#SBATCH -C gpu
#SBATCH -q debug
#SBATCH -t 00:30:00
#SBATCH -N 2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=4

# Don't load system pytorch - use venv instead
source /pscratch/sd/t/te137/mega-env/bin/activate
cd /pscratch/sd/t/te137/practice_setup

export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=29500

ssrun bash -c "
    export PATH=/pscratch/sd/t/te137/mega-env/bin:\$PATH
    python -m torch.distributed.run \
        --nproc_per_node=4 \
        --nnodes=2 \
        --node_rank=\$SLURM_PROCID \
        --master_addr=$MASTER_ADDR \
        --master_port=$MASTER_PORT \
        test_parallel_blip.py \
        --tensor-parallel-size 8"