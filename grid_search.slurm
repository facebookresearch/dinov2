#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=20
#SBATCH --gpus-per-node=1
#SBATCH --gres=gpumem:24g
#SBATCH --time=10:00:00
#SBATCH --mem-per-cpu=6000
#SBATCH --tmp=4000 # per node!!
#SBATCH --job-name=dino_run
#SBATCH --output=dino_run.out
#SBATCH --error=dino_run.err


# Load the required modules
# module load gcc/8.2.0 python_gpu/3.10.4 eth_proxy git
module load gcc/8.2.0 python_gpu/3.9.9 eth_proxy git
# module load cudnn/8.2.4.15
source .env//bin/activate

echo "GPUs allocated by SLURM: ${CUDA_VISIBLE_DEVICES}"

FREE_PORT=$(python -c 'import socket; sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM); sock.bind(("", 0)); print(sock.getsockname()[1]); sock.close()')
echo "Free Port: ${FREE_PORT}"

num_tasks=$SLURM_NTASKS
echo "Number of workers: ${num_tasks}"

export PYTHONPATH=.

torchrun --nproc_per_node=1 --master_port=$FREE_PORT dinov2/train/train.py --config-file dinov2/configs/$1.yaml --output_dir /cluster/scratch/cmerk/testdinov2/dinov2_finetune_$1