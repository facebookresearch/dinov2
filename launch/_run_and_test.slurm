#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=20
#SBATCH --gpus-per-node=1
#SBATCH --gres=gpumem:24g
#SBATCH --time=24:00:00
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

torchrun --nproc_per_node=1 --master_port=$FREE_PORT dinov2/train/train.py --config-file dinov2/configs/$1.yaml --output_dir $2

####################################################################################################
function parse_yaml {
   local prefix=$2
   local s='[[:space:]]*' w='[a-zA-Z0-9_]*' fs=$(echo @|tr @ '\034')
   sed -ne "s|^\($s\):|\1|" \
        -e "s|^\($s\)\($w\)$s:$s[\"']\(.*\)[\"']$s\$|\1$fs\2$fs\3|p" \
        -e "s|^\($s\)\($w\)$s:$s\(.*\)$s\$|\1$fs\2$fs\3|p"  $1 |
   awk -F$fs '{
      indent = length($1)/2;
      vname[indent] = $2;
      for (i in vname) {if (i > indent) {delete vname[i]}}
      if (length($3) > 0) {
         vn=""; for (i=0; i<indent; i++) {vn=(vn)(vname[i])("_")}
         printf("%s%s%s=\"%s\"\n", "'$prefix'",vn, $2, $3);
      }
   }'
}
# Run DR Grading Test Training
module load gcc/8.2.0 python_gpu/3.11.2
module load eth_proxy

source ${4}/.env//bin/activate

echo "GPUs allocated by SLURM: ${CUDA_VISIBLE_DEVICES}"

checkpoint_file=$(ls ${2}/best_total_loss_*.rank_0.pth | head -n 1)

# Get block expansion from .yaml file
export $(parse_yaml ${6}/dinov2/configs/${1}.yaml "config_")

# Remove [ and ] from config_block_expansion_expanded_blocks
block_expansion_positions=$(echo $config_block_expansion_expanded_blocks | tr -d '[]')

# Few Shot
torchrun --nproc_per_node=1 --master_port=$FREE_PORT ${4}/main_finetune.py \
 --batch_size 8 --world_size 1 --model dinov2_vitb14 \
 --epochs 100 --warmup_epochs 10 --blr 4e-2 --weight_decay 1e-3 \
 --nb_classes 5 --num_workers $num_tasks \
 --data_path ${3} \
 --task ${2}/cross_eval/few_shot_${5} \
 --input_size 224 --loss_weights True\
 --loss_function DistanceWeightedCrossEntropyLoss --loss_weight_factor 0.1 --loss_penalty_type squared\
 --RandomResizeCrop_upperBound 1.2 --RandomResizeCrop_lowerBound 0.8 \
 --RandomRotation_degrees 5 \
 --preprocessing_CropRoundImage False --preprocessing_CLAHETransform False --preprocessing_MedianFilterTransform False \
 --pretrained_checkpoint ${checkpoint_file} \
 --color_jitter_param_brightness 0.2 --color_jitter_param_contrast 0.2 --color_jitter_param_saturation 0.1 --color_jitter_param_hue 0.1 \
 --fix_backbone True --n_classification_heads 1 \
 --block_expansion_positions ${block_expansion_positions} \
 --few_shot_learning ${5} --n_few_shot_folds 5

# Full
torchrun --nproc_per_node=1 --master_port=$FREE_PORT ${4}/main_finetune.py \
 --batch_size 64 --world_size 1 --model dinov2_vitb14 \
 --epochs 100 --warmup_epochs 10 --blr 4e-2 --weight_decay 1e-3 \
 --nb_classes 5 --num_workers $num_tasks \
 --data_path ${3} \
 --task ${2}/cross_eval/full \
 --input_size 224 --loss_weights True\
 --loss_function DistanceWeightedCrossEntropyLoss --loss_weight_factor 0.1 --loss_penalty_type squared\
 --RandomResizeCrop_upperBound 1.2 --RandomResizeCrop_lowerBound 0.8 \
 --RandomRotation_degrees 5 \
 --preprocessing_CropRoundImage False --preprocessing_CLAHETransform False --preprocessing_MedianFilterTransform False \
 --pretrained_checkpoint ${checkpoint_file} \
 --color_jitter_param_brightness 0.2 --color_jitter_param_contrast 0.2 --color_jitter_param_saturation 0.1 --color_jitter_param_hue 0.1 \
 --fix_backbone True --n_classification_heads 1 \
 --block_expansion_positions ${5}