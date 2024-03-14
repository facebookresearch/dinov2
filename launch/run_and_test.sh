#!bin/bash

user="cmerk"

# DINOv2
code_location="/cluster/home/${user}/dinov2"
config_files=(base_3_blocks)
output_dir="/cluster/scratch/${user}/dinov2_tests_05032024/fewer_epochs_teacher_warmup_path_dropout_old_model_init"

# DR Training and Test
aptos_dataset_dir="/cluster/scratch/${user}/datasets/aptos2019-blindness-detection"
code_dir="/cluster/home/${user}/Fundus_Foundation_Models"
n_samples=16

# Get directory of this file
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
# Get the slurm script
slurm_script=( $(ls $DIR/*.slurm) )

for config in "${config_files[@]}"
do
    sbatch $slurm_script $config $output_dir $aptos_dataset_dir $code_dir $n_samples $code_location
done