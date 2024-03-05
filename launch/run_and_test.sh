# config_files=(ssl_default_config_small_noKoleo ssl_default_config_base_noKoleo ssl_default_config_base_withKoleo ssl_default_config_small_withKoleo)

# DINOv2
code_location="/cluster/home/cmerk/dinov2"
config_files=(base_3_blocks)
output_dir="/cluster/scratch/cmerk/test_dinov2"
block_expansion_positions="3,7,11"
n_samples=16

# DR Training and Test
aptos_dataset_dir="/cluster/scratch/cmerk/datasets/aptos2019-blindness-detection"
code_dir="/cluster/home/cmerk/Fundus_Foundation_Models"


# Get directory of this file
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
# Get the slurm script
slurm_script=( $(ls $DIR/*.slurm) )

for config in "${config_files[@]}"
do
    sbatch $slurm_script $config $output_dir $aptos_dataset_dir $code_dir $n_samples $code_location
done