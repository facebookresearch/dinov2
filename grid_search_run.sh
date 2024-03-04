# config_files=(ssl_default_config_small_noKoleo ssl_default_config_base_noKoleo ssl_default_config_base_withKoleo ssl_default_config_small_withKoleo)

config_files=(base_3_blocks)
output_dir="/cluster/scratch/cmerk/dinov2_accum_12_low_lr_2_filtered_dataset_3_block_no_patches"

for config in "${config_files[@]}"
do
    sbatch grid_search.slurm $config $output_dir
done