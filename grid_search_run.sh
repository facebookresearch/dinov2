config_files=(ssl_default_config_small_noKoleo ssl_default_config_base_noKoleo ssl_default_config_base_withKoleo ssl_default_config_small_withKoleo)

for config in "${config_files[@]}"
do
    sbatch grid_search.slurm $config
done