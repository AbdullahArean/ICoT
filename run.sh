#!/bin/bash

config_dir="./config"
configs=($(ls $config_dir/*.yaml))
configs=("$config_dir/base_zero.yaml"  "$config_dir/base_one.yaml")

gpus=(1 2)
echo "Number of gpus: ${#gpus[@]}"
for i in "${!configs[@]}"; do
    config="${configs[$i]}"
    gpu_id=${gpus[$((i % ${#gpus[@]}))]}
    log_file="log_${gpu_id}_$(basename "$config" .yaml).log"

    echo "Running main.py with $config on GPU $gpu_id..."

    CUDA_VISIBLE_DEVICES=$gpu_id nohup python ./main_m3cot_chameleon_release.py --config $config > "$log_file" 2>&1 &

done

wait
