#!/bin/bash

config_dir="./config"
configs=($(ls $config_dir/*.yaml))

gpus=(0 1 2 3 4 5 6 7)
echo "Number of gpus: ${#gpus[@]}"
for i in "${!configs[@]}"; do
    config="${configs[$i]}"
    gpu_id=${gpus[$((i % ${#gpus[@]}))]}
    log_file="log_${gpu_id}_$(basename "$config" .yaml).log"

    echo "Running main.py with $config on GPU $gpu_id..."

    CUDA_VISIBLE_DEVICES=$gpu_id nohup python ./run.py --config $config > "$log_file" 2>&1 &

done

wait
