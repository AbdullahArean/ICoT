#!/bin/bash

config_dir="config"
# shellcheck disable=SC2207
configs=($(ls $config_dir/*.yaml))

# Check actual available GPUs
nvidia-smi --list-gpus > /dev/null 2>&1
if [ $? -eq 0 ]; then
    # Get actual number of GPUs
    gpu_count=$(nvidia-smi --list-gpus | wc -l)
    gpus=($(seq 0 $((gpu_count-1))))
else
    # Fallback to single GPU
    gpu_count=1
    gpus=(0)
fi

echo "Number of gpus: ${#gpus[@]}"
for i in "${!configs[@]}"; do
    config="${configs[$i]}"
    gpu_id=${gpus[$((i % ${#gpus[@]}))]}
    log_file="log_${gpu_id}_$(basename "$config" .yaml).log"

    echo "Running main.py with $config on GPU $gpu_id..."

    CUDA_VISIBLE_DEVICES=$gpu_id nohup python run.py --config "$config" > "$log_file" 2>&1 &

done

wait
