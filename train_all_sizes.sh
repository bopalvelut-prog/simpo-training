#!/bin/bash
# SimPO training across Qwen2.5 model sizes
# SimPO: Simple Preference Optimization (reference-free)

set -e

MODEL_SIZES=("1.5b" "3b" "7b" "14b")
BASE_MODEL="Qwen/Qwen2.5"

for size in "${MODEL_SIZES[@]}"; do
    model="${BASE_MODEL}-${size}-Instruct"
    output="./simpo-qwen2.5-${size}"

    echo "============================================"
    echo "Training SimPO on ${model}"
    echo "============================================"

    if [ "$size" == "14b" ] || [ "$size" == "7b" ]; then
        use_lora="--use_lora"
    else
        use_lora=""
    fi

    python train_simpo.py \
        --model "$model" \
        --output_dir "$output" \
        --model_size "$size" \
        --gamma 1.0 \
        --beta 0.01 \
        --training_hours 5.0 \
        --batch_size 4 \
        --gradient_accumulation_steps 8 \
        --learning_rate 5e-7 \
        --max_seq_length 2048 \
        --save_steps 200 \
        $use_lora \
        --wandb_project "simpo-qwen2.5"

    echo "Saved: $output"
done

echo "All SimPO training complete!"
