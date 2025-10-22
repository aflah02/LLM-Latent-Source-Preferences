#!/bin/bash

models=(
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
    "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
    "meta-llama/Llama-3.2-1B-Instruct"
    "meta-llama/Llama-3.1-8B-Instruct"
    "microsoft/phi-4"
    "mistralai/Ministral-8B-Instruct-2410"
    "mistralai/Mistral-Nemo-Instruct-2407"
    "Qwen/Qwen2.5-1.5B-Instruct"
    "Qwen/Qwen2.5-7B-Instruct"
    "microsoft/Phi-4-mini-instruct"
    # "azure--gpt-4.1-mini"
    # "azure--gpt-4.1-nano"
    "gpt-4.1-nano"
    "gpt-4.1-mini"
)

badges=("X_Handle" "X_Followers" "X_URL" "Instagram_Handle" "Instagram_Followers" "Instagram_URL" "URL" "Year_of_Establishment" "Years_Since_Establishment" "Base")

data_domains=("Electronics" "Books" "Grocery" "Clothing" "Beauty")

# Number of iterations - This will run every model-badge combination this many times
ITERATIONS=1

gpu="0" # GPU to use if multiple GPUs are available


for model in "${models[@]}"; do
    echo "======== MODEL: $model ========"
    for i in $(seq 1 "$ITERATIONS"); do
        echo "======== ITERATION $i/$ITERATIONS ========"
        for badge in "${badges[@]}"; do
            for data_domain in "${data_domains[@]}"; do
                echo "Running for Model: $model | Badge: $badge | Data Domain: $data_domain | Iteration: $i"
                CUDA_VISIBLE_DEVICES=$gpu python3 run_ecommerce.py \
                    --data_domain "$data_domain" \
                    --badge_to_use "$badge" \
                    --seed 107 \
                    --mode prod \
                    --model_name "$model"

                cmd_to_kill="nvidia-smi | grep sglang | awk -v gpu=\"$gpu\" '\$2 == gpu { print \$5 }' | xargs -r kill -9"
                echo "Killing processes using GPU $gpu..."
                echo "Command: $cmd_to_kill"
                eval "$cmd_to_kill"

                nvidia-smi

                sleep 5

                if [ $? -ne 0 ]; then
                    echo "Error occurred | Model: $model | Badge: $badge | Iteration: $i, continuing..."
                fi
        done
    done
done

cleanup() {
    echo "🧹 Cleaning up..."
    pkill -f "python -m sglang.serve"
}
trap cleanup EXIT
