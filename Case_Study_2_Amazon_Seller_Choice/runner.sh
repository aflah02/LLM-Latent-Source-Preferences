#!/bin/bash

# Define the list of experiment types
experiment_types=(
  "unguided_run"
  "speed_optimized_run"
  "money_optimized_run"
)

# Define the list of model names

model_names=(
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

# seed
seed=107

# mode
mode="prod"

# gpu
gpu="0"

# Error out 
set -e

# Loop through experiment types and models
for experiment in "${experiment_types[@]}"; do

  # Map experiment type to file name
  case "$experiment" in
    "unguided_run")
      file="unguided_run.py"
      ;;
    "speed_optimized_run")
      file="speed_optimized_run.py"
      ;;
    "money_optimized_run")
      file="money_optimized_run.py"
      ;;
    *)
      echo "Unknown experiment type: $experiment"
      continue
      ;;
  esac

  echo "Running experiment: $experiment using script: $file"

   for model in "${model_names[@]}"; do
    # echo " -> Model: $model"

    # Echo the command instead of running it
    echo "Command: CUDA_VISIBLE_DEVICES=\"$gpu\" python3 \"$file\" --model \"$model\" --seed \"$seed\" --mode \"$mode\""

    eval "CUDA_VISIBLE_DEVICES=\"$gpu\" python3 \"$file\" --model \"$model\" --seed \"$seed\" --mode \"$mode\""
    # status=$? # Capture status of the echo command

    cmd_to_kill="nvidia-smi | grep sglang | awk -v gpu=\"$gpu\" '\$2 == gpu { print \$5 }' | xargs -r kill -9"
    echo "Killing processes using GPU $gpu..."
    echo "Command: $cmd_to_kill"
    eval "$cmd_to_kill"

    nvidia-smi

    sleep 5

    if [ $? -ne 0 ]; then
        echo "Error occurred for Badge: $badge | Domain: $domain | Iteration: $i, continuing..."
    fi
  done
done

# Print summary
echo "---------------------------------"
if [ ${#failed_runs[@]} -eq 0 ]; then
  echo "🎉 All runs completed successfully!"
else
  echo "❗ Some runs failed:"
  for run in "${failed_runs[@]}"; do
    echo "   - $run"
  done
fi

cleanup() {
echo "🧹 Cleaning up..."
pkill -f "python -m sglang.serve"
}
trap cleanup EXIT
