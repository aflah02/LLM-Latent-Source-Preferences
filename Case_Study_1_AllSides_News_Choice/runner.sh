#!/bin/bash

# Define the list of experiment types
experiment_types=(
  "source_hidden"
  "center_right_swap"
  "left_right_swap"
  "do_not_be_biased"
  "base"
  "left_center_swap"
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
gpu="1"

# Error out 
set -e

# Loop through experiment types and models
for experiment in "${experiment_types[@]}"; do

  # Map experiment type to file name
  case "$experiment" in
    "do_not_be_biased_influence_check")
      file="base_run_do_not_be_biased_influence_check.py"
      ;;
    "infuence_check")
      file="base_run_infuence_check.py"
      ;;
    "source_hidden")
      file="base_run_source_hidden.py"
      ;;
    "center_right_swap")
      file="center_right_swap_run.py"
      ;;
    "left_right_swap")
      file="left_right_swap_run.py"
      ;;
    "do_not_be_biased")
      file="base_run_do_not_be_biased.py"
      ;;
    "base")
      file="base_run.py"
      ;;
    "left_center_swap")
      file="left_center_swap_run.py"
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
