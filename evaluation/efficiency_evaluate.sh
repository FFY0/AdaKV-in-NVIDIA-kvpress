#!/bin/bash
press_name=(fullkv snapkv ada_snapkv)

# Meta-Llama-3.1-8B-Instruct
model_name="Mistral-7B-Instruct-v0.3"
model=${MODELS_DIR}/${model_name}

for press in "${press_name[@]}"; do
    python efficiency_evaluate.py --press_name $press --model_path $model
done