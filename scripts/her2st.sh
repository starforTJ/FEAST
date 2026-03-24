#!/bin/bash

DEVICE=0
export CUDA_VISIBLE_DEVICES=$DEVICE

mkdir -p logs

YAML_CONFIG="configs/config_her2st.yaml"

python main.py --config "$YAML_CONFIG" > logs/her2st.log 2>&1