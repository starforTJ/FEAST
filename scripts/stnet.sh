#!/bin/bash

DEVICE=0
export CUDA_VISIBLE_DEVICES=$DEVICE

mkdir -p logs

YAML_CONFIG="configs/config_stnet.yaml"

python main.py --config "$YAML_CONFIG" > logs/stnet.log 2>&1