#!/bin/bash

DEVICE=0
export CUDA_VISIBLE_DEVICES=$DEVICE

mkdir -p logs

YAML_CONFIG="configs/config_skin.yaml"

python main.py --config "$YAML_CONFIG" > logs/skin.log 2>&1