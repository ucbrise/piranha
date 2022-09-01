#!/bin/bash

# Runs Piranha clients locally on 3 different GPUs
CUDA_VISIBLE_DEVICES=1 ./piranha -p 1 -c files/samples/localhost_config.json >/dev/null &
CUDA_VISIBLE_DEVICES=2 ./piranha -p 2 -c files/samples/localhost_config.json >/dev/null &
CUDA_VISIBLE_DEVICES=0 ./piranha -p 0 -c files/samples/localhost_config.json

