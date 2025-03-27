#!/bin/bash

srun \
    --account=project_462000353\
    --partition=dev-g \
    --nodes=1 \
    --ntasks-per-node=1 \
    --gres=gpu:mi250:1 \
    --time=01:00:00 \
    --mem=100G\
    --pty \
    bash
