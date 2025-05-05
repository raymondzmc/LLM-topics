#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=0

python process_dataset_new.py \
    --dataset data/stackoverflow.csv \
    --content_key text \
    --label_key label \
    --split all \
    --vocab_size 2000 \
    --model_name meta-llama/Llama-3.2-1B-Instruct \
    --hidden_state_layer -1 \
    --single_token_only \
    --batch_size 32 \
    --target_method summary \
    --bow_dataset

python process_dataset_new.py \
    --dataset data/stackoverflow.csv \
    --content_key text \
    --label_key label \
    --split all \
    --vocab_size 2000 \
    --model_name meta-llama/Llama-3.2-3B-Instruct \
    --hidden_state_layer -1 \
    --single_token_only \
    --batch_size 32 \
    --target_method summary \
    --bow_dataset