#!/usr/bin/env bash

for num_topics in 25 50 75 100; do
    CUDA_VISIBLE_DEVICES=0 python run_topic_model_baselines.py \
        --model lda \
        --data_path data/dbpedia_14_Llama-3.2-3B-Instruct_vocab_2000_last \
        --results_path results/dbpedia/lda_K${num_topics} \
        --num_topics ${num_topics} \
        --num_seeds 5
    
    CUDA_VISIBLE_DEVICES=0 python run_topic_model_baselines.py \
        --model prodlda \
        --data_path data/dbpedia_14_Llama-3.2-3B-Instruct_vocab_2000_last \
        --results_path results/dbpedia/prodlda_K${num_topics} \
        --num_topics ${num_topics} \
        --num_hidden_layers 2 \
        --num_seeds 5

    CUDA_VISIBLE_DEVICES=0 python run_topic_model_baselines.py \
        --model zeroshot \
        --data_path data/dbpedia_14_Llama-3.2-3B-Instruct_vocab_2000_last \
        --results_path results/dbpedia/zeroshot_K${num_topics} \
        --num_topics ${num_topics} \
        --num_hidden_layers 2 \
        --num_seeds 5
    
    CUDA_VISIBLE_DEVICES=0 python run_topic_model_baselines.py \
        --model combined \
        --data_path data/dbpedia_14_Llama-3.2-3B-Instruct_vocab_2000_last \
        --results_path results/dbpedia/combined_K${num_topics} \
        --num_topics ${num_topics} \
        --num_hidden_layers 2 \
        --num_seeds 5
    
    CUDA_VISIBLE_DEVICES=0 python run_topic_model_baselines.py \
        --model etm \
        --data_path data/dbpedia_14_Llama-3.2-3B-Instruct_vocab_2000_last \
        --results_path results/dbpedia/etm_K${num_topics} \
        --num_topics ${num_topics} \
        --num_hidden_layers 2 \
        --num_seeds 5
done