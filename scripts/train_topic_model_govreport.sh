#!/usr/bin/env bash
# python process_dataset.py --dataset ccdv/govreport-summarization \
#                           --content_key summary \
#                           --split test \
#                           --vocab_size 2000 \
#                           --model_name meta-llama/Llama-3.2-1B \
#                           --single_token_only \
#                           --hidden_state_layer -1 \
#                           --batch_size 32 \
#                           --bow_dataset


# python run_topic_modeling.py \
#   --data_path ./data/govreport-summarization_Llama-3.2-1B_2000 \
#   --model zeroshot \
#   --K 25 \
#   --test_corpus_path data/govreport-summarization_Llama-3.2-1B_2000/bow_dataset.txt \
#   --num_seeds 3 \
#   --num_epochs 20
  
# python run_topic_modeling.py \
#   --data_path ./data/govreport-summarization_Llama-3.2-1B_2000 \
#   --model zeroshot \
#   --K 50 \
#   --test_corpus_path data/govreport-summarization_Llama-3.2-1B_2000/bow_dataset.txt \
#   --num_seeds 3 \
#   --num_epochs 20
  
# python run_topic_modeling.py \
#   --data_path ./data/govreport-summarization_Llama-3.2-1B_2000 \
#   --model combined \
#   --K 25 \
#   --test_corpus_path data/govreport-summarization_Llama-3.2-1B_2000/bow_dataset.txt \
#   --num_seeds 3 \
#   --num_epochs 20

# python run_topic_modeling.py \
#   --data_path ./data/govreport-summarization_Llama-3.2-1B_2000 \
#   --model combined \
#   --K 50 \
#   --test_corpus_path data/govreport-summarization_Llama-3.2-1B_2000/bow_dataset.txt \
#   --num_seeds 3 \
#   --num_epochs 20 

# python run_topic_modeling.py \
#   --data_path ./data/govreport-summarization_Llama-3.2-1B_2000 \
#   --model generative \
#   --K 25 \
#   --test_corpus_path data/govreport-summarization_Llama-3.2-1B_2000/bow_dataset.txt \
#   --num_seeds 3 \
#   --num_epochs 20

# python run_topic_modeling.py \
#   --data_path ./data/govreport-summarization_Llama-3.2-1B_2000 \
#   --model generative \
#   --K 50 \
#   --test_corpus_path data/govreport-summarization_Llama-3.2-1B_2000/bow_dataset.txt \
#   --num_seeds 3 \
#   --num_epochs 20 
