python process_dataset.py --dataset SetFit/bbc-news \
                          --content_key text \
                          --split test \
                          --vocab_size 2000 \
                          --model_name meta-llama/Llama-3.2-1B \
                          --single_token_only \
                          --batch_size 32 \
                          --bow_dataset

DATA_PATH="./data/bbc-news_Llama-3.2-1B_2000"
SEEDS=(42 43 44 45 46)

# python run_topic_modeling.py \
#       --data_path ./data/bbc-news_Llama-3.2-1B_2000 \
#       --model generative \
#       --K 25 \
#       --hidden_state_layer -1 \
#       --num_epochs 20 \
#       --test_corpus_path data/bbc-news_Llama-3.2-1B_2000/bow_dataset.txt \
#       --seed 42 \
#       --eval_prompt_path topic_ratings/bbcnews_system.jinja > train.log 2>&1

# # Loop over loss weights
# LOSS_WEIGHTS=(0.1 1 10 100 1000)
# for WL in "${LOSS_WEIGHTS[@]}"; do  
#   for SEED in "${SEEDS[@]}"; do
#     python run_topic_modeling.py \
#       --data_path "${DATA_PATH}" \
#       --model generative \
#       --K 25 \
#       --hidden_state_layer -1 \
#       --loss_weights "${WL}" \
#       --test_corpus_path data/bbc-news_Llama-3.2-1B_2000/bow_dataset.txt \
#       --seed "${SEED}"
#   done
# done


# # Loop over hidden layers
# HIDDEN_LAYERS=(0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16)
# for HL in "${HIDDEN_LAYERS[@]}"; do  
#   for SEED in "${SEEDS[@]}"; do
#     python run_topic_modeling.py \
#       --data_path "${DATA_PATH}" \
#       --model generative \
#       --K 25 \
#       --hidden_state_layer "${HL}" \
#       --test_corpus_path data/bbc-news_Llama-3.2-1B_2000/bow_dataset.txt \
#       --seed "${SEED}"
#   done
# done

for SEED in "${SEEDS[@]}"; do
  python run_topic_modeling.py \
    --data_path "${DATA_PATH}" \
    --model zeroshot \
    --K 25 \
    --test_corpus_path data/bbc-news_Llama-3.2-1B_2000/bow_dataset.txt \
    --seed "${SEED}" \
    --num_epochs 100 \
    --eval_prompt_path topic_ratings/bbcnews_system.jinja 
done


for SEED in "${SEEDS[@]}"; do
  python run_topic_modeling.py \
    --data_path "${DATA_PATH}" \
    --model combined \
    --K 25 \
    --test_corpus_path data/bbc-news_Llama-3.2-1B_2000/bow_dataset.txt \
    --seed "${SEED}" \
    --num_epochs 100 \
    --eval_prompt_path topic_ratings/bbcnews_system.jinja 
done
