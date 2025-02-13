# #!/usr/bin/env bash
python process_dataset.py --dataset fancyzhx/dbpedia_14 \
                          --content_key content \
                          --split test \
                          --vocab_size 2000 \
                          --model_name meta-llama/Llama-3.2-1B \
                          --single_token_only \
                          --hidden_state_layer 16 \
                          --batch_size 32 \
                          --bow_dataset


# python run_topic_modeling.py \
#   --data_path ./data/dbpedia_14_Llama-3.2-1B_2000 \
#   --model zeroshot \
#   --K 25 \
#   --test_corpus_path data/dbpedia_14_Llama-3.2-1B_2000/bow_dataset.txt \
#   --num_seeds 5 \
#   --num_epochs 20
  
# python run_topic_modeling.py \
#   --data_path ./data/dbpedia_14_Llama-3.2-1B_2000 \
#   --model zeroshot \
#   --K 50 \
#   --test_corpus_path data/dbpedia_14_Llama-3.2-1B_2000/bow_dataset.txt \
#   --num_seeds 5 \
#   --num_epochs 20
  
# python run_topic_modeling.py \
#   --data_path ./data/dbpedia_14_Llama-3.2-1B_2000 \
#   --model combined \
#   --K 25 \
#   --test_corpus_path data/dbpedia_14_Llama-3.2-1B_2000/bow_dataset.txt \
#   --num_seeds 5 \
#   --num_epochs 20

# python run_topic_modeling.py \
#   --data_path ./data/dbpedia_14_Llama-3.2-1B_2000 \
#   --model combined \
#   --K 50 \
#   --test_corpus_path data/dbpedia_14_Llama-3.2-1B_2000/bow_dataset.txt \
#   --num_seeds 5 \
#   --num_epochs 20 

python run_topic_modeling.py \
  --data_path ./data/dbpedia_14_Llama-3.2-1B_2000 \
  --model generative \
  --K 50 \
  --test_corpus_path data/dbpedia_14_Llama-3.2-1B_2000/bow_dataset.txt \
  --num_seeds 5 \
  --num_epochs 20 

  
  

# # Hidden layers based on your commands
# HIDDEN_LAYERS=(0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16)

# # Loop over hidden layers
# for HL in "${HIDDEN_LAYERS[@]}"; do
#   python run_topic_modeling.py \
#     --data_path ./data/dbpedia_14_Llama-3.2-1B_2000 \
#     --model generative \
#     --K 25 \
#     --test_corpus_path data/dbpedia_14_Llama-3.2-1B_2000/bow_dataset.txt \
#     --num_seeds 5 \
#     --num_epochs 20 

#   python run_topic_modeling.py \
#     --data_path ./data/dbpedia_14_Llama-3.2-1B_2000 \
#     --model generative \
#     --K 25 \
#     --test_corpus_path data/dbpedia_14_Llama-3.2-1B_2000/bow_dataset.txt \
#     --num_seeds 5 \
#     --num_epochs 50 
# done