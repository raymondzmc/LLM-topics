# python process_dataset.py --dataset fancyzhx/dbpedia_14 \
#                           --content_key content \
#                           --split test \
#                           --vocab_size 2000 \
#                           --model_name meta-llama/Llama-3.2-1B \
#                           --single_token_only \
#                           --batch_size 32 \
#                           --bow_dataset \
#                           --examples_per_vocab 100

# # Train topic model with each hidden layer as input
# python run_topic_modeling.py --data_path ./data/dbpedia_14_Llama-3.2-1B_2000/test \
#                              --model generative \
#                              --K 25 \
#                              --test_corpus_path data/dbpedia_14_Llama-3.2-1B_2000/test/bow_dataset.txt

# python run_topic_modeling.py --data_path ./data/dbpedia_14_Llama-3.2-1B_2000/train \
#                              --model generative \
#                              --K 25 \
#                              --hidden_state_layer 2 \
#                              --test_corpus_path data/dbpedia_14_Llama-3.2-1B_2000/test/bow_dataset.txt

# python run_topic_modeling.py --data_path ./data/dbpedia_14_Llama-3.2-1B_2000/train \
#                              --model generative \
#                              --K 25 \
#                              --hidden_state_layer 3 \
#                              --test_corpus_path data/dbpedia_14_Llama-3.2-1B_2000/test/bow_dataset.txt

# python run_topic_modeling.py --data_path ./data/dbpedia_14_Llama-3.2-1B_2000/train \
#                              --model generative \
#                              --K 25 \
#                              --hidden_state_layer 4  \
#                              --test_corpus_path data/dbpedia_14_Llama-3.2-1B_2000/test/bow_dataset.txt

# python run_topic_modeling.py --data_path ./data/dbpedia_14_Llama-3.2-1B_2000/train \
#                              --model generative \
#                              --K 25 \
#                              --hidden_state_layer 5 \
#                              --test_corpus_path data/dbpedia_14_Llama-3.2-1B_2000/test/bow_dataset.txt

# python run_topic_modeling.py --data_path ./data/dbpedia_14_Llama-3.2-1B_2000/train \
#                              --model generative \
#                              --K 25 \
#                              --hidden_state_layer 6 \
#                              --test_corpus_path data/dbpedia_14_Llama-3.2-1B_2000/test/bow_dataset.txt

# python run_topic_modeling.py --data_path ./data/dbpedia_14_Llama-3.2-1B_2000/train \
#                              --model generative \
#                              --K 25 \
#                              --hidden_state_layer 7 \
#                              --test_corpus_path data/dbpedia_14_Llama-3.2-1B_2000/test/bow_dataset.txt

# python run_topic_modeling.py --data_path ./data/dbpedia_14_Llama-3.2-1B_2000/train \
#                              --model generative \
#                              --K 25 \
#                              --hidden_state_layer 8 \
#                              --test_corpus_path data/dbpedia_14_Llama-3.2-1B_2000/test/bow_dataset.txt

# python run_topic_modeling.py --data_path ./data/dbpedia_14_Llama-3.2-1B_2000/train \
#                              --model generative \
#                              --K 25 \
#                              --hidden_state_layer 9 \
#                              --test_corpus_path data/dbpedia_14_Llama-3.2-1B_2000/test/bow_dataset.txt

# python run_topic_modeling.py --data_path ./data/dbpedia_14_Llama-3.2-1B_2000/train \
#                              --model generative \
#                              --K 25 \
#                              --hidden_state_layer 10 \
#                              --test_corpus_path data/dbpedia_14_Llama-3.2-1B_2000/test/bow_dataset.txt

# python run_topic_modeling.py --data_path ./data/dbpedia_14_Llama-3.2-1B_2000/train \
#                              --model generative \
#                              --K 25 \
#                              --hidden_state_layer 11 \
#                              --test_corpus_path data/dbpedia_14_Llama-3.2-1B_2000/test/bow_dataset.txt

# python run_topic_modeling.py --data_path ./data/dbpedia_14_Llama-3.2-1B_2000/train \
#                              --model generative \
#                              --K 25 \
#                              --hidden_state_layer 12 \
#                              --test_corpus_path data/dbpedia_14_Llama-3.2-1B_2000/test/bow_dataset.txt

# python run_topic_modeling.py --data_path ./data/dbpedia_14_Llama-3.2-1B_2000/train \
#                              --model generative \
#                              --K 25 \
#                              --hidden_state_layer 13  \
#                              --test_corpus_path data/dbpedia_14_Llama-3.2-1B_2000/test/bow_dataset.txt

# python run_topic_modeling.py --data_path ./data/dbpedia_14_Llama-3.2-1B_2000/train \
#                              --model generative \
#                              --K 25 \
#                              --hidden_state_layer 14 \
#                              --test_corpus_path data/dbpedia_14_Llama-3.2-1B_2000/test/bow_dataset.txt

# python run_topic_modeling.py --data_path ./data/dbpedia_14_Llama-3.2-1B_2000/train \
#                              --model generative \
#                              --K 25 \
#                              --hidden_state_layer 15 \
#                              --test_corpus_path data/dbpedia_14_Llama-3.2-1B_2000/test/bow_dataset.txt

# python run_topic_modeling.py --data_path ./data/dbpedia_14_Llama-3.2-1B_2000/train \
#                              --model generative \
#                              --K 25 \
#                              --hidden_state_layer 16 \
#                              --test_corpus_path data/dbpedia_14_Llama-3.2-1B_2000/test/bow_dataset.txt

python run_topic_modeling.py --data_path ./data/dbpedia_14_Llama-3.2-1B_2000/test \
                             --model zeroshot \
                             --K 25 \
                             --test_corpus_path data/dbpedia_14_Llama-3.2-1B_2000/test/bow_dataset.txt \
                             --num_epochs 100

python run_topic_modeling.py --data_path ./data/dbpedia_14_Llama-3.2-1B_2000/test \
                             --model combined \
                             --K 25 \
                             --test_corpus_path data/dbpedia_14_Llama-3.2-1B_2000/test/bow_dataset.txt \
                             --num_epochs 100