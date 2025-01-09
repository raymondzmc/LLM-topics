# python process_dataset.py --dataset google-research-datasets/newsgroup \
#                           --content_key text \
#                           --split train \
#                           --vocab_path data/20newsgroup_vocab.json \
#                           --model_name meta-llama/Llama-3.2-1B \
#                           --data_path data \
#                           --batch_size 32 \
#                           --examples_per_vocab 100

# Train topic model with each hidden layer as input
python run_topic_modeling.py --data_path ./data/newsgroup_Llama-3.2-1B_2000/train \
                             --model generative \
                             --K 25 \
                             --hidden_state_layer 1 \
                             --test_corpus_path data/20newsgroup_corpus.txt

python run_topic_modeling.py --data_path ./data/newsgroup_Llama-3.2-1B_2000/train \
                             --model generative \
                             --K 25 \
                             --hidden_state_layer 2 \
                             --test_corpus_path data/20newsgroup_corpus.txt

python run_topic_modeling.py --data_path ./data/newsgroup_Llama-3.2-1B_2000/train \
                             --model generative \
                             --K 25 \
                             --hidden_state_layer 3 \
                             --test_corpus_path data/20newsgroup_corpus.txt

python run_topic_modeling.py --data_path ./data/newsgroup_Llama-3.2-1B_2000/train \
                             --model generative \
                             --K 25 \
                             --hidden_state_layer 4 \
                             --test_corpus_path data/20newsgroup_corpus.txt

python run_topic_modeling.py --data_path ./data/newsgroup_Llama-3.2-1B_2000/train \
                             --model generative \
                             --K 25 \
                             --hidden_state_layer 5 \
                             --test_corpus_path data/20newsgroup_corpus.txt

python run_topic_modeling.py --data_path ./data/newsgroup_Llama-3.2-1B_2000/train \
                             --model generative \
                             --K 25 \
                             --hidden_state_layer 6 \
                             --test_corpus_path data/20newsgroup_corpus.txt

python run_topic_modeling.py --data_path ./data/newsgroup_Llama-3.2-1B_2000/train \
                             --model generative \
                             --K 25 \
                             --hidden_state_layer 7 \
                             --test_corpus_path data/20newsgroup_corpus.txt

python run_topic_modeling.py --data_path ./data/newsgroup_Llama-3.2-1B_2000/train \
                             --model generative \
                             --K 25 \
                             --hidden_state_layer 8 \
                             --test_corpus_path data/20newsgroup_corpus.txt

python run_topic_modeling.py --data_path ./data/newsgroup_Llama-3.2-1B_2000/train \
                             --model generative \
                             --K 25 \
                             --hidden_state_layer 9 \
                             --test_corpus_path data/20newsgroup_corpus.txt

python run_topic_modeling.py --data_path ./data/newsgroup_Llama-3.2-1B_2000/train \
                             --model generative \
                             --K 25 \
                             --hidden_state_layer 10 \
                             --test_corpus_path data/20newsgroup_corpus.txt

python run_topic_modeling.py --data_path ./data/newsgroup_Llama-3.2-1B_2000/train \
                             --model generative \
                             --K 25 \
                             --hidden_state_layer 11 \
                             --test_corpus_path data/20newsgroup_corpus.txt

python run_topic_modeling.py --data_path ./data/newsgroup_Llama-3.2-1B_2000/train \
                             --model generative \
                             --K 25 \
                             --hidden_state_layer 12 \
                             --test_corpus_path data/20newsgroup_corpus.txt

python run_topic_modeling.py --data_path ./data/newsgroup_Llama-3.2-1B_2000/train \
                             --model generative \
                             --K 25 \
                             --hidden_state_layer 13 \
                             --test_corpus_path data/20newsgroup_corpus.txt

python run_topic_modeling.py --data_path ./data/newsgroup_Llama-3.2-1B_2000/train \
                             --model generative \
                             --K 25 \
                             --hidden_state_layer 14 \
                             --test_corpus_path data/20newsgroup_corpus.txt

python run_topic_modeling.py --data_path ./data/newsgroup_Llama-3.2-1B_2000/train \
                             --model generative \
                             --K 25 \
                             --hidden_state_layer 15 \
                             --test_corpus_path data/20newsgroup_corpus.txt

python run_topic_modeling.py --data_path ./data/newsgroup_Llama-3.2-1B_2000/train \
                             --model generative \
                             --K 25 \
                             --hidden_state_layer 16 \
                             --test_corpus_path data/20newsgroup_corpus.txt
