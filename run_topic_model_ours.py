import os
import csv
import json
import argparse
from utils.io import load_processed_dataset
from utils.embeddings import get_openai_embedding
from utils.metrics import compute_aggregate_results
from models.CTM import CTM as GenerativeTM
from utils.metrics import evaluate_topic_model
import torch
import pdb


def run(args):
    if args.results_path is None:
        dataset_name = os.path.basename(args.data_path).split('_')[0]
        args.results_path = f'results/{dataset_name}/generative_K{args.num_topics}'

    os.makedirs(args.results_path, exist_ok=True)

    with open(os.path.join(args.data_path, 'bow_dataset.txt'), 'r', encoding='utf-8') as f:
        bow_corpus = [line.strip().split() for line in f if line.strip()]

    processed_data_path = os.path.join(args.data_path, 'processed_dataset')
    processed_dataset = load_processed_dataset(processed_data_path, layer_idx=args.input_hidden_layer)
    vocab_path = os.path.join(args.data_path, 'vocab.json')
    processed_dataset['vocab'] = json.load(open(vocab_path, encoding='utf-8'))
    vocab_embedding_path = os.path.join(args.data_path, 'vocab_embeddings.json')
    if os.path.exists(vocab_embedding_path):
        vocab_embeddings = json.load(open(vocab_embedding_path, encoding='utf-8'))
    else:
        vocab_embeddings = get_openai_embedding(processed_dataset['vocab'])
        with open(vocab_embedding_path, 'w', encoding='utf-8') as f:
            json.dump(vocab_embeddings, f)

    for seed in range(5):
        seed_dir = os.path.join(args.results_path, f"seed_{seed}")
        os.makedirs(seed_dir, exist_ok=True)
        model_output_path = os.path.join(seed_dir, 'model_output.pt')
        if os.path.exists(model_output_path):
            model_output = torch.load(model_output_path, weights_only=False)
        else:
            model = GenerativeTM(
                num_topics=args.num_topics,
                num_layers=args.num_hidden_layers,
                num_neurons=args.hidden_size,
                num_epochs=args.num_epochs,
                use_partitions=False,
                seed=seed,
                loss_weight=args.loss_weight,
                sparsity_ratio=args.sparsity_ratio,
                loss_type=args.loss_type,
                temperature=args.temperature
            )
            model.train_model(dataset=processed_dataset, top_words=args.top_words)
            torch.save(model.model.model.state_dict(), os.path.join(seed_dir, 'checkpoint.pt'))
            model_output = model.model.get_info()
            torch.save(model_output, os.path.join(seed_dir, 'model_output.pt'))

        # checkpoints = os.listdir(checkpoint_dir)
        # if len(checkpoints) > 0:
        #     if len(checkpoints) > 1:
        #         print(f"Found {len(checkpoints)} checkpoints for seed {seed}. Using the latest one.")
        #     checkpoint_path = os.path.join(checkpoint_dir, checkpoints[-1])
        #     epoch_files = [os.path.join(checkpoint_path, f) for f in os.listdir(checkpoint_path) if f.endswith('.pth')]
        #     epochs = [int(f.split('_')[-1].split('.')[0]) for f in epoch_files]
        #     epochs.sort()
        #     model.model.load(checkpoint_path, epochs[-1])
        #     model_output = model.model.get_info()
        # else:
        #     model_output = model.train_model(dataset=processed_dataset, top_words=args.top_words) # Train the model
        #     model.model.save(checkpoint_dir)
        topics = model_output['topics']
        with open(os.path.join(seed_dir, 'topics.json'), 'w', encoding="utf-8") as f:
            json.dump(topics, f)
        evaluation_results = evaluate_topic_model(model_output, top_words=args.top_words, test_corpus=bow_corpus, embeddings=vocab_embeddings)
        with open(os.path.join(seed_dir, 'evaluation_results.json'), 'w', encoding="utf-8") as f:
            json.dump(evaluation_results, f)

    averaged_results = compute_aggregate_results(args.results_path)
    with open(os.path.join(args.results_path, 'averaged_results.json'), 'w', encoding='utf-8') as f:
        json.dump(averaged_results, f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default=None)
    parser.add_argument('--results_path', type=str, default=None)
    parser.add_argument('--num_topics', type=int, default=25, help='Number of topics')
    parser.add_argument('--input_hidden_layer', type=int, default=-1)
    parser.add_argument('--num_hidden_layers', type=int, default=2)
    parser.add_argument('--hidden_size', type=int, default=200)
    parser.add_argument('--num_seeds', type=int, default=5)
    parser.add_argument('--num_epochs', type=int, default=50)
    parser.add_argument('--top_words', type=int, default=20)
    parser.add_argument('--loss_weight', type=float, default=1.0)
    parser.add_argument('--sparsity_ratio', type=float, default=1.0)
    parser.add_argument('--loss_type', type=str, default='KL', choices=['KL', 'CE'])
    parser.add_argument('--temperature', type=float, default=1.0)
    args = parser.parse_args()
    run(args)
