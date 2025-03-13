import os
import csv
import json
import argparse
from utils.io import load_processed_dataset
from utils.metrics import compute_aggregate_results
from models.CTM import CTM as GenerativeTM
from run_topic_modeling import evaluate_topic_model
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
    processed_dataset = load_processed_dataset(processed_data_path, layer_idx=16)
    vocab_path = os.path.join(args.data_path, 'vocab.json')
    processed_dataset['vocab'] = json.load(open(vocab_path, encoding='utf-8'))
    for seed in range(5):
        seed_dir = os.path.join(args.results_path, f"seed_{seed}")
        os.makedirs(seed_dir, exist_ok=True)
        model = GenerativeTM(
            num_topics=args.num_topics,
            num_layers=args.num_hidden_layers,
            num_neurons=args.hidden_size,
            num_epochs=args.num_epochs,
            use_partitions=False,
            seed=seed,
            loss_weight=args.loss_weight,
            sparsity_ratio=args.sparsity_ratio,
        )
        model_output = model.train_model(dataset=processed_dataset, top_words=args.top_words) # Train the model
        topics = model_output['topics']
        torch.save(model_output, os.path.join(seed_dir, 'model_output.pt'))

        with open(os.path.join(seed_dir, 'topics.json'), 'w', encoding="utf-8") as f:
            json.dump(topics, f)
        evaluation_results = evaluate_topic_model(model, top_words=args.top_words, test_corpus=bow_corpus, topics=topics)[1]
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
    parser.add_argument('--num_hidden_layers', type=int, default=2)
    parser.add_argument('--hidden_size', type=int, default=200)
    parser.add_argument('--num_seeds', type=int, default=5)
    parser.add_argument('--num_epochs', type=int, default=50)
    parser.add_argument('--top_words', type=int, default=20)
    parser.add_argument('--loss_weight', type=float, default=1.0)
    parser.add_argument('--sparsity_ratio', type=float, default=1.0)
    args = parser.parse_args()
    run(args)
