import os
import csv
import json
import argparse
from octis.dataset.dataset import Dataset
from octis.models.CTM import CTM
from utils.metrics import compute_aggregate_results
from run_topic_modeling import evaluate_topic_model
import torch
import pdb


def run(args):
    if args.results_path is None:
        dataset_name = os.path.basename(args.data_path).split('_')[0]
        args.results_path = f'results/{dataset_name}/{args.model}_K{args.num_topics}'

    os.makedirs(args.results_path, exist_ok=True)
    
    with open(os.path.join(args.data_path, 'bow_dataset.txt'), 'r', encoding='utf-8') as f:
        bow_corpus = [line.strip().split() for line in f if line.strip()]

    if args.model in ['zeroshot', 'combined']:
        corpus_file = os.path.join(args.data_path, 'corpus.tsv')
        vocab_file = os.path.join(args.data_path, 'vocab.txt')

        if not os.path.exists(corpus_file):
            with open(corpus_file, 'w', encoding='utf-8', newline='') as tsvfile:
                writer = csv.writer(tsvfile, delimiter='\t')
                for doc in bow_corpus:
                    # Each row: document, partition, label (empty)
                    writer.writerow([' '.join(doc), 'train', ''])

        if not os.path.exists(vocab_file):
            with open(os.path.join(args.data_path, 'vocab.json'), encoding="utf-8") as f:
                vocab = json.load(f)

            with open(os.path.join(args.data_path, "vocabulary.txt"), "w", encoding="utf-8") as vocab_file:
                for word in vocab:
                    vocab_file.write(f"{word}\n")

        dataset = Dataset()
        dataset.load_custom_dataset_from_folder(args.data_path)
        for seed in range(5):
            seed_dir = os.path.join(args.results_path, f"seed_{seed}")
            os.makedirs(seed_dir, exist_ok=True)
            model = CTM(
                num_topics=args.num_topics,
                num_layers=args.num_hidden_layers,
                num_neurons=args.hidden_size,
                num_epochs=args.num_epochs,
                inference_type=args.model,
                use_partitions=False,
            )
            model_output = model.train_model(dataset=dataset, top_words=args.top_words) # Train the model
            topics = model_output['topics']
            torch.save(model_output, os.path.join(seed_dir, 'model_output.pt'))

            with open(os.path.join(seed_dir, 'topics.json'), 'w', encoding="utf-8") as f:
                json.dump(topics, f)
            topics, evaluation_results = evaluate_topic_model(model, top_words=args.top_words, test_corpus=bow_corpus, topics=topics)[1]
            with open(os.path.join(seed_dir, 'evaluation_results.json'), 'w', encoding="utf-8") as f:
                json.dump(evaluation_results, f)

    averaged_results = compute_aggregate_results(args.results_path)
    with open(os.path.join(args.results_path, 'averaged_results.json'), 'w', encoding='utf-8') as f:
        json.dump(averaged_results, f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default=None)
    parser.add_argument('--results_path', type=str, default=None)
    parser.add_argument('--model', type=str, default='zeroshot', choices=['zeroshot', 'combined', 'prodlda', 'lda', 'etm'])
    parser.add_argument('--num_topics', type=int, default=25, help='Number of topics')
    parser.add_argument('--num_hidden_layers', type=int, default=2)
    parser.add_argument('--hidden_size', type=int, default=200)
    parser.add_argument('--num_epochs', type=int, default=50)
    parser.add_argument('--top_words', type=int, default=20)
    parser.add_argument('--num_seeds', type=int, default=5)
    args = parser.parse_args()
    run(args)
