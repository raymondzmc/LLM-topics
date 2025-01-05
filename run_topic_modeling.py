import os
import json
import torch
import argparse
from datasets import load_from_disk
from train import train_topic_model
from contextualized_topic_models.evaluation.measures import (
    TopicDiversity,
    CoherenceNPMI,
    CoherenceWordEmbeddings,
    InvertedRBO,
)


def main(args):
    vocab = json.load(open(args.vocab_path, 'r'))
    processed_dataset = load_from_disk(args.processed_data_path)

    model_weight_path: str | None = os.path.join(args.checkpoint_path, 'model.pt')
    model_weights: dict | None = None
    if os.path.exists(model_weight_path):
        print(f"Found model weights at {model_weight_path}.")
        model_weights = torch.load(model_weight_path)

    topic_model = train_topic_model(args.model,
                                    vocab,
                                    processed_dataset,
                                    args.K,
                                    hidden_sizes=args.hidden_sizes,
                                    activation=args.activation,
                                    dropout=args.dropout,
                                    learn_priors=args.learn_priors,
                                    batch_size=args.batch_size,
                                    lr=args.lr,
                                    momentum=args.momentum,
                                    solver=args.solver,
                                    num_epochs=args.num_epochs,
                                    reduce_on_plateau=args.reduce_on_plateau,
                                    label_size=args.label_size,
                                    loss_weights=args.loss_weights,
                                    model_weights=model_weights,
                                    continue_training=args.continue_training)    

    torch.save(topic_model.model, model_weight_path)
    print(f"Saved model weights to {model_weight_path}.")

    topics = topic_model.get_topics(10)
    topics = list(topics.values())
    td = TopicDiversity(topics)
    print("Topic Diversity:", td.score(topk=10))

    texts = [example['content'].split() for example in processed_dataset]
    npmi = CoherenceNPMI(topics, texts)
    print("NPMI:", npmi.score(topk=10))

    we = CoherenceWordEmbeddings(topics)
    print("Word Embeddings:", we.score(topk=10))

    irbo = InvertedRBO(topics)
    print("Inverted RBO:", irbo.score(topk=10))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default=None)
    parser.add_argument('--model', type=str, default='gtm', choices=['generative', 'zeroshot', 'combined'])
    parser.add_argument('--K', type=int, default=10, help='Number of topics')

    # Model hyperparameters
    parser.add_argument('--hidden_sizes', type=tuple, default=(200, 200))
    parser.add_argument('--activation', type=str, default='softplus')
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--learn_priors', type=bool, default=True)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=2e-3)
    parser.add_argument('--momentum', type=float, default=0.99)
    parser.add_argument('--solver', type=str, default='adam')
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--reduce_on_plateau', type=bool, default=False)
    parser.add_argument('--label_size', type=int, default=0)
    parser.add_argument('--loss_weights', type=list, default=None)
    parser.add_argument('--continue_training', action='store_true')
    args = parser.parse_args()

    args.vocab_path = os.path.join(args.data_path, 'vocab.json')
    if args.model == 'generative':
        args.processed_data_path = os.path.join(args.data_path, 'processed_dataset')
    else:
        args.processed_data_path = os.path.join(args.data_path, 'tokenized_dataset')
    
    if not os.path.isfile(args.vocab_path):
        raise ValueError("Vocab file does not exist. Run process_dataset.py first.")
    if not os.path.isdir(args.processed_data_path):
        raise ValueError("Processed data directory does not exist. Run process_dataset.py first.")

    args.checkpoint_path = os.path.join(args.data_dir, 'checkpoints', f'{args.model}_K{args.K}.pt')
    if not os.path.exists(args.checkpoint_path):
        args.checkpoint_path = os.path.join(args.checkpoint_path, 'run_0')
        os.makedirs(args.checkpoint_path, exist_ok=True)
    else:
        num_runs = len(os.listdir(args.checkpoint_path))
        print(f"Found {num_runs} runs in {args.checkpoint_path}.")

        # Check if a run dir has the same args as the current run
        previous_run_path: None | str = None
        for run_dir in os.listdir(args.checkpoint_path):
            if run_dir.startswith('run_'):
                run_dir_path = os.path.join(args.checkpoint_path, run_dir)
                if os.path.isdir(run_dir_path):
                    run_args = json.load(open(os.path.join(run_dir_path, 'args.json'), 'r'))
                    if run_args == vars(args):
                        previous_run_path = run_dir_path
                        print(f"Saved checkpoint with identical configs already exists.",
                              f"Loading from {args.checkpoint_path}")
                        break
        if previous_run_path is None:
            print(f"No previous run found with identical configs. Creating new run.")
            args.checkpoint_path = os.path.join(args.checkpoint_path, f'run_{num_runs}')
            os.makedirs(args.checkpoint_path, exist_ok=True)
            with open(os.path.join(args.data_path, 'args.json'), 'w') as f:
                json.dump(vars(args), f)
        else:
            args.checkpoint_path = previous_run_path

    main(args)
