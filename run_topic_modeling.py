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
from utils.io import load_processed_dataset


def evaluate_topic_model(topic_model, processed_dataset, vocab):
    evaluation_results = {}
    topics = topic_model.get_topics(10)
    topics = list(topics.values())
    td = TopicDiversity(topics)
    td_score = td.score(topk=10)
    print("Topic Diversity:", td_score)
    evaluation_results['topic_diversity'] = td_score

    texts = [example['content'].split() for example in processed_dataset]
    npmi = CoherenceNPMI(topics, texts)
    npmi_score = npmi.score(topk=10)
    print("NPMI:", npmi_score)
    evaluation_results['npmi'] = npmi_score

    we = CoherenceWordEmbeddings(topics)
    we_score = we.score(topk=10)
    print("Word Embeddings:", we_score)
    evaluation_results['word_embeddings'] = we_score

    irbo = InvertedRBO(topics)
    irbo_score = irbo.score(topk=10)
    print("Inverted RBO:", irbo_score)
    evaluation_results['inverted_rbo'] = irbo_score
    return evaluation_results


def main(args):
    vocab = json.load(open(args.vocab_path, 'r'))
    processed_dataset = load_processed_dataset(args.processed_data_path)
    model_weight_path: str | None = os.path.join(args.checkpoint_path, 'model.pt')
    model_weights: dict | None = None
    if os.path.exists(model_weight_path):
        print(f"Found model weights at {model_weight_path}.")
        model_weights = torch.load(model_weight_path)

    topic_model = train_topic_model(args.model,
                                    vocab,
                                    processed_dataset,
                                    args.K,
                                    embedding_type=args.embedding_type,
                                    hidden_sizes=(args.hidden_sizes, args.hidden_sizes),
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

    evaluation_results = evaluate_topic_model(topic_model, processed_dataset, vocab)
    with open(os.path.join(args.checkpoint_path, 'evaluation_results.json'), 'w') as f:
        json.dump(evaluation_results, f)
    print(f"Saved evaluation results to {os.path.join(args.checkpoint_path, 'evaluation_results.json')}.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default=None)
    parser.add_argument('--model', type=str, default='gtm', choices=['generative', 'zeroshot', 'combined'])
    parser.add_argument('--K', type=int, default=10, help='Number of topics')

    # Model hyperparameters
    parser.add_argument('--hidden_sizes', type=int, default=200)
    parser.add_argument('--embedding_type', type=str, default='hidden_states', choices=['hidden_states', 'sbert'])
    parser.add_argument('--hidden_state_layer', type=int, default=0, help='Layer of hidden states to use for embedding')
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

    args.checkpoint_path = os.path.join(args.data_path, 'checkpoints', f'{args.model}_K{args.K}.pt')
    if not os.path.exists(args.checkpoint_path):
        args.checkpoint_path = os.path.join(args.checkpoint_path, 'run_0')
        os.makedirs(args.checkpoint_path, exist_ok=True)
    else:
        run_dirs = [
            os.path.join(args.checkpoint_path, f) for f in os.listdir(args.checkpoint_path)
            if f.startswith('run_') and 
               os.path.isdir(os.path.join(args.checkpoint_path, f)) and 
               len(os.listdir(os.path.join(args.checkpoint_path, f))) > 0
        ]
        num_runs = len(run_dirs)
        print(f"Found {num_runs} runs in {args.checkpoint_path}.")

        # Check if a run dir has the same args as the current run
        previous_run_path: None | str = None
        for run_dir_path in run_dirs:
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
