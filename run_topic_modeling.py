import os
import json
import torch
import argparse
import numpy as np
from datasets import load_from_disk
from train import train_topic_model
from contextualized_topic_models.evaluation.measures import (
    CoherenceNPMI,
    TopicDiversity,
    CoherenceWordEmbeddings,
    InvertedRBO,
)
from utils.io import load_processed_dataset
from utils.metrics import compute_llm_rating
import pdb


def evaluate_topic_model(topic_model, top_words=10, test_corpus=None):
    evaluation_results = {}
    topics = topic_model.get_topics(top_words)
    topics = list(topics.values())

    td = TopicDiversity(topics)
    td_score = td.score(topk=top_words)
    print("Topic Diversity:", td_score)
    evaluation_results['topic_diversity'] = float(td_score)

    we = CoherenceWordEmbeddings(topics)
    we_score = we.score(topk=top_words)
    print("Word Embeddings:", we_score)
    evaluation_results['word_embeddings'] = float(we_score)

    irbo = InvertedRBO(topics)
    irbo_score = irbo.score(topk=top_words)
    print("Inverted RBO:", irbo_score)
    evaluation_results['inverted_rbo'] = float(irbo_score)

    npmi = CoherenceNPMI(topics, test_corpus)
    npmi_score = npmi.score(topk=top_words)
    print("NPMI:", npmi_score)
    evaluation_results['npmi'] = float(npmi_score)

    llm_ratings = compute_llm_rating(topics)
    llm_average_rating = float(np.mean(llm_ratings))
    print("LLM Rating:", llm_average_rating)
    evaluation_results['llm_rating'] = llm_average_rating
    return topics, evaluation_results


def main(args, vocab, processed_dataset):
    model_checkpoint_path: str | None = os.path.join(args.checkpoint_path, 'model.pt')
    model_checkpoint: dict | None = None
    if os.path.exists(model_checkpoint_path):
        print(f"Found model weights at \"{model_checkpoint_path}\".")
        model_checkpoint = torch.load(model_checkpoint_path, weights_only=True)

    hidden_sizes = tuple([args.hidden_sizes for _ in range(args.num_hidden_layers)])
    topic_model = train_topic_model(args.model,
                                    vocab,
                                    processed_dataset,
                                    args.K,
                                    embedding_type=args.embedding_type,
                                    hidden_state_layer=args.hidden_state_layer,
                                    hidden_sizes=hidden_sizes,
                                    loss_type=args.loss_type,
                                    activation=args.activation,
                                    dropout=args.dropout,
                                    learn_priors=args.learn_priors,
                                    temperature=args.temperature,
                                    batch_size=args.batch_size,
                                    lr=args.lr,
                                    momentum=args.momentum,
                                    solver=args.solver,
                                    num_epochs=args.num_epochs,
                                    reduce_on_plateau=args.reduce_on_plateau,
                                    label_size=args.label_size,
                                    loss_weight=args.loss_weight,
                                    model_checkpoint=model_checkpoint,
                                    continue_training=args.continue_training)    
    torch.save(topic_model.model.state_dict(), model_checkpoint_path)
    print(f"Saved model weights to \"{model_checkpoint_path}\".")

    test_corpus = None
    if args.test_corpus_path is not None and os.path.isfile(args.test_corpus_path):
        with open(args.test_corpus_path, 'r') as f:
            test_corpus = [line.strip().split() for line in f.readlines()]

    topics, evaluation_results = evaluate_topic_model(
        topic_model,
        top_words=args.top_words,
        test_corpus=test_corpus,
    )

    with open(os.path.join(args.checkpoint_path, 'evaluation_results.json'), 'w') as f:
        json.dump(evaluation_results, f)
    print(f"Saved evaluation results to \"{os.path.join(args.checkpoint_path, 'evaluation_results.json')}\".")

    with open(os.path.join(args.checkpoint_path, 'topics.json'), 'w') as f:
        json.dump(topics, f, indent=4)
    print(f"Saved topics to \"{os.path.join(args.checkpoint_path, 'topics.json')}\".")
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default=None)

    # Model hyperparameters
    train_group = parser.add_argument_group("Model Hyperparameters for Training")
    train_group.add_argument('--model', type=str, default='gtm', choices=['generative', 'zeroshot', 'combined'])
    train_group.add_argument('--K', type=int, default=25, help='Number of topics')
    train_group.add_argument('--embedding_type', type=str, default='hidden_states', choices=['hidden_states', 'sbert'])
    train_group.add_argument('--hidden_state_layer', type=int, default=-1, help='Layer of hidden states to use for embedding')
    train_group.add_argument('--loss_type', type=str, default='ce', choices=['ce', 'kl'])
    train_group.add_argument('--hidden_sizes', type=int, default=200)
    train_group.add_argument('--num_hidden_layers', type=int, default=2)
    train_group.add_argument('--activation', type=str, default='gelu')
    train_group.add_argument('--dropout', type=float, default=0.2)
    train_group.add_argument('--learn_priors', type=bool, default=True)
    train_group.add_argument('--batch_size', type=int, default=64)
    train_group.add_argument('--lr', type=float, default=2e-3)
    train_group.add_argument('--temperature', type=float, default=1.0)
    train_group.add_argument('--momentum', type=float, default=0.99)
    train_group.add_argument('--solver', type=str, default='adam')
    train_group.add_argument('--num_epochs', type=int, default=20)
    train_group.add_argument('--reduce_on_plateau', type=bool, default=False)
    train_group.add_argument('--label_size', type=int, default=0)
    train_group.add_argument('--loss_weight', type=float, default=None)
    train_group.add_argument('--continue_training', action='store_true')
    train_group.add_argument('--seed', type=int, default=42)

    # Evaluation parameters
    parser.add_argument('--num_seeds', type=int, default=1, help='Number of seeds to run')
    parser.add_argument('--test_corpus_path', type=str, default=None, help='Path to test corpus')
    parser.add_argument('--top_words', type=int, default=20, help='Number of top words to evaluate for each topic')
    args = parser.parse_args()

    train_arg_names = {action.dest for action in train_group._group_actions}

    args.vocab_path = os.path.join(args.data_path, 'vocab.json')
    vocab = json.load(open(args.vocab_path, 'r'))

    if args.model == 'generative':
        args.processed_data_path = os.path.join(args.data_path, 'processed_dataset')
        processed_dataset = load_processed_dataset(args.processed_data_path, layer_idx=args.hidden_state_layer)
    else:
        args.processed_data_path = os.path.join(args.data_path, 'tokenized_dataset')
        processed_dataset = load_from_disk(args.processed_data_path)

    random_seeds = list(range(args.seed, args.seed + args.num_seeds))
    for seed in random_seeds:
        args.seed = seed
        args.checkpoint_path = os.path.join(args.data_path, 'checkpoints', f'{args.model}_K{args.K}')
        if not os.path.exists(args.checkpoint_path):
            checkpoint_path = os.path.join(args.checkpoint_path, f'run_0')
            os.makedirs(checkpoint_path, exist_ok=True)
            with open(os.path.join(checkpoint_path, 'args.json'), 'w') as f:
                json.dump(vars(args), f, indent=4)
            args.checkpoint_path = checkpoint_path
        else:
            previous_runs = [
                os.path.join(args.checkpoint_path, d) for d in os.listdir(args.checkpoint_path)
                if d.startswith('run_') and 
                os.path.isdir(os.path.join(args.checkpoint_path, d)) and 
                os.path.isfile(os.path.join(args.checkpoint_path, d, 'model.pt'))
            ]
            num_runs = len(previous_runs)
            print(f"Found {num_runs} runs in \"{args.checkpoint_path}\".")

            # Check if a run dir has the same args as the current run
            previous_run_path: None | str = None
            for run_path in previous_runs:
                run_args = json.load(open(os.path.join(run_path, 'args.json'), 'r'))

                # Compare only the training args to check if the run has identical configs
                filtered_previous_args = {k: run_args[k] for k in train_arg_names if k in run_args}
                filtered_current_args = {k: vars(args)[k] for k in train_arg_names if k in vars(args)}
                if filtered_previous_args == filtered_current_args:
                    previous_run_path = run_path
                    print(f"Found checkpoint with identical configs in \"{previous_run_path}\".")
                    break

            if previous_run_path is None:
                print(f"No previous checkpoint found with identical configs. Creating new run.")
                checkpoint_path = os.path.join(args.checkpoint_path, f'run_{num_runs}')
                os.makedirs(checkpoint_path, exist_ok=True)
                with open(os.path.join(checkpoint_path, 'args.json'), 'w') as f:
                    json.dump(vars(args), f, indent=4)
                args.checkpoint_path = checkpoint_path
            else:
                args.checkpoint_path = previous_run_path

        # Set seed for torch and numpy
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        np.random.seed(args.seed)
        
        main(args, vocab, processed_dataset)
