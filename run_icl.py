import json
import torch
import argparse
import numpy as np
from datasets import load_from_disk
from train import train_generative_tm, train_ctm
from contextualized_topic_models.evaluation.measures import (
    TopicDiversity,
    CoherenceNPMI,
    CoherenceWordEmbeddings,
    InvertedRBO,
)


def main(args):
    vocab = json.load(open(args.vocab_path, 'r'))
    processed_dataset = load_from_disk(args.processed_data_path)
    if args.model == 'generative':
        tm = train_generative_tm(vocab,
                                processed_dataset,
                                args.K,
                                args.num_epochs,
                                args.hidden_sizes)
    else:
        tm = train_ctm(args.model,
                       vocab,
                       processed_dataset,
                       args.K,
                       args.num_epochs,
                       args.hidden_sizes)

    torch.save(tm.model, args.checkpoint_path)

    topics = tm.get_topics(args.K)
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
    parser.add_argument('--vocab_path', type=str, default=None)
    parser.add_argument('--processed_data_path', type=str, default=None)
    parser.add_argument('--model', type=str, default='gtm', choices=['generative', 'zeroshot', 'combined'])
    parser.add_argument('--checkpoint_path', type=str, default=None)
    parser.add_argument('--K', type=int, default=10)
    parser.add_argument('--hidden_sizes', type=tuple, default=(200, 200))
    parser.add_argument('--num_epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=8)
    args = parser.parse_args()
    main(args)