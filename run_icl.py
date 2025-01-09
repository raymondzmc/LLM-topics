import json
import torch
import argparse
import numpy as np
from datasets import load_dataset, load_from_disk
from train import train_generative_tm, train_ctm
from transformers import AutoModelForCausalLM, AutoTokenizer
from llm import jinja_template_manager
from contextualized_topic_models.evaluation.measures import (
    TopicDiversity,
    CoherenceNPMI,
    CoherenceWordEmbeddings,
    InvertedRBO,
)


def get_retrieval_mappings(document, model, tokenizer, method):
    input_ids = tokenizer.encode(document, return_tensors='pt').to(device)
    outputs = model(
        input_ids,
        use_cache=True,
        output_hidden_states=True
    )
    if args.method == 'next_token_distribution':
        logits = outputs.logits
        next_token_logits = logits[:, -1, :]
        next_token_probs = torch.softmax(next_token_logits, dim=-1).squeeze(0)
        next_token_probs = next_token_probs.cpu().tolist()
        return next_token_probs
    elif args.method == 'hidden_states':
        embeddings = [h[0, -1].cpu().tolist() for h in outputs.hidden_states]
        return embeddings
    elif args.method == 'topic_distribution':
        pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default=None)
    parser.add_argument('--model_name', type=str, default='meta-llama/Llama-3.2-1B')
    parser.add_argument('--use_flash_attention_2', action='store_true', default=True)
    parser.add_argument('--num_demonstrations', type=int, default=1)
    parser.add_argument('--method', type=str, default=None, choices=['hidden_states', 'next_token_distribution', 'topic_distribution'])
    args = parser.parse_args()

    dataset = load_dataset('EdinburghNLP/xsum')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
        use_flash_attention_2=args.use_flash_attention_2,
    ).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    
    train_dataset = dataset['train']
    test_dataset = dataset['test']

    # Get retrieval mappings for train dataset
    keys = []
    for idx, example in enumerate(train_dataset):
        keys.append(get_retrieval_mappings(example['document'], model, tokenizer, args.method))
    
    references = []
    predictions = []
    for example in test_dataset:
        query = get_retrieval_mappings(example['document'], model, tokenizer, args.method)
        if args.method == 'next_token_distribution':
            query = torch.tensor(query)
            kl_divergences = torch.nn.functional.kl_div(query, torch.tensor(keys), reduction='none')
            top_indices = torch.argsort(kl_divergences, dim=-1, descending=True)[:args.num_demonstrations]
            demonstrations = [train_dataset[i] for i in top_indices]
        elif args.method == 'hidden_states':
            query = torch.tensor(query)
            cosine_similarities = torch.nn.functional.cosine_similarity(query, torch.tensor(keys), dim=-1)
            top_indices = torch.argsort(cosine_similarities, dim=-1, descending=True)[:args.num_demonstrations]
            demonstrations = [train_dataset[i] for i in top_indices]
        elif args.method == 'topic_distribution':
            pass
        
        prompt = jinja_template_manager.render(
            'icl_summarization.jinja',
            examples=demonstrations,
            target_document=example['document']
        )
        # Generate the summary with \n as end of sentence token
        summary = model.generate(
            prompt,
            max_new_tokens=100,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id
        )
        print(summary)
        predictions.append(summary)
        references.append(example['summary'])

    results = rouge_metrics(references, predictions)
    print(results)
            