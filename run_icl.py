import os
import numpy as np
import pickle
import torch
import argparse
import json
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM, AutoTokenizer,
    StoppingCriteriaList, StoppingCriteria,
)
from llm import jinja_template_manager
import evaluate
from tqdm import tqdm
import pdb


def get_retrieval_dimension(model, tokenizer, method):
    if method == 'last_hidden_state' or method == 'mean_hidden_states':
        return model.config.hidden_size
    elif method == 'next_token_distribution':
        return model.config.vocab_size
    else:
        raise NotImplementedError


def get_retrieval_mappings(document, model, tokenizer, method):
    input_ids = tokenizer.encode(document, return_tensors='pt').to(device)
    outputs = model(
        input_ids,
        use_cache=True,
        output_hidden_states=True
    )
    if method == 'next_token_distribution':
        logits = outputs.logits
        next_token_logits = logits[:, -1, :]
        next_token_probs = torch.softmax(next_token_logits, dim=-1).squeeze(0)
        next_token_probs = next_token_probs.cpu().tolist()
        return next_token_probs
    elif method == 'last_hidden_state':
        embeddings = outputs.hidden_states[-1][0, -1].cpu().detach().to(torch.float16).numpy()
        return embeddings
    elif method == 'mean_hidden_states':
        embeddings = outputs.hidden_states[-1][0].mean(dim=0).cpu().detach().to(torch.float16).numpy()
        return embeddings
    elif method == 'topic_distribution':
        pass

def get_retrieval_mappings_batched(documents, model, tokenizer, method, batch_size=4, max_tokens=512):
    document_lengths = tokenizer(documents, truncation=False)['input_ids']
    document_lengths = [len(ids) for ids in document_lengths]
    _documents = [doc for doc, length in zip(documents, document_lengths) if length <= max_tokens]
    print(f"Kept {len(_documents)}/{len(documents)} documents with {max_tokens} tokens or less")
    
    all_embeddings = []
    for i in tqdm(range(0, len(_documents), batch_size)):
        batch_docs = documents[i:i + batch_size]

        # Set pad token if not set
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Tokenize and pad the batch
        inputs = tokenizer(batch_docs, return_tensors='pt', padding=True, truncation=True).to(device)
        attention_mask = inputs.attention_mask
        with torch.no_grad():
            outputs = model(
                input_ids=inputs.input_ids,
                attention_mask=attention_mask,
                use_cache=True,
                output_hidden_states=True
            )
        
        if method == 'next_token_distribution':
            logits = outputs.logits
            next_token_logits = logits[:, -1, :]
            next_token_probs = torch.softmax(next_token_logits, dim=-1)
            next_token_probs = next_token_probs.cpu().tolist()
            all_embeddings.extend(next_token_probs)
            
        elif method == 'last_hidden_state':
            # Get last hidden state for each sequence
            embeddings = outputs.hidden_states[-1][:, -1].cpu().to(torch.float16).numpy()
            all_embeddings.extend(embeddings)
            
        elif method == 'mean_hidden_states':
            # Get mean of hidden states for each sequence
            embeddings = outputs.hidden_states[-1].mean(dim=1).cpu().to(torch.float16).numpy()
            all_embeddings.extend(embeddings)
            
        elif method == 'topic_distribution':
            pass
            
    return np.array(all_embeddings)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default=None)
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--model_name', type=str, default='meta-llama/Llama-3.2-1B')
    parser.add_argument('--use_flash_attention_2', action='store_true', default=True)
    parser.add_argument('--num_demonstrations', type=int, default=1)
    parser.add_argument('--method', type=str, default=None, choices=['zeroshot', 'mean_hidden_states', 'last_hidden_state', 'next_token_distribution', 'topic_distribution'])
    args = parser.parse_args()

    if args.output_dir is None:
        raise ValueError('Output directory is required')
    
    os.makedirs(args.output_dir, exist_ok=True)

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
    if args.method != 'zeroshot':
        key_file_path = os.path.join(args.output_dir, f'train_keys_{args.method}.pkl')
        if os.path.exists(key_file_path):
            keys = pickle.load(open(key_file_path, 'rb'))
            print('Loaded retrieval mappings for train dataset')
            keys = torch.tensor(keys)
        else:
            print('Computing retrieval mappings for train dataset')
            keys = get_retrieval_mappings_batched(train_dataset['document'], model, tokenizer, args.method)
            pickle.dump(keys, open(key_file_path, 'wb'))
            print(f'Saved retrieval mappings for train dataset to {key_file_path}')
            keys = torch.tensor(keys)

    references = []
    predictions = []
    for example in tqdm(test_dataset):
        demonstrations = []
        if args.method != 'zeroshot':
            query = get_retrieval_mappings(example['document'], model, tokenizer, args.method)
            if args.method == 'next_token_distribution':
                query = torch.tensor(query).unsqueeze(0)
                kl_divergences = torch.nn.functional.kl_div(query, torch.tensor(keys), reduction='none')
                top_indices = torch.argsort(kl_divergences, dim=-1, descending=True)[:args.num_demonstrations]
                demonstrations = [train_dataset[i] for i in top_indices]
            elif args.method in ['last_hidden_state', 'mean_hidden_states']:
                query = torch.tensor(query).unsqueeze(0)
                cosine_similarities = torch.nn.functional.cosine_similarity(query, keys, dim=-1)
                top_indices = torch.argsort(cosine_similarities, dim=-1, descending=True)[:args.num_demonstrations].tolist()
                demonstrations = [train_dataset[i] for i in top_indices]
            elif args.method == 'topic_distribution':
                raise NotImplementedError
        
        prompt = jinja_template_manager.render(
            'icl_summarization_xsum.jinja',
            examples=demonstrations,
            target_document=example['document']
        )
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        stopping_criteria = tokenizer.convert_tokens_to_ids(['.', '!', '?'])


        class EndOfSentenceStoppingCriteria(StoppingCriteria):
            def __init__(self, end_token_ids, tokenizer):
                super().__init__()
                self.end_token_ids = end_token_ids
                self.tokenizer = tokenizer

            def __call__(self, input_ids, scores, **kwargs):
                last_token = input_ids[:, -1]
                return last_token in self.end_token_ids

        stop_tokens = [382]
        eos_token_id = tokenizer.convert_tokens_to_ids(['.', '!', '?', '.Ċ', '!Ċ', '?Ċ', ').Ċ', 'ĊĊ', '".Ċ', '".ĊĊ'])
        end_token_ids = torch.tensor(eos_token_id + stop_tokens).to(device)
        stopping_criteria = StoppingCriteriaList([EndOfSentenceStoppingCriteria(end_token_ids, tokenizer)])

        # Generate the summary with \n as end of sentence token
        with torch.no_grad():
            outputs = model.generate(
                inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_new_tokens=100,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

        summary = tokenizer.decode(
            outputs[0], skip_special_tokens=True
        )[len(prompt):].strip()

        if summary.split('\n')[0] != summary:
            summary = summary.split('\n')[0]

        predictions.append(summary)
        references.append(example['summary'])

    rouge = evaluate.load('rouge')
    results = rouge.compute(predictions=predictions,
                            references=references)
    with open(f'xsum_icl_results_{args.method}.json', 'w') as f:
        json.dump(results, f, indent=4)
    print(results)
            