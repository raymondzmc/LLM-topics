import os
import pickle
import torch
import argparse
import json
import evaluate
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from datasets import load_from_disk, load_dataset, concatenate_datasets
from utils.io import load_processed_dataset
from utils.dataset import get_ctm_dataset
from utils.enums import ModelType, model_classes
from utils.distributions import combine_distributions
from transformers import (
    AutoModelForCausalLM, AutoTokenizer,
    StoppingCriteriaList, StoppingCriteria,
)
from contextualized_topic_models.datasets.dataset import CTMDataset

from llm import jinja_template_manager
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
            
        elif method == 'topic':
            pass
            
    return np.array(all_embeddings)


def get_topic_model(topic_model_path, vocab, args):
    model_checkpoint_path = os.path.join(topic_model_path, 'model.pt')
    model_checkpoint = torch.load(model_checkpoint_path, weights_only=True)
    contextual_size = model_checkpoint['inf_net.input_layer.weight'].shape[1]
    print(f"Contextual size: {contextual_size} for \"{args['model']}\" topic model")
    model_type = ModelType(args['model'])
    model_cls = model_classes[model_type]
    idx2token = {i: token for i, token in enumerate(vocab)}
    hidden_sizes = tuple([args['hidden_sizes'] for _ in range(args['num_hidden_layers'])])
    model = model_cls(bow_size=len(vocab),
                      contextual_size=contextual_size,
                      n_components=args['K'],
                      hidden_sizes=hidden_sizes,
                      activation=args['activation'],
                      label_size=args['label_size'])
    model.model.load_state_dict(model_checkpoint, strict=False)
    model.idx2token = idx2token
    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--dataset_name', type=str, default='EdinburghNLP/xsum')
    parser.add_argument('--model_name', type=str, default='meta-llama/Llama-3.2-1B')
    parser.add_argument('--use_flash_attention_2', action='store_true', default=True)
    parser.add_argument('--num_demonstrations', type=int, default=1)
    parser.add_argument('--method', type=str, default=None, choices=['zeroshot', 'mean_hidden_states', 'last_hidden_state', 'next_token_distribution', 'topic'])
    parser.add_argument('--topic_model_path', type=str, default=None)
    parser.add_argument('--train_dataset_path', type=str, default=None)
    parser.add_argument('--test_dataset_path', type=str, default=None)
    parser.add_argument('--input_key', type=str, default='document')
    parser.add_argument('--target_key', type=str, default='summary')
    parser.add_argument('--inference_prompt', type=str, default='icl_summarization_xsum.jinja')
    parser.add_argument('--label_mapping', type=str, default='icl_summarization_xsum.jinja')
    parser.add_argument('--evaluation_metric', type=str, default='rouge', choices=['rouge', 'accuracy'])
    parser.add_argument('--category_names', type=str, default=None)
    args = parser.parse_args()

    if args.output_dir is None:
        raise ValueError('Output directory is required')
    
    os.makedirs(args.output_dir, exist_ok=True)
    

    if args.method == 'topic':
        tm_args = json.load(open(os.path.join(args.topic_model_path, 'args.json'), 'r'))

        # Load vocab from topic model path
        test_vocab = json.load(open(os.path.join(args.test_dataset_path, 'vocab.json'), 'r'))
        train_vocab = json.load(open(os.path.join(args.train_dataset_path, 'vocab.json'), 'r'))
        assert test_vocab == train_vocab, "Vocab from topic model and train dataset do not match"
        vocab = test_vocab

    # Initialize model and tokenizer
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
        use_flash_attention_2=args.use_flash_attention_2,
    ).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    # Initialize topic model
    if args.method == 'topic':
        topic_model = get_topic_model(args.topic_model_path, vocab, tm_args)
    
    # Load huggingface datasets
    dataset = load_dataset(args.dataset_name)
    train_dataset = dataset['train']
    test_dataset = dataset['test']
    labels = list(set(test_dataset['label']))
    test_dataset = concatenate_datasets([test_dataset.filter(lambda ex: ex["label"] == x).select(range(100)) for x in labels])

    # Load label mapping for classification tasks
    if args.category_names is not None:
        with open(args.category_names, 'r') as f:
            categories = json.load(f)
        label_mapping = {label: i for i, label in enumerate(categories)}

    # Get retrieval mappings for train dataset
    key_file_path = os.path.join(args.output_dir, f'train_keys_{args.method}.pkl')
    if args.method in ['mean_hidden_states', 'last_hidden_state', 'next_token_distribution']:
        if os.path.exists(key_file_path):
            keys = pickle.load(open(key_file_path, 'rb'))
            print('Loaded retrieval mappings for train dataset')
            keys = torch.tensor(keys)
        else:
            print('Computing retrieval mappings for train dataset')
            keys = get_retrieval_mappings_batched(train_dataset[args.input_key], model, tokenizer, args.method)
            pickle.dump(keys, open(key_file_path, 'wb'))
            print(f'Saved retrieval mappings for train dataset to {key_file_path}')
            keys = torch.tensor(keys)

    elif args.method == 'topic':
        tokenized_test_dataset = load_from_disk(os.path.join(args.test_dataset_path, 'tokenized_dataset'))
        bow_test_dataset = open(os.path.join(args.test_dataset_path, 'bow_dataset.txt'), 'r').readlines()
        if os.path.exists(key_file_path):
            keys = pickle.load(open(key_file_path, 'rb'))
            print('Loaded train keys')
        else:
            # Load processed train and test datasets
            if tm_args['model'] == 'generative':
                processed_dataset_path = os.path.join(args.train_dataset_path, 'processed_dataset')
                num_chuncks = [len(os.listdir(os.path.join(processed_dataset_path, subdir))) for subdir in os.listdir(processed_dataset_path)]
                assert len(set(num_chuncks)) == 1, "All subdirectories should have the same number of chunks"
                thetas = []
                for chunk_idx in tqdm(range(num_chuncks[0])):
                    processed_train_dataset = load_processed_dataset(processed_dataset_path, tm_args['hidden_state_layer'], chunk_idx, verbose=False)
                    ctm_train_dataset = get_ctm_dataset(processed_train_dataset, vocab, tm_args['model'], tm_args['embedding_type'])
                    thetas.append(topic_model.get_doc_topic_distribution(ctm_train_dataset))
                thetas = np.concatenate(thetas, axis=0)
                inference_train_dataset = load_from_disk(os.path.join(args.train_dataset_path, 'inference_dataset'))
                assert len(thetas) == len(inference_train_dataset), "Thetas and tokenized train dataset should have the same number of documents"

                # Group thetas with same id
                mappings = defaultdict(list)
                for i, _id in enumerate(tqdm(inference_train_dataset['id'])):
                    mappings[_id].append(thetas[i])

                # For each example in train dataset, get thetas at each inference step
                unprocessed_theta_path = os.path.join(args.output_dir, f'train_thetas_unprocessed_{args.method}.pkl')
                keys = []
                for example in tqdm(train_dataset):
                    keys.append(mappings[example['id']])
                pickle.dump(keys, open(unprocessed_theta_path, 'wb'))

                # Combine thetas for each document
                num_topics = thetas.shape[1]
                keys = [combine_distributions(dist) for dist in keys]
                keys = np.stack([k if k is not None else np.zeros(num_topics) for k in keys])
                pickle.dump(keys, open(key_file_path, 'wb'))
                print(f'Saved train keys to {key_file_path}')

            else:
                tokenized_train_dataset = load_from_disk(os.path.join(args.train_dataset_path, 'tokenized_dataset'))
                ctm_train_dataset = get_ctm_dataset(tokenized_train_dataset, vocab, tm_args['model'], tm_args['embedding_type'])
                thetas = topic_model.get_doc_topic_distribution(ctm_train_dataset)
                keys = thetas
                pickle.dump(keys, open(key_file_path, 'wb'))
                print(f'Saved train keys to {key_file_path}')

    predictions = []
    targets = []
    
    for idx, example in enumerate(tqdm(test_dataset)):
        demonstrations = []
        if args.method != 'zeroshot':
            query = get_retrieval_mappings(example[args.input_key], model, tokenizer, args.method)
            if args.method == 'next_token_distribution':
                query = torch.tensor(query).unsqueeze(0)
                kl_divergences = torch.nn.functional.kl_div(query, torch.tensor(keys), reduction='none')
                top_indices = torch.argsort(kl_divergences, dim=-1, descending=True)[:args.num_demonstrations]
                demonstrations = [
                    {'document': train_dataset[i][args.input_key],
                    'target': train_dataset[i][args.target_key] if args.category_names is None else categories[train_dataset[i][args.target_key]]}
                    for i in top_indices
                ]
            elif args.method in ['last_hidden_state', 'mean_hidden_states']:
                query = torch.tensor(query).unsqueeze(0)
                cosine_similarities = torch.nn.functional.cosine_similarity(query, keys, dim=-1)
                top_indices = torch.argsort(cosine_similarities, dim=-1, descending=True)[:args.num_demonstrations].tolist()
                demonstrations = [
                    {'document': train_dataset[i][args.input_key],
                    'target': train_dataset[i][args.target_key] if args.category_names is None else categories[train_dataset[i][args.target_key]]}
                    for i in top_indices
                ]
            elif args.method == 'topic':
                tokenized_example = tokenized_test_dataset[idx]
                embeddings = []
                for word, (start, _) in zip(tokenized_example['words'], tokenized_example['offsets']):
                    if word in vocab:
                        input_ids = tokenizer.encode(tokenized_example['content'][start:], return_tensors='pt').to(device)
                        with torch.no_grad():
                            outputs = model(
                                input_ids,
                                use_cache=True,
                                output_hidden_states=True
                            )
                        embeddings.append(outputs.hidden_states[tm_args['hidden_state_layer']][0, -1].cpu().tolist())
                if len(embeddings) == 0:
                    # Use random indices
                    top_indices = np.random.randint(0, len(train_dataset), size=args.num_demonstrations)
                    demonstrations = [train_dataset[int(i)] for i in top_indices]
                else:
                    embeddings = np.stack(embeddings)
                    targets = np.zeros((len(embeddings), len(vocab)))   # Use dummy targets for inference
                    dataset = CTMDataset(X_contextual=embeddings, X_bow=targets, idx2token=topic_model.idx2token)
                    query = torch.tensor(combine_distributions(topic_model.get_doc_topic_distribution(dataset))).unsqueeze(0)
                    kl_div = torch.nn.functional.kl_div(
                        query.log(),
                        torch.tensor(keys),
                        reduction='none'
                    ).sum(dim=-1)
                    top_indices = torch.argsort(kl_div, descending=False)[:args.num_demonstrations].tolist()
                    demonstrations = [
                        {'document': train_dataset[i][args.input_key],
                        'target': train_dataset[i][args.target_key] if args.category_names is None else categories[train_dataset[i][args.target_key]]}
                        for i in top_indices
                    ]
            else:
                raise NotImplementedError(f"Method {args.method} not implemented!")
        
        prompt = jinja_template_manager.render(
            args.inference_prompt,
            examples=demonstrations,
            target_document=example[args.input_key]
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
        eos_token_id = tokenizer.convert_tokens_to_ids(['.', '!', '?', '.Ċ', '!Ċ', '?Ċ', ').Ċ', 'ĊĊ', '".Ċ', '".ĊĊ', 'ĊĊĊ'])
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
                stopping_criteria=stopping_criteria,
            )
        
        prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)[len(prompt):].strip()
        if prediction.split('\n')[0] != prediction:
            prediction = prediction.split('\n')[0]

        predictions.append(prediction)
        targets.append(example[args.target_key])

    if args.evaluation_metric == 'rouge':
        rouge = evaluate.load('rouge')
        results = rouge.compute(predictions=predictions,
                                references=targets)
        with open(os.path.join(args.output_dir, f'results_{args.method}.json'), 'w') as f:
            json.dump(results, f, indent=4)
        print(results)
    elif args.evaluation_metric == 'accuracy':
        
        
        predicted_labels_index = [label_mapping.get(pred, -1) for pred in predictions]
        accuracy = evaluate.load('accuracy')
        results = accuracy.compute(predictions=predicted_labels_index,
                                   references=targets)
        with open(os.path.join(args.output_dir, f'results_{args.method}.json'), 'w') as f:
            json.dump(results, f, indent=4)

    # Save predictions and targets
    with open(os.path.join(args.output_dir, f'predictions_{args.method}.json'), 'w') as f:
        json.dump(predictions, f, indent=4)
    with open(os.path.join(args.output_dir, f'targets_{args.method}.json'), 'w') as f:
        json.dump(targets, f, indent=4)
    
            