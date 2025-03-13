import os
import json
import argparse
import multiprocessing
from collections import Counter
import torch
import spacy
from tqdm import tqdm
from huggingface_hub import login
from datasets import (
    load_dataset,
    load_from_disk,
    get_dataset_config_names,
    concatenate_datasets,
)
from transformers import AutoTokenizer, AutoModelForCausalLM
from llm import compute_word_log_prob, jinja_template_manager
from settings import settings
import pdb

try:
    nlp = spacy.load("en_core_web_lg")
except OSError:
    import subprocess
    print("Installing spacy model 'en_core_web_lg'...")
    subprocess.run(["python", "-m", "spacy", "download", "en_core_web_lg"], check=True)
    nlp = spacy.load("en_core_web_lg")


def get_hf_dataset(args):
    configs = get_dataset_config_names(args.dataset, trust_remote_code=True)
    if 'default' in configs:
        configs = ['default']

    if len(configs) == 0:
        raise ValueError(f"No dataset configs found for {args.dataset}")

    datasets = []
    for cfg in configs:
        dataset = load_dataset(args.dataset, cfg, trust_remote_code=True)
        if args.split is None or args.split == 'all':
            all_splits = list(dataset.keys())
        else:
            all_splits = [args.split]

        for split in all_splits:
            datasets.append(dataset[split])

    dataset = concatenate_datasets(datasets)
    return dataset


def get_local_dataset(args):
    if not os.path.basename(args.dataset).endswith('.csv'):
        raise ValueError(f"Dataset {args.dataset} is not a CSV file")
    dataset = load_dataset("csv", data_files=args.dataset, delimiter='\t')
    dataset = dataset['train']
    return dataset



def tokenize_dataset(batch, tokenizer, content_key: str, single_token_only: bool = False, vocab: list[str] | None = None):
    words = []
    token_ids = []
    offsets = []
    
    for text in batch[content_key]:
        doc = nlp(text)
        # encoding = tokenizer(text, return_offsets_mapping=True, return_attention_mask=False)
        word_list, token_list, word_offsets = [], [], []
        for word in doc:
            # Start of Selection
            if (
                (vocab is not None and word.text in vocab)
                or (
                    vocab is None
                    and word.is_alpha
                    and not word.is_stop
                    and not word.is_sent_start
                    and len(word.text) > 2
                    and word.is_lower
                    # and word.pos_ in ['NOUN']
                )
            ):
                start, end = word.idx, word.idx + len(word)

                # Get all token offsets that overlap with the word span
                # word_token_ids = [encoding.input_ids[i] for i, offset in enumerate(encoding.offset_mapping)
                #                   if offset[0] < end and offset[1] > start]

                word_token_ids = tokenizer.encode(f" {word.text}", add_special_tokens=False)
                if len(word_token_ids) == 0:
                    raise ValueError(f"Word {word.text} not found in tokenizer")
                
                if not single_token_only or len(word_token_ids) == 1:
                    word_list.append(word.text)
                    token_list.append(word_token_ids)
                    word_offsets.append((start, end))

        words.append(word_list)
        token_ids.append(token_list)
        offsets.append(word_offsets)
    return {'words': words, 'token_ids': token_ids, 'offsets': offsets, 'content': batch[content_key]}


def create_bow_dataset(batch, vocab, content_key):
    bow_lines = []
    for words in batch["words"]:
        # Keep only words in vocab (including duplicates if they appear multiple times)
        filtered_words = [w for w in words if w in vocab]
        bow_lines.append(" ".join(filtered_words))

    return {"bow_line": bow_lines}

def save_processed_examples(processed_examples: list[dict], data_path: str, num_saved: int, layers: list[int] = None):
    keys = processed_examples[0].keys()

    # Build a dict of lists from the list of dicts
    data_dict = {key: [ex[key] for ex in processed_examples] for key in keys}
    save_path = os.path.join(data_path, "processed_dataset")
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)

    for key in keys:
        values = data_dict.get(key, [])
        key_dir = os.path.join(save_path, key)
        if not os.path.exists(key_dir):
            os.makedirs(key_dir, exist_ok=True)
        
        if layers is not None and key == 'input_embeddings':
            if len(layers) == 1:
                layer_dir = os.path.join(key_dir, f"{layers[0]}")
                os.makedirs(layer_dir, exist_ok=True)
                torch.save(values, os.path.join(layer_dir, f"{num_saved}.pt"))
            else:
                for i, layer in enumerate(layers):
                    layer_dir = os.path.join(key_dir, f"{layer}")
                    os.makedirs(layer_dir, exist_ok=True)
                    layer_embeddings = [embedding[i] for embedding in values]
                    torch.save(layer_embeddings, os.path.join(layer_dir, f"{num_saved}.pt"))
        else:
            torch.save(values, os.path.join(key_dir, f"{num_saved}.pt"))
    return save_path


def main(args):

    login(settings.hf_token)

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
        # use_flash_attention_2=args.use_flash_attention_2,
    ).eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenized_dataset_path = os.path.join(args.data_path, 'tokenized_dataset')
    vocab_path = os.path.join(args.data_path, 'vocab.json')
    num_proc = multiprocessing.cpu_count()

    if args.vocab_path:
        vocab = json.load(open(args.vocab_path, 'r'))
    else:
        vocab = None

    if os.path.exists(tokenized_dataset_path) and os.path.exists(vocab_path):
        print(f"Loading preprocessed dataset from {tokenized_dataset_path}")
        dataset = load_from_disk(tokenized_dataset_path)
        with open(vocab_path, 'r') as f:
            vocab = json.load(f)
    else:
        if os.path.exists(args.dataset):
            dataset = get_local_dataset(args)
        else:
            dataset = get_hf_dataset(args)
        dataset = dataset.map(
            lambda x: tokenize_dataset(
                x,
                tokenizer,
                args.content_key,
                single_token_only=args.single_token_only,
                vocab=vocab
            ),
            batched=True,
            batch_size=1000,
            num_proc=1
        )
        dataset.save_to_disk(tokenized_dataset_path)
        # If there are no specified vocab, create based on word frequency
        if vocab is None:
            all_tokens = [word for tokens_list in dataset['words'] for word in tokens_list]
            counter = Counter(all_tokens)
            # Visualize top 25 most frequent words
            for word, freq in counter.most_common(25):
                print(word, freq)
            vocab = list(set(word for word, _ in counter.most_common(args.vocab_size)))

        with open(vocab_path, 'w') as f:
            json.dump(vocab, f)

    if args.bow_dataset:
        dataset = dataset.map(
            lambda x: create_bow_dataset(x, vocab, args.content_key),
            batched=True,
            batch_size=1000,
            num_proc=num_proc
        )
        # Save the resulting BoW lines to a text file
        bow_output_path = os.path.join(args.data_path, "bow_dataset.txt")
        print(f"Saving BoW dataset to {bow_output_path} ...")
        with open(bow_output_path, "w", encoding="utf-8") as f:
            for bow_line in dataset["bow_line"]:
                f.write(bow_line + "\n")
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    processed_examples = []
    vocab_token_ids = [tokenizer.encode(f" {word}", add_special_tokens=False) for word in vocab]
    vocab_token_prefix = [ids[0] for ids in vocab_token_ids]
    token_lengths = [len(token_ids) for token_ids in vocab_token_ids]
    single_token_word_idx = [i for i, token_len in enumerate(token_lengths) if token_len == 1]
    multi_token_word_idx = [i for i, token_len in enumerate(token_lengths) if token_len > 1]

    if args.hidden_state_layer is None:
        hidden_state_layers = list(range(model.config.num_hidden_layers + 1))
    else:
        hidden_state_layers = [args.hidden_state_layer]

    if (len(single_token_word_idx) + len(multi_token_word_idx)) != len(vocab):
        raise ValueError(
            "The total number of single-token and multi-token words does not match the vocabulary size. Please check the vocabulary and tokenization process.")

    if args.word_prob_method == 'prefix' and len(vocab_token_prefix) > len(set(vocab_token_prefix)):
        print(
            f"Warning: Vocab token prefix is not unique, {len(vocab_token_prefix) - len(set(vocab_token_prefix))} duplicates.",
            "Consider using 'product' method to compute word probabilities."
        )
    
    if args.single_token_only and len(multi_token_word_idx) > 0:
        raise ValueError("Single token only is set to True, but there are multi-token words in the vocabulary.")
    
    num_saved = 0
    for idx, example in enumerate(tqdm(dataset)):
        context = jinja_template_manager.render('document_topic.jinja', document=example[args.content_key])
        context_input_ids = tokenizer.encode(context.rstrip(), return_tensors='pt').to(device)
        context_length = context_input_ids.shape[1]

        # Compute the probabilities for single token words
        single_token_probs = {}
        with torch.no_grad():
            outputs = model(context_input_ids, use_cache=True, output_hidden_states=True)
        logits = outputs.logits
        next_token_logits = logits[:, -1, :]
        next_word = tokenizer.decode(torch.argmax(next_token_logits, dim=-1))

        # Save the hidden states from the specified layer
        if args.hidden_state_layer is not None:
            if args.embedding_method == 'mean':
                embeddings = outputs.hidden_states[args.hidden_state_layer][0].mean(dim=0).cpu().tolist()
            elif args.embedding_method == 'last':
                embeddings = outputs.hidden_states[args.hidden_state_layer][0, -1].cpu().tolist()
            else:
                raise ValueError(f"Unsupported embedding method: {args.embedding_method}")
        else:
            # By default, save the hidden states from all layers
            if args.embedding_method == 'mean':
                embeddings = [h[0].mean(dim=0).cpu().tolist() for h in outputs.hidden_states]
            elif args.embedding_method == 'last':
                embeddings = [h[0, -1].cpu().tolist() for h in outputs.hidden_states]
            else:
                raise ValueError(f"Unsupported embedding method: {args.embedding_method}")
        
        # Compute next word probs for words in the vocab
        if args.single_token_only and len(single_token_word_idx) == len(vocab):
            vocab_logits = next_token_logits[0, vocab_token_prefix]
            all_probs = {vocab[i]: vocab_logits[i].item() for i in range(len(vocab))}
            next_word_probs = [all_probs[word] for word in vocab]
        else:
            single_token_probs = {}
            for i in single_token_word_idx:
                single_token_probs[vocab[i]] = next_token_probs[vocab_token_prefix[i]].item()

            if args.temperature is not None:
                next_token_probs = torch.softmax(next_token_logits / args.temperature, dim=-1).squeeze(0)
            else:
                next_token_probs = torch.softmax(next_token_logits, dim=-1).squeeze(0)

            multi_token_probs = {}
            if not args.single_token_only and len(multi_token_word_idx) > 0:
                if args.word_prob_method == 'product':
                    multi_token_probs = compute_word_log_prob(
                        model, tokenizer, device, multi_token_word_idx, context_input_ids,
                        vocab_token_ids, vocab, args.batch_size, context_length)
                elif args.word_prob_method == 'prefix':
                    prefix_groups = {}
                    for i in multi_token_word_idx:
                        prefix = vocab_token_prefix[i]
                        prefix_groups.setdefault(prefix, []).append(i)

                    # Split the probability among words sharing the same prefix
                    for prefix, indices in prefix_groups.items():
                        total_prefix_prob = next_token_probs[prefix].item()
                        split_prob = total_prefix_prob / len(indices)
                        for i in indices:
                            multi_token_probs[vocab[i]] = split_prob
                else:
                    raise ValueError(f"Invalid word probability method: {args.word_prob_method}")

            all_probs = {**single_token_probs, **multi_token_probs}
            total_prob = sum(all_probs.values())
            normalized_next_word_probs = {word: prob / total_prob for word, prob in all_probs.items()}
            next_word_probs = [normalized_next_word_probs[word] for word in vocab]

        processed_examples.append({
            'id': example['id'] if 'id' in example else idx,
            'context': context,
            'next_word': next_word,
            'next_word_probs': next_word_probs,
            'input_embeddings': embeddings,
        })

        # Save every 2000 examples to reduce memory usage
        CHUNK_SIZE = 2000
        if len(processed_examples) >= CHUNK_SIZE:
            save_path = save_processed_examples(processed_examples, args.data_path, num_saved, hidden_state_layers)
            print(f"Saved {len(processed_examples)} processed examples at: {save_path}")
            num_saved += CHUNK_SIZE
            processed_examples = []
         
    if len(processed_examples) > 0:
        save_path = save_processed_examples(processed_examples, args.data_path, num_saved, hidden_state_layers)
        print(f"Saved {len(processed_examples)} processed examples at: {save_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='fancyzhx/dbpedia_14')
    parser.add_argument('--vocab_path', type=str, default=None)
    parser.add_argument('--content_key', type=str, default='content')
    parser.add_argument('--id_key', type=str, default='id')
    parser.add_argument('--split', type=str, default='all')
    parser.add_argument('--vocab_size', type=int, default=2000)
    parser.add_argument('--model_name', type=str, default='meta-llama/Llama-3.2-1B')
    parser.add_argument('--use_flash_attention_2', action='store_true')
    parser.add_argument('--target_method', type=str, choices=['generative', 'summary'])
    parser.add_argument('--single_token_only', action='store_true', help="Whether to only include single token words.")
    parser.add_argument('--word_prob_method', type=str, default='prefix', choices=['prefix', 'product'])
    parser.add_argument('--hidden_state_layer', type=int, default=None, help="The hidden state layer to save (default: all)")
    parser.add_argument('--embedding_method', type=str, default='last', choices=['mean', 'last'], help="The method to compute the embedding of the context.")
    parser.add_argument('--examples_per_vocab', type=int, default=None, help="Number of examples to sample per vocab word.")
    parser.add_argument('--bow_dataset', action='store_true', help="Whether to compute the bag-of-words dataset.")
    parser.add_argument('--data_path', type=str, default='data')
    parser.add_argument('--dir_name', type=str, default=None)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--subset', type=int, default=None)
    parser.add_argument('--max_tokens', type=int, default=None)
    args = parser.parse_args()
    print(f'Processing dataset \"{args.dataset}\"')

    if args.dir_name is None:
        data_dir_name = f"{os.path.basename(args.dataset)}_{os.path.basename(args.model_name)}_{args.vocab_size}"
    else:
        data_dir_name = args.dir_name

    print(f'Processing dataset \"{args.dataset}\"')

    data_dir_name = f"{os.path.basename(args.dataset).split('.')[0]}_{os.path.basename(args.model_name)}_vocab_{args.vocab_size}_{args.embedding_method}"
    if args.subset is not None:
        data_dir_name += f"_subset_{args.subset}"
    args.data_path = os.path.join(args.data_path, data_dir_name)
    if not os.path.exists(args.data_path):
        os.makedirs(args.data_path, exist_ok=True)

    main(args)
