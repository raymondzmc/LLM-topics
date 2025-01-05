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
    Dataset,
    load_dataset,
    load_from_disk,
    get_dataset_config_names,
    concatenate_datasets,
)
from transformers import AutoTokenizer, AutoModelForCausalLM
import pdb

try:
    nlp = spacy.load("en_core_web_lg")
except OSError:
    import subprocess
    print("Installing spacy model 'en_core_web_lg'...")
    subprocess.run(["python", "-m", "spacy", "download", "en_core_web_lg"], check=True)
    nlp = spacy.load("en_core_web_lg")


def get_dataset(args):
    configs = get_dataset_config_names(args.dataset, trust_remote_code=True)
    if len(configs) > 0:
        datasets = []
        if args.split is None:
            raise ValueError("Split must be specified when combining multiple configs")
        for cfg in configs:
            dataset = load_dataset(args.dataset, cfg, trust_remote_code=True)
            datasets.append(dataset[args.split])
        dataset = concatenate_datasets(datasets)
    else:
        dataset = load_dataset(args.dataset, trust_remote_code=True)
        dataset = dataset[args.split] if args.split else dataset
    return dataset


def tokenize_dataset(batch, tokenizer, content_key: str, single_token_only: bool = False, vocab: list[str] = None):
    words = []
    token_ids = []
    offsets = []
    
    for text in batch[content_key]:
        text = text.split('\n\n')[1]
        doc = nlp(text)
        encoding = tokenizer(text, return_offsets_mapping=True, return_attention_mask=False)
        word_list, token_list, word_offsets = [], [], []
        for word in doc:
            if (len(vocab) and (word.text in vocab)) or (vocab is None and word.is_alpha and not word.is_stop and not word.is_sent_start and len(word.text) > 2):
                start, end = word.idx, word.idx + len(word)

                # Get all token offsets that overlap with the word span
                word_token_ids = [encoding.input_ids[i] for i, offset in enumerate(encoding.offset_mapping)
                                  if offset[0] < end and offset[1] > start]
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


def create_inference_examples(batch, vocab, content_key):
    special_cols = {content_key, "words", "token_ids", "offsets"}
    other_cols = [col for col in batch.keys() if col not in special_cols]

    contexts = []
    next_tokens = []
    next_words = []
    repeated_cols = {col: [] for col in other_cols}
    for i, (content, words, token_ids, offsets) in enumerate(zip(
        batch['content'],
        batch['words'],
        batch['token_ids'],
        batch['offsets']
    )):
        for word, word_token_ids, (start, _) in zip(words, token_ids, offsets):
            if word in vocab:
                contexts.append(content[:start])
                next_tokens.append(word_token_ids)
                next_words.append(word)
                for col in repeated_cols.keys():
                    repeated_cols[col].append(batch[col][i])
    return {
        'context': contexts,
        'next_word_token_ids': next_tokens,
        'next_word': next_words,
        **repeated_cols,
    }


def main(args):
    token = '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'
    login(token)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
        use_flash_attention_2=args.use_flash_attention_2,
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
        dataset = get_dataset(args)
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
            num_proc=num_proc
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

    # Create LM inference examples when the next word is in the vocab
    inference_dataset = dataset.map(
        lambda x: create_inference_examples(x, vocab, args.content_key),
        batched=True,
        batch_size=200,
        num_proc=4,
        remove_columns=dataset.column_names,
    )

    if args.examples_per_vocab and args.examples_per_vocab > 0:
        df = inference_dataset.to_pandas()
        grouped = df.groupby('next_word', group_keys=False)
        df_sampled = grouped.apply(
            lambda x: x.sample(
                n=min(len(x), args.examples_per_vocab),
                random_state=42
            ),
            include_groups=True
        ).reset_index(drop=True)
        sampled_inference_dataset = Dataset.from_pandas(df_sampled, preserve_index=False)
    else:
        sampled_inference_dataset = inference_dataset

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    processed_examples = []
    for example in tqdm(sampled_inference_dataset):
        context = example['context']
        next_word = example['next_word']
        context_input_ids = tokenizer.encode(context.rstrip(), return_tensors='pt').to(device)
        context_length = context_input_ids.shape[1]
        word_token_ids = [tokenizer.encode(context + word)[context_length:] for word in vocab]
        token_lengths = [len(token_ids) for token_ids in word_token_ids]
        single_token_word_idx = [i for i, token_len in enumerate(token_lengths) if token_len == 1]
        multi_token_word_idx = [i for i, token_len in enumerate(token_lengths) if token_len > 1]
        if (len(single_token_word_idx) + len(multi_token_word_idx)) != len(vocab):
            continue

        # Skip multi-token words if single_token_only is True
        if args.single_token_only and len(multi_token_word_idx) > 0:
            continue

        # Compute the probabilities for single token words
        single_token_probs = {}
        with torch.no_grad():
            outputs = model(
                context_input_ids,
                use_cache=True,
                output_hidden_states=True
            )
        logits = outputs.logits
        next_token_logits = logits[:, -1, :]
        next_token_probs = torch.softmax(next_token_logits, dim=-1).squeeze(0)
        embeddings = [h[0, -1].cpu().tolist() for h in outputs.hidden_states]

        single_token_probs = {}
        for i in single_token_word_idx:
            prob = next_token_probs[word_token_ids[i][0]].item()
            single_token_probs[vocab[i]] = prob

        multi_token_probs = {}
        if len(multi_token_word_idx) > 0:
            for start in range(0, len(multi_token_word_idx), args.batch_size):
                end = start + args.batch_size
                chunk_indices = multi_token_word_idx[start:end]

                chunk_input_ids = []
                for idx in chunk_indices:
                    combined_ids = context_input_ids[0].tolist() + word_token_ids[idx]
                    chunk_input_ids.append(combined_ids)
                padded_batch = tokenizer.pad(
                    [{"input_ids": seq} for seq in chunk_input_ids],
                    return_tensors='pt'
                ).to(device)

                with torch.no_grad():
                    outputs = model(**padded_batch, use_cache=True, output_hidden_states=True)
                all_logits = outputs.logits  # [batch_size, seq_len, vocab_size]

                # Calculate probabilities for each sequence in this chunk
                for batch_idx, idx in enumerate(chunk_indices):
                    token_ids_for_word = word_token_ids[idx]
                    log_p = 0.0
                    for k, subtoken_id in enumerate(token_ids_for_word):
                        pred_pos = context_length + k - 1
                        if pred_pos < 0:
                            raise ValueError("The subtoken index is out of range. Check offsets.")

                        token_logits = all_logits[batch_idx, pred_pos, :]
                        token_prob = torch.softmax(token_logits, dim=-1)[subtoken_id]
                        log_p += torch.log(token_prob).item()

                    # Convert log-prob to probability
                    word_probability = float(torch.exp(torch.tensor(log_p)))
                    multi_token_probs[vocab[idx]] = word_probability
        all_probs = {**single_token_probs, **multi_token_probs}
        total_prob = sum(all_probs.values())
        normalized_next_word_probs = {word: prob / total_prob for word, prob in all_probs.items()}
        next_word_probs = [normalized_next_word_probs[word] for word in vocab]

    processed_examples.append({
        'context': context,
        'next_word': next_word,
        'next_word_probs': next_word_probs,
        'input_embeddings': embeddings,
    })
    # Build a dict of lists from the list of dicts
    keys = processed_examples[0].keys()
    data_dict = {key: [ex[key] for ex in processed_examples] for key in keys}

    # Create the Dataset object from our data dictionary
    processed_dataset = Dataset.from_dict(data_dict)

    # Save the dataset to disk
    save_path = os.path.join(args.data_path, "processed_dataset")
    processed_dataset.save_to_disk(save_path)
    print(f"Processed examples saved to disk at: {save_path}")



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='fancyzhx/dbpedia_14')
    parser.add_argument('--vocab_path', type=str, default=None)
    parser.add_argument('--content_key', type=str, default='content')
    parser.add_argument('--split', type=str, default='test')
    parser.add_argument('--vocab_size', type=int, default=2000)
    parser.add_argument('--model_name', type=str, default='meta-llama/Llama-3.2-1B')
    parser.add_argument('--use_flash_attention_2', action='store_true')
    parser.add_argument('--single_token_only', action='store_true', help="Whether to only include single token words.")
    parser.add_argument('--examples_per_vocab', type=int, default=None, help="Number of examples to sample per vocab word.")
    parser.add_argument('--data_path', type=str, default='data')
    parser.add_argument('--batch_size', type=int, default=32)
    args = parser.parse_args()
    print(f'Processing dataset {args.dataset}')

    data_dir_name = f"{os.path.basename(args.dataset)}_{os.path.basename(args.model_name)}_{args.vocab_size}"
    args.data_path = os.path.join(args.data_path, data_dir_name, args.split)
    if not os.path.exists(args.data_path):
        os.makedirs(args.data_path, exist_ok=True)

    main(args)
