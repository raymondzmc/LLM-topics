# Script to keep the top 2000 most frequent words in the test dataset and remove stopwords
# Optimized script to keep the top 2000 most frequent words in the test dataset and remove stopwords

import os
import json
from datasets import load_dataset
from collections import Counter
import pandas as pd
import multiprocessing
import spacy
from transformers import AutoTokenizer
import pdb

nlp = spacy.load("en_core_web_lg")

ds = load_dataset("fancyzhx/dbpedia_14")
test_set = ds["test"]
num_proc = multiprocessing.cpu_count()

model_name = "meta-llama/Llama-3.2-1B"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Modified preprocess function to separate words and positions
def preprocess(batch):
    processed_tokens = []
    original_tokens = []
    token_positions = []
    
    for text in batch['content']:
        doc = nlp(text)
        words, tokens, positions = [], [], []
        for token in doc:
            if token.is_alpha and not token.is_stop and not token.is_sent_start and len(token.text) > 2:
                start, end = token.idx, token.idx + len(token)
                context = text[:start]
                context_length = len(tokenizer.encode(context.strip()))
                contextual_token_length = len(tokenizer.encode(context + token.text)) - context_length
                token_length = len(tokenizer.encode(f" {token.text}", add_special_tokens=False))
                if token_length == 1 and contextual_token_length == 1:
                    assert text[start:end] == token.text, print(text[start:end], token)
                    words.append(token.text)
                    tokens.append(token.text)
                    positions.append((start, end))
                

        original_tokens.append(words)
        processed_tokens.append(tokens)
        token_positions.append(positions)
    return {'processed_tokens': processed_tokens, 'tokens': original_tokens, 'positions': token_positions}

# Apply preprocessing to the dataset
test_set = test_set.map(
    preprocess,
    batched=True,
    batch_size=1000,
    num_proc=num_proc,
)

# Build the frequency counter from all tokens
all_tokens = [word for tokens_list in test_set['processed_tokens'] for word in tokens_list]
counter = Counter(all_tokens)

# Get the top 2000 most frequent words
vocab = set(word for word, _ in counter.most_common(2000))
for word, freq in counter.most_common(2000)[:100]:
    print(word, freq)

# Modified create_training_examples function
def create_training_examples(batch):
    contexts = []
    next_tokens = []
    next_words = []
    labels = []
    titles = []
    for content, processed_tokens_list, original_tokens_list, positions_list, label, title in zip(
        batch['content'], batch['processed_tokens'], batch['tokens'], batch['positions'], batch['label'], batch['title']):
        # Iterate over tokens
        for token, word, (start, _) in zip(processed_tokens_list[1:], original_tokens_list[1:], positions_list[1:]):
            if token in vocab:  
                contexts.append(content[:start])
                next_tokens.append(token)
                next_words.append(word)
                labels.append(label)
                titles.append(title)
    return {
        'context': contexts,
        'next_token': next_tokens,
        'next_word': next_words,
        'label': labels,
        'title': titles
    }

# Apply the function to create the training set
training_set = test_set.map(
    create_training_examples,
    batched=True,
    batch_size=1000,
    remove_columns=test_set.column_names,
    num_proc=num_proc,
)

# Remove empty examples (if any)
training_set = training_set.filter(lambda example: len(example['context']) > 0)

# Save the most_common_words in the training set vocab
save_path = 'dbpedia_14'
training_set.save_to_disk(save_path)
with open(os.path.join(save_path, 'vocab.json'), 'w') as f:
    json.dump(list(vocab), f)

df = pd.DataFrame(training_set)
print(df.head())
