# Script to keep the top 2000 most frequent words in the test dataset and remove stopwords
# Optimized script to keep the top 2000 most frequent words in the test dataset and remove stopwords

import re
from datasets import load_dataset
from collections import Counter
import nltk
from nltk.corpus import stopwords
import multiprocessing

# Download NLTK stopwords if not already downloaded
nltk.download('stopwords', quiet=True)

# Load English stopwords
stop_words = set(stopwords.words('english'))

# Load the dataset
ds = load_dataset("fancyzhx/dbpedia_14")
test = ds["test"]

# Number of CPU cores for multiprocessing
num_proc = multiprocessing.cpu_count()

# Function to preprocess text: tokenize, remove stopwords and non-alphabetic tokens
def preprocess(batch):
    processed_texts = []
    for text in batch['content']:
        # Tokenize the text using regex to avoid NLTK dependencies
        tokens = re.findall(r'\b[a-zA-Z]+\b', text.lower())
        # Filter tokens: remove stopwords
        tokens = [word for word in tokens if word not in stop_words]
        processed_texts.append(tokens)
    return {'tokens': processed_texts}

# Apply preprocessing to the dataset in batches with multiprocessing
test = test.map(
    preprocess,
    batched=True,
    batch_size=1000,
    num_proc=num_proc,
    remove_columns=test.column_names  # Remove original columns to save memory
)

# Flatten the list of tokens to build the frequency counter
all_tokens = [token for tokens_list in test['tokens'] for token in tokens_list]

# Build a frequency counter
counter = Counter(all_tokens)

# Get the top 2000 most frequent words
most_common_words = set(word for word, _ in counter.most_common(2000))

# Function to filter tokens to only keep the most common words
def filter_tokens(batch):
    filtered_texts = []
    for tokens in batch['tokens']:
        filtered_tokens = [word for word in tokens if word in most_common_words]
        filtered_texts.append(' '.join(filtered_tokens))
    return {'filtered_content': filtered_texts}

# Apply the filtering to the dataset in batches with multiprocessing
test = test.map(
    filter_tokens,
    batched=True,
    batch_size=1000,
    num_proc=num_proc,
    remove_columns=['tokens']  # Remove tokens column to save memory
)

import random

# Existing code up to getting 'most_common_words'
# ...

# Convert 'most_common_words' set to a list
words_list = list(most_common_words)

# Randomly select 140 words from the top 2000
sample_size = 14 * 10  # 14 lists * 10 words each
sampled_words = random.sample(words_list, sample_size)

# Shuffle the sampled words
random.shuffle(sampled_words)

# Split the sampled words into 14 lists of 10 words each
num_lists = 14
words_per_list = 10
word_lists = [sampled_words[i*words_per_list:(i+1)*words_per_list] for i in range(num_lists)]

for topic in word_lists: print(topic)

# Now `test` contains a new field 'filtered_content' with only the top 2000 words
import pdb; pdb.set_trace()