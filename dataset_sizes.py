import os
import spacy
import argparse
from tqdm import tqdm
from process_dataset_new import get_hf_dataset, get_local_dataset

try:
    nlp = spacy.load("en_core_web_lg")
except OSError:
    import subprocess
    print("Installing spacy model 'en_core_web_lg'...")
    subprocess.run(["python", "-m", "spacy", "download", "en_core_web_lg"], check=True)
    nlp = spacy.load("en_core_web_lg")


datasets = ['SetFit/20_newsgroups', 'fancyzhx/dbpedia_14', 'data/stackoverflow.csv']
context_keys = ['text', 'content', 'text']
for dataset_path, context_key in list(zip(datasets, context_keys))[1:]:
    print(f"Processing dataset: {dataset_path}...")
    args = argparse.Namespace(dataset=dataset_path, split='all', content_key=context_key)
    if os.path.exists(dataset_path):
        print("Loading local dataset...")
        dataset = get_local_dataset(args)
    else:
        print("Loading Hugging Face dataset...")
        dataset = get_hf_dataset(args)
    # Get average number of words in each document
    texts = (doc[context_key] for doc in dataset)
    print("Processing documents with spaCy (this might take a while)...")
    num_words = [len(doc) for doc in tqdm(nlp.pipe(texts, n_process=-1), total=len(dataset) if hasattr(dataset, '__len__') else None)]
    print(f"Average number of words in {dataset_path}: {sum(num_words) / len(num_words)}")
    import pdb; pdb.set_trace() # Consider removing or commenting out debugger for normal runs