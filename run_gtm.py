from contextualized_topic_models.models.ctm import GenerativeTM
from contextualized_topic_models.utils.data_preparation import TopicModelDataPreparation
from contextualized_topic_models.utils.data_preparation import bert_embeddings_from_list
from contextualized_topic_models.datasets.dataset import CTMDataset
import json
from datasets import load_dataset
import pdb
import numpy as np
import torch
import multiprocessing
import os
from scipy.sparse import lil_matrix
from tqdm import tqdm
os.environ["TOKENIZERS_PARALLELISM"] = "false"

K = 10
dataset_path = 'dbpedia_14/dbpedia_14_ctm_dataset.pt'
processed_dataset_path = 'dbpedia_14/processed_dataset.pt'
processed_dataset = torch.load(processed_dataset_path)

vocab = json.load(open('dbpedia_14/vocab.json', 'r'))

idx2token = {i: token for i, token in enumerate(vocab)}
num_proc = multiprocessing.cpu_count()

X_bows = np.stack([example['next_token_probs'] for example in processed_dataset])
X_contextual = np.stack([example['input_embeddings'] for example in processed_dataset])
dataset = CTMDataset(X_contextual=X_contextual, X_bow=X_bows, idx2token=idx2token,)
ds = load_dataset("fancyzhx/dbpedia_14")['test']

topic_model_path = f'dbpedia_14/generativetm_{K}_topics.pt'
if os.path.exists(topic_model_path):
    topics = torch.load(topic_model_path)
else:
    ctm = GenerativeTM(bow_size=len(dataset.idx2token), contextual_size=2048, n_components=K, hidden_sizes=(200, 200), num_epochs=20)
    ctm.fit(dataset)
    torch.save(ctm.get_topics(25), topic_model_path)
    topics = ctm.get_topics(25)

from contextualized_topic_models.evaluation.measures import TopicDiversity, CoherenceNPMI, CoherenceWordEmbeddings, InvertedRBO

topics = list(topics.values())
td = TopicDiversity(topics)
print("Topic Diversity:", td.score(topk=10))

texts = [example['content'].split() for example in ds]
npmi = CoherenceNPMI(topics, texts)
print("NPMI:", npmi.score(topk=10))

we = CoherenceWordEmbeddings(topics)
print("Word Embeddings:", we.score(topk=10))

irbo = InvertedRBO(topics)
print("Inverted RBO:", irbo.score(topk=10))
