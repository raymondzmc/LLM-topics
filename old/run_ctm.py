from contextualized_topic_models.models.ctm import ZeroShotTM, CombinedTM
from contextualized_topic_models.utils.data_preparation import TopicModelDataPreparation
from contextualized_topic_models.utils.data_preparation import bert_embeddings_from_list
from contextualized_topic_models.datasets.dataset import CTMDataset
import json
from datasets import load_dataset
import pdb
import torch
import multiprocessing
import os
from scipy.sparse import lil_matrix
from tqdm import tqdm
os.environ["TOKENIZERS_PARALLELISM"] = "false"

model = "zeroshottm"
K = 10
dataset_path = 'dbpedia_14/dbpedia_14_ctm_dataset.pt'
ds = load_dataset("fancyzhx/dbpedia_14")['test']
if os.path.exists(dataset_path):
    dataset = torch.load(dataset_path)
else:
    vocab = json.load(open('dbpedia_14/vocab.json', 'r'))

    idx2token = {i: token for i, token in enumerate(vocab)}
    num_proc = multiprocessing.cpu_count()

    X_bows = lil_matrix((len(ds), len(vocab)))
    text_for_contextual = []
    for i, example in enumerate(tqdm(ds)):
        text = example['content']
        for j, token in idx2token.items():
            if token in text:
                X_bows[i, j] = text.count(token)
        text_for_contextual.append(text)

    X_bows = X_bows.tocsr()
    X_contextual = bert_embeddings_from_list(
        text_for_contextual,
        sbert_model_to_load='paraphrase-multilingual-mpnet-base-v2',
        max_seq_length=128,
    )
    dataset = CTMDataset(X_contextual=X_contextual, X_bow=X_bows, idx2token=idx2token,)
    torch.save(dataset, 'dbpedia_14/dbpedia_14_ctm_dataset.pt')


topic_model_path = f'dbpedia_14/{model}_{K}_topics.pt'
if os.path.exists(topic_model_path):
    topics = torch.load(topic_model_path)
else:
    if model == "combinedtm":
        ctm = CombinedTM(bow_size=len(dataset.idx2token), contextual_size=768, n_components=K, num_epochs=20)
    else:
        ctm = ZeroShotTM(bow_size=len(dataset.idx2token), contextual_size=768, n_components=K, num_epochs=20)
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
