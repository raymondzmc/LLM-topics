from collections import Counter
from datasets import Dataset
from contextualized_topic_models.utils.data_preparation import bert_embeddings_from_list
from scipy.sparse import lil_matrix
from tqdm import tqdm
import numpy as np
from utils.enums import ModelType, EmbeddingType
from contextualized_topic_models.datasets.dataset import CTMDataset
from typing import Any


def get_bows(words, vocab):
    bows = lil_matrix((len(words), len(vocab)), dtype=int)
    vocab_to_idx = {token: idx for idx, token in enumerate(vocab)}
    for i, words in enumerate(tqdm(words)):
        token_counts = Counter(words)
        for token, count in token_counts.items():
            if token in vocab_to_idx:
                j = vocab_to_idx[token]
                bows[i, j] = count
    return bows.tocsr()


def get_sbert_embeddings(processed_dataset: Dataset,
                         embedding_model: str = 'paraphrase-multilingual-mpnet-base-v2',
                         max_seq_length: int = 512) -> np.ndarray:
    text_for_contextual = [' '.join(example['words']) for example in processed_dataset]
    sbert_embeddings = bert_embeddings_from_list(
        text_for_contextual,
        sbert_model_to_load=embedding_model,
        max_seq_length=max_seq_length,
    )
    return sbert_embeddings


def get_ctm_dataset(data: dict[str, Any], vocab: list[str], model_type: str, embedding_type: str, verbose: bool = True):
    try:
        model_type = ModelType(model_type)
    except ValueError:
        raise ValueError(f"Unsupported model: {model_type}")

    try:
        embedding_type = EmbeddingType(embedding_type)
    except ValueError:
        raise ValueError(f"Unsupported embedding type: {embedding_type}")

    if model_type == ModelType.GENERATIVE:
        if embedding_type == EmbeddingType.HIDDEN_STATES:
            embeddings = np.stack(data['input_embeddings'])
        else:
            embeddings = get_sbert_embeddings(data)
    else:
        embeddings = get_sbert_embeddings(data)

    if model_type == ModelType.GENERATIVE:
        if isinstance(data['next_word_probs'][0], dict):
            print("Converting next_word_probs dicts to numpy array")
            data['next_word_probs'] = [[prob[word] for word in vocab] for prob in data['next_word_probs']]
        targets = np.stack(data['next_word_probs'])
    else:
        targets = get_bows(data['words'], vocab)

    if verbose:
        print(f"Input shape: {embeddings.shape}, target shape: {targets.shape}")
    idx2token = {i: token for i, token in enumerate(vocab)}
    dataset = CTMDataset(X_contextual=embeddings,
                         X_bow=targets,
                         idx2token=idx2token)
    return dataset