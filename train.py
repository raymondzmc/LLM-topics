import numpy as np
from datasets import Dataset
from contextualized_topic_models.models.ctm import CTM, GenerativeTM, ZeroShotTM, CombinedTM
from contextualized_topic_models.datasets.dataset import CTMDataset
from contextualized_topic_models.utils.data_preparation import bert_embeddings_from_list
from scipy.sparse import lil_matrix
from tqdm import tqdm


def train_generative_tm(
    vocab: list[str],
    processed_dataset: Dataset,
    K: int,
    num_epochs: int,
    hidden_sizes: tuple[int]
) -> CTM:
    idx2token = {i: token for i, token in enumerate(vocab)}
    input_embeddings = np.stack([example['input_embeddings'] for example in processed_dataset])
    target_distributions = np.stack([example['next_word_probs'] for example in processed_dataset])
    dataset = CTMDataset(X_contextual=input_embeddings,
                         X_bow=target_distributions,
                         idx2token=idx2token)

    contextual_size = input_embeddings.shape[1]
    model = GenerativeTM(bow_size=len(dataset.idx2token),
                         contextual_size=contextual_size,
                         n_components=K,
                         hidden_sizes=hidden_sizes,
                         num_epochs=num_epochs)
    model.fit(dataset)
    return model


def train_ctm(
    model: str,
    vocab: list[str],
    processed_dataset: Dataset,
    K: int,
    num_epochs: int,
    hidden_sizes: tuple[int],
    embedding_model: str = 'paraphrase-multilingual-mpnet-base-v2'
) -> CTM:
    idx2token = {i: token for i, token in enumerate(vocab)}
    X_bows = lil_matrix((len(processed_dataset), len(vocab)))
    text_for_contextual = []
    for i, example in enumerate(tqdm(processed_dataset)):
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
    dataset = CTMDataset(X_contextual=X_contextual,
                         X_bow=X_bows,
                         idx2token=idx2token)
    contextual_size = X_contextual.shape[1]
    if model == 'combined':
        model_cls = CombinedTM
    elif model == 'zeroshot':
        model_cls = ZeroShotTM
    else:
        raise ValueError(f"Unsupported model: {model}")

    model = model_cls(
        bow_size=len(dataset.idx2token),
        contextual_size=contextual_size,
        hidden_sizes=hidden_sizes,
        n_components=K,
        num_epochs=num_epochs,
    )
    model.fit(dataset)
    return model
