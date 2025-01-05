import torch
import numpy as np
from datasets import Dataset
from contextualized_topic_models.models.ctm import CTM, GenerativeTM, ZeroShotTM, CombinedTM
from contextualized_topic_models.datasets.dataset import CTMDataset
from contextualized_topic_models.utils.data_preparation import bert_embeddings_from_list
from scipy.sparse import lil_matrix
from tqdm import tqdm
from enum import Enum

class ModelType(Enum):
    GENERATIVE = 'generative'
    ZEROSHOT = 'zeroshot'
    COMBINED = 'combined'


class EmbeddingType(Enum):
    HIDDEN_STATES = 'hidden_states'
    SBERT = 'sbert'


model_classes = {
    ModelType.GENERATIVE: GenerativeTM,
    ModelType.ZEROSHOT: ZeroShotTM,
    ModelType.COMBINED: CombinedTM,
}


def get_bows(processed_dataset: Dataset,
             vocab: list[str]) -> tuple[np.ndarray, np.ndarray]:
    
    bows = lil_matrix((len(processed_dataset), len(vocab))).tocsr()
    for i, example in enumerate(tqdm(processed_dataset)):
        text = example['content'] if 'content' in example else example['context']
        for j, token in vocab:
            if token in text:
                bows[i, j] = text.count(token)
    return bows


def get_sbert_embeddings(processed_dataset: Dataset,
                         embedding_model: str = 'paraphrase-multilingual-mpnet-base-v2',
                         max_seq_length: int = 128) -> np.ndarray:
    text_for_contextual = [example['content'] for example in processed_dataset]
    sbert_embeddings = bert_embeddings_from_list(
        text_for_contextual,
        sbert_model_to_load=embedding_model,
        max_seq_length=max_seq_length,
    )
    return sbert_embeddings


def train_topic_model(model_type: str,
                      vocab: list[str],
                      processed_dataset: Dataset,
                      K: int,
                      embedding_type: str = 'hidden_states',
                      hidden_sizes: tuple[int] = (100, 100),
                      activation: str = "softplus",
                      dropout: float = 0.2,
                      learn_priors: bool = True,
                      batch_size: int = 64,
                      lr: float = 2e-3,
                      momentum: float = 0.99,
                      solver: str = "adam",
                      num_epochs: int = 100,
                      reduce_on_plateau: bool = False,
                      label_size: int = 0,
                      loss_weights: list[float] | None = None,
                      model_weights: dict | None = None,
                      continue_training: bool = False) -> CTM:
    try:
        model_type = ModelType(model_type)
    except ValueError:
        raise ValueError(f"Unsupported model: {model_type}")

    try:
        embedding_type = EmbeddingType(embedding_type)
    except ValueError:
        raise ValueError(f"Unsupported embedding type: {embedding_type}")

    if embedding_type == EmbeddingType.HIDDEN_STATES:
        embeddings = np.stack([example['input_embeddings'] for example in processed_dataset])
    else:
        embeddings = get_sbert_embeddings(processed_dataset)

    if model_type == ModelType.GENERATIVE:
        targets = np.stack([example['next_word_probs'] for example in processed_dataset])
    else:
        targets = get_bows(processed_dataset, vocab)

    idx2token = {i: token for i, token in enumerate(vocab)}
    dataset = CTMDataset(X_contextual=embeddings,
                         X_bow=targets,
                         idx2token=idx2token)
    model_cls = model_classes[model_type]

    model = model_cls(bow_size=len(vocab),
                      contextual_size=embeddings.shape[1],
                      n_components=K,
                      hidden_sizes=hidden_sizes,
                      activation=activation,
                      dropout=dropout,
                      learn_priors=learn_priors,
                      batch_size=batch_size,
                      lr=lr,
                      momentum=momentum,
                      solver=solver,
                      num_epochs=num_epochs,
                      reduce_on_plateau=reduce_on_plateau,
                      label_size=label_size,
                      loss_weights=loss_weights)
    if model_weights is not None:
        model.model.load_state_dict(model_weights)
        print("Successfully loaded model weights.")
        if continue_training:
            print("Continuing training.")
            model.fit(dataset)
    else:
        model.fit(dataset)
    return model
