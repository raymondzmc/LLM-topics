import os
import numpy as np
from collections import Counter
from datasets import Dataset
from contextualized_topic_models.models.ctm import CTM, GenerativeTM, ZeroShotTM, CombinedTM
from contextualized_topic_models.datasets.dataset import CTMDataset
from contextualized_topic_models.utils.data_preparation import bert_embeddings_from_list
from scipy.sparse import lil_matrix, csr_matrix
from tqdm import tqdm
from enum import Enum
import pdb

os.environ["TOKENIZERS_PARALLELISM"] = "false"


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


def get_bows(processed_dataset, vocab):
    bows = lil_matrix((len(processed_dataset), len(vocab)), dtype=int)
    vocab_to_idx = {token: idx for idx, token in enumerate(vocab)}
    for i, words in enumerate(tqdm(processed_dataset['words'])):
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


def train_topic_model(model_type: str,
                      vocab: list[str],
                      processed_dataset: Dataset | list[list[str]],
                      K: int,
                      embedding_type: str = 'hidden_states',
                      hidden_state_layer: int | None = None,
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
                      loss_weight: float | None = None,
                      model_checkpoint: dict | None = None,
                      continue_training: bool = False) -> CTM:
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
            embeddings = np.stack(processed_dataset['input_embeddings'])
        else:
            embeddings = get_sbert_embeddings(processed_dataset)
    else:
        embeddings = get_sbert_embeddings(processed_dataset)

    if model_type == ModelType.GENERATIVE:
        if isinstance(processed_dataset['next_word_probs'][0], dict):
            print("Converting next_word_probs dicts to numpy array")
            processed_dataset['next_word_probs'] = [[prob[word] for word in vocab] for prob in processed_dataset['next_word_probs']]
        targets = np.stack(processed_dataset['next_word_probs'])
    else:
        targets = get_bows(processed_dataset, vocab)

    print(f"Input shape: {embeddings.shape}, target shape: {targets.shape}")

    if loss_weight is not None:
        loss_weights = {"beta": loss_weight}
    else:
        loss_weights = None

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

    if model_checkpoint is not None:
        model.model.load_state_dict(model_checkpoint, strict=False)
        model.idx2token = idx2token
        print("Successfully loaded model weights.")
        if continue_training:
            print("Continuing training.")
            model.fit(dataset)
    else:
        model.fit(dataset)
    return model
