import os
from datasets import Dataset
from utils.dataset import get_ctm_dataset
from utils.enums import model_classes
from contextualized_topic_models.models.ctm import CTM
import pdb

os.environ["TOKENIZERS_PARALLELISM"] = "false"


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

    if loss_weight is not None:
        loss_weights = {"beta": loss_weight}
    else:
        loss_weights = None

    dataset = get_ctm_dataset(processed_dataset, vocab, model_type, embedding_type)
    contextual_size = dataset.X_contextual.shape[1]

    model_cls = model_classes[model_type]
    model = model_cls(bow_size=len(vocab),
                      contextual_size=contextual_size,
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
        model.idx2token = dataset.idx2token
        print("Successfully loaded model weights.")
        if continue_training:
            print("Continuing training.")
            model.fit(dataset)
    else:
        model.fit(dataset)
    return model
