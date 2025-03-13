import torch
import numpy as np
import random
from octis.models.model import AbstractModel
from octis.models.contextualized_topic_models.datasets.dataset import CTMDataset
from models import ctm as ctm


class CTM(AbstractModel):

    def __init__(
        self, num_topics=10, model_type='prodLDA', activation='softplus',
        dropout=0.2, learn_priors=True, batch_size=64, lr=2e-3, momentum=0.99,
        solver='adam', num_epochs=100, reduce_on_plateau=False, prior_mean=0.0,
        prior_variance=None, num_layers=2, num_neurons=100, seed=None,
        use_partitions=False, num_samples=10, loss_weight=None, sparsity_ratio=None):
        """
        initialization of CTM

        :param num_topics : int, number of topic components, (default 10)
        :param model_type : string, 'prodLDA' or 'LDA' (default 'prodLDA')
        :param activation : string, 'softplus', 'relu', 'sigmoid', 'swish',
            'tanh', 'leakyrelu', 'rrelu', 'elu', 'selu' (default 'softplus')
        :param num_layers : int, number of layers (default 2)
        :param dropout : float, dropout to use (default 0.2)
        :param learn_priors : bool, make priors a learnable parameter
            (default True)
        :param batch_size : int, size of batch to use for training (default 64)
        :param lr : float, learning rate to use for training (default 2e-3)
        :param momentum : float, momentum to use for training (default 0.99)
        :param solver : string, optimizer 'adam' or 'sgd' (default 'adam')
        :param num_epochs : int, number of epochs to train for, (default 100)
        :param num_samples: int, number of times theta needs to be sampled
            (default: 10)
        :param seed : int, the random seed. Not used if None (default None).
        :param use_partitions: bool, if true the model will be trained on the
            training set and evaluated on the test set (default: true)
        :param reduce_on_plateau : bool, reduce learning rate by 10x on
            plateau of 10 epochs (default False)
        """

        super().__init__()

        self.hyperparameters['num_topics'] = num_topics
        self.hyperparameters['model_type'] = model_type
        self.hyperparameters['activation'] = activation
        self.hyperparameters['dropout'] = dropout
        self.hyperparameters['learn_priors'] = learn_priors
        self.hyperparameters['batch_size'] = batch_size
        self.hyperparameters['lr'] = lr
        self.hyperparameters['num_samples'] = num_samples
        self.hyperparameters['momentum'] = momentum
        self.hyperparameters['solver'] = solver
        self.hyperparameters['num_epochs'] = num_epochs
        self.hyperparameters['reduce_on_plateau'] = reduce_on_plateau
        self.hyperparameters["prior_mean"] = prior_mean
        self.hyperparameters["prior_variance"] = prior_variance
        self.hyperparameters["num_neurons"] = num_neurons
        self.hyperparameters["num_layers"] = num_layers
        self.hyperparameters["seed"] = seed
        self.hyperparameters["loss_weight"] = loss_weight
        self.hyperparameters["sparsity_ratio"] = sparsity_ratio
        self.use_partitions = use_partitions

        hidden_sizes = tuple([num_neurons for _ in range(num_layers)])
        self.hyperparameters['hidden_sizes'] = tuple(hidden_sizes)

        self.model = None
        self.vocab = None

    def train_model(self, dataset, hyperparameters=None, top_words=10):
        """
        trains CTM model

        :param dataset: octis Dataset for training the model
        :param hyperparameters: dict, with optionally) the following information:
        :param top_words: number of top-n words of the topics (default 10)

        """
        if hyperparameters is None:
            hyperparameters = {}

        self.set_params(hyperparameters)
        self.vocab = dataset['vocab']
        self.set_seed(seed=self.hyperparameters['seed'])
        x_train, input_size = self.preprocess(vocab=self.vocab, train=dataset)

        self.model = ctm.CTM(
             input_size=input_size, bert_input_size=x_train.X_bert.shape[1], model_type='prodLDA',
             num_topics=self.hyperparameters['num_topics'], dropout=self.hyperparameters['dropout'],
             activation=self.hyperparameters['activation'], lr=self.hyperparameters['lr'],
             inference_type='zeroshot',
             hidden_sizes=self.hyperparameters['hidden_sizes'], solver=self.hyperparameters['solver'],
             momentum=self.hyperparameters['momentum'], num_epochs=self.hyperparameters['num_epochs'],
             learn_priors=self.hyperparameters['learn_priors'],
             batch_size=self.hyperparameters['batch_size'],
             num_samples=self.hyperparameters['num_samples'],
             topic_prior_mean=self.hyperparameters["prior_mean"],
             reduce_on_plateau=self.hyperparameters['reduce_on_plateau'],
             topic_prior_variance=self.hyperparameters["prior_variance"],
             top_words=top_words,
             loss_weight=self.hyperparameters["loss_weight"],
             sparsity_ratio=self.hyperparameters['sparsity_ratio'])

        self.model.fit(x_train, None, verbose=False)
        result = self.model.get_info()
        return result

    def set_params(self, hyperparameters):
        for k in hyperparameters.keys():
            if k in self.hyperparameters.keys() and k != 'hidden_sizes':
                self.hyperparameters[k] = hyperparameters.get(
                    k, self.hyperparameters[k])

        self.hyperparameters['hidden_sizes'] = tuple(
            [self.hyperparameters["num_neurons"] for _ in range(
                self.hyperparameters["num_layers"])])

    def inference(self, x_test):
        assert isinstance(self.use_partitions, bool) and self.use_partitions
        results = self.model.predict(x_test)
        return results

    def partitioning(self, use_partitions=False):
        self.use_partitions = use_partitions

    @staticmethod
    def set_seed(seed=None):
        if seed is not None:
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)
            torch.backends.cudnn.enabled = False
            torch.backends.cudnn.deterministic = True

    @staticmethod
    def preprocess(vocab, train):
        X_bert = np.stack(train['input_embeddings'])
        X = np.stack(train['next_word_probs'])
        idx2token = {i: token for i, token in enumerate(vocab)}
        train_data = CTMDataset(X_bert=X_bert, X=X, idx2token=idx2token)
        input_size = len(idx2token.keys())
        return train_data, input_size
