from collections import OrderedDict
from torch import nn
import torch


class ContextualInferenceNetwork(nn.Module):

    """Inference Network."""

    def __init__(self, input_size, bert_size, output_size, hidden_sizes,
                 activation='gelu', dropout=0.2, label_size=0, layer_type='linear'):
        """
        # TODO: check dropout in main caller
        Initialize InferenceNetwork.

        Args
            input_size : int, dimension of input
            output_size : int, dimension of output
            hidden_sizes : tuple, length = n_layers
            activation : string, 'softplus', 'gelu', or 'relu', default 'gelu'
            dropout : float, default 0.2, default 0.2
        """
        super(ContextualInferenceNetwork, self).__init__()
        assert isinstance(input_size, int), "input_size must by type int."
        assert isinstance(output_size, int), "output_size must be type int."
        assert isinstance(hidden_sizes, tuple), \
            "hidden_sizes must be type tuple."
        assert activation in ['softplus', 'relu', 'gelu'], \
            "activation must be 'softplus', 'relu' or 'gelu'."
        assert dropout >= 0, "dropout must be >= 0."

        self.input_size = input_size
        self.output_size = output_size
        self.hidden_sizes = hidden_sizes
        self.dropout = dropout

        if activation == 'softplus':
            self.activation = nn.Softplus()
        elif activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'gelu':
            self.activation = nn.GELU()
        else:
            raise ValueError(f"Activation function {activation} not supported.")

        self.input_layer = nn.Linear(bert_size + label_size, hidden_sizes[0])
        #self.adapt_bert = nn.Linear(bert_size, hidden_sizes[0])

        # Create sequential layers for all hidden sizes
        layers = []
        for i in range(len(hidden_sizes)-1):
            h_in = hidden_sizes[i]
            h_out = hidden_sizes[i+1]

            if layer_type == 'linear':
                layers.append((f'l_{i}', nn.Sequential(nn.Linear(h_in, h_out), self.activation)))
            elif layer_type == 'attention':
                layers.append((f'l_{i}', nn.TransformerEncoderLayer(d_model=h_in, num_heads=4, dim_feedforward=h_in, activation=activation)))

        self.hiddens = nn.Sequential(OrderedDict(layers))

        self.f_mu = nn.Linear(hidden_sizes[-1], output_size)
        self.f_mu_batchnorm = nn.BatchNorm1d(output_size, affine=False)

        self.f_sigma = nn.Linear(hidden_sizes[-1], output_size)
        self.f_sigma_batchnorm = nn.BatchNorm1d(output_size, affine=False)

        self.dropout_enc = nn.Dropout(p=self.dropout)

    def forward(self, x, x_bert, labels=None):
        """Forward pass."""

        x = x_bert
        if labels:
            x = torch.cat((x_bert, labels), 1)

        x = self.input_layer(x)

        x = self.activation(x)
        x = self.hiddens(x)
        x = self.dropout_enc(x)
        mu = self.f_mu_batchnorm(self.f_mu(x))
        log_sigma = self.f_sigma_batchnorm(self.f_sigma(x))

        return mu, log_sigma


class CombinedInferenceNetwork(nn.Module):

    """Inference Network."""

    def __init__(self, input_size, bert_size, output_size, hidden_sizes,
                 activation='gelu', dropout=0.2, label_size=0):
        """
        Initialize InferenceNetwork.

        Args
            input_size : int, dimension of input
            output_size : int, dimension of output
            hidden_sizes : tuple, length = n_layers
            activation : string, 'softplus', 'relu', 'gelu', default 'gelu'
            dropout : float, default 0.2, default 0.2
        """
        super(CombinedInferenceNetwork, self).__init__()
        assert isinstance(input_size, int), "input_size must by type int."
        assert isinstance(output_size, int), "output_size must be type int."
        assert isinstance(hidden_sizes, tuple), \
            "hidden_sizes must be type tuple."
        assert activation in ['softplus', 'relu', 'gelu'], \
            "activation must be 'softplus' or 'relu' or 'gelu'."
        assert dropout >= 0, "dropout must be >= 0."

        self.input_size = input_size
        self.output_size = output_size
        self.hidden_sizes = hidden_sizes
        self.dropout = dropout

        if activation == 'softplus':
            self.activation = nn.Softplus()
        elif activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'gelu':
            self.activation = nn.GELU()
        else:
            raise ValueError(f"Activation function {activation} not supported.")


        self.adapt_bert = nn.Linear(bert_size, input_size)
        #self.bert_layer = nn.Linear(hidden_sizes[0], hidden_sizes[0])
        self.input_layer = nn.Linear(input_size + input_size + label_size, hidden_sizes[0])

        layers = []
        for i in range(len(hidden_sizes)-1):
            h_in = hidden_sizes[i]
            h_out = hidden_sizes[i+1]
            layers.append((f'l_{i}', nn.Sequential(nn.Linear(h_in, h_out), self.activation)))
        self.hiddens = nn.Sequential(OrderedDict(layers))

        self.f_mu = nn.Linear(hidden_sizes[-1], output_size)
        self.f_mu_batchnorm = nn.BatchNorm1d(output_size, affine=False)

        self.f_sigma = nn.Linear(hidden_sizes[-1], output_size)
        self.f_sigma_batchnorm = nn.BatchNorm1d(output_size, affine=False)

        self.dropout_enc = nn.Dropout(p=self.dropout)

    def forward(self, x, x_bert, labels=None):
        """Forward pass."""
        x_bert = self.adapt_bert(x_bert)

        x = torch.cat((x, x_bert), 1)

        if labels is not None:
            x = torch.cat((x, labels), 1)

        x = self.input_layer(x)

        x = self.activation(x)
        x = self.hiddens(x)
        x = self.dropout_enc(x)
        mu = self.f_mu_batchnorm(self.f_mu(x))
        log_sigma = self.f_sigma_batchnorm(self.f_sigma(x))

        return mu, log_sigma
