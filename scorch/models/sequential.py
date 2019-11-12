import torch
import torch.nn as nn

from scorch.models import Model
from scorch.nn import Input, Embedding, FeedForward
import scorch.utils.parameters as default_parameters
from scorch.utils.cuda import get_device


class SequentialNet(Model):
    """
    Neural network for processing sequential categorical data such as text.

    - The elements of the sequences must be integers, so categorical values such as words must be label
      encoded before passing them to the network.

    - The elements of the sequences are embedded before being passed through one or more recurrent layers.

    - Can optionally include structured input data. That is, each sequence can be complemented by standard
      numeric and categorical features.

    - Any complementary numeric features are concatenated with the output of the recurrent layers.

    - Any complementary categorical features are embedded and then concatenated with the output of
      the recurrent layers.

    - Data can then be further processed by one or more hidden layers.

    - Multiple output layers are possible for multi-task networks.

    - Can optionally return final hidden layer activations instead of network outputs.

    Parameters
    ----------

    num_seq_embeddings: int
        Size of the dictionary of sequence embeddings.
        This should correspond to the number of unique values
        (such as word) in the sequence.

    seq_embedding_dim: int
        The size of the embedding vectors for the values
        (such as words) in the sequence.

    padding_idx: int, optional (default=None)
        Value used to pad the end of variable length sequences in the
        data which will be passed to the model.
        If the network encounters this value in a sequence
        it will effectively ignore it.

    recurrent_module: torch module, optional (default=nn.GRU)
        Recurrent module from torch.nn.

    recurrent_dim: int or 'auto', optional (default='auto')
        Hidden dimension of recurrent layers.
        If 'auto', will be set to seq_embedding_dim.

    num_recurrent_layers: int, optional (default=1)
        Number of stacked recurrent layers.

    bidirectional: boolean, optional (default=False)
        If True, recurrent layer becomes bidirectional and the output
        of the recurrent layers will have dimension (2 * recurrent_dim).

    numeric_dim: int, optional (default=0)
        Number of numeric features in complementary structured data.

    num_cat_embeddings: dict, optional (default=None)
        Size of the dictionary of embeddings for each complementary categorical feature.
        Of the form {feature name (str): dictionary size (int)}.
        Must have the same keys as embedding_dims.
        e.g. {'cat1': 32, 'cat2': 7}

    cat_embedding_dims: dict, optional (default=None)
        The size of each embedding vector for each complementary categorical feature.
        Of the form {feature name (str): dictionary size (int)}.
        Must have the same keys as num_embeddings.
        e.g. {'cat1': 128, 'cat2': 64}

    hidden_dims: list, optional (default=None)
        A dense layer will be added for each element in the list,
        where hidden_dims[i] specifies the size of the i-th hidden layer.
        Elements in the list must either be integers or 'auto'.
        If 'auto', the size of the corresponding layer will be equal to the size of the previous layer.
        If None, the recurrent layers will be connected directly to the output layer.
        e.g. ['auto', 128, 64] will implement three hidden layer, the first having the combined size of
        the recurrent layer and structured input layer, followed by two layers of size 128 and 64 respectively.

    output_dims: int or list of ints, optional (default=1)
        Number of outputs.
        If list, len(output_dims) output layers are implemented,
        where output_dims[i] is the size of the i-th output layer.
        e.g. [1, 10] will implement two output layers, one of size 1 and another of size 10.

    numeric_input_layer_params: dict, optional (default=None)
        Named parameters for structured numeric input layer.
        Of the form {param name (str): param value}.
        Correspond to parameters of scorch.nn.NumericInput module.
        If None, default parameters from scorch.utils.parameters will be used.

    seq_embed_layer_params: dict, optional (default=None)
        Named parameters for sequence embedding layer.
        Of the form {param name (str): param value}.
        Correspond to parameters of scorch.nn.Embedding module.
        If None, default parameters from scorch.utils.parameters will be used.

    cat_embed_layer_params: dict, optional (default=None)
        Named parameters for structured categorical embedding layer.
        Of the form {param name (str): param value}.
        Correspond to parameters of scorch.nn.MultiColumnEmbedding module.
        If None, default parameters from scorch.utils.parameters will be used.

    recurrent_layer_params: dict, optional (default=None)
        Named parameters for the recurrent layer(s).
        Of the form {param name (str): param value}.
        Correspond to parameters of recurrent_module.
        If None, default parameters from scorch.utils.parameters will be used.

    dense_layer_params: dict, optional (default=None)
        Named parameters for dense layers.
        Of the form {param name (str): param value}.
        Correspond to parameters of scorch.nn.MultiDense module.
        If None, default parameters from scorch.utils.parameters will be used.

    output_layer_params: dict, optional (default=None)
        Named parameters for output layer.
        Of the form {param name (str): param value}.
        Correspond to parameters of scorch.nn.Output module.
        If None, default parameters from scorch.utils.parameters will be used.

    device: torch.device or string, optional (default=None)
        Device on which module will be placed.
        If string, must specify the device e.g. 'cpu' or 'cuda'.
        If None, GPU will be used if available, else CPU.
    """

    def __init__(self,
                 num_seq_embeddings,
                 seq_embedding_dim,
                 padding_idx=None,
                 recurrent_module=nn.GRU,
                 recurrent_dim='auto',
                 num_recurrent_layers=1,
                 bidirectional=False,
                 numeric_dim=0,
                 num_cat_embeddings=None,
                 cat_embedding_dims=None,
                 hidden_dims=None,
                 output_dims=1,
                 numeric_input_layer_params=None,
                 seq_embed_layer_params=None,
                 cat_embed_layer_params=None,
                 recurrent_layer_params=None,
                 dense_layer_params=None,
                 output_layer_params=None,
                 device=None):

        super(SequentialNet, self).__init__()

        # initialise some instance variables
        self.num_seq_embeddings = num_seq_embeddings
        self.seq_embedding_dim = seq_embedding_dim
        self.padding_idx = padding_idx
        self.recurrent_dim = seq_embedding_dim if recurrent_dim == 'auto' else recurrent_dim
        self.num_recurrent_layers = num_recurrent_layers
        self.bidirectional = bidirectional
        self.numeric_dim = numeric_dim
        self.num_cat_embeddings = num_cat_embeddings or {}
        self.cat_embedding_dims = cat_embedding_dims or {}
        self.hidden_dims = hidden_dims
        self.output_dims = output_dims
        self.device = get_device(device)

        # get default layer parameters if None
        numeric_input_layer_params = numeric_input_layer_params or default_parameters.numeric_input_layer_params
        seq_embed_layer_params = seq_embed_layer_params or default_parameters.embed_layer_params
        seq_embed_layer_params['batch_norm'] = False
        cat_embed_layer_params = cat_embed_layer_params or default_parameters.embed_layer_params
        recurrent_layer_params = recurrent_layer_params or default_parameters.recurrent_layer_params
        dense_layer_params = dense_layer_params or default_parameters.dense_layer_params
        output_layer_params = output_layer_params or default_parameters.output_layer_params

        # initialise input layer for structured data
        self.struct_input = Input(numeric_dim=self.numeric_dim,
                                  num_embeddings=self.num_cat_embeddings,
                                  embedding_dims=self.cat_embedding_dims,
                                  numeric_input_layer_params=numeric_input_layer_params,
                                  embed_layer_params=cat_embed_layer_params,
                                  device=self.device)

        # initialise embedding layer for sequence data
        self.seq_embedder = Embedding(num_embeddings=self.num_seq_embeddings,
                                      embedding_dim=self.seq_embedding_dim,
                                      padding_idx=self.padding_idx,
                                      **seq_embed_layer_params)

        # initialise recurrent layers
        self.recurrent = recurrent_module(input_size=self.seq_embedding_dim,
                                          hidden_size=self.recurrent_dim,
                                          num_layers=self.num_recurrent_layers,
                                          bidirectional=self.bidirectional,
                                          **recurrent_layer_params)

        # initialise feed forward layers
        d = self.struct_input.out_features + (self.recurrent_dim * (2 if self.bidirectional else 1))
        self.feed_forward = FeedForward(in_features=d,
                                        hidden_dims=hidden_dims,
                                        output_dims=output_dims,
                                        dense_layer_params=dense_layer_params,
                                        output_layer_params=output_layer_params,
                                        device=self.device)

        # move model to device
        self.to(self.device)

    def forward(self,
                data,
                activate_output=False,
                return_activations=False):
        """
        Executes the forward pass.

        Parameters
        -------

        data: dict
            Batch data with the following keys:
                'S': torch.LongTensor, shape (max sequence length, batch size)
                    Label encoded sequences.
                    If sequences are of variable length, their ends should be
                    padded with self.padding_idx.

                'i_end': np.array, shape (batch size,)
                    i_end[i] is the index of the last non-padded element of sequence S[:, i].
                    Equivalent to the true length of each sequence minus 1.

                'X_num': torch.FloatTensor
                    Complementary numeric features of shape (batch size, self.numeric_dim).
                    If no numeric features, use empty torch.FloatTensor.

                'X_cat': torch.LongTensor
                    Complementary categorical features of shape (batch size, len(self.num_embeddings)).
                    If no categorical features, use empty torch.LongTensor.

        activate_output: bool, optional (default=False)
            If True, activation function will be applied to outputs.

        return_activations: bool, optional (default=False)
            If True the activations from the final hidden layer will be returned
            instead of network outputs.

        Returns
        -------

        out: torch.Tensor
            If return_activations=False, returns network output for each sequence,
            else returns final hidden layer activations.
            out[i] is the output or activation corresponding to
            data['X_num'][i] and data['X_cat'][i].

        Examples
        --------

        >>> import numpy as np
        >>> import torch
        >>> import torch.nn.functional as F
        >>> from scorch.models import SequentialNet
        >>> num_rows = 5
        >>> sequence_lengths = np.random.randint(1, 10, num_rows)
        >>> num_seq_embeddings = 10
        >>> seq_embedding_dim = 8
        >>> padding_idx = 0
        >>> max_length = max(sequence_lengths)
        >>> data = {}
        >>> data['S'] = torch.LongTensor([list(np.random.randint(1, num_seq_embeddings, i))
        >>>                               + ([padding_idx] * (max_length - i))
        >>>                               for i in sequence_lengths]).t()
        >>> data['i_end'] = np.array(sequence_lengths) - 1
        >>> numeric_dim = 3
        >>> num_cat_embeddings = {'cat_1': 5, 'cat_2': 10}
        >>> cat_embedding_dims = {'cat_1': 2, 'cat_2': 4}
        >>> data['X_num'] = torch.rand(num_rows, 3, dtype=torch.float)
        >>> data['X_cat'] = torch.cat([torch.randint(0, num_cat_embeddings['cat_1'], (num_rows, 1)),
        >>>                            torch.randint(0, num_cat_embeddings['cat_2'], (num_rows, 1))],
        >>>                           dim=1)
        >>> mdl = SequentialNet(num_seq_embeddings,
        >>>                     seq_embedding_dim,
        >>>                     padding_idx=padding_idx,
        >>>                     recurrent_module=torch.nn.GRU,
        >>>                     recurrent_dim=16,
        >>>                     num_recurrent_layers=2,
        >>>                     bidirectional=True,
        >>>                     numeric_dim=numeric_dim,
        >>>                     num_cat_embeddings=num_cat_embeddings,
        >>>                     cat_embedding_dims=cat_embedding_dims,
        >>>                     hidden_dims=[32, 32],
        >>>                     output_dims=1,
        >>>                     dense_layer_params={'activation': F.relu, 'dropout_rate': 0.5},
        >>>                     output_layer_params={'activation': torch.sigmoid},
        >>>                     device='cpu')
        >>> out = mdl(data, activate_output=True)
        >>> print(out)
        tensor([0.5843, 0.5440, 0.4096, 0.5668, 0.4968], grad_fn=<SigmoidBackward>)
        """

        # process structured input
        X_struct = self.struct_input(data['X_num'], data['X_cat'])

        # embed sequences
        E_seq = self.seq_embedder(data['S'])

        # get recurrent activations corresponding to last element of each sequence
        H, _ = self.recurrent(E_seq)
        h = H[data['i_end'], range(H.size(1)), :self.recurrent_dim]

        if self.bidirectional:
            # get recurrent activations corresponding to opposite direction
            h = torch.cat([h, H[0, :, self.recurrent_dim:]], dim=1)

        # concatenate with structured input
        H = torch.cat([h, X_struct], dim=1)

        # get outputs
        out = self.feed_forward(H, activate_output=activate_output,
                                return_activations=return_activations)

        return out
