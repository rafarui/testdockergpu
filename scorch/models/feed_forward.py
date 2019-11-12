from scorch.models import Model
from scorch.nn import Input, FeedForward
import scorch.utils.parameters as default_parameters
from scorch.utils.cuda import get_device


class FeedForwardNet(Model):
    """
    Feed forward neural network which implements:

    input layer (with optional embeddings) > hidden layer(s) > output layer(s).

    - Input features can be numeric, categorical or both.

    - Any categorical features are embedded and then concatenated with any numeric features.

    - Data can then be further processed by one or more hidden layers.

    - Multiple output layers are possible for multi-task networks.

    - Can optionally return final hidden layer activations instead of network outputs.

    Parameters
    ----------

    numeric_dim: int, optional (default=0)
        Number of numeric features in data.
        If 0, num_embeddings and embedding_dims must not be None.

    num_embeddings: dict, optional (default=None)
        Size of the dictionary of embeddings for each categorical feature.
        Of the form {feature name (str): dictionary size (int)}.
        Must have the same keys as embedding_dims.
        If None, numeric_dim must not be 0.
        e.g. {'cat1': 32, 'cat2': 7}

    embedding_dims: dict, optional (default=None)
        The size of each embedding vector for each categorical feature.
        Of the form {feature name (str): dictionary size (int)}.
        Must have the same keys as num_embeddings.
        If None, numeric_dim must not be 0.
        e.g. {'cat1': 128, 'cat2': 64}

    hidden_dims: list, optional (default=None)
        A dense layer will be added for each element in the list,
        where hidden_dims[i] specifies the size of the i-th hidden layer.
        Elements in the list must either be integers or 'auto'.
        If 'auto', the size of the corresponding layer will be equal to the size of the previous layer.
        If None, the input layer will be connected directly to the output layer.
        e.g. ['auto', 128, 64] will implement three hidden layer, the first having the same size as
        the input layer followed by two layers of size 128 and 64 respectively.

    output_dims: int or list of ints, optional (default=1)
        Number of outputs.
        If list, len(output_dims) output layers are implemented,
        where output_dims[i] is the size of the i-th output layer.
        e.g. [1, 10] will implement two output layers, one of size 1 and another of size 10.

    numeric_input_layer_params: dict, optional (default=None)
        Named parameters for numeric input layer.
        Of the form {param name (str): param value}.
        Correspond to parameters of scorch.nn.NumericInput module.
        If None, default parameters from scorch.utils.parameters will be used.

    embed_layer_params: dict, optional (default=None)
        Named parameters for embedding layer.
        Of the form {param name (str): param value}.
        Correspond to parameters of scorch.nn.MultiColumnEmbedding module.
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

    Examples
    --------

    >>> import torch
    >>> import torch.nn.functional as F
    >>> from scorch.models import FeedForwardNet
    >>> num_rows = 5
    >>> numeric_dim = 3
    >>> num_embeddings = {'cat_1': 5, 'cat_2': 10}
    >>> embedding_dims = {'cat_1': 2, 'cat_2': 4}
    >>> data = dict(
    >>>     X_num=torch.rand(num_rows, numeric_dim, dtype=torch.float),
    >>>     X_cat=torch.cat([torch.randint(0, num_embeddings['cat_1'], (num_rows, 1)),
    >>>                      torch.randint(0, num_embeddings['cat_2'], (num_rows, 1))], dim=1)
    >>> )
    >>> mdl = FeedForwardNet(numeric_dim,
    >>>                      num_embeddings,
    >>>                      embedding_dims,
    >>>                      hidden_dims=['auto', 16],
    >>>                      output_dims=1,
    >>>                      numeric_input_layer_params={'batch_norm': True},
    >>>                      embed_layer_params={'dropout_rate': 0.1},
    >>>                      dense_layer_params={'activation': F.relu, 'dropout_rate': 0.5},
    >>>                      output_layer_params={'activation': torch.sigmoid},
    >>>                      device='cpu')
    >>> out = mdl(data, activate_output=True)
    >>> print(out)
    tensor([0.3063, 0.3822, 0.4825, 0.3762, 0.5318], grad_fn=<SigmoidBackward>)
    """

    def __init__(self,
                 numeric_dim=0,
                 num_embeddings=None,
                 embedding_dims=None,
                 hidden_dims=None,
                 output_dims=1,
                 numeric_input_layer_params=None,
                 embed_layer_params=None,
                 dense_layer_params=None,
                 output_layer_params=None,
                 device=None):

        super(FeedForwardNet, self).__init__()

        # initialise some instance variables
        self.numeric_dim = numeric_dim
        self.num_embeddings = num_embeddings or {}
        self.embedding_dims = embedding_dims or {}
        self.hidden_dims = hidden_dims
        self.output_dims = output_dims
        self.device = get_device(device)

        # get default layer parameters if None
        numeric_input_layer_params = numeric_input_layer_params or default_parameters.numeric_input_layer_params
        embed_layer_params = embed_layer_params or default_parameters.embed_layer_params
        dense_layer_params = dense_layer_params or default_parameters.dense_layer_params
        output_layer_params = output_layer_params or default_parameters.output_layer_params

        # initialise input layer
        self.input = Input(numeric_dim=self.numeric_dim,
                           num_embeddings=self.num_embeddings,
                           embedding_dims=self.embedding_dims,
                           numeric_input_layer_params=numeric_input_layer_params,
                           embed_layer_params=embed_layer_params,
                           device=self.device)

        # initialise feed forward layers
        self.feed_forward = FeedForward(in_features=self.input.out_features,
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
            Data is stored in the following keys:
                'X_num': torch.FloatTensor shape (n_rows, self.numeric_dim)
                    Numeric features.
                    If self.numeric_dim = 0, must be empty tensor.
                'X_cat': torch.LongTensor, shape (n, len(self.num_embeddings))
                    Categorical features
                    The columns must correspond to the column names specified in
                    self.num_embeddings, and the columns must be ordered according
                    to the alphabetical order of their names.
                    If self.num_embeddings is None, must be empty tensor.

        activate_output: bool, optional (default=False)
            If True, activation function will be applied to outputs.

        return_activations: bool, optional (default=False)
            If True the activations from the final hidden layer will be returned
            instead of network outputs.

        Returns
        -------

        out: torch.Tensor
            If return_activations=False, returns network output for each input,
            else returns final hidden layer activations.
            out[i] is the output or activation corresponding to
            data['X_num'][i] and data['X_cat'][i].
        """

        X = self.input(data['X_num'], data['X_cat'])

        out = self.feed_forward(X, activate_output=activate_output,
                                return_activations=return_activations)

        return out
