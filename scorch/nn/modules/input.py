import torch
import torch.nn as nn

from scorch.nn.modules.utils import PostLinear
from scorch.nn.modules.embedding import MultiColumnEmbedding, MultiTableSharedEmbedding, MultiTableUnsharedEmbedding
import scorch.utils.parameters as default_parameters
from scorch.utils.cuda import get_device


class NumericInput(nn.Module):
    """
    Input layer for numeric inputs.

    - Option to apply batch norm > dropout.

    Parameters
    ----------

    num_features: int
        Number of input features.

    batch_norm: bool, optional (default=False)
        If True, numeric inputs will be batch normalised.

    dropout_rate: float, optional (default=None)
        If not None, dropout layer with specified dropout rate will
        be applied to numeric inputs.

    device: torch.device or string, optional (default=None)
        Device on which module will be placed.
        If string, must specify the device e.g. 'cpu' or 'cuda'.
        If None, GPU will be used if available, else CPU.

    Examples
    --------

    >>> import torch
    >>> from scorch.nn import NumericInput
    >>> X = torch.rand(5, 3, dtype=torch.float)
    >>> input_layer = NumericInput(X.shape[1],
    >>>                            batch_norm=True,
    >>>                            dropout_rate=0.5,
    >>>                            device='cpu')
    >>> H = input_layer(X)
    >>> print(H.shape)
    """

    def __init__(self,
                 num_features,
                 batch_norm=False,
                 dropout_rate=None,
                 device=None):

        super(NumericInput, self).__init__()

        # initialise some instance variables
        self.num_features = num_features
        self.device = get_device(device)

        if num_features > 0:

            # initialise batch norm and dropout
            self.layers = PostLinear(in_features=num_features,
                                     batch_norm=batch_norm,
                                     activation=None,
                                     dropout_rate=dropout_rate,
                                     device=self.device)

        # move module to device
        self.to(self.device)

    def forward(self, X):
        """
        Executes forward pass.

        Applies batch norm > dropout.

        If batch norm or dropout is None, will be missed out.

        Parameters
        ----------

        X: torch.FloatTensor, shape (n, self.num_features)
            Numeric input data.

        Returns
        -------

        H: torch.FloatTensor, shape (n, self.num_features)
            Input data after having possibly been passed through
            batch norm and dropout layers.
        """

        if len(X) == 0:
            return torch.tensor([], dtype=torch.float, device=self.device)

        H = self.layers(X)

        return H


class Input(nn.Module):
    """
    Input layer for both numeric and categorical inputs.

    Embeds categorical features and concatenates them with numeric features.

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

    device: torch.device or string, optional (default=None)
        Device on which module will be placed.
        If string, must specify the device e.g. 'cpu' or 'cuda'.
        If None, GPU will be used if available, else CPU.

    Examples
    --------

    >>> import torch
    >>> from scorch.nn import Input
    >>> num_rows = 5
    >>> numeric_dim = 3
    >>> num_embeddings = {'cat_1': 5, 'cat_2': 10}
    >>> embedding_dims = {'cat_1': 2, 'cat_2': 4}
    >>> X_num = torch.rand(num_rows, numeric_dim, dtype=torch.float)
    >>> X_cat = torch.cat([torch.randint(0, num_embeddings['cat_1'], (num_rows, 1)),
    >>>                    torch.randint(0, num_embeddings['cat_2'], (num_rows, 1))], dim=1)
    >>> input_layer = Input(numeric_dim,
    >>>                     num_embeddings,
    >>>                     embedding_dims,
    >>>                     numeric_input_layer_params={'batch_norm': True},
    >>>                     embed_layer_params={'dropout_rate': 0.1},
    >>>                     device='cpu')
    >>> H = input_layer(X_num, X_cat)
    >>> print(H.shape)
    torch.Size([5, 9])
    """

    def __init__(self,
                 numeric_dim,
                 num_embeddings,
                 embedding_dims,
                 numeric_input_layer_params=None,
                 embed_layer_params=None,
                 device=None):

        super(Input, self).__init__()

        # initialise some instance variables
        self.numeric_dim = numeric_dim
        self.num_embeddings = num_embeddings
        self.embedding_dims = embedding_dims
        self.device = get_device(device)

        # get default layer parameters if None
        numeric_input_layer_params = numeric_input_layer_params or default_parameters.numeric_input_layer_params
        embed_layer_params = embed_layer_params or default_parameters.embed_layer_params

        # initialise numeric input layer
        self.numeric_input = NumericInput(num_features=numeric_dim,
                                          **numeric_input_layer_params,
                                          device=self.device)

        # initialise embedding layers
        self.embedder = MultiColumnEmbedding(num_embeddings=num_embeddings,
                                             embedding_dims=embedding_dims,
                                             **embed_layer_params,
                                             device=self.device)

        # output dimension is equal to numeric dimension plus embedding dimension
        self.out_features = self.numeric_input.num_features + self.embedder.out_features

        # move module to device
        self.to(self.device)

    def forward(self, X_num, X_cat):
        """
        Executes forward pass.

        Embeds categorical features and concatenates with numeric features.

        Parameters
        ----------

        X_num: torch.FloatTensor, shape (n_rows, self.numeric_dim)
            Numeric features.
            If self.numeric_dim = 0, must be empty tensor.

        X_cat: torch.LongTensor, shape (n_rows, len(self.num_embeddings))
            Categorical features.
            The columns must correspond to the column names specified in
            self.num_embeddings, and the columns must be ordered according
            to the alphabetical order of their names.
            If self.num_embeddings is None, must be empty tensor.

        Returns
        -------

        H: torch.FloatTensor, shape (n_rows, self.out_features)
            Concatenated numeric features and embeddings.
            H[i] is the concatenation of X_num[i] and the
            embeddings of X_cat[i].
        """

        X_num = self.numeric_input(X_num)
        X_embed = self.embedder(X_cat)
        H = torch.cat([X_num, X_embed], dim=1)

        return H


class MultiTableNumericInput(nn.Module):
    """
    Input layer for numeric features of multiple tables.

    - Option to apply batch norm > dropout to each set of features.

    Parameters
    ----------

    num_features: list
        Number of numeric features in each input table.
        len(num_features) must be equal to the number of tables,
        where num_features[i] is the number of numeric features in the i-th table.
        e.g. num_features = [10, 20, 30, 40]

    batch_norm: bool, optional (default=False)
        If True, numeric inputs will be batch normalised.

    dropout_rate: float, optional (default=None)
        If not None, dropout layer with specified dropout rate will
        be applied to numeric inputs.

    device: torch.device or string, optional (default=None)
        Device on which module will be placed.
        If string, must specify the device e.g. 'cpu' or 'cuda'.
        If None, GPU will be used if available, else CPU.

    Examples
    --------

    >>> import torch
    >>> from scorch.nn import MultiTableNumericInput
    >>> num_rows = [5, 8, 4]
    >>> numeric_dims = [3, 2, 0]
    >>> X0 = torch.rand(num_rows[0], numeric_dims[0], dtype=torch.float)
    >>> X1 = torch.rand(num_rows[1], numeric_dims[1], dtype=torch.float)
    >>> X2 = torch.FloatTensor()
    >>> Xs = [X0, X1, X2]
    >>> input_layer = MultiTableNumericInput(numeric_dims,
    >>>                                      batch_norm=True,
    >>>                                      dropout_rate=0.5,
    >>>                                      device='cpu')
    >>> Hs = input_layer(Xs)
    >>> print(Hs[0].shape, Hs[1].shape, Hs[2].shape)
    torch.Size([5, 3]) torch.Size([8, 2]) torch.Size([0])
    """

    def __init__(self,
                 num_features,
                 batch_norm=False,
                 dropout_rate=None,
                 device=None):

        super(MultiTableNumericInput, self).__init__()

        # initialise some instance variables
        self.num_features = num_features
        self.device = get_device(device)

        # initialise layers
        self.layers = nn.ModuleList()
        for d in num_features:
            self.layers.append(NumericInput(num_features=d,
                                            batch_norm=batch_norm,
                                            dropout_rate=dropout_rate,
                                            device=self.device))

        # move module to device
        self.to(self.device)

    def forward(self, Xs):
        """
        Executes the forward pass.

        Parameters
        -------

        Xs: list of torch.FloatTensor.
            Xs[i] contains the numeric features of the i-th table.
            Xs[i] has shape (ni, self.num_features[i]) and
            len(Xs) = len(self.num_features).

        Returns
        -------

        Hs: list of torch.FloatTensor.
            Input tensors after having possibly been passed through
            batch norm and dropout layers.
        """

        Hs = [layer(X) for X, layer in zip(Xs, self.layers)]

        return Hs


class MultiTableInput(nn.Module):
    """
    Input layer for both numeric and categorical features of multiple tables.

    For each table, embeds the categorical features and concatenates them
    with the numeric features of the same table.

    Parameters
    ----------

    numeric_dims: list
        Number of numeric features in each input table.
        len(numeric_dims) must be equal to the number of unique indices in dag,
        where numeric_dims[i] is the number of numeric features in the i-th table.
        e.g. numeric_dims = [10, 20, 30, 40]

    num_embeddings: list of dicts
        For each table, the size of the dictionary of embeddings for each
        categorical feature in the table.
        Each dict is of the form {feature name (str): dictionary size (int)}.
        len(num_embeddings) must be equal to the number of unique indices in dag,
        where num_embeddings[i] defines the number of embedding for each
        categorical feature in the i-th table.
        If a table does not contain any categorical features,
        use an empty dictionary.
        e.g. [{'cat1': 10, 'cat2': 20}, {}, {'cat3':15}, {}]
        defines the number of embeddings for features 'cat1' and 'cat2' of table 0
        and feature 'cat3' of table 2, where tables 1 and 3 do not have any
        categorical features.

    embedding_dims: list of dicts
        For each table, the dimension of the embedding vectors for each
        categorical feature in the table.
        Each dict is of the form {feature name (str): embedding dim (int)}.
        len(embedding_dims) must be equal to the number of unique indices in dag,
        where embedding_dims[i] defines the embedding dimension for each
        categorical feature in the i-th table.
        If a table does not contain any categorical features,
        use an empty dictionary.
        e.g. [{'cat1': 16, 'cat2': 8}, {}, {'cat3': 32}, {}]
        defines the embedding dimension for features 'cat1' and 'cat2' of table 0
        and feature 'cat3' of table 2, where tables 1 and 3 do not have any
        categorical features.

    share_embeds: bool, optional (default=False)
        If True, then if different tables contain the same feature name
        (as defined in num_embeddings and embedding_dims), both tables
        will share the same embedding layer.
        If False, different embedding layer will be created for each table
        even if the names match.

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

    device: torch.device or string, optional (default=None)
        Device on which module will be placed.
        If string, must specify the device e.g. 'cpu' or 'cuda'.
        If None, GPU will be used if available, else CPU.

    Examples
    --------

    >>> import torch
    >>> from scorch.nn import MultiTableInput
    >>> num_rows = [5, 8, 4]
    >>> numeric_dims = [3, 2, 0]
    >>> num_embeddings = [{'cat1': 5, 'cat2': 10}, {}, {'cat3': 8}]
    >>> embedding_dims = [{'cat1': 4, 'cat2': 8}, {}, {'cat3': 4}]
    >>> X0_num = torch.rand(num_rows[0], numeric_dims[0], dtype=torch.float)
    >>> X0_cat = torch.cat([torch.randint(0, num_embeddings[0]['cat1'], (num_rows[0], 1)),
    >>>                     torch.randint(0, num_embeddings[0]['cat2'], (num_rows[0], 1))], dim=1)
    >>> X1_num = torch.rand(num_rows[1], numeric_dims[1], dtype=torch.float)
    >>> X1_cat = torch.LongTensor([])
    >>> X2_num = torch.FloatTensor()
    >>> X2_cat = torch.randint(0, num_embeddings[2]['cat3'], (num_rows[2], 1))
    >>> Xs_num = [X0_num, X1_num, X2_num]
    >>> Xs_cat = [X0_cat, X1_cat, X2_cat]
    >>> input_layer = MultiTableInput(numeric_dims,
    >>>                               num_embeddings,
    >>>                               embedding_dims,
    >>>                               numeric_input_layer_params={'batch_norm': True},
    >>>                               embed_layer_params={'dropout_rate': 0.5},
    >>>                               device='cpu')
    >>> Hs = input_layer(Xs_num, Xs_cat)
    >>> print(Hs[0].shape, Hs[1].shape, Hs[2].shape)
    torch.Size([5, 15]) torch.Size([8, 2]) torch.Size([4, 4])
    """

    def __init__(self,
                 numeric_dims,
                 num_embeddings,
                 embedding_dims,
                 share_embeds=False,
                 numeric_input_layer_params=None,
                 embed_layer_params=None,
                 device=None):

        super(MultiTableInput, self).__init__()

        # initialise some instance variables
        self.numeric_dims = numeric_dims
        self.num_embeddings = num_embeddings
        self.embedding_dims = embedding_dims
        self.device = get_device(device)

        # get default layer parameters if None
        numeric_input_layer_params = numeric_input_layer_params or default_parameters.numeric_input_layer_params
        embed_layer_params = embed_layer_params or default_parameters.embed_layer_params

        # initialise numeric input layers
        self.numeric_input = MultiTableNumericInput(num_features=numeric_dims,
                                                    **numeric_input_layer_params,
                                                    device=self.device)

        # initialise categorical embedding layers
        if share_embeds:
            embed_module = MultiTableSharedEmbedding
        else:
            embed_module = MultiTableUnsharedEmbedding
        self.embedder = embed_module(num_embeddings=num_embeddings,
                                     embedding_dims=embedding_dims,
                                     **embed_layer_params,
                                     device=self.device)

        # total input dims are numeric dims plus embedding dims
        self.out_features = [
            i + j for i, j in zip(self.numeric_input.num_features, self.embedder.out_features)]

        # move module to device
        self.to(self.device)

    def forward(self,
                Xs_num,
                Xs_cat):
        """
        Executes the forward pass.

        For each table, concatenates the numeric features with the
        embeddings of the categorical features.

        Parameters
        -------

        Xs_num: list of torch.FloatTensor
            Xs[i] contains the numeric features of the i-th table.
            Xs[i] has shape (ni, self.num_features[i]) and
            len(Xs) = len(self.num_features).

        Xs_cat: list of torch.LongTensor
            Xs[i] contains the categorical features of the i-th table.
            The columns in Xs[i] must correspond to the column names
            specified in self.num_embeddings[i] and self.embedding_dims[i],
            and the columns must be ordered according to the alphabetical
            order of their names.
            Xs[i] has shape (ni, len(self.num_embeddings[i])).

        Returns
        -------

        Xs: list of torch.FloatTensor
            Xs[i] is the concatenation of Xs_num[i] and the
            categorical embedding of Xs_cat[i] and has shape
            (ni, self.out_features).
            len(Xs) = len(Xs_num) = len(Xs_cat).
        """

        # process numeric data
        Xs_num = self.numeric_input(Xs_num)

        # embed categorical data
        Xs_embed = self.embedder(Xs_cat)

        # concatenate numeric data and embeddings
        Xs = [torch.cat((X_num, X_embed), dim=1)
              for X_num, X_embed in zip(Xs_num, Xs_embed)]

        return Xs
