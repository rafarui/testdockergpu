import torch
import torch.nn as nn

from scorch.nn.modules.utils import PostLinear
from scorch.utils.cuda import get_device


class Embedding(nn.Module):
    """
    Builds on torch.nn.Embedding.

    See https://pytorch.org/docs/stable/nn.html#embedding.

    Embeds a single categorical feature with the following extra functionality:

    - Option to apply batch norm > activation > dropout after embedding layer.

    Parameters
    ----------

    num_embeddings: int
        Size of the dictionary of embeddings.

    embedding_dim: int
        The size of each embedding vector.

    padding_idx: int, optional (default=None)
        If not None, an embedding vector corresponding to this int
        is initialised with zeros.
        Can be used to essentially ignore a specific value which has
        been used to pad variable length sequences.

    batch_norm: bool, optional (default=False)
        If True, batch norm layer will be added after embedding layer
        in the order defined above.

    activation: torch activation function, optional (default=None)
        If not None, activation function will be added after embedding layer
        in the order defined above.

    dropout_rate: float, optional (default=None)
        If not None, dropout layer with specified dropout rate will be added
        after embedding layer in the order defined above.

    weight_init: initialiser from torch.nn.init, optional (default=torch.nn.init.xavier_uniform_)
        Used to initialise the weights in the embedding layer.

    device: torch.device or string, optional (default=None)
        Device on which module will be placed.
        If string, must specify the device e.g. 'cpu' or 'cuda'.
        If None, GPU will be used if available, else CPU.

    Examples
    --------

    >>> import torch
    >>> import torch.nn.functional as F
    >>> from scorch.nn import Embedding
    >>> num_rows = 5
    >>> num_embeddings = 16
    >>> embedding_dim = 8
    >>> X = torch.randint(0, num_embeddings, (num_rows,))
    >>> embedder = Embedding(num_embeddings,
    >>>                      embedding_dim,
    >>>                      batch_norm=True,
    >>>                      activation=F.relu,
    >>>                      dropout_rate=0.5,
    >>>                      device='cpu')
    >>> E = embedder(X)
    >>> print(E.shape)
    torch.Size([5, 8])
    """

    def __init__(self,
                 num_embeddings,
                 embedding_dim,
                 padding_idx=None,
                 batch_norm=False,
                 activation=None,
                 dropout_rate=None,
                 weight_init=nn.init.xavier_uniform_,
                 device=None):

        super(Embedding, self).__init__()

        # initialise some instance variables
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.device = get_device(device)

        # initialise embedding layer
        self.embedder = nn.Embedding(num_embeddings=num_embeddings,
                                     embedding_dim=embedding_dim,
                                     padding_idx=padding_idx)
        weight_init(self.embedder.weight)

        # initialise post-embedding layer
        self.post_embed = PostLinear(in_features=embedding_dim,
                                      batch_norm=batch_norm,
                                      activation=activation,
                                      dropout_rate=dropout_rate,
                                      device=self.device)

        # move module to device
        self.to(self.device)

    def forward(self, X):
        """
        Executes forward pass.

        Applies embedding > batch norm > activation > dropout.

        If batch norm or activation or dropout is None, will be missed out.

        Parameters
        ----------

        X: torch.LongTensor, shape (n_rows,)
            Categorical feature.

        Returns
        -------

        A: torch.FloatTensor, shape (n_rows, self.embedding_dim)
            Embeddings, where A[i] is the embedding vector for X[i].
        """

        if len(X) == 0:
            return torch.tensor([], dtype=torch.float, device=self.device)

        H = self.embedder(X)
        A = self.post_embed(H)

        return A


class MultiColumnEmbedding(nn.Module):
    """
    Implements a set of scorch.nn.Embedding layers for multiple categorical features.

    - Option to apply batch norm > activation > dropout after each embedding layer.

    Parameters
    ----------

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

    batch_norm: bool, optional (default=False)
        If True, batch norm layer will be added after embedding layers
        in the order defined above.

    activation: torch activation function, optional (default=None)
        If not None, activation function will be added after embedding layers
        in the order defined above.

    dropout_rate: float, optional (default=None)
        If not None, dropout layer with specified dropout rate will be added
        after embedding layers in the order defined above.

    weight_init: initialiser from torch.nn.init, optional (default=torch.nn.init.xavier_uniform_)
        Used to initialise the weights in the embedding layers.

    device: torch.device or string, optional (default=None)
        Device on which module will be placed.
        If string, must specify the device e.g. 'cpu' or 'cuda'.
        If None, GPU will be used if available, else CPU.

    Examples
    --------

    >>> import torch
    >>> import torch.nn.functional as F
    >>> from scorch.nn import MultiColumnEmbedding
    >>> num_rows = 5
    >>> num_embeddings = {'cat_1': 5, 'cat_2': 10, 'cat_3': 7}
    >>> embedding_dims = {'cat_1': 2, 'cat_2': 4, 'cat_3': 6}
    >>> X = torch.cat([torch.randint(0, num_embeddings['cat_1'], (num_rows, 1)),
    >>>                torch.randint(0, num_embeddings['cat_2'], (num_rows, 1)),
    >>>                torch.randint(0, num_embeddings['cat_3'], (num_rows, 1))], dim=1)
    >>> embedder = MultiColumnEmbedding(num_embeddings,
    >>>                                 embedding_dims,
    >>>                                 batch_norm=True,
    >>>                                 activation=F.relu,
    >>>                                 dropout_rate=0.5,
    >>>                                 device='cpu')
    >>> E = embedder(X)
    >>> print(E.shape)
    torch.Size([5, 12])
    """

    def __init__(self,
                 num_embeddings,
                 embedding_dims,
                 batch_norm=False,
                 activation=None,
                 dropout_rate=None,
                 weight_init=nn.init.xavier_uniform_,
                 device=None):

        super(MultiColumnEmbedding, self).__init__()

        # initialise some instance variables
        self.num_embeddings = num_embeddings
        self.embedding_dims = embedding_dims
        self.device = get_device(device)

        # get a sorted list of feature names
        # this is the order that the forward pass expects
        self.cat_cols = sorted(list(num_embeddings.keys()))

        # initialise layers attributes
        self.out_features = 0
        self.layers = nn.ModuleList()

        # for each column...
        for col in self.cat_cols:

            # initialise embedding layer
            n, d = num_embeddings[col], embedding_dims[col]
            self.layers.append(Embedding(num_embeddings=n,
                                         embedding_dim=d,
                                         batch_norm=batch_norm,
                                         activation=activation,
                                         dropout_rate=dropout_rate,
                                         weight_init=weight_init,
                                         device=self.device))
            self.out_features += d

        # move module to device
        self.to(self.device)

    def forward(self, X):
        """
        For each layer, applies embedding > batch norm > activation > dropout.

        If batch norm or activation or dropout is None, will be missed out.

        Parameters
        ----------

        X: torch.LongTensor, shape (n_rows, len(self.num_embeddings))
            Categorical features.
            The columns must correspond to the column names specified in
            self.num_embeddings, and the columns must be ordered according
            to the alphabetical order of their names.

        Returns
        -------

        H: torch.FloatTensor, shape (n_rows, self.out_features)
            Layer embeddings, where H[i] is the concatenated embeddings of
            all columns in X[i].
        """

        if len(X) == 0:
            return torch.tensor([], dtype=torch.float, device=self.device)

        # initialise embedding tensor
        H = torch.zeros(len(X), self.out_features,
                        dtype=torch.float, device=self.device)

        # for each column...
        col_marker = 0
        for i, layer in enumerate(self.layers):

            # get embeddings and add to output tensor
            E = layer(X[:, i])
            d = E.shape[1]
            H[:, col_marker:col_marker + d] = E
            col_marker += d

        return H


class MultiTableUnsharedEmbedding(nn.Module):
    """
    Implements a set of unshared scorch.nn.Embedding layers for multiple tables.

    If columns in different tables have the same name they will each be allocated
    a different embedding layer.

    - Option to apply batch norm > activation > dropout after each embedding layer.

    Parameters
    ----------

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

    batch_norm: bool, optional (default=False)
        If True, batch norm layer will be added after embedding layers
        in the order defined above.

    activation: torch activation function, optional (default=None)
        If not None, activation function will be added after embedding layers
        in the order defined above.

    dropout_rate: float, optional (default=None)
        If not None, dropout layer with specified dropout rate will be added
        after embedding layers in the order defined above.

    weight_init: initialiser from torch.nn.init, optional (default=torch.nn.init.xavier_uniform_)
        Used to initialise the weights in the embedding layers.

    device: torch.device or string, optional (default=None)
        Device on which module will be placed.
        If string, must specify the device e.g. 'cpu' or 'cuda'.
        If None, GPU will be used if available, else CPU.

    Examples
    --------

    >>> import torch
    >>> import torch.nn.functional as F
    >>> from scorch.nn import MultiTableUnsharedEmbedding
    >>> num_rows = [5, 8, 6]
    >>> num_embeddings = [{'cat1': 5, 'cat2': 10}, {}, {'cat3': 8}]
    >>> embedding_dims = [{'cat1': 4, 'cat2': 8}, {}, {'cat3': 4}]
    >>> X0 = torch.cat([torch.randint(0, num_embeddings[0]['cat1'], (num_rows[0], 1)),
    >>>                 torch.randint(0, num_embeddings[0]['cat2'], (num_rows[0], 1))], dim=1)
    >>> X1 = torch.LongTensor([])
    >>> X2 = torch.randint(0, num_embeddings[2]['cat3'], (num_rows[2], 1))
    >>> Xs = [X0, X1, X2]
    >>> embedder = MultiTableUnsharedEmbedding(num_embeddings,
    >>>                                        embedding_dims,
    >>>                                        batch_norm=True,
    >>>                                        activation=F.relu,
    >>>                                        dropout_rate=0.5,
    >>>                                        device='cpu')
    >>> Es = embedder(Xs)
    >>> print(Es[0].shape, Es[1].shape, Es[2].shape)
    torch.Size([5, 12]) torch.Size([0]) torch.Size([6, 4])
    """

    def __init__(self,
                 num_embeddings,
                 embedding_dims,
                 batch_norm=False,
                 activation=None,
                 dropout_rate=None,
                 weight_init=nn.init.xavier_uniform_,
                 device=None):

        super(MultiTableUnsharedEmbedding, self).__init__()

        # initialise some instance variables
        self.num_embeddings = num_embeddings
        self.embedding_dims = embedding_dims
        self.device = get_device(device)

        # initialise embedder attributes
        self.layers = nn.ModuleList()
        self.out_features = []

        # for each table...
        for i in range(len(num_embeddings)):

            # initialise multi-embedding layer
            self.layers.append(MultiColumnEmbedding(num_embeddings=num_embeddings[i],
                                                    embedding_dims=embedding_dims[i],
                                                    batch_norm=batch_norm,
                                                    activation=activation,
                                                    dropout_rate=dropout_rate,
                                                    weight_init=weight_init,
                                                    device=self.device))

            self.out_features.append(self.layers[i].out_features)

        # move module to device
        self.to(self.device)

    def forward(self, Xs):
        """
        For each layer of each table, applies embedding > batch norm > activation > dropout.

        If batch norm or activation or dropout is None, will be missed out.

        Parameters
        ----------

        Xs: list of torch.LongTensor.
            Xs[i] contains the categorical features of the i-th table.
            The columns in Xs[i] must correspond to the column names
            specified in self.num_embeddings[i] and self.embedding_dims[i],
            and the columns must be ordered according to the alphabetical
            order of their names.
            Xs[i] has shape (ni, len(self.num_embeddings[i])).

        Returns
        -------

        Hs: list of torch.FloatTensor
            Hs[i] contains the concatenated embeddings of all column in Xs[i]
            and has shape (ni, self.out_features[i]).
        """

        Hs = [layer(X) for X, layer in zip(Xs, self.layers)]

        return Hs


class MultiTableSharedEmbedding(nn.Module):
    """
    Implements a set of shared scorch.nn.Embedding layers for multiple tables.

    If columns in different tables have the same name the will share the same embedding layer.

    Parameters
    ----------

    num_embeddings: list of dicts
        For each table, the size of the dictionary of embeddings for each
        categorical feature in the table.
        Each dict is of the form {feature name (str): dictionary size (int)}.
        len(num_embeddings) must be equal to the number of unique indices in dag,
        where num_embeddings[i] defines the number of embedding for each
        categorical feature in the i-th table.
        The number of embeddings must be the same for columns from different
        tables with the same name.
        If a table does not contain any categorical features,
        use an empty dictionary.
        e.g. [{'cat1': 10, 'cat2': 20}, {}, {'cat2':20}, {}]
        defines the number of embeddings for features 'cat1' and 'cat2' of table 0
        and feature 'cat3' of table 2, where tables 1 and 3 do not have any
        categorical features.
        In this case the the embedding layer for 'cat2' would be shared
        between tables 1 and 2.

    embedding_dims: list of dicts
        For each table, the dimension of the embedding vectors for each
        categorical feature in the table.
        Each dict is of the form {feature name (str): embedding dim (int)}.
        len(embedding_dims) must be equal to the number of unique indices in dag,
        where embedding_dims[i] defines the embedding dimension for each
        categorical feature in the i-th table.
        The dimension must be the same for columns from different
        tables with the same name.
        If a table does not contain any categorical features,
        use an empty dictionary.
        e.g. [{'cat1': 16, 'cat2': 8}, {}, {'cat2': 8}, {}]
        defines the embedding dimension for features 'cat1' and 'cat2' of table 0
        and feature 'cat3' of table 2, where tables 1 and 3 do not have any
        categorical features.
        In this case the the embedding layer for 'cat2' would be shared
        between tables 1 and 2.

    batch_norm: bool, optional (default=False)
        If True, batch norm layer will be added after embedding layers
        in the order defined above.

    activation: torch activation function, optional (default=None)
        If not None, activation function will be added after embedding layers
        in the order defined above.

    dropout_rate: float, optional (default=None)
        If not None, dropout layer with specified dropout rate will be added
        after embedding layers in the order defined above.

    weight_init: initialiser from torch.nn.init, optional (default=torch.nn.init.xavier_uniform_)
        Used to initialise the weights in the embedding layers.

    device: torch.device or string, optional (default=None)
        Device on which module will be placed.
        If string, must specify the device e.g. 'cpu' or 'cuda'.
        If None, GPU will be used if available, else CPU.

    Examples
    --------

    >>> import torch
    >>> import torch.nn.functional as F
    >>> from scorch.nn import MultiTableSharedEmbedding
    >>> num_rows = [5, 8, 6]
    >>> num_embeddings = [{'cat1': 5, 'cat2': 10}, {}, {'cat2': 10, 'cat3': 8}]
    >>> embedding_dims = [{'cat1': 4, 'cat2': 8}, {}, {'cat2': 8, 'cat3': 6}]
    >>> X0 = torch.cat([torch.randint(0, num_embeddings[0]['cat1'], (num_rows[0], 1)),
    >>>                 torch.randint(0, num_embeddings[0]['cat2'], (num_rows[0], 1))], dim=1)
    >>> X1 = torch.LongTensor([])
    >>> X2 = torch.cat([torch.randint(0, num_embeddings[2]['cat2'], (num_rows[2], 1)),
    >>>                 torch.randint(0, num_embeddings[2]['cat3'], (num_rows[2], 1))], dim=1)
    >>> Xs = [X0, X1, X2]
    >>> embedder = MultiTableSharedEmbedding(num_embeddings,
    >>>                                      embedding_dims,
    >>>                                      batch_norm=True,
    >>>                                      activation=F.relu,
    >>>                                      dropout_rate=0.5,
    >>>                                      device='cpu')
    >>> Es = embedder(Xs)
    >>> print(Es[0].shape, Es[1].shape, Es[2].shape)
    torch.Size([5, 12]) torch.Size([0, 0]) torch.Size([6, 14])
    """

    def __init__(self,
                 num_embeddings,
                 embedding_dims,
                 batch_norm=False,
                 activation=None,
                 dropout_rate=None,
                 weight_init=nn.init.xavier_uniform_,
                 device=None):

        super(MultiTableSharedEmbedding, self).__init__()

        # initialise some instance variables
        self.num_embeddings = num_embeddings
        self.embedding_dims = embedding_dims
        self.device = get_device(device)

        # get a sorted list of categorical columns in each table
        # this is the order that the forward pass expects
        self.table_cols = {i: sorted(list(d.keys()))
                           for i, d in enumerate(num_embeddings)}

        # get a list of all categorical columns in all tables
        all_cat_cols = list(set().union(*(x.keys() for x in num_embeddings)))

        # initialise layer attributes
        self.layers = nn.ModuleDict()
        self.layer_labels = {}
        self.out_features = [0] * len(num_embeddings)

        # for each categorical column...
        for col in all_cat_cols:

            # get indices of all tables which contain this column
            i_tables = [i for i, x in enumerate(num_embeddings) if col in x]

            # the number of embeddings and dimension of should be the same for all
            # columns with this name, so just use the first
            # TODO: assert this
            n, d = num_embeddings[i_tables[0]][col], embedding_dims[i_tables[0]][col]

            # construct label for this embedder
            # it is the concatenation of the table indices and the column name,
            # separated by underscores
            label = '_'.join([str(i) for i in i_tables] + [col])

            # initialise embedding layer
            self.layers[label] = Embedding(num_embeddings=n,
                                           embedding_dim=d,
                                           batch_norm=batch_norm,
                                           activation=activation,
                                           dropout_rate=dropout_rate,
                                           weight_init=weight_init,
                                           device=self.device)

            # add dimension to total embedding dimension for tables which contain this column
            self.out_features = [
                x + d if i in i_tables else x for i, x in enumerate(self.out_features)]

            # record layer label for this column
            self.layer_labels[col] = label

        # move module to device
        self.to(self.device)

    def forward(self, Xs):
        """
        For each layer of each table, applies embedding > batch norm > activation > dropout.

        If batch norm or activation or dropout is None, will be missed out.

        Parameters
        ----------

        Parameters
        ----------

        Xs: list of torch.LongTensor.
            Xs[i] contains the categorical features of the i-th table.
            The columns in Xs[i] must correspond to the column names
            specified in self.num_embeddings[i] and self.embedding_dims[i],
            and the columns must be ordered according to the alphabetical
            order of their names.
            Xs[i] has shape (ni, len(self.num_embeddings[i])).

        Returns
        -------

        Hs: list of torch.FloatTensor
            Hs[i] contains the concatenated embeddings of all column in Xs[i]
            and has shape (ni, self.out_features[i]).
        """

        Hs = []

        # for each table...
        for i, X in enumerate(Xs):

            # initialise embedding tensor
            n, k = len(X), self.out_features[i]
            H = torch.zeros(n, k, dtype=torch.float, device=self.device)

            if (n > 0):

                # for each column...
                col_marker = 0
                for j, col in enumerate(self.table_cols[i]):

                    # get embeddings and add to embedding tensor
                    layer = self.layers[self.layer_labels[col]]
                    d = self.embedding_dims[i][col]
                    H[:, col_marker:col_marker + d] = layer(X[:, j])
                    col_marker += d

            Hs.append(H)

        return Hs
