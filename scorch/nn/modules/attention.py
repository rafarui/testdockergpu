import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from scorch.nn.modules.dense import Dense
import scorch.utils.parameters as default_parameters
from scorch.utils.recursion import get_sons
from scorch.utils.cuda import get_device


class Transformer(nn.Module):
    """
    Implementation of the Transformer layer, based on the implementation described in
    "Attention Is All You Need" (https://arxiv.org/pdf/1706.03762.pdf).

    Parameters
    ----------

    query_dim_in: int
        Input dimension of data which will be transformed to queries.

    query_dim_out: int
        Dimension of queries and keys.

    value_dim_in: int
        Input dimension of data which will be transformed to keys and values.

    value_dim_out: int
        Dimension of values.

    batch_norm: boolean, optional (default=False)
        If True, batch norm will be applied to queries, values and keys.

    activation: torch activation function, optional (default=None)
        If not None, activation function will be applied to queries, keys and values.

    dropout_rate: float, optional (default=None)
        If not None, dropout with specified dropout rate will be applied to queries, keys and values.

    weight_init: initialiser from torch.nn.init, optional (default=torch.nn.init.xavier_uniform_)
        Used to initialise the weights in the linear layers
        which compute the queries, values and keys.

    device: torch.device or string, optional (default=None)
        Device on which module will be placed.
        If string, must specify the device e.g. 'cpu' or 'cuda'.
        If None, GPU will be used if available, else CPU.

    Examples
    --------

    >>> import numpy as np
    >>> import torch
    >>> from scorch.nn import Transformer
    >>> X1 = torch.rand(5, 4)
    >>> X2 = torch.rand(20, 6)
    >>> row_map = np.random.randint(-1, len(X1), len(X2))
    >>> transformer = Transformer(query_dim_in=X1.shape[1],
    >>>                           query_dim_out=8,
    >>>                           value_dim_in=X2.shape[1],
    >>>                           value_dim_out=10,
    >>>                           device='cpu')
    >>> C = transformer(X1, X2, row_map)
    >>> print(C.shape)
    torch.Size([5, 10])
    """

    def __init__(self,
                 query_dim_in,
                 query_dim_out,
                 value_dim_in,
                 value_dim_out,
                 batch_norm=False,
                 activation=None,
                 dropout_rate=None,
                 weight_init=nn.init.xavier_uniform_,
                 device=None):

        super(Transformer, self).__init__()

        # initialise some instance variables
        self.query_dim_in = query_dim_in
        self.query_dim_out = query_dim_out
        self.value_dim_in = value_dim_in
        self.value_dim_out = value_dim_out
        self.device = get_device(device)

        # initialise query dense layer
        self.dense_query = Dense(in_features=query_dim_in,
                                 out_features=query_dim_out,
                                 batch_norm=batch_norm,
                                 activation=activation,
                                 dropout_rate=dropout_rate,
                                 weight_init=weight_init,
                                 device=self.device)

        # initialise key dense layer
        self.dense_key = Dense(in_features=value_dim_in,
                               out_features=query_dim_out,
                               batch_norm=batch_norm,
                               activation=activation,
                               dropout_rate=dropout_rate,
                               weight_init=weight_init,
                               device=self.device)

        # initialise value dense layer
        self.dense_value = Dense(in_features=value_dim_in,
                                 out_features=value_dim_out,
                                 batch_norm=batch_norm,
                                 activation=activation,
                                 dropout_rate=dropout_rate,
                                 weight_init=weight_init,
                                 device=self.device)

        # move module to device
        self.to(self.device)

    def forward(self,
                X1,
                X2,
                row_map):
        """
        Executes Transformer forward pass.

        Parameters
        ----------

        X1: torch.FloatTensor, shape (n_queries, self.query_dim_in)
            Data which will be transformed to queries.

        X2: torch.FloatTensor, shape (n_keys, self.value_dim_in)
            Data which will be transformed to keys and values.

        row_map: np.array, shape (n_keys,)
            Specifies the mapping between rows of X1 and X2,
            that is, between queries and keys.
            row_map[j] = i means that X2[j] is related to X1[i].
            If row_map[j] < 0 then X2[j] is not related to any row in X1.

        Returns
        -------

        C: torch.FloatTensor, shape (n_queries, self.value_dim_out)
            Attention weighted sums for rows in X1.
            C[i] is the weighted average of the values, computed from X2,
            which are related to X1[i].
            If X1[i] is not related to any rows in X2[i],
            C[i] will be a row of zeros.
        """

        # initialise the output tensor with zeros
        n = len(X1)
        m = self.value_dim_out
        C = torch.zeros(n, m, dtype=torch.float, device=self.device)

        # get rows of X1 which are relate to some row in X2 and
        # get rows of X2 which are related to some row in X1 and
        # re-index the row map
        X1_rows, X2_rows, row_map = self.get_used_rows(row_map)

        if (len(X1_rows) == 0) | (len(X2_rows) == 0):
            # return zero tensor
            return C

        else:
            # filter rows
            X1 = X1[X1_rows]
            X2 = X2[X2_rows]

            # get queries, values and keys
            Q = self.dense_query(X1)
            K = self.dense_key(X2)
            V = self.dense_value(X2)

            # compute the scaled dot product of every row in Q with every row in K
            # the result will be a tensor S of shape (len(Q), len(K)),
            # where S[i, j] is the dot product of Q[i] and K[j]
            S = Q.mm(K.t()) / np.sqrt(Q.size(1))

            # the dot product of Q[i] and K[j] is only relevant if row_map[j] = i, that is,
            # only weights S[i, j] for [row_map[1], 1], ..., [row_map[len(row_map)], len(row_map)]
            # are relevant
            # to ignore the non-relevant values when the softmax is applied,
            # set them to a large negative number
            cols = np.arange(len(row_map))
            S_masked = -1000 * torch.ones_like(S).to(self.device)
            S_masked[row_map, cols] = S[row_map, cols]

            # normalise weights with softmax
            W = F.softmax(S_masked, dim=1)

            # compute the weighted sums
            C[X1_rows] = W.mm(V)

            return C

    @staticmethod
    def get_used_rows(row_map):
        """
        Gets the rows of X1 and X2 (from the forward pass)
        which are actually going to be used in the attention calculation.
        That is, we discard rows of X1 which are not related to
        and rows in X2 and discard rows of X2 which are not related
        to any rows in X1, as defined by the row map.

        Parameters
        ----------

        row_map: np.array, shape (n_keys,)
            Specifies the mapping between rows of X1 and X2,
            that is, between queries and keys.
            row_map[j] = i means that X2[j] is related to X1[i].
            If row_map[j] < 0 then X2[j] is not related to any row in X1.

        Returns
        -------

        X1_rows: np.array
            Specifies which rows of X1 will be used in the forward pass.

        X2_rows: np.array
            Specifies which rows of X2 will be used in the forward pass.

        new_row_map: np.array
            Re-indexed row map.
        """

        # get rows of X2 which are related to some row in X1
        X2_rows = np.where(row_map >= 0)[0]

        # get relevant rows of row map
        row_map = row_map[X2_rows]

        if len(row_map) > 0:

            # get rows of X1 which are related to some row in X2
            X1_rows = np.unique(row_map)

            # re-index row map
            index_map = dict(zip(X1_rows, range(len(X1_rows))))
            new_row_map = np.array([index_map[i] for i in row_map])

        else:
            X1_rows = np.array([])
            new_row_map = np.array([])

        return X1_rows, X2_rows, new_row_map


class MultiHeadedAttention(nn.Module):
    """
    Multi-headed attention.

    !!! Not currently implemented correctly !!!

    Currently single-headed attention.

    Parameters
    ----------

    query_dim_in: int
        Input dimension of data which will be transformed to queries.

    query_dim_out: int
        Dimension of queries and keys.

    value_dim_in: int
        Input dimension of data which will be transformed to keys and values.

    value_dim_out: int
        Dimension of values.

    num_heads: int, optional (default=1)
        Number of attention heads.
        !!! This currently doesn't do anything !!!

    attention_mechanism: nn.Module, optional (default=scorch.nn.Transformer)
        The specific form of attention.
        Only current option is scorch.nn.Transformer.

    attention_layer_params: dict, optional (default=None)
        Named parameters for attention mechanism.
        Of the form {param name (str): param value}.
        Correspond to parameters of scorch.nn.MultiColumnEmbedding module.
        Correspond to parameters of module specified for attention_mechanism.
        If None, default parameters from scorch.utils.parameters will be used.

    device: torch.device or string, optional (default=None)
        Device on which module will be placed.
        If string, must specify the device e.g. 'cpu' or 'cuda'.
        If None, GPU will be used if available, else CPU.
    """

    def __init__(self,
                 query_dim_in,
                 query_dim_out,
                 value_dim_in,
                 value_dim_out,
                 num_heads=1,
                 attention_mechanism=Transformer,
                 attention_layer_params=None,
                 device=None):

        super(MultiHeadedAttention, self).__init__()

        # initialise some instance variables
        self.query_dim_in = query_dim_in
        self.query_dim_out = query_dim_out
        self.value_dim_in = value_dim_in
        self.value_dim_out = value_dim_out
        self.out_features = value_dim_out
        self.num_heads = num_heads
        self.device = get_device(device)

        # get default layer parameters if None
        attention_layer_params = attention_layer_params or default_parameters.attention_layer_params

        # initialise layers (attention mechanism with a single head)
        self.layer = attention_mechanism(query_dim_in=query_dim_in,
                                         query_dim_out=query_dim_out,
                                         value_dim_in=value_dim_in,
                                         value_dim_out=value_dim_out,
                                         **attention_layer_params,
                                         device=self.device)

        # move module to device
        self.to(self.device)

    def forward(self,
                X1,
                X2,
                row_map):
        """
        Executes multi-headed attention forward pass.

        !!! Not currently implemented correctly !!!

        Currently single-headed attention.

        Parameters
        ----------

        X1: torch.FloatTensor, shape (n_queries, self.query_dim_in)
            Data which will be transformed to queries.

        X2: torch.FloatTensor, shape (n_keys, self.value_dim_in)
            Data which will be transformed to keys and values.

        row_map: np.array, shape (n_keys,)
            Specifies the mapping between rows of X1 and X2,
            that is, between queries and keys.
            row_map[j] = i means that X2[j] is related to X1[i].
            If row_map[j] < 0 then X2[j] is not related to any row in X1.

        Returns
        -------

        C: torch.FloatTensor, shape (n_queries, self.value_dim_out)
            Attention weighted sums for rows in X1.
            C[i] is the weighted average of the values, computed from X2,
            which are related to X1[i].
            If X1[i] is not related to any rows in X2[i],
            C[i] will be a row of zeros.
        """

        C = self.layer(X1, X2, row_map)

        return C


class RecursiveAttention(nn.Module):
    """
    Recursive attention for related tensors.

    - The tensors are assumed to be indexed from 0 to k, with table 0 assumed to be the main tensors.

    - The relations are defined by a directed acyclic graph (DAG).

    - The module returns the concatenation of the main tensor with the attention components of its children.

    - The result is a single tensor with the same number of rows as the main tensor but greater dimension.

    Parameters
    ----------

    dag: dict
        Defines relationships between tables in a directed acyclic graph format.
        Each element is of the form {table index (int): [indices of child tables]}.
        e.g. dag = {
                    0: [1, 2, 3],
                    1: [2, 3],
                    2: [3],
                }
        means that tables 1, 2 and 3 are children of table 0,
        tables 2 and 3 are children of table 1
        and table 3 is a child of table 2.

    input_dims: list
        Number of features in each input tensor.
        len(input_dims) must be equal to the number of unique indices in dag,
        where input_dims[i] is the number of features in the i-th table.
        e.g. input_dims = [10, 20, 30, 40] for the dag above

    layer_dims: dict
        The dimension of the queries and keys in each attention layer.
        Must have an entry for each parent-child relationship in dag.
        Each element is of the form
        {(index of parent table (int), index of child table (int)):
                    {'query': query dimension, 'value': value dimension}}
        The query dimensions can either be integers or 'mean', 'max', 'min'.
            - If 'mean', query dimension will be set equal to the mean of the parent and child dimensions.
            - If 'max', query dimension will be set equal to the max of the parent and child dimensions.
            - If 'min', query dimension will be set equal to the min of the parent and child dimensions.
        The value dimensions can either be integers or 'same'.
            - If 'same', value dimension will be set equal to child dimension.
        e.g. for the dag above we could have
                attention_dims = {
                        (0, 1): {'query': 'max', 'value': 10},
                        (0, 2): {'query': 10, 'value': 'same'},
                        (0, 3): {'query': 'mean', 'value': 15},
                        (1, 2): {'query': 20, 'value': 10},
                        (1, 3): {'query': 10, 'value': 5},
                        (2, 3): {'query': 'min', 'value': 'same'},
                    }

    num_heads: int, optional (default=1)
        Number of attention heads.
        !!! This currently doesn't do anything !!!

    attention_mechanism: nn.Module, optional (default=scorch.nn.Transformer)
        The specific form of attention.
        Only current option is scorch.nn.Transformer.

    attention_layer_params: dict, optional (default=None)
        Named parameters for attention mechanism.
        Of the form {param name (str): param value}.
        Correspond to parameters of scorch.nn.MultiColumnEmbedding module.
        Correspond to parameters of module specified for attention_mechanism.
        If None, default parameters from scorch.utils.parameters will be used.

    device: torch.device or string, optional (default=None)
        Device on which module will be placed.
        If string, must specify the device e.g. 'cpu' or 'cuda'.
        If None, GPU will be used if available, else CPU.

    Examples
    --------

    >>> import numpy as np
    >>> import torch
    >>> from scorch.nn import Transformer, RecursiveAttention
    >>> dag = {0: [1, 2], 1: [2]}
    >>> num_rows = [5, 8, 10]
    >>> input_dims = [3, 2, 5]
    >>> layer_dims = {
    >>>     (0, 1): {'query': 'max', 'value': 10},
    >>>     (0, 2): {'query': 10, 'value': 8},
    >>>     (1, 2): {'query': 20, 'value': 'same'},
    >>> }
    >>> X0 = torch.rand(num_rows[0], input_dims[0], dtype=torch.float)
    >>> X1 = torch.rand(num_rows[1], input_dims[1], dtype=torch.float)
    >>> X2 = torch.rand(num_rows[2], input_dims[2], dtype=torch.float)
    >>> Xs = [X0, X1, X2]
    >>> row_maps = {}
    >>> row_maps[(0, 1)] = np.random.randint(-1, num_rows[0], num_rows[1])
    >>> row_maps[(0, 2)] = np.random.randint(-1, num_rows[0], num_rows[2])
    >>> row_maps[(0, 1, 2)] = np.random.randint(-1, num_rows[1], num_rows[2])
    >>> attention = RecursiveAttention(dag,
    >>>                                input_dims,
    >>>                                layer_dims,
    >>>                                attention_mechanism=Transformer,
    >>>                                device='cpu')
    >>> H = attention(Xs, row_maps)
    >>> print(H.shape)
    torch.Size([5, 21])
    """

    def __init__(self,
                 dag,
                 input_dims,
                 layer_dims,
                 num_heads=1,
                 attention_mechanism=Transformer,
                 attention_layer_params=None,
                 device=None):

        super(RecursiveAttention, self).__init__()

        # initialise some instance variables
        self.dag = dag
        self.input_dims = input_dims
        self.layer_dims = layer_dims
        self.num_heads = num_heads
        self.device = get_device(device)

        # get default layer parameters if None
        attention_layer_params = attention_layer_params or default_parameters.attention_layer_params

        # initialise layers
        self.layers = nn.ModuleDict()
        for relation, dims in layer_dims.items():

            # get dimension of parent
            dp = input_dims[relation[0]]

            # get dimension of child
            dc = self.node_dim(relation[1])

            # get query/key dim
            dq = self.query_key_dim(dp, dc, dims['query'])

            # get value dim
            dv = self.value_dim(dc, dims['value'])

            # get layer label
            label = self.layer_label(relation[0], relation[1])

            # initialise layer
            self.layers[label] = MultiHeadedAttention(dp, dq, dc, dv, num_heads,
                                                      attention_mechanism,
                                                      attention_layer_params,
                                                      self.device)

        # compute output dimension
        self.out_features = self.node_dim(0)

        # move module to device
        self.to(self.device)

    def node_dim(self, node):
        """
        Computes the dimension of the specified node in the DAG after concatenation with
        all the attention components of its children.

        Parameters
        ----------

        node: int
            Index of the node in the DAG.

        Returns
        -------

        d: int
            Dimension of the node.
        """

        # get dimension of concatenated attention components
        sons = get_sons(self.dag, node)
        attention_dim = sum(
            [
                self.value_dim(self.node_dim(son),
                               self.layer_dims[(node, son)]['value']) for son in sons
            ]
        )

        # add node dimension
        d = self.input_dims[node] + attention_dim

        return d

    @staticmethod
    def query_key_dim(dp, dc, dq):
        """
        Gets the dimension of queries/keys.


        The value dimensions can either be integers or 'same'.


        Parameters
        ----------

        dp: int
            Dimension of parent.

        dc: int
            Dimension of child.

        dq: int or string
            Dimension of queries/keys.
            If string, must be one of 'mean', 'max' or 'min'.
                - If 'mean', output will be set equal to the mean of dp and dc.
                - If 'max', output will be set equal to the max of dp and dc.
                - If 'min', output will be set equal to the min of dp and dc.

        Returns
        -------

        dq: int
            Dimension of queries/keys.
        """

        if dq == 'mean':
            dq = int(np.ceil((dp + dc) / 2))
        elif dq == 'max':
            dq = max(dp, dc)
        elif dq == 'min':
            dq = min(dp, dc)

        return dq

    @staticmethod
    def value_dim(dc, dv):
        """
        Gets the dimension of values.

        Parameters
        ----------

        dc: int
            Dimension of child.

        dv: int or string
            Dimension of values
            If string, must be 'same'.
                - If 'same', output will be set equal to dc.

        Returns
        -------

        dv: int
            Dimension of values
        """

        if dv == 'same':
            dv = dc

        return dv

    @staticmethod
    def layer_label(parent, child):
        """
        Constructs attention layer label.

        It is the concatenation of the parent and child table indices,
        separated by an underscore.

        Parameters
        ----------

        parent: int
            Parent table index.

        child: int
            Child table index

        Returns
        -------

        label: str
            Attention layer label.
        """

        label = str(parent) + '_' + str(child)

        return label

    def forward(self, Xs, row_maps):
        """
        Recursive attention forward pass.

        Parameters
        ----------

        Xs: list of torch.FloatTensor
            A tensor for each node in the DAG.
            Xs[i] is the tensor corresponding to node represented by i in self.dag.

        row_maps: dict
            Has the form {tuple: np.array}, where tuple defines a path in the DAG and the array
            specifies the mapping between the rows of the pair of tensors corresponding to the final
            two indices in the tuple.
            e.g. {(0, 1, 2, 3): v} specifies the mapping between the rows of Xs[2] and Xs[3]
            in the path 0 --> 1 --> 2 --> 3, where v has the same length as Xs[3] and
            v[i] = j means that Xs[3][i] is related to Xs[2][j].
            If v[i] < 0 then Xs[3][i] is not related to any row in Xs[i].

        Returns
        -------

        H: torch.FloatTensor
            Xs[0] concatenated with all its attention components.
        """

        H = self.recursive_attention(Xs, row_maps, (0,))

        return H

    def recursive_attention(self, Xs, row_maps, trace):
        """
        Recursively executes all nested attention layers.

        Parameters
        ----------

        Xs: list of torch.FloatTensor
            A tensor for each node in the DAG.
            Xs[i] is the tensor corresponding to node represented by i in self.dag.

        row_maps: dict
            Has the form {tuple: np.array}, where tuple defines a path in the DAG and the array
            specifies the mapping between the rows of the pair of tensors corresponding to the final
            two indices in the tuple.
            e.g. {(0, 1, 2, 3): v} specifies the mapping between the rows of Xs[2] and Xs[3]
            in the path 0 --> 1 --> 2 --> 3, where v has the same length as Xs[3] and
            v[i] = j means that Xs[3][i] is related to Xs[2][j].
            If v[i] < 0 then Xs[3][i] is not related to any row in Xs[i].

        trace: tuple
            Specifies the current path in the DAG.
            e.g. (0, 1, 2, 3) specifies the path 0 --> 1 --> 2 --> 3.

        Returns
        -------

        H: torch.FloatTensor
            The concatenation of the node with all its attention components,
            or the original node if it has no attention components.
        """

        node = trace[-1]
        if get_sons(self.dag, node):
            # recursively execute all nested attention layers for this node and concatenate with node
            H = torch.cat(
                (Xs[node],
                 *[self.layers[self.layer_label(node, son)](Xs[node],
                                                            self.recursive_attention(Xs, row_maps, trace + (son,)),
                                                            row_maps[trace + (son,)])
                   for son in get_sons(self.dag, node)]), dim=1)
            return H

        else:
            # return the node
            H = Xs[node]
            return H
