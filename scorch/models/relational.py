from scorch.models import Model
from scorch.nn import MultiTableInput, Transformer, RecursiveAttention, FeedForward
import scorch.utils.parameters as default_parameters
from scorch.utils.cuda import get_device
from scorch.utils.recursion import get_nodes


class RelationalNet(Model):
    """
    Neural network with nested attention layers for processing related data tables.

    - The tables are assumed to be indexed from 0 to k, with table 0 assumed to be the main table.

    - The relations are defined by a directed acyclic graph (DAG).

    - The network outputs a prediction for each row in the main table.

    - Input features in each table can be numeric, categorical or both.

    - Any categorical features are embedded and then concatenated with any numeric features of the same table.

    - The tables are then fed to a nested attention layer which outputs a single tensor with the same number of
      rows as the main table.

    - Data can then be further processed by one or more hidden layers.

    - Multiple output layers are possible for multi-task networks.

    - Can optionally return final hidden layer activations instead of network outputs.

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

    numeric_dims: list
        Number of numeric features in each input table.
        len(numeric_dims) must be equal to the number of unique indices in dag,
        where numeric_dims[i] is the number of numeric features in the i-th table.
        e.g. numeric_dims = [10, 20, 30, 40] for the dag above

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

    attention_dims: dict
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
        If greater than 1, model will be a multi-headed attention network.

    hidden_dims: list, optional (default=None)
        A dense layer will be added after attention layers for each element in the list,
        where hidden_dims[i] specifies the size of the i-th hidden layer.
        Elements in the list must either be integers or 'auto'.
        If 'auto', the size of the corresponding layer will be equal to the size of the previous layer.
        If None, the attention layer will be connected directly to the output layer.
        e.g. ['auto', 128, 64] will implement three hidden layer, the first having the same size as
        the output of the attention layer followed by two layers of size 128 and 64 respectively.

    output_dims: int or list of ints, optional (default=1)
        Number of outputs.
        If list, len(output_dims) output layers are implemented,
        where output_dims[i] is the size of the i-th output layer.
        e.g. [1, 10] will implement two output layers, one of size 1 and another of size 10.

    attention_mechanism: nn.Module, optional (default=scorch.nn.Transformer)
        The specific form of attention.
        Only current option is scorch.nn.Transformer.

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

    attention_layer_params: dict, optional (default=None)
        Named parameters for attention mechanism.
        Of the form {param name (str): param value}.
        Correspond to parameters of scorch.nn.MultiColumnEmbedding module.
        Correspond to parameters of module specified for attention_mechanism.
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

    >>> import numpy as np
    >>> import torch
    >>> import torch.nn.functional as F
    >>> from scorch.models import RelationalNet
    >>> from scorch.nn import Transformer
    >>> dag = {0: [1, 2], 1: [2]}
    >>> num_rows = [5, 8, 4]
    >>> numeric_dims = [3, 2, 0]
    >>> num_embeddings = [{'cat1': 5, 'cat2': 10}, {}, {'cat3': 8}]
    >>> embedding_dims = [{'cat1': 4, 'cat2': 8}, {}, {'cat3': 4}]
    >>> attention_dims = {
    >>>     (0, 1): {'query': 'max', 'value': 10},
    >>>     (0, 2): {'query': 10, 'value': 'same'},
    >>>     (1, 2): {'query': 20, 'value': 10},
    >>> }
    >>> X0 = dict(
    >>>     num=torch.rand(num_rows[0], numeric_dims[0], dtype=torch.float),
    >>>     cat=torch.cat([torch.randint(0, num_embeddings[0]['cat1'], (num_rows[0], 1)),
    >>>                    torch.randint(0, num_embeddings[0]['cat2'], (num_rows[0], 1))], dim=1)
    >>> )
    >>> X1 = dict(
    >>>     num=torch.rand(num_rows[1], numeric_dims[1], dtype=torch.float),
    >>>     cat=torch.LongTensor([])
    >>> )
    >>> X2 = dict(
    >>>     num=torch.FloatTensor(),
    >>>     cat=torch.randint(0, num_embeddings[2]['cat3'], (num_rows[2], 1))
    >>> )
    >>> Xs = {0: X0, 1: X1, 2: X2}
    >>> maps = {}
    >>> maps[(0, 1)] = np.random.randint(-1, num_rows[0], num_rows[1])
    >>> maps[(0, 2)] = np.random.randint(-1, num_rows[0], num_rows[2])
    >>> maps[(0, 1, 2)] = np.random.randint(-1, num_rows[1], num_rows[2])
    >>> data = {}
    >>> data['Xs'] = Xs
    >>> data['maps'] = maps
    >>> mdl = RelationalNet(dag,
    >>>                     numeric_dims,
    >>>                     num_embeddings,
    >>>                     embedding_dims,
    >>>                     attention_dims,
    >>>                     num_heads=1,
    >>>                     hidden_dims=[20, 20],
    >>>                     output_dims=1,
    >>>                     dense_layer_params={'activation': F.relu, 'dropout_rate': 0.5},
    >>>                     output_layer_params={'activation': torch.sigmoid},
    >>>                     attention_mechanism=Transformer,
    >>>                     device='cpu')
    >>> out = mdl(data, activate_output=True)
    >>> print(out)
    tensor([0.4834, 0.4488, 0.5953, 0.4284, 0.5227], grad_fn=<SigmoidBackward>)
    """

    def __init__(self,
                 dag,
                 numeric_dims,
                 num_embeddings,
                 embedding_dims,
                 attention_dims,
                 num_heads=1,
                 hidden_dims=None,
                 output_dims=1,
                 attention_mechanism=Transformer,
                 share_embeds=False,
                 numeric_input_layer_params=None,
                 embed_layer_params=None,
                 attention_layer_params=None,
                 dense_layer_params=None,
                 output_layer_params=None,
                 device=None):

        super(RelationalNet, self).__init__()

        # initialise some instance variables
        self.dag = dag
        self.table_indices = get_nodes(dag)
        self.n_tables = len(self.table_indices)
        self.numeric_dims = numeric_dims or [0] * self.n_tables
        self.num_embeddings = num_embeddings or [{}] * self.n_tables
        self.embedding_dims = embedding_dims or [{}] * self.n_tables
        self.attention_dims = attention_dims
        self.num_heads = num_heads
        self.hidden_dims = hidden_dims
        self.output_dims = output_dims
        self.device = get_device(device)

        # get default layer parameters if None
        numeric_input_layer_params = numeric_input_layer_params or default_parameters.numeric_input_layer_params
        embed_layer_params = embed_layer_params or default_parameters.embed_layer_params
        attention_layer_params = attention_layer_params or default_parameters.attention_layer_params
        dense_layer_params = dense_layer_params or default_parameters.dense_layer_params
        output_layer_params = output_layer_params or default_parameters.output_layer_params

        # initialise input layers
        self.input = MultiTableInput(numeric_dims=numeric_dims,
                                     num_embeddings=num_embeddings,
                                     embedding_dims=embedding_dims,
                                     share_embeds=share_embeds,
                                     numeric_input_layer_params=numeric_input_layer_params,
                                     embed_layer_params=embed_layer_params,
                                     device=self.device)

        # initialise attention layers
        self.attention = RecursiveAttention(dag=dag,
                                            input_dims=self.input.out_features,
                                            layer_dims=attention_dims,
                                            num_heads=num_heads,
                                            attention_mechanism=attention_mechanism,
                                            attention_layer_params=attention_layer_params,
                                            device=self.device)

        # initialise feed forward layers
        self.feed_forward = FeedForward(in_features=self.attention.out_features,
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
                'Xs': dict
                    Keys are table indices.
                    data['Xs'][i] is a dictionary containing the numeric
                    and categorical features of the i-th table, of the form
                    {'num': torch.FloatTensor, 'cat': torch.LongTensor}.
                    The columns in data['Xs'][i]['cat'] must correspond to the
                    column names specified in self.num_embeddings[i], and the
                    columns must be ordered according to the alphabetical order
                    of their names.
                    If table does not contain any numeric feature, use an empty torch.FloatTensor.
                    If table does not contain any categorical feature, use an empty torch.LongTensor.

                'maps': dict
                    Has the form {tuple: np.array}, where tuple defines a path in the DAG and the array
                    specifies the mapping between the rows of the pair of tensors corresponding to the final
                    two indices in the tuple.
                    e.g. {(0, 1, 2, 3): v} specifies the mapping between the rows of Xs[2] and Xs[3]
                    in the path 0 --> 1 --> 2 --> 3, where v has the same length as Xs[3] and
                    v[i] = j means that Xs[3][i] is related to Xs[2][j].
                    If v[i] < 0 then Xs[3][i] is not related to any row in Xs[i].

        activate_output: bool, optional (default=False)
            If True, activation function will be applied to outputs.

        return_activations: bool, optional (default=False)
            If True the activations from the final hidden layer will be returned
            instead of network outputs.

        Returns
        -------

        out: torch.Tensor
            If return_activations=False, returns network output for each row of main table,
            else returns final hidden layer activations.
            out[i] is the output or activation corresponding to
            data['Xs'][0]['num'][i] and data['Xs'][0]['cat'][i].
        """

        Xs = self.input([data['Xs'][i]['num'] for i in range(self.n_tables)],
                        [data['Xs'][i]['cat'] for i in range(self.n_tables)])

        H = self.attention(Xs, data['maps'])

        out = self.feed_forward(H, activate_output=activate_output,
                                return_activations=return_activations)

        return out
