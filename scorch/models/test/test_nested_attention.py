import numpy as np
import torch

from scorch.models.nested_attention import *
from scorch.utils.data_loader import RelationalDataLoader

# Define toy tensors
min_rows = 4
max_rows = 5
min_cols = 2
max_cols = 3
X0 = torch.rand(np.random.randint(min_rows, max_rows),
                np.random.randint(min_cols, max_cols))
X1 = torch.rand(np.random.randint(min_rows, max_rows),
                np.random.randint(min_cols, max_cols))
X2 = torch.rand(np.random.randint(min_rows, max_rows),
                np.random.randint(min_cols, max_cols))
X3 = torch.rand(np.random.randint(min_rows, max_rows),
                np.random.randint(min_cols, max_cols))
X4 = torch.rand(np.random.randint(min_rows, max_rows),
                np.random.randint(min_cols, max_cols))
Xs = [X0, X1, X2, X3, X4]

# Define entity relations

dag = {
    0: [1, 2, ],
    1: [],
    2: [3, 4],
}

# Define row relations

np.random.seed(0)
row_relations = {}
for left_i, right_is in dag.items():
    for right_i in right_is:
        row_relations[(left_i, right_i)] = [np.random.randint(0, len(Xs[right_i]), np.random.randint(0, 5)).tolist()
                                            for _ in range(len(Xs[left_i]))]

# Define hidden layers dimensions

dims = [X.size(1) for X in Xs]

query_dims = {
    (0, 1): np.random.randint(10, 21),
    (0, 2): np.random.randint(10, 21),
    (0, 3): np.random.randint(10, 21),
    (0, 4): np.random.randint(10, 21),
    (1, 2): np.random.randint(10, 21),
    (1, 3): np.random.randint(10, 21),
    (2, 3): np.random.randint(10, 21),
    (2, 4): np.random.randint(10, 21),
    (3, 4): np.random.randint(10, 21),
}

value_dims = {
    (0, 1): np.random.randint(10, 21),
    (0, 2): np.random.randint(10, 21),
    (0, 3): np.random.randint(10, 21),
    (0, 4): np.random.randint(10, 21),
    (1, 2): np.random.randint(10, 21),
    (1, 3): np.random.randint(10, 21),
    (2, 3): np.random.randint(10, 21),
    (2, 4): np.random.randint(10, 21),
    (3, 4): np.random.randint(10, 21),
}

attention_dims = {
    k: {
        'query': query_dims[k],
        'value': value_dims[k]
    }
    for k in query_dims
}
hidden_dim = 20

torch.manual_seed(0)

# initialise network


def disabled_test_initialize_network_no_categoricals():
    net = NestedAttentionNet(
        dag=dag,
        num_dims=dims,
        num_embeds=None,
        embed_dims=None,
        attention_dims=attention_dims
    )

    assert net


def disabled_test_batch_loader():

    batch_loader = RelationalDataLoader(Xs, dag, row_relations,)
    Xs_batch = batch_loader.next_batch()
    print(Xs_batch)
