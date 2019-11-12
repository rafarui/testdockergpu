import numpy as np
import torch
import torch.nn.functional as F
from scorch.models import RelationalNet
from scorch.nn import Transformer
dag = {0: [1, 2], 1: [2]}
num_rows = [5, 8, 4]
numeric_dims = [3, 2, 0]
num_embeddings = [{'cat1': 5, 'cat2': 10}, {}, {'cat3': 8}]
embedding_dims = [{'cat1': 4, 'cat2': 8}, {}, {'cat3': 4}]
attention_dims = {
    (0, 1): {'query': 'max', 'value': 10},
    (0, 2): {'query': 10, 'value': 'same'},
    (1, 2): {'query': 20, 'value': 10},
}
X0 = dict(
    num=torch.rand(num_rows[0], numeric_dims[0], dtype=torch.float),
    cat=torch.cat([torch.randint(0, num_embeddings[0]['cat1'], (num_rows[0], 1)),
                   torch.randint(0, num_embeddings[0]['cat2'], (num_rows[0], 1))], dim=1)
)
X1 = dict(
    num=torch.rand(num_rows[1], numeric_dims[1], dtype=torch.float),
    cat=torch.LongTensor([])
)
X2 = dict(
    num=torch.FloatTensor(),
    cat=torch.randint(0, num_embeddings[2]['cat3'], (num_rows[2], 1))
)
Xs = {0: X0, 1: X1, 2: X2}
maps = {}
maps[(0, 1)] = np.random.randint(-1, num_rows[0], num_rows[1])
maps[(0, 2)] = np.random.randint(-1, num_rows[0], num_rows[2])
maps[(0, 1, 2)] = np.random.randint(-1, num_rows[1], num_rows[2])
data = {}
data['Xs'] = Xs
data['maps'] = maps
mdl = RelationalNet(dag,
                    numeric_dims,
                    num_embeddings,
                    embedding_dims,
                    attention_dims,
                    num_heads=1,
                    hidden_dims=[20, 20],
                    output_dims=1,
                    dense_layer_params={'activation': F.relu, 'dropout_rate': 0.5},
                    output_layer_params={'activation': torch.sigmoid},
                    attention_mechanism=Transformer,
                    device='cuda')
mdl(data, activate_output=True)