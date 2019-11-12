import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scorch.utils.data_loader import RelationalDataFrameLoader
from scorch.models import RelationalNet
from scorch.nn import Transformer
from scorch.utils.cuda import get_device
device = get_device()
dag = {0: [1, 2], 1: [2]}
num_rows = [200, 100, 50]
X0 = pd.DataFrame({
    'num1': np.random.rand(num_rows[0]),
    'num2': np.random.rand(num_rows[0]),
    'num3': np.random.rand(num_rows[0]),
    'cat1': np.random.randint(0, 10, num_rows[0]),
    'cat2': np.random.randint(0, 10, num_rows[0]),
    'y': (np.random.rand(num_rows[0])).astype(int) > 0.5,
    't': np.random.rand(num_rows[0]),
    'w': np.random.rand(num_rows[0]),
})
X1 = pd.DataFrame({
    'num1': np.random.rand(num_rows[1]),
    'num2': np.random.rand(num_rows[1]),
    't': np.random.rand(num_rows[1]),
})
X2 = pd.DataFrame({
    'cat1': np.random.randint(0, 10, num_rows[2]),
    't': np.random.rand(num_rows[2]),
})
Xs = [X0, X1, X2]
features = {}
features[0] = {'num': ['num1', 'num2', 'num3'], 'cat': ['cat1', 'cat2']}
features[1] = {'num': ['num1', 'num2'], 'cat': []}
features[2] = {'num': [], 'cat': ['cat1']}
target = 'y'
row_relations = {}
for r in [(0, 1), (0, 2), (1, 2)]:
    row_relations[r] = [np.random.choice(num_rows[r[1]], np.random.randint(num_rows[r[1]]), replace=False).tolist()
                         for _ in range(num_rows[r[0]])]
data_loader = RelationalDataFrameLoader(Xs,
                                        dag,
                                        row_relations,
                                        features,
                                        target='y',
                                        time_cols=['t', 't', 't'],
                                        sample_weight='w',
                                        device=device)
numeric_dims = [3, 3, 1]
num_embeddings = [{'cat1': 10, 'cat2': 10}, {}, {'cat1': 10}]
embedding_dims = [{'cat1': 4, 'cat2': 8}, {}, {'cat1': 4}]
attention_dims = {
    (0, 1): {'query': 'max', 'value': 10},
    (0, 2): {'query': 10, 'value': 'same'},
    (1, 2): {'query': 20, 'value': 10},
}
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
                    device=device)
history = mdl.fit(data_loader,
                  criterion=torch.nn.MSELoss,
                  optimiser=torch.optim.Adam,
                  optimiser_params={'lr': 0.001},
                  num_epochs=5,
                  metrics=['loss'],
                  compute_training_metrics=True,
                  verbose=True)

print(history)