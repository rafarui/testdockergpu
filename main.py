import numpy as np
import pandas as pd
import torch
from scorch.models import FeedForwardNet
from scorch.utils.data_loader import DataFrameLoader
from scorch.utils.cuda import get_device
device = get_device()
num_cols = ['x1', 'x2', 'x3']
target_col = 'y'
all_cols = num_cols + [target_col]
n_rows_train = 100
n_rows_val = 100
df_train = pd.DataFrame(np.random.rand(n_rows_train, len(num_cols) + 1), columns=all_cols)
df_val = pd.DataFrame(np.random.rand(n_rows_val, len(num_cols) + 1), columns=all_cols)
data_loader_train =  DataFrameLoader(df_train, num_cols, target_cols=target_col, device=device)
data_loader_val =  DataFrameLoader(df_train, num_cols, target_cols=target_col, device=device)
mdl = FeedForwardNet(numeric_dim=len(num_cols),  output_dims=1, device=device)
early_stopping_criteria={'metric': 'loss', 'more': False, 'min_delta': 0.001, 'patience': 2}
history = mdl.fit(data_loader_train,
                  criterion=torch.nn.MSELoss,
                  optimiser=torch.optim.Adam,
                  optimiser_params={'lr': 0.001},
                  num_epochs=5,
                  data_loader_val=data_loader_val,
                  metrics=['loss'],
                  early_stopping_criteria=early_stopping_criteria,
                  verbose=False)
history