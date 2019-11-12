import torch.nn as nn

from scorch.nn.modules.batchnorm import BatchNorm1d
from scorch.utils.cuda import get_device


class PostLinear(nn.Module):
    """
    A combinations of modules which are commonly added after linear layers.

        - Option to apply batch norm > activation > dropout.

    Parameters
    ----------

    in_features: int
        Number of input features.

    batch_norm: bool, optional (default=False)
        If True, batch norm layer will be added in the order defined above.

    activation: torch activation function, optional (default=None)
        If not None, activation function will be added in the order defined above.

    dropout_rate: float, optional (default=None)
        If not None, dropout layer with specified dropout rate will be added in the order defined above.

    device: torch.device or string, optional (default=None)
        Device on which module will be placed.
        If string, must specify the device e.g. 'cpu' or 'cuda'.
        If None, GPU will be used if available, else CPU.

    Examples
    --------

    >>> import torch
    >>> import torch.nn.functional as F
    >>> from scorch.nn import PostLinear
    >>> X = torch.rand(5, 3)
    >>> post_linear = PostLinear(in_features=X.shape[1],
    >>>                          batch_norm=True,
    >>>                          activation=F.relu,
    >>>                          dropout_rate=0.5,
    >>>                          device='cpu')
    >>> H = post_linear(X)
    >>> print(H.shape)
    torch.Size([5, 3])
    """

    def __init__(self,
                 in_features,
                 batch_norm=False,
                 activation=None,
                 dropout_rate=None,
                 device=None):

        super(PostLinear, self).__init__()

        # initialise some instance variables
        self.in_features = in_features
        self.activation = activation
        self.device = get_device(device)

        if batch_norm:
            # initialise batch norm
            self.bn = BatchNorm1d(in_features)
        else:
            self.bn = None

        if dropout_rate is not None:
            # initialise dropout
            self.dropout = nn.Dropout(dropout_rate)
        else:
            self.dropout = None

        # move module to device
        self.to(self.device)

    def forward(self, X):
        """
        Executes forward pass.

        Applies batch norm > activation > dropout.

        If batch norm or activation or dropout is None, will be missed out.

        Parameters
        ----------

        X: torch.FloatTensor, shape (n_rows, self.in_features)
            Input data.

        Returns
        -------

        X: torch.FloatTensor, shape (n_rows, self.in_features)
            Inputs after applying specified operations.
        """

        for layer in [self.bn, self.activation, self.dropout]:
            if layer is not None:
                X = layer(X)

        return X