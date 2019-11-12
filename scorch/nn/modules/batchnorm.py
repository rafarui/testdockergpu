import torch.nn as nn

from scorch.utils.cuda import get_device


class BatchNorm1d(nn.Module):
    """
    Builds on torch.nn.BatchNorm1d.

    See https://pytorch.org/docs/stable/nn.html#batchnorm1d.

    Extra functionality:

        - If a single data point is passed when in training mode it will be returned unchanged
          instead of throwing an error.

    Parameters
    ----------

    num_features: int
        Number of input features.

    device: torch.device or string, optional (default=None)
        Device on which module will be placed.
        If string, must specify the device e.g. 'cpu' or 'cuda'.
        If None, GPU will be used if available, else CPU.

    Examples
    --------

    >>> import torch
    >>> from scorch.nn import BatchNorm1d
    >>> X = torch.rand(5, 3)
    >>> bn = BatchNorm1d(num_features=X.shape[1],
                     device='cpu')
    >>> A = bn(X)
    >>> print(A.shape)
    torch.Size([5, 3])
    """

    def __init__(self,
                 num_features,
                 device=None):

        super(BatchNorm1d, self).__init__()

        self.num_features = num_features
        self.device = get_device(device)

        # initialise layer
        self.bn = nn.BatchNorm1d(num_features)

        # move module to device
        self.to(self.device)

    def forward(self, X):
        """
        Executes batch normalisation forward pass.

        If we are in training mode and only a single data point is passed,
        batch norm will not be applied and the data point will be returned unchanged.

        Parameters
        ----------

        X: torch.FloatTensor
            Input data of shape (n_rows, self.num_features).

        Returns
        -------

        X: torch.FloatTensor
            Batch normalised inputs of shape (n_rows, self.num_features).
        """

        if (len(X) > 1) | (not self.training):
            X = self.bn(X)

        return X
