import torch
import torch.nn as nn


class ExponentialRankingLoss(nn.Module):
    """
    Exponential ranking loss function.

    Computes the loss

        exp(-y * (x1 - x2)),

    where y=1 means that x1 should be greater than x2,
    and y=-1 means that x2 should be greater than x1.

    Parameters
    ----------

    reduction: str, optional (default=None)
        The reduction strategy. Options are 'mean' and 'sum'.

    Examples
    --------

    >>> import torch
    >>> from scorch.nn import ExponentialRankingLoss
    >>> x1 = torch.rand(5)
    >>> x2 = torch.rand(5)
    >>> y = torch.tensor([1, -1, 1, -1, 1]).float()
    >>> loss = ExponentialRankingLoss(reduction='mean')
    >>> print(loss(x1, x2, y))
    tensor(0.9712)
    """

    def __init__(self, reduction=None):
        super(ExponentialRankingLoss, self).__init__()
        self.reduction = reduction

    def forward(self, x1, x2, y):
        """

        Parameters
        ----------

        x1: torch.FloatTensor, shape (n,)
            Ranking score for first input.

        x2: torch.FloatTensor, shape (n,)
            Ranking score for second input.

        y: torch.FloatTensor, shape (n,)
            y[i] = 1.0 if x1[i] should be greater than x2[i],
            else y[i] = -1.0.

        Returns
        -------

        loss: torch.FloatTensor
            If self.reduction is None, loss for each sample is returned
            in a tensor of shape (n,), else the reduced loss is returned
            in a tensor of shape (,).
        """

        loss = torch.exp(-y * (x1 - x2))
        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()

        return loss
