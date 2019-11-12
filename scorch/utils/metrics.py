import numpy as np
import torch
from sklearn.metrics import roc_auc_score


def update_metrics(y, y_pred, metric_scores, metric_name, metric_func, **kwargs):
    """
    Uses the given labels, predictions and metric function to compute the
    metric score and then adds it to metric scores dictionary.

    If labels and predictions are lists of tensors (for multi-task networks),
    the score for each set of labels as well as the average over all sets are
    added to metric scores.

    Parameters
    ----------

    y: torch.Tensor or list of torch.Tensor
        True labels.
        If tensor, y[i] is the label for y_pred[i].
        If list of tensors, each tensor is a separate set of labels,
        where y[i][j] is the label for y_pred[i][j]

    y_pred: torch.Tensor or list of torch.Tensor
        Model predictions.
        If tensor, y_pred[i] is the prediction for y[i].
        If list of tensors, each tensor is a separate set of predictions,
        where y_pred[i][j] is the predictions for y[i][j]

    metric_scores: dict
        Metric scores of the form {metric name (str): metric score (float)}.
        New metric score will be added to this dictionary.

    metric_name: str
        Name of the metric which we want to compute and add to existing scores.

    metric_func: callable
        Function for computing metric score.
        Must have parameters y and y_pred plus possible others
        which can be specified in kwargs.

    kwargs: keyword arguments for metric_func

    Examples
    --------

    >>> import torch
    >>> from scorch.utils.metrics import precision_at_k, update_metrics
    >>> metric_scores = {'auc': 0.7}
    >>> n_rows = 100
    >>> y = [(torch.rand(n_rows) > 0.5).float(), (torch.rand(n_rows) > 0.5).float()]
    >>> y_pred = [torch.rand(n_rows), torch.rand(n_rows)]
    >>> update_metrics(y, y_pred, metric_scores, 'precision@5', precision_at_k, k=5)
    >>> print(metric_scores)
    {'auc': 0.7, 'precision@5_0': 0.6, 'precision@5_1': 0.4, 'precision@5': 0.5}
    """

    if isinstance(y, list):
        # get metric score for each set of targets
        task_scores = np.zeros(len(y))
        for i in range(len(y)):
            task_scores[i] = apply_metric(
                y[i], y_pred[i], metric_func, **kwargs)
            metric_scores["{}_{}".format(metric_name, i)] = task_scores[i]
        metric_scores[metric_name] = task_scores.mean()
    else:
        # get metric score for single single of targets
        metric_scores[metric_name] = apply_metric(
            y, y_pred, metric_func, **kwargs)


def apply_metric(y, y_pred, metric_func, **kwargs):
    """
    Apply the given metric functions to compute the metric score given the
    labels and their predictions.

    Parameters
    ----------

    y: torch.Tensor, shape (n_rows,)
        True labels.

    y_pred: torch.Tensor, shape (n_rows, *)
        Predictions, where y_pred[i] is the prediction for y[i].
        Number of columns depends on metric.

    metric_func: callable
        Function for computing metric score.
        Must have parameters y and y_pred plus possible others
        which can be specified in kwargs.

    kwargs: keyword arguments for metric_func

    Returns
    -------

    score: float
        The computed metric score.

    Examples
    --------

    >>> import torch
    >>> from scorch.utils.metrics import precision_at_k, apply_metric
    >>> n_rows = 100
    >>> y = (torch.rand(n_rows) > 0.5).float()
    >>> y_pred = torch.rand(n_rows)
    >>> print(apply_metric(y, y_pred, precision_at_k, k=5))
    0.6
    """

    score = metric_func(y, y_pred, **kwargs)
    if not np.isscalar(score):
        score = score.item()

    return score


def loss(y, y_pred, loss_func):
    """
    Computes the loss given the labels and predictions.

    Parameters
    ----------

    y: torch.Tensor, shape (n_rows,)
        True labels.

    y_pred: torch.Tensor, shape (n_rows, *)
        Predictions, where y_pred[i] is the prediction for y[i].
        Number of columns depends on loss function.

    loss_func: torch or scorch loss function
        Function for computing the loss.
        Must have already been initialised with no reduction strategy.
        e.g. torch.nn.MSELoss(reduction='none')

    Returns
    -------

    score: float
        The loss.

    Examples
    --------

    >>> import torch
    >>> from torch.nn import CrossEntropyLoss
    >>> from scorch.utils.metrics import loss
    >>> n_rows = 100
    >>> n_classes = 10
    >>> y = torch.randint(0, n_classes, (n_rows, ))
    >>> y_pred = torch.rand(n_rows, n_classes)
    >>> loss_func = CrossEntropyLoss(reduction='none')
    >>> print(loss(y, y_pred, loss_func))
    2.3383893966674805
    """

    score = loss_func(y_pred, y).mean().item()

    return score


def accuracy(y, y_pred):
    """
    Computes the accuracy given the labels and predictions.

    - Only suitable for classification tasks.

    Parameters
    ----------

    y: torch.LongTensor, shape (n_rows,)
        True labels.

    y_pred: torch.FloatTensor, shape (n_rows, n_classes)
        Target scores, where y_pred[i][j] is the probability
        or confidence that y[i] = j.

    Returns
    -------

    score: float
        The accuracy.

    Examples
    --------

    >>> import torch
    >>> from scorch.utils.metrics import accuracy
    >>> n_rows = 100
    >>> n_classes = 10
    >>> y = torch.randint(0, n_classes, (n_rows, ))
    >>> y_pred = torch.rand(n_rows, n_classes)
    >>> print(accuracy(y, y_pred))
    0.08
    """

    # convert to numpy arrays
    y_pred_np = y_pred.detach().cpu().numpy()
    y_np = y.detach().cpu().numpy()

    # get predicted classes
    i = np.argmax(y_pred_np, axis=1)

    # compute score
    score = (y_np == i).mean()

    return score


def auc(y, y_pred):
    """
    Computes ROC-AUC for binary classification given the labels and predictions.

    Parameters
    ----------

    y: torch.FloatTensor, shape (n_rows,)
        True binary labels.

    y_pred: torch.Tensor, shape (n_rows,)
        Target scores, where y_pred[i] is the positive class score
        or probability corresponding to y[i].

    Returns
    -------

    score: float
        The ROC-AUC score.

    Examples
    --------

    >>> import torch
    >>> from scorch.utils.metrics import auc
    >>> n_rows = 100
    >>> y = (torch.rand(n_rows) > 0.5).float()
    >>> y_pred = torch.rand(n_rows)
    >>> print(auc(y, y_pred))
    0.5833333333333334
    """

    # convert to numpy arrays
    y_pred_np = y_pred.detach().cpu().numpy()
    y_np = y.detach().cpu().numpy()

    score = roc_auc_score(y_np, y_pred_np)

    return score


def accuracy_top_k(y, y_pred, k):
    """
    Computes the top-k accuracy given the labels and predictions.

    For each sample we check if the true label is among the top-k predictions,
    and if so we consider it correct.

    - Only suitable for classification tasks.

    Parameters
    ----------

    y: torch.LongTensor, shape (n_rows,)
        True labels.

    y_pred: torch.FloatTensor, shape (n_rows, n_classes)
        Target scores, where y_pred[i][j] is the probability
        or confidence that y[i] = j.

    k: int

    Returns
    -------

    score: float
        The top-k accuracy.

    Examples
    --------

    >>> import torch
    >>> from scorch.utils.metrics import accuracy_top_k
    >>> n_rows = 100
    >>> n_classes = 10
    >>> y = torch.randint(0, n_classes, (n_rows, ))
    >>> y_pred = torch.rand(n_rows, n_classes)
    >>> print(accuracy_top_k(y, y_pred, k=5))
    0.5199999809265137
    """

    # get top k classes
    _, i = torch.topk(y_pred, k, dim=1, sorted=False)

    # compute score
    score = (y.view(-1, 1) == i).any(dim=1).float().mean().item()

    return score


def auc_micro_average(y, y_pred):
    """
    Computes AUC micro-average given the labels and predictions.
    https://datascience.stackexchange.com/questions/15989/micro-average-vs-macro-average-performance-in-a-multiclass-classification-settin

    - Only suitable for multi-class classification.

    Parameters
    ----------

    y: torch.LongTensor, shape (n_rows,)
        True labels.

    y_pred: torch.FloatTensor, shape (n_rows, n_classes)
        Target scores, where y_pred[i][j] is the probability
        or confidence that y[i] = j.

    Returns
    -------

    score: float
        The AUC micro-average.

    Examples
    --------

    >>> import torch
    >>> from scorch.utils.metrics import auc_micro_average
    >>> n_rows = 100
    >>> n_classes = 10
    >>> y = torch.randint(0, n_classes, (n_rows, ))
    >>> y_pred = torch.rand(n_rows, n_classes)
    >>> print(auc_micro_average(y, y_pred))
    0.4846
    """

    # convert to numpy arrays
    y_pred_np = y_pred.detach().cpu().numpy()
    y_np = y.detach().cpu().numpy()

    # compute score
    y_one_hot = np.zeros(y_pred_np.shape)
    y_one_hot[range(len(y_one_hot)), y_np] = 1
    score = roc_auc_score(y_one_hot.ravel(), y_pred_np.ravel())

    return score


def auc_macro_average(y, y_pred):
    """
    Computes AUC macro-average given the labels and predictions.
    https://datascience.stackexchange.com/questions/15989/micro-average-vs-macro-average-performance-in-a-multiclass-classification-settin

    - Only suitable for multi-class classification.

    Parameters
    ----------

    y: torch.LongTensor, shape (n_rows,)
        True labels.

    y_pred: torch.FloatTensor, shape (n_rows, n_classes)
        Target scores, where y_pred[i][j] is the probability
        or confidence that y[i] = j.

    Returns
    -------

    score: float
        The AUC macro-average.

    Examples
    --------

    >>> import torch
    >>> from scorch.utils.metrics import auc_macro_average
    >>> n_rows = 100
    >>> n_classes = 10
    >>> y = torch.randint(0, n_classes, (n_rows, ))
    >>> y_pred = torch.rand(n_rows, n_classes)
    >>> print(auc_macro_average(y, y_pred))
    >>> 0.48618757416055286
    """

    # convert to numpy arrays
    y_pred_np = y_pred.detach().cpu().numpy()
    y_np = y.detach().cpu().numpy()

    # compute score
    one_v_all_scores = np.zeros(y_pred_np.shape[1])
    for i in range(y_pred_np.shape[1]):
        one_v_all_scores[i] = roc_auc_score(y_np == i, y_pred_np[:, i])
    score = one_v_all_scores.mean()

    return score


def precision_at_k(y, y_pred, k):
    """
    Computes precision at k given the labels and predictions.
    https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Precision_at_K

    - Only suitable for binary classification.

    Parameters
    ----------

    y: torch.FloatTensor, shape (n_rows,)
        True binary labels.

    y_pred: torch.Tensor, shape (n_rows,)
        Target scores, where y_pred[i] is the positive class score
        or probability corresponding to y[i].

    k: int

    Returns
    -------

    score: float
        The precision at k.

    Examples
    --------

    >>> import torch
    >>> from scorch.utils.metrics import precision_at_k
    >>> n_rows = 100
    >>> y = (torch.rand(n_rows) > 0.5).float()
    >>> y_pred = torch.rand(n_rows)
    >>> print(precision_at_k(y, y_pred, k=5))
    0.6
    """

    # compute score
    _, i_sort = torch.sort(y_pred, descending=True)
    num_pos = y.sum().item()
    num_true_pos = y[i_sort[:k]].sum().item()
    if num_true_pos == num_pos:
        score = 1
    else:
        score = num_true_pos / k

    return score


def mean_average_precision_at_k(y, y_pred, k):
    """
    Computes mean average precision at k given the labels and predictions.
    https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Mean_average_precision

    - Only suitable for binary classification.

    Parameters
    ----------

    y: torch.FloatTensor, shape (n_rows,)
        True binary labels.

    y_pred: torch.Tensor, shape (n_rows,)
        Target scores, where y_pred[i] is the positive class score
        or probability corresponding to y[i].

    k: int

    Returns
    -------

    score: float
        The mean average precision at k.

    Examples
    --------

    >>> import torch
    >>> from scorch.utils.metrics import mean_average_precision_at_k
    >>> n_rows = 100
    >>> y = (torch.rand(n_rows) > 0.5).float()
    >>> y_pred = torch.rand(n_rows)
    >>> print(mean_average_precision_at_k(y, y_pred, k=5))
    0.6133333444595337
    """

    # compute score
    _, i_sort = torch.sort(y_pred, descending=True)
    precision_at_ks = torch.cumsum(
        y[i_sort[:k]], dim=0) / torch.arange(1, k+1, dtype=y.dtype, device=y.device)
    score = precision_at_ks.mean().item()

    return score



