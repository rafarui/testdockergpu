import torch
from time import time
import copy

import scorch.utils.metrics as metric_utils


def fit(model,
        data_loader_train,
        criterion,
        optimiser,
        num_epochs=1,
        reg=0.0,
        l1_ratio=0.5,
        data_loader_val=None,
        metrics=None,
        early_stopping_criteria=None,
        compute_training_metrics=False,
        verbose=True):
    """
    Fits the given model by using the given optimiser to minimise the given criterion
    on the given training data.

    - If no validation data is provided, the model is trained for the specified
    number of epochs.

    - If validation data is provided, validation metrics are computed at the end
    of each epoch and training is stopped if early stopping criteria is met.
    
    Parameters
    ----------
    
    model: inherits from torch.nn.Module
        Model which will be fit to training data.
        
    data_loader_train: a data loader object from scorch.utils.data_loader
        Returns batches of training data.

    criterion: torch or scorch loss function
        Criterion to use for measuring the error of the network during training.
        Must have already been initialised with no reduction strategy.
        e.g. torch.nn.MSELoss(reduction='none')
        
    optimiser: torch or scorch optimiser
        Optimiser for updating network parameters during training.
        Must have already been initialised with model parameters.
        e.g. torch.optim.Adam(model.parameters())
        
    num_epochs: integer, optional (default=1)
        Maximum number of passes through full training set
        (possibly less if validation data provided and early stopping triggered).
        
    reg: float, optional (default=0.0)
        Strength of elastic net regularisation on the network's learnable parameters.
        https://en.wikipedia.org/wiki/Elastic_net_regularization

    l1_ratio: float, optional (default=0.5)
        Ratio of L1-regularisation to L2-regularisation.
        All of the networks learnable parameters are regularised as follows:
            reg * (l1_ratio * torch.norm(param, 1) + ((1 - l1_ratio) / 2) * torch.norm(param, 2))
        
    data_loader_val: a data loader object from scorch.utils.data_loader, optional (default=None)
        Returns batches of validation data.

    metrics: list of strings, optional (default=None)
        Indicates which metrics to compute at the end of each epoch.
        Options can be found in scorch.utils.train.evaluate_model.
        If None, only the criterion will be computed.

    early_stopping_criteria: dict, optional (default=None)
        If not None and data_loader_val not None, training will be stopped if
        early stopping criteria met.
        If dict, must have the following keys:
            'metric' (str): name of the metric to use for early stopping (must be in metrics parameter)
            'more' (bool): if True, indicates that a higher metric score is better, else lower is better
            'min_delta' (float): required epoch increase/decrease to be considered an improvement
            'patience' (int): number of epochs to wait without improvement before stopping
           
    compute_training_metrics: boolean, optional (default=False)
        Indicates whether or not to compute training metrics after every training epoch.

    verbose: boolean, optional (default=True)
        Whether or no to print training progress after every epoch.
                
    Returns
    -------
    
    history: dict
        For each of the given metrics contains the following keys:
            train_<metric>: training score after every epoch (only if compute_training_metrics=True)
            val_<metric>: validation score after every epoch (only if validation data provided)

    Examples
    --------

    >>> import numpy as np
    >>> import pandas as pd
    >>> import torch
    >>> from scorch.models import FeedForwardNet
    >>> from scorch.utils.data_loader import DataFrameLoader
    >>> from scorch.utils.train import fit
    >>> n_rows_train = 100
    >>> n_rows_val = 50
    >>> num_cols = ['x1', 'x2', 'x3']
    >>> target_col = 'y'
    >>> all_cols = num_cols + [target_col]
    >>> df_train = pd.DataFrame(np.random.rand(n_rows_train, len(num_cols)), columns=num_cols)
    >>> df_train[target_col] = (np.random.rand(n_rows_train) > 0.5).astype(int)
    >>> df_val = pd.DataFrame(np.random.rand(n_rows_val, len(num_cols)), columns=num_cols)
    >>> df_val[target_col] = (np.random.rand(n_rows_val) > 0.5).astype(int)
    >>> data_loader_train =  DataFrameLoader(df_train, num_cols, target_cols=target_col, device='cpu')
    >>> data_loader_val =  DataFrameLoader(df_val, num_cols, target_cols=target_col, device='cpu')
    >>> mdl = FeedForwardNet(numeric_dim=len(num_cols),  hidden_dims=[4, 4], output_dims=1, device='cpu')
    >>> criterion = torch.nn.MSELoss(reduction='none')
    >>> optimiser = torch.optim.Adam(mdl.parameters(), lr=0.005)
    >>> early_stopping_criteria={'metric': 'auc', 'more': True, 'min_delta': 0.001, 'patience': 5}
    >>> history = fit(mdl,
    >>>               data_loader_train,
    >>>               criterion=criterion,
    >>>               optimiser=optimiser,
    >>>               num_epochs=5,
    >>>               data_loader_val=data_loader_val,
    >>>               reg=0.001,
    >>>               l1_ratio=0.5,
    >>>               metrics=['loss', 'auc'],
    >>>               early_stopping_criteria=early_stopping_criteria,
    >>>               verbose=False)
    >>> print(np.array(history['val_auc']))
    [0.47739096 0.47979192 0.4789916  0.47859144 0.47659064]
    """

    metrics = metrics or ['loss']

    # initialise training/validation history and early stopping flag
    history = {}
    stop = False

    if (data_loader_val is not None) & (early_stopping_criteria is not None):

        # initialise best score, epoch and model weights
        es_metric = early_stopping_criteria['metric']

        if verbose:
            print("-" * 50)
            print("Computing initial validation {}...".format(es_metric))

        best_epoch = -1
        best_score = evaluate_model(model, data_loader_val, metrics=[es_metric], criterion=criterion)[es_metric]
        best_weights = copy.deepcopy(model.state_dict())

        if verbose:
            print("Validation {}: {:.4f}".format(es_metric, best_score))
            print("-" * 50)

    # train in epochs
    for epoch in range(num_epochs):

        if verbose:
            t0 = time()
            print("Epoch {}/{}".format(epoch + 1, num_epochs))
            print("Training...")

        # train in batches
        training_epoch(model, criterion, optimiser, data_loader_train, reg, l1_ratio)

        if compute_training_metrics:
            if verbose:
                print("Computing training metrics...")

            metric_scores_train = evaluate_model(model, data_loader_train, metrics, criterion=criterion)
            add_metrics_to_history(history, metric_scores_train, prefix="train_")

            if verbose:
                print_metrics(metric_scores_train)

        if data_loader_val is not None:
            if verbose:
                print("Computing validation metrics...")

            metric_scores_val = evaluate_model(model, data_loader_val, metrics, criterion=criterion)
            add_metrics_to_history(history, metric_scores_val, prefix="val_")

            if verbose:
                print_metrics(metric_scores_val)

            if early_stopping_criteria is not None:
                # check early stopping criteria
                best_epoch, best_score, best_weights, stop = check_early_stopping(model,
                                                                                  history,
                                                                                  best_epoch,
                                                                                  best_score,
                                                                                  best_weights,
                                                                                  early_stopping_criteria,
                                                                                  verbose)
        if verbose:
            print("Elapsed time: {0:.2f}s".format(time() - t0))
            print("-" * 50)

        if stop:
            break

    if (data_loader_val is not None) & (early_stopping_criteria is not None):

        # load model weights corresponding to best epoch
        model.load_state_dict(best_weights)

        if verbose:
            print("Best validation {} = {:.4f} at epoch {}".format(es_metric, best_score, best_epoch + 1))
            print("Returning best model.")
            print("-" * 50)

    return history


def training_epoch(model,
                   criterion,
                   optimiser,
                   data_loader,
                   reg=0.0,
                   l1_ratio=0.5):
    """
    Executes one training epoch.
    
    For the given model, updates parameters by performing one pass
    through the entire training set, using the specified optimiser to minimise
    the specified criterion on the given training data.
    
    - Training is done in batches, with each batch loaded from the data loader.
    
    Parameters
    ----------
    
    model: inherits from torch.nn.Module
        Model which will be fit to training data.
        
    criterion: torch or scorch loss function
        Criterion to use for measuring the error of the network during training.
        Must have already been initialised with no reduction strategy.
        e.g. torch.nn.MSELoss(reduction='none')

    optimiser: torch or scorch optimiser
        Optimiser for updating network parameters during training.
        Must have already been initialised with model parameters.
        e.g. torch.optim.Adam(model.parameters())

    data_loader: a data loader object from scorch.utils.data_loader, optional (default=None)
        Returns batches of training data.
        
    reg: float, optional (default=0.0)
        Strength of elastic net regularisation on the network's learnable parameters.
        https://en.wikipedia.org/wiki/Elastic_net_regularization

    l1_ratio: float, optional (default=0.5)
        Ratio of L1-regularisation to L2-regularisation.
        All of the networks learnable parameters are regularised as follows:
            reg * (l1_ratio * torch.norm(param, 1) + ((1 - l1_ratio) / 2) * torch.norm(param, 2))

    Examples
    --------

    >>> import numpy as np
    >>> import pandas as pd
    >>> import torch
    >>> from scorch.models import FeedForwardNet
    >>> from scorch.utils.data_loader import DataFrameLoader
    >>> from scorch.utils.train import training_epoch
    >>> n_rows = 100
    >>> num_cols = ['x1', 'x2', 'x3']
    >>> target_col = 'y'
    >>> all_cols = num_cols + [target_col]
    >>> df = pd.DataFrame(np.random.rand(n_rows, len(num_cols)), columns=num_cols)
    >>> df[target_col] = (np.random.rand(n_rows) > 0.5).astype(int)
    >>> data_loader =  DataFrameLoader(df, num_cols, target_cols=target_col, device='cpu')
    >>> mdl = FeedForwardNet(numeric_dim=len(num_cols),  hidden_dims=[4, 4], output_dims=1, device='cpu')
    >>> criterion = torch.nn.MSELoss(reduction='none')
    >>> optimiser = torch.optim.Adam(mdl.parameters(), lr=0.005)
    >>> training_epoch(mdl,
    >>>                criterion,
    >>>                optimiser,
    >>>                data_loader,
    >>>                reg=0.001,
    >>>                l1_ratio=0.5)
    >>> print(data_loader.epoch_complete)
    True
    """

    # we are in training mode
    model.train()

    # reset data loader
    data_loader.reset()

    # train in batches
    while not data_loader.epoch_complete:

        # zero the parameter gradients
        optimiser.zero_grad()

        # get next batch
        batch_data = data_loader.next_batch()

        # forward pass
        out = model(batch_data)

        # compute data + regularisation loss
        loss = compute_loss(batch_data['y'],
                            out,
                            criterion,
                            sample_weight=batch_data.get('sample_weight', None),
                            model=model,
                            reg=reg,
                            l1_ratio=l1_ratio)

        # backward pass + optimise
        loss.backward()
        optimiser.step()


def compute_loss(y,
                 y_pred,
                 criterion,
                 sample_weight=None,
                 model=None,
                 reg=0.0,
                 l1_ratio=0.5):
    """
    Computes the loss given the labels, predictions and criterion.

    - Option to weight each sample in the loss function.

    - Option to add elastic net regularisation loss.

    - Can provide more than one set of targets and predictions for multi-task loss.

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
        Number of columns in each tensor depends on criterion.

    criterion: torch or scorch loss function
        Criterion to use for measuring the error of the network.
        Must have already been initialised with no reduction strategy.
        e.g. torch.nn.MSELoss(reduction='none')

    sample_weight: torch.Tensor or list of torch.Tensor, optional (default=None)
        Sample weights for weighting the loss.
        If tensor, sample_weight[i] is the weight of y[i].
        If list of tensors, each tensor is a separate set of weights,
        where sample_weight[i][j] is the weight of y[i][j]

    model: inherits from torch.nn.Module, optional (default=None)
        If provided and reg > 0, regularisation loss will be added
        for model's parameters.

    reg: float, optional (default=0.0)
        Strength of elastic net regularisation on the network's learnable parameters.
        https://en.wikipedia.org/wiki/Elastic_net_regularization

    l1_ratio: float, optional (default=0.5)
        Ratio of L1-regularisation to L2-regularisation.
        All of the networks learnable parameters are regularised as follows:
            reg * (l1_ratio * torch.norm(param, 1) + ((1 - l1_ratio) / 2) * torch.norm(param, 2))

    Returns
    -------

    loss: torch.FloatTensor, shape (,)
        The total loss.

    Examples
    --------

    >>> import numpy as np
    >>> import torch
    >>> from scorch.models import FeedForwardNet
    >>> from scorch.utils.train import compute_loss
    >>> n_rows = 100
    >>> y = [torch.rand(n_rows), torch.rand(n_rows)]
    >>> y_pred = [torch.rand(n_rows), torch.rand(n_rows)]
    >>> sample_weight = [torch.rand(n_rows), torch.rand(n_rows)]
    >>> mdl = FeedForwardNet(numeric_dim=10,  hidden_dims=[4, 4], output_dims=2, device='cpu')
    >>> criterion = torch.nn.MSELoss(reduction='none')
    >>> loss = compute_loss(y,
    >>>                     y_pred,
    >>>                     criterion,
    >>>                     sample_weight=sample_weight,
    >>>                     model=mdl,
    >>>                     reg=0.001,
    >>>                     l1_ratio=0.5)
    >>> print(loss)
    tensor(0.3005, grad_fn=<AddBackward0>)
    """

    if isinstance(y_pred, list):
        loss = compute_multi_task_data_loss(y, y_pred, criterion, sample_weight=sample_weight)
    else:
        loss = compute_data_loss(y, y_pred, criterion, sample_weight=sample_weight)

    if reg > 0:
        loss = loss + compute_regularisation_loss(model, reg, l1_ratio)

    return loss


def compute_data_loss(y,
                      y_pred,
                      criterion,
                      sample_weight=None):
    """
    Computes the mean loss between the true labels and predictions
    using the specified criterion.

    - Option to weight the loss of each sample.

    Parameters
    ----------

    y: torch.Tensor, shape (n_rows, )
        True labels.

    y_pred: torch.Tensor, shape (n_rows, *)
        Predictions, where y_pred[i] is the prediction for y[i].
        Number of columns depends on criterion.

    criterion: torch or scorch loss function
        Criterion to use for measuring the error of the network.
        Must have already been initialised with no reduction strategy.
        e.g. torch.nn.MSELoss(reduction='none')

    sample_weight: torch.Tensor, shape(n_rows, ), optional (default=None)
        Sample weights for weighting the loss,
        where sample_weight[i] is the weight of y[i].

    Returns
    -------

    loss: torch.FloatTensor, shape (,)
        The (possibly weighted) average loss between labels and predictions.

    Examples
    --------

    >>> import numpy as np
    >>> import torch
    >>> from scorch.utils.train import compute_data_loss
    >>> n_rows = 100
    >>> y = torch.rand(n_rows)
    >>> y_pred = torch.rand(n_rows)
    >>> sample_weight = torch.rand(n_rows)
    >>> criterion = torch.nn.MSELoss(reduction='none')
    >>> loss = compute_data_loss(y,
    >>>                          y_pred,
    >>>                          criterion,
    >>>                          sample_weight=sample_weight)
    >>> print(loss)
    tensor(0.1748)
    """

    sample_loss = criterion(y_pred, y)
    if sample_weight is not None:
        sample_loss = sample_loss * sample_weight
        loss = sample_loss.sum() / sample_weight.sum()
    else:
        loss = sample_loss.mean()

    return loss


def compute_multi_task_data_loss(y,
                                 y_pred,
                                 criterion,
                                 sample_weight=None):
    """
    Computes the multi-task loss given the labels, predictions and criterion.

    - Multi-class loss is the sum of the average loss of each set of labels.

    - Option to weight each sample in the loss function.

    Parameters
    ----------

    y: list of torch.Tensor
        True labels.
        Each tensor is a separate set of labels,
        where y[i][j] is the label for y_pred[i][j]

    y_pred: torch.Tensor or list of torch.Tensor
        Model predictions.
        Each tensor is a separate set of predictions,
        where y_pred[i][j] is the predictions for y[i][j]
        Number of columns in each tensor depends on criterion.

    criterion: torch or scorch loss function
        Criterion to use for measuring the error of the network.
        Must have already been initialised with no reduction strategy.
        e.g. torch.nn.MSELoss(reduction='none')

    sample_weight: list of torch.Tensor, optional (default=None)
        Each tensor is a separate set of weights,
        where sample_weight[i][j] is the weight of y[i][j]

    Returns
    -------

    loss: torch.FloatTensor, shape (,)
        The (possible weighted) multi-task loss.

    Examples
    --------

    >>> import numpy as np
    >>> import torch
    >>> from scorch.utils.train import compute_multi_task_data_loss
    >>> n_rows = 100
    >>> y = [torch.rand(n_rows), torch.rand(n_rows)]
    >>> y_pred = [torch.rand(n_rows), torch.rand(n_rows)]
    >>> sample_weight = [torch.rand(n_rows), torch.rand(n_rows)]
    >>> criterion = torch.nn.MSELoss(reduction='none')
    >>> loss = compute_multi_task_data_loss(y,
    >>>                                     y_pred,
    >>>                                     criterion,
    >>>                                     sample_weight=sample_weight)
    >>> print(loss)
    tensor(0.3060)
    """

    loss = 0.0
    for i in range(len(y)):
        if sample_weight is not None:
            loss = loss + compute_data_loss(y[i], y_pred[i], criterion, sample_weight=sample_weight[i])
        else:
            loss = loss + compute_data_loss(y[i], y_pred[i], criterion)
    return loss


def compute_regularisation_loss(model,
                                reg,
                                l1_ratio=0.5):
    """
    Computes elastic net regularisation loss for model's parameters.
    https://en.wikipedia.org/wiki/Elastic_net_regularization

    Parameters
    ----------

    model: inherits from torch.nn.Module
        Regularisation loss will be computed for model's parameters.

    reg: float, optional (default=0.0)
        Strength of elastic net regularisation on the network's learnable parameters.

    l1_ratio: float, optional (default=0.5)
        Ratio of L1-regularisation to L2-regularisation.
        All of the networks learnable parameters are regularised as follows:
            reg * (l1_ratio * torch.norm(param, 1) + ((1 - l1_ratio) / 2) * torch.norm(param, 2))

    Returns
    -------

    loss: torch.FloatTensor, shape (,)
        Total regularisation loss of model's parameters.

    Examples
    --------

    >>> from scorch.models import FeedForwardNet
    >>> from scorch.utils.train import compute_regularisation_loss
    >>> mdl = FeedForwardNet(numeric_dim=10,  hidden_dims=[4, 4], output_dims=1, device='cpu')
    >>> loss = compute_regularisation_loss(mdl,
    >>>                                    reg=0.001,
    >>>                                    l1_ratio=0.5)
    >>> print(loss)
    tensor(0.0138, grad_fn=<AddBackward0>)
    """

    loss = 0.0
    for param in model.parameters():
        l1_loss = reg * l1_ratio * torch.norm(param, 1)
        l2_loss = reg * ((1 - l1_ratio) / 2) * torch.norm(param, 2)
        loss = loss + l1_loss + l2_loss

    return loss


def apply_model(model,
                data_loader,
                activate_output=False):
    """
    Uses the given model to predict labels for samples in the data loader.
    
    - The model is applied in batches.
    
    Parameters
    ----------
    
    model: inherits from torch.nn.Module
        Model which will be used to make predictions.
        
    data_loader: a data loader object from scorch.utils.data_loader, optional (default=None)
        Returns batches of data.
        
    activate_output: bool (default=False)
        If True, model's final layer activation function
        will be applied to outputs.
          
    Returns
    -------
    
    out: torch.Tensor or list of torch.Tensor
        Model outputs.
        If model is a multi-task network the output will be
        a list of tensors, else a single tensor.

    Examples
    --------

    >>> import numpy as np
    >>> import pandas as pd
    >>> from scorch.models import FeedForwardNet
    >>> from scorch.utils.data_loader import DataFrameLoader
    >>> from scorch.utils.train import apply_model
    >>> n_rows = 5
    >>> num_cols = ['x1', 'x2', 'x3']
    >>> all_cols = num_cols + [target_col]
    >>> df = pd.DataFrame(np.random.rand(n_rows, len(num_cols)), columns=num_cols)
    >>> data_loader =  DataFrameLoader(df, num_cols, device='cpu')
    >>> mdl = FeedForwardNet(numeric_dim=len(num_cols),  hidden_dims=[4, 4], output_dims=1, device='cpu')
    >>> out = apply_model(mdl,
    >>>                   data_loader)
    >>> print(out)
    tensor([0.5527, 0.3201, 0.7180, 0.3256, 0.5321])
    """

    # get device the data is on
    device = data_loader.device

    with torch.no_grad():

        # we are in inference mode
        model.eval()

        # reset data loader
        data_loader.reset()

        if isinstance(model.output_dims, list):
            # create list of tensors for storing outputs
            out = [torch.zeros(data_loader.num_rows, d, dtype=torch.float, device=device).squeeze(dim=1)
                   for d in model.output_dims]
        else:
            # create single tensor for storing predictions
            out = torch.zeros(data_loader.num_rows, model.output_dims, dtype=torch.float, device=device).squeeze(dim=1)

        # apply in batches   
        while not data_loader.epoch_complete:

            # get next batch
            batch_data = data_loader.next_batch()

            # forward pass
            out_batch = model(batch_data, activate_output=activate_output)

            # add predictions to output tensors
            if isinstance(out_batch, list):
                for i in range(len(out_batch)):
                    out[i][batch_data['rows']] = out_batch[i]
            else:
                out[batch_data['rows']] = out_batch

    return out


def evaluate_model(model,
                   data_loader,
                   metrics,
                   criterion=None):
    """
    Evaluates the given model on the given data by computing the specified metrics.

    Parameters
    ----------
    
    model: inherits from torch.nn.Module
        Model which will be evaluated.

    data_loader: a data loader object from scorch.utils.data_loader, optional (default=None)
        Returns batches of data.

    metrics: list of strings
        Indicates which metrics to compute.
        Options are:
            - 'loss'
            - 'accuracy' (classificaton)
            - 'auc' (binary classification)
            - 'accuracy_top_k' where k is some integer (classification)
            - 'auc_micro_average' (multi-task binary classification)
            - 'auc_macro_average' (multi-task binary classification)
            - 'precision@k' where k is some integer (binary classification)
            - 'map@k' where k is some integer (binary classification)
        If metrics contains 'loss' then criterion must be specified.

    criterion: torch or scorch loss function, optional (default=None)
        Criterion to use for computing the loss.
        Must have already been initialised with no reduction strategy.
        e.g. torch.nn.MSELoss(reduction='none')
          
    Returns
    -------
    
    metric_scores: dict
        Keys are metric names and values are the corresponding scores.

    Examples
    --------

    >>> import numpy as np
    >>> import pandas as pd
    >>> import torch
    >>> from scorch.models import FeedForwardNet
    >>> from scorch.utils.data_loader import DataFrameLoader
    >>> from scorch.utils.train import evaluate_model
    >>> n_rows = 100
    >>> num_cols = ['x1', 'x2', 'x3']
    >>> target_col = 'y'
    >>> df = pd.DataFrame(np.random.rand(n_rows, len(num_cols)), columns=num_cols)
    >>> df[target_col] = (np.random.rand(n_rows) > 0.5).astype(int)
    >>> data_loader =  DataFrameLoader(df, num_cols, target_cols=target_col)
    >>> mdl = FeedForwardNet(numeric_dim=len(num_cols),  hidden_dims=[4, 4], output_dims=1,
    >>>                      output_layer_params={'activation': torch.sigmoid})
    >>> criterion = torch.nn.BCEWithLogitsLoss(reduction='none')
    >>> scores = evaluate_model(mdl,
    >>>                         data_loader,
    >>>                         metrics=['loss', 'auc'],
    >>>                         criterion=criterion)
    >>> print(scores)
    {'loss': 0.6827784776687622, 'auc': 0.5422824302134648}
    """

    # apply model
    y_pred = apply_model(model, data_loader, activate_output=False)

    # get true labels
    y = data_loader.y

    metric_scores = {}
    if metrics is not None:
        # compute metrics        
        for name in metrics:

            if (name == "loss") & (criterion is not None):
                metric_utils.update_metrics(y, y_pred, metric_scores, name,
                                            metric_utils.loss, loss_func=criterion)

            if name == "accuracy":
                metric_utils.update_metrics(y, y_pred, metric_scores, name,
                                            metric_utils.accuracy)

            if name.startswith('accuracy_top_'):
                k = int(name.split('_')[2])
                metric_utils.update_metrics(y, y_pred, metric_scores, name,
                                            metric_utils.accuracy_top_k, k=k)

            if name == "auc":
                metric_utils.update_metrics(y, y_pred, metric_scores, name,
                                            metric_utils.auc)

            if name == "auc_micro_average":
                metric_utils.update_metrics(y, y_pred, metric_scores, name,
                                            metric_utils.auc_micro_average)

            if name == "auc_macro_average":
                metric_utils.update_metrics(y, y_pred, metric_scores, name,
                                            metric_utils.auc_macro_average)

            if name.startswith('precision@'):
                k = int(name.split('@')[1])
                metric_utils.update_metrics(y, y_pred, metric_scores, name,
                                            metric_utils.precision_at_k, k=k)

            if name.startswith('map@'):
                k = int(name.split('@')[1])
                metric_utils.update_metrics(y, y_pred, metric_scores, name,
                                            metric_utils.mean_average_precision_at_k, k=k)

    return metric_scores


def add_metrics_to_history(history, metric_scores, prefix=''):
    """
    Adds the metric scores to the history.

    Parameters
    ----------

    history: dict
        Training history of the form {metric name (str): metric scores (list}.
        May be empty.

    metric_scores: dict
        Metric scores of the form {metric name (str): metric score (float)}.

    prefix: str, optional (default='')
        Prefix to add to metric names to create history keys.
        e.g. 'train'

    Examples
    --------

    >>> from scorch.utils.train import add_metrics_to_history
    >>> train_scores = {'loss': 0.2, 'auc': 0.6}
    >>> history = {}
    >>> add_metrics_to_history(history, train_scores, prefix='train_')
    >>> print(history)
    {'train_loss': [0.2], 'train_auc': [0.6]}

    >>> val_scores = {'loss': 0.4, 'auc': 0.5}
    >>> add_metrics_to_history(history, val_scores, prefix='val_')
    >>> print(history)
    {'train_loss': [0.2], 'train_auc': [0.6], 'val_loss': [0.4], 'val_auc': [0.5]}

    >>> train_scores = {'loss': 0.1, 'auc': 0.7}
    >>> add_metrics_to_history(history, train_scores, prefix='train_')
    >>> print(history)
    {'train_loss': [0.2, 0.1], 'train_auc': [0.6, 0.7], 'val_loss': [0.4], 'val_auc': [0.5]}
    """

    for name, score in metric_scores.items():
        key = prefix + name
        if key not in history:
            history[key] = []
        history[key].append(score)


def print_metrics(metric_scores):
    """
    Print out the metric scores to the command line.

    Parameters
    ----------

    metric_scores: dict
        Metric scores of the form {metric name (str): metric score (float)}.

    Examples
    --------

    >>> from scorch.utils.train import print_metrics
    >>> scores = {'loss': 0.4, 'auc': 0.6}
    >>> print_metrics(scores)
    loss: 0.4000
    auc: 0.6000
    """

    for name, score in metric_scores.items():
        print("{}: {:.4f}".format(name, score))


def check_early_stopping(model,
                         history,
                         best_epoch,
                         best_score,
                         best_weights,
                         early_stopping_criteria,
                         verbose=True):
    """
    Gets the best score, epoch and model weights so far and checks whether early stopping
    criteria has been met.
    
    Parameters
    ----------
    
    model: inherits from torch.nn.Module
        Model which is to be checked.

    history: dict
        Validation history of the form {metric name (str): metric scores (list},
        where the metric names are of the form 'val_*' where * is the name of
        a metric.
        e.g. {'val_loss': [0.3, 0.1], 'val_auc': [0.5, 0.6]}
        
    best_epoch: int
        Best epoch.
        
    best_score: float
        Validation score corresponding to best epoch.
        
    best_weights: state dict
        Model weights corresponding to best epoch.
        
    early_stopping_criteria: dict (default=None)
        If not None and data_loader_val not None, training will be stopped if
        early stopping criteria met.
        If dict, must have the following keys...
            - 'metric': name of the metric to use for early stopping,
                        where 'val_<metric name>' must be in history
            - 'more': if True, indicates that a higher metric score is better
            - 'min_delta': positive float indicating required improvement
            - 'patience': number of epochs to wait without improvement before stopping
        
    verbose: boolean (default=True)
        Whether or not to print updates.
          
    Returns
    -------
    
    best_epoch: int
        New best epoch.
        
    best_score: float
        Validation score corresponding to new best epoch.
        
    best_weights: dict
        Model weights corresponding to new best epoch.
        
    stop: bool
        Whether or not early stopping criteria have been met.

    Examples
    --------

    >>> import copy
    >>> from scorch.models import FeedForwardNet
    >>> from scorch.utils.train import check_early_stopping
    >>> mdl = FeedForwardNet(numeric_dim=6,  hidden_dims=[4, 4], output_dims=1, device='cpu')
    >>> history = {'val_loss': [0.4, 0.39, 0.38, 0.37, 0.36], 'val_auc': [0.5, 0.6, 0.7, 0.705, 0.707]}
    >>> best_epoch = 2
    >>> best_score = 0.7
    >>> best_weights = copy.deepcopy(mdl.state_dict())
    >>> early_stopping_criteria = {
    >>>     'metric': 'auc',
    >>>     'more': True,
    >>>     'min_delta': 0.01,
    >>>     'patience': 2
    >>> }
    >>> best_epoch, best_score, best_weights, stop = check_early_stopping(mdl,
    >>>                                                                   history,
    >>>                                                                   best_epoch,
    >>>                                                                   best_score,
    >>>                                                                   best_weights,
    >>>                                                                   early_stopping_criteria,
    >>>                                                                   verbose=True)
    >>> print(stop)
    Epoch(s) since minimum improvement: 2
    True

    >>> early_stopping_criteria['min_delta'] = 0
    >>> best_epoch, best_score, best_weights, stop = check_early_stopping(mdl,
    >>>                                                                   history,
    >>>                                                                   best_epoch,
    >>>                                                                   best_score,
    >>>                                                                   best_weights,
    >>>                                                                   early_stopping_criteria,
    >>>                                                                   verbose=True)
    >>> print(stop)
    Epoch(s) since minimum improvement: 0
    False
    """

    # get the current epoch
    key = 'val_' + early_stopping_criteria['metric']
    epoch = len(history[key]) - 1

    # compute the improvement
    new_score = history[key][epoch]
    if early_stopping_criteria['more']:
        improvement = new_score - best_score
    else:
        improvement = best_score - new_score

    # has the score improved sufficiently?
    if improvement > early_stopping_criteria['min_delta']:
        # record new best score, epoch and model weights
        best_score = new_score
        best_epoch = epoch
        best_weights = copy.deepcopy(model.state_dict())
        epochs_since_improvement = 0
    else:
        epochs_since_improvement = epoch - best_epoch

    if verbose:
        print("Epoch(s) since minimum improvement: {}".format(epochs_since_improvement))

    # stop training early?
    if epochs_since_improvement == early_stopping_criteria['patience']:
        stop = True
    else:
        stop = False

    return best_epoch, best_score, best_weights, stop


def get_activations(model,
                    data_loader,
                    half=False):
    """
    Gets the activations from the final hidden layer of the model.
    
    Parameters
    ----------
    
    model: inherits from torch.nn.Module
        Model which will be used to compute activations.

    data_loader: a data loader object from scorch.utils.data_loader, optional (default=None)
        Returns batches of data.
        
    half: bool, optional (default=False)
        If True, half precision activations will be returned instead of float.

    Returns
    -------
    
    A: torch.Tensor
        Final hidden layer activations for data in data_loader, in same order.
        If half=True then torch.HalfTensor will be returned, else torch.FloatTensor.

    Examples
    --------

    >>> import numpy as np
    >>> import pandas as pd
    >>> from scorch.models import FeedForwardNet
    >>> from scorch.utils.data_loader import DataFrameLoader
    >>> from scorch.utils.train import get_activations
    >>> n_rows = 100
    >>> num_cols = ['x1', 'x2', 'x3']
    >>> df = pd.DataFrame(np.random.rand(n_rows, len(num_cols)), columns=num_cols)
    >>> data_loader =  DataFrameLoader(df, num_cols)
    >>> mdl = FeedForwardNet(numeric_dim=len(num_cols),  hidden_dims=[4, 4], output_dims=1)
    >>> A = get_activations(mdl,
    >>>                     data_loader,
    >>>                     half=False)
    >>> print(A.shape)
    torch.Size([100, 4])
    """

    with torch.no_grad():

        # we are in inference mode
        model.eval()

        # get activations in batches
        first_batch = True
        while not data_loader.epoch_complete:

            # get next batch
            batch_data = data_loader.next_batch()

            # get activations for this batch
            A_batch = model(batch_data, return_activations=True)

            if half:
                # convert to half precision
                A_batch = A_batch.half()

            if first_batch:
                # initialise tensor for storing all activations
                A = torch.zeros(data_loader.num_rows, A_batch.size(1), dtype=A_batch.dtype,
                                device=data_loader.device)
                first_batch = False

            # add batch activations
            A[batch_data['rows'], :] = A_batch

    return A
