import torch.nn as nn

import scorch.utils.train as train_utils


class Model(nn.Module):
    """
    Abstract class for all neural network models.

    Implements training and predict functionality.

    Examples
    --------

    >>> import numpy as np
    >>> import pandas as pd
    >>> import torch
    >>> from scorch.models import FeedForwardNet
    >>> from scorch.utils.data_loader import DataFrameLoader
    >>> num_cols = ['x1', 'x2', 'x3']
    >>> target_col = 'y'
    >>> all_cols = num_cols + [target_col]
    >>> df_train = pd.DataFrame(np.random.rand(5, len(num_cols) + 1), columns=all_cols)
    >>> df_val = pd.DataFrame(np.random.rand(5, len(num_cols) + 1), columns=all_cols)
    >>> data_loader_train =  DataFrameLoader(df_train, num_cols, target_cols=target_col, device='cpu')
    >>> data_loader_val =  DataFrameLoader(df_train, num_cols, target_cols=target_col, device='cpu')
    >>> mdl = FeedForwardNet(numeric_dim=len(num_cols),  output_dims=1, device='cpu')
    >>> early_stopping_criteria = {'metric': 'loss', 'more': False, 'min_delta': 0.001, 'patience': 2}
    >>> history = mdl.fit(data_loader_train,
    >>>               criterion=torch.nn.MSELoss,
    >>>               optimiser=torch.optim.Adam,
    >>>               optimiser_params={'lr': 0.001},
    >>>               num_epochs=5,
    >>>               data_loader_val=data_loader_val,
    >>>               metrics=['loss'],
    >>>               early_stopping_criteria=early_stopping_criteria,
    >>>               verbose=False)
    >>> print(np.array(history['val_loss']))
    [0.23005927 0.22870973 0.22736962 0.22603922 0.22471862]
    """

    def __init__(self):
        super(Model, self).__init__()

    def fit(self,
            data_loader_train,
            criterion,
            optimiser,
            optimiser_params,
            num_epochs=1,
            reg=0.0,
            l1_ratio=0.5,
            data_loader_val=None,
            metrics=None,
            early_stopping_criteria=None,
            compute_training_metrics=False,
            verbose=True):
        """
        Fits the model by using the given optimiser to minimise the given criterion
        on the given training data.

        - If no validation data is provided, the model is trained for the specified
        number of epochs.

        - If validation data is provided, validation metrics are computed at the end
        of each epoch and training is stopped if early stopping criteria is met.

        Parameters
        ----------

        data_loader_train: a data loader object from scorch.utils.data_loader
            Returns batches of training data.

        criterion: torch or scorch loss function
            Criterion to use for measuring the error of the network during training.
            e.g. torch.nn.MSELoss

        optimiser: torch or scorch optimiser
            Optimiser for updating network parameters during training.
            e.g. torch.optim.Adam

        optimiser_params: dict
            Parameter settings for the optimiser.
            Of the form {parameter name (str): parameter value}.
            Keys must match optimiser parameter names.
            e.g. {'lr': 0.001}

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
        """

        # initialise criterion with no reduction strategy
        criterion = criterion(reduction='none')

        # initialise optimiser
        optimiser = optimiser(self.parameters(), **optimiser_params)

        # fit
        history = train_utils.fit(model=self,
                                  data_loader_train=data_loader_train,
                                  criterion=criterion,
                                  optimiser=optimiser,
                                  num_epochs=num_epochs,
                                  reg=reg,
                                  l1_ratio=l1_ratio,
                                  data_loader_val=data_loader_val,
                                  metrics=metrics,
                                  early_stopping_criteria=early_stopping_criteria,
                                  compute_training_metrics=compute_training_metrics,
                                  verbose=verbose)

        return history

    def predict(self,
                data_loader,
                activate_output=False):
        """
        Uses the model to make predictions for samples in the data loader.

        - The model is applied in batches to avoid a memory error.

        Parameters
        ----------

        data_loader: a data loader object from scorch.utils.data_loader
            Returns batches of data.

        activate_output: bool, optional (default=False)
            If True, model's activation function will be applied to outputs.

        Returns
        -------

        y_pred: torch.Tensor or list of torch.Tensor
            Model predictions for each sample in data loader.
            If self.output_dims is an integer, will return a torch.Tensor
            of shape (data_loader.num_rows, self.output_dims).
            If self.output_dims is a list of integers, will return a list
            of torch.Tensor where the i-th tensor has shape
            (data_loader.num_rows, self.output_dims[i]).
        """

        y_pred = train_utils.apply_model(model=self,
                                         data_loader=data_loader,
                                         activate_output=activate_output)

        return y_pred

    def evaluate(self,
                 data_loader,
                 metrics,
                 criterion=None):
        """
        Evaluates the model on the samples in the data loader.

        Uses the model to make predictions for samples in the data loader
        and then computes the specified metrics.

        - The model is applied in batches to avoid memory error.

        Parameters
        ----------

        data_loader: a data loader object from scorch.utils.data_loader
            Returns batches of data.

        metrics: list of strings
            Indicates which metrics to compute.
            Options can be found in scorch.utils.train.evaluate_model.
            If metrics contains 'loss' then criterion must be specified.

        criterion: torch or scorch loss function, optional (default=None)
            Criterion for computing model loss.
            e.g. torch.nn.MSELoss

        Returns
        -------

        metric_scores: dict
            Keys are metric names and values are the corresponding scores.
        """

        metric_scores = train_utils.evaluate_model(model=self,
                                                   data_loader=data_loader,
                                                   metrics=metrics,
                                                   criterion=criterion)

        return metric_scores

    def get_activations(self,
                        data_loader,
                        half=False):
        """
        Gets the activations from the final hidden layer of the model.

        - Activations are computed in batches to avoid memory error.

        Parameters
        ----------

        data_loader: a data loader object from scorch.utils.data_loader
            Returns batches of data.

        half: bool, optional (default=False)
            If True, half precision activations will be returned
            instead of full precision.

        Returns
        -------

        A: torch.Tensor, shape (data_loader.num_rows, final hidden layer dimension)
            Final hidden layer activations for samples in data loader.
        """

        A = train_utils.get_activations(model=self,
                                        data_loader=data_loader,
                                        half=half)

        return A
