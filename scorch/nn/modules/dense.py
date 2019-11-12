import torch.nn as nn

from scorch.nn.modules.utils import PostLinear
from scorch.utils.cuda import get_device
import scorch.utils.parameters as default_parameters


class Dense(nn.Module):
    """
    Builds on torch.nn.Linear.

    See https://pytorch.org/docs/stable/nn.html#linear.

    Extra functionality:

        - Option to apply batch norm > activation > dropout after linear layer.

    Parameters
    ----------

    in_features: int
        Number of input features.

    out_features: int
        Number of output features.

    batch_norm: bool, optional (default=False)
        If True, batch norm layer will be added after linear layer
        in the order defined above.

    activation: torch activation function, optional (default=None)
        If not None, activation function will be added after linear layer
        in the order defined above.

    dropout_rate: float, optional (default=None)
        If not None, dropout layer with specified dropout rate will be added
        after linear layer in the order defined above.

    weight_init: initialiser from torch.nn.init, optional (default=torch.nn.init.xavier_uniform_)
        Used to initialise the weights in the linear layer.

    device: torch.device or string, optional (default=None)
        Device on which module will be placed.
        If string, must specify the device e.g. 'cpu' or 'cuda'.
        If None, GPU will be used if available, else CPU.

    Examples
    --------

    >>> import torch
    >>> import torch.nn.functional as F
    >>> from scorch.nn import Dense
    >>> X = torch.rand(5, 3)
    >>> dense = Dense(in_features=X.shape[1],
    >>>               out_features=4,
    >>>               batch_norm=True,
    >>>               activation=F.relu,
    >>>               dropout_rate=0.5,
    >>>               device='cpu')
    >>> H = dense(X)
    >>> print(H.shape)
    torch.Size([5, 4])
    """

    def __init__(self,
                 in_features,
                 out_features,
                 batch_norm=False,
                 activation=None,
                 dropout_rate=None,
                 weight_init=nn.init.xavier_uniform_,
                 device=None):

        super(Dense, self).__init__()

        # initialise some instance variables
        self.in_features = in_features
        self.out_features = out_features
        self.device = get_device(device)

        # initialise linear layer
        self.linear = nn.Linear(in_features=in_features,
                                out_features=out_features)
        weight_init(self.linear.weight)

        # initialise post-linear layer
        self.post_linear = PostLinear(in_features=out_features,
                                      batch_norm=batch_norm,
                                      activation=activation,
                                      dropout_rate=dropout_rate,
                                      device=self.device)

        # move module to device
        self.to(self.device)

    def forward(self, X):
        """
        Executes forward pass.

        Applies linear transformation > batch norm > activation > dropout.

        If batch norm or activation or dropout is None, will be missed out.

        Parameters
        ----------

        X: torch.FloatTensor, shape (n_rows, self.in_features)
            Input data.

        Returns
        -------

        A: torch.FloatTensor, shape (n_rows, self.out_features)
            Activations, where A[i] is the activation of X[i].
        """

        H = self.linear(X)
        A = self.post_linear(H)

        return A


class MultiDense(nn.Module):
    """
    Implements a series of scorch.nn.Dense layers.

    Parameters
    ----------

    in_features: int
        Number of input features.

    hidden_features: list
        A hidden layer will be added for each element in the list,
        where hidden_features[i] specifies the size of the i-th hidden layer.
        Elements in the list must either be integers or 'auto'.
        If 'auto', the size of the corresponding layer will be equal to the size of the previous layer.
        e.g. ['auto', 128, 64] will result in three layers with sizes
        self.in_features, 128 and 64 respectively.

    batch_norm: bool, optional (default=False)
        If True, batch norm layer will be added after each linear layer
        in the order defined in scorch.nn.Dense.

    activation: torch activation function, optional (default=None)
        If not None, activation function will be added after each linear layer
        in the order defined in scorch.nn.Dense.

    dropout_rate: float, optional (default=None)
        If not None, dropout layer with specified dropout rate will be added
        after each linear layer in the order defined in scorch.nn.Dense.

    weight_init: initialiser from torch.nn.init, optional (default=torch.nn.init.xavier_uniform_)
        Used to initialise the weights in the linear layers.

    device: torch.device or string, optional (default=None)
        Device on which module will be placed.
        If string, must specify the device e.g. 'cpu' or 'cuda'.
        If None, GPU will be used if available, else CPU.

    Examples
    --------

    >>> import torch
    >>> import torch.nn.functional as F
    >>> from scorch.nn import MultiDense
    >>> X = torch.rand(5, 3)
    >>> dense = MultiDense(in_features=X.shape[1],
    >>>                    hidden_features=['auto', 4, 6],
    >>>                    batch_norm=True,
    >>>                    activation=F.relu,
    >>>                    dropout_rate=0.5,
    >>>                    device='cpu')
    >>> H = dense(X)
    >>> print(H.shape)
    torch.Size([5, 6])
    """

    def __init__(self,
                 in_features,
                 hidden_features,
                 batch_norm=False,
                 activation=None,
                 dropout_rate=None,
                 weight_init=nn.init.xavier_uniform_,
                 device=None):

        super(MultiDense, self).__init__()

        # initialise some instance variables
        self.in_features = in_features
        self.hidden_features = hidden_features
        self.device = get_device(device)

        # initialise layers
        self.layers = nn.ModuleList()
        d_in = in_features
        for d_out in self.hidden_features:
            if d_out == 'auto':
                d_out = d_in
            self.layers.append(Dense(in_features=d_in,
                                     out_features=d_out,
                                     batch_norm=batch_norm,
                                     activation=activation,
                                     dropout_rate=dropout_rate,
                                     weight_init=weight_init,
                                     device=self.device))
            d_in = d_out

        self.out_features = d_out

        # move module to device
        self.to(self.device)

    def forward(self, X):
        """
        Executes forward pass.

        For each layer, applies linear transformation > batch norm > activation > dropout.

        If batch norm or activation or dropout is None, will be missed out.

        Parameters
        ----------

        X: torch.FloatTensor, shape (n_rows, self.in_features)
            Input data.

        Returns
        -------

        X: torch.FloatTensor, shape (n_rows, self.out_features)
            Activations, where X[i] is the activation of the input X[i].
        """

        for layer in self.layers:
            X = layer(X)

        return X


class Output(nn.Module):
    """
    Linear output layer with option to add activation function and specify weight initialisation.

    Parameters
    ----------

    in_features: int
        Number of input features.

    out_dim: int
        Dimension of output.

    activation: torch activation function, optional (default=None)
        If not None, activation function will be applied to output.

    weight_init: initialiser from torch.nn.init, optional (default=torch.nn.init.xavier_uniform_)
        Used to initialise the weights in the linear layer.

    device: torch.device or string, optional (default=None)
        Device on which module will be placed.
        If string, must specify the device e.g. 'cpu' or 'cuda'.
        If None, GPU will be used if available, else CPU.

    Examples
    --------

    >>> import torch
    >>> from scorch.nn import Output
    >>> X = torch.rand(4, 3)
    >>> output = Output(in_features=X.shape[1],
    >>>                 out_dim=1,
    >>>                 activation=torch.sigmoid,
    >>>                 device='cpu')
    >>> y = output(X, activate=True)
    >>> print(y)
    tensor([0.3188, 0.1656, 0.1854, 0.4329], grad_fn=<SigmoidBackward>)
    """

    def __init__(self,
                 in_features,
                 out_dim,
                 activation=None,
                 weight_init=nn.init.xavier_uniform_,
                 device=None):

        super(Output, self).__init__()

        # initialise some instance variables
        self.in_features = in_features
        self.out_dim = out_dim
        self.activation = activation
        self.device = get_device(device)

        # initialise layer
        self.linear = Dense(in_features=in_features,
                            out_features=out_dim,
                            weight_init=weight_init,
                            device=self.device)

        # move module to device
        self.to(self.device)

    def forward(self, X, activate=False):
        """
        Executes forward pass.

        Applies linear transformation > activation.

        If self.activation is None or activate=False, activation will be missed out.

        Parameters
        ----------

        X: torch.FloatTensor, shape (n_rows, self.in_features)
            Input data.

        activate: bool, optional (default=False)
            If True, activation function will be applied to outputs.

        Returns
        -------

        y: torch.FloatTensor, shape (n_rows, self.out_dim)
            y[i] is the output for X[i].
        """

        y = self.linear(X).squeeze()
        if activate & (self.activation is not None):
            y = self.activation(y)

        return y


class MultiOutput(nn.Module):
    """
    Multiple output layers for multi-task networks.

    Parameters
    ----------

    in_features: int
        Number of input features.

    out_dims: int or list of ints
        Number of outputs.
        If list, len(output_dims) output layers are implemented,
        where output_dims[i] is the size of the i-th output layer.
        e.g. [1, 10] will implement two output layers, one of size 1 and another of size 10.

    activation: torch activation function, optional (default=None)
        If not None, activation function will be applied to each output.

    weight_init: initialiser from torch.nn.init, optional (default=torch.nn.init.xavier_uniform_)
        Used to initialise the weights in the linear layers.

    device: torch.device or string, optional (default=None)
        Device on which module will be placed.
        If string, must specify the device e.g. 'cpu' or 'cuda'.
        If None, GPU will be used if available, else CPU.

    Examples
    --------

    >>> import torch
    >>> from scorch.nn import MultiOutput
    >>> X = torch.rand(5, 3)
    >>> output = MultiOutput(in_features=X.shape[1],
    >>>                      out_dims=[1, 2],
    >>>                      activation=torch.sigmoid,
    >>>                      device='cpu')
    >>> y = output(X)
    >>> print(y[0].shape, y[1].shape)
    torch.Size([5]) torch.Size([5, 2])
    """

    def __init__(self,
                 in_features,
                 out_dims,
                 activation=None,
                 weight_init=nn.init.xavier_uniform_,
                 device=None):

        super(MultiOutput, self).__init__()

        if isinstance(out_dims, int):
            out_dims = [out_dims]

        # initialise some instance variables
        self.in_features = in_features
        self.out_dims = out_dims
        self.activation = activation
        self.device = get_device(device)
        
        # initialise layers
        self.linear = nn.ModuleList()
        for d in out_dims:
            self.linear.append(Output(in_features=in_features,
                                      out_dim=d,
                                      activation=activation,
                                      weight_init=weight_init,
                                      device=self.device))

        # move module to device
        self.to(self.device)

    def forward(self, X, activate=False):
        """
        Executes forward pass.

        For each output applies linear transformation > activation.

        If self.activation is None or activate=False, activation will be missed out.

        Parameters
        ----------

        X: torch.FloatTensor, shape (n_rows, self.in_features)
            Input data.

        activate: bool, optional (default=False)
            If True, activation function will be applied to outputs.

        Returns
        -------

        y: torch.FloatTensor or list of torch.FloatTensor
            If len(self.out_dims) == 1, y is a torch.FloatTensor of shape
            (n_rows, self.out_dims[0]), where y[j] is the output for X[j].
            If self.out_dims is a list, y is a list of length len(self.out_dims),
            where y[i] is a torch.FloatTensor of shape (n_rows, self.out_dims[i])
            and y[i][j] is the i-th layer's output for X[j].
        """

        y = [layer(X, activate=activate).squeeze() for layer in self.linear]
        if len(y) == 1:
            y = y[0]

        return y


class FeedForward(nn.Module):
    """
    Hidden layer(s) followed by output layer(s).

    Parameters
    ----------

    in_features: int
        Number of input features.

    hidden_dims: list, optional (default=None)
        A dense layer will be added for each element in the list,
        where hidden_dims[i] specifies the size of the i-th hidden layer.
        Elements in the list must either be integers or 'auto'.
        If 'auto', the size of the corresponding layer will be equal to the size of the previous layer.
        If None, the input layer will be connected directly to the output layer.
        e.g. ['auto', 128, 64] will implement three hidden layer, the first having the same size as
        the input layer followed by two layers of size 128 and 64 respectively.

    output_dims: int or list of ints, optional (default=1)
        Number of outputs.
        If list, len(output_dims) output layers are implemented,
        where output_dims[i] is the size of the i-th output layer.
        e.g. [1, 10] will implement two output layers, one of size 1 and another of size 10.

    dense_layer_params: dict, optional (default=None)
        Named parameters for dense layers.
        Of the form {param name (str): param value}.
        Correspond to parameters of scorch.nn.MultiDense module.
        If None, default parameters from scorch.utils.parameters will be used.

    output_layer_params: dict, optional (default=None)
        Named parameters for output layer.
        Of the form {param name (str): param value}.
        Correspond to parameters of scorch.nn.Output module.
        If None, default parameters from scorch.utils.parameters will be used.

    device: torch.device or string, optional (default=None)
        Device on which module will be placed.
        If string, must specify the device e.g. 'cpu' or 'cuda'.
        If None, GPU will be used if available, else CPU.

    Examples
    --------

    >>> import torch
    >>> import torch.nn.functional as F
    >>> from scorch.nn import FeedForward
    >>> X = torch.rand(5, 3)
    >>> output = FeedForward(in_features=X.shape[1],
    >>>                      hidden_dims=[8, 4],
    >>>                      output_dims=1,
    >>>                      dense_layer_params={'activation': F.relu, 'dropout_rate': 0.5},
    >>>                      output_layer_params={'activation': torch.sigmoid},
    >>>                      device='cpu')
    >>> y = output(X, activate_output=True)
    >>> print(y)
    tensor([0.4189, 0.1752, 0.2741, 0.3840, 0.4611], grad_fn=<SqueezeBackward0>)
    """

    def __init__(self,
                 in_features,
                 hidden_dims=None,
                 output_dims=1,
                 dense_layer_params=None,
                 output_layer_params=None,
                 device=None):

        super(FeedForward, self).__init__()

        # initialise some instance variables
        self.in_features = in_features
        self.hidden_dims = hidden_dims
        self.output_dims = output_dims
        self.device = get_device(device)

        # get default layer parameters if None
        dense_layer_params = dense_layer_params or default_parameters.dense_layer_params
        output_layer_params = output_layer_params or default_parameters.output_layer_params

        # initialise hidden layer(s)
        if hidden_dims is not None:
            self.hidden = MultiDense(in_features=in_features,
                                     hidden_features=hidden_dims,
                                     **dense_layer_params,
                                     device=self.device)
            d = self.hidden.out_features
        else:
            self.hidden = None
            d = in_features

        # initialise output layer(s)
        self.output = MultiOutput(in_features=d,
                                  out_dims=output_dims,
                                  **output_layer_params,
                                  device=self.device)

        # move model to device
        self.to(self.device)

    def forward(self,
                X,
                activate_output=False,
                return_activations=False):
        """
        Executes the forward pass.

        Parameters
        -------

        X: torch.FloatTensor, shape (n_rows, self.in_features)
            Input data.

        activate_output: bool, optional (default=False)
            If True, activation function will be applied to outputs.

        return_activations: bool, optional (default=False)
            If True the activations from the final hidden layer will be returned
            instead of outputs.

        Returns
        -------

        out: torch.FloatTensor or list of torch.FloatTensor
            If len(self.out_dims) == 1, y is a torch.FloatTensor of shape
            (n_rows, self.out_dims[0]), where y[j] is the output for X[j].
            If self.out_dims is a list, y is a list of length len(self.out_dims),
            where y[i] is a torch.FloatTensor of shape (n_rows, self.out_dims[i])
            and y[i][j] is the i-th layer's output for X[j].
        """

        if self.hidden is not None:
            X = self.hidden(X)

        if return_activations:
            return X

        out = self.output(X, activate=activate_output)

        return out
