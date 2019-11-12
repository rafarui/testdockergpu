import torch


def get_device(device=None):
    """
    Initialises torch device given string, or default device if None.

    Parameters
    ----------

    device: torch.device or str, optional (default=None)
        If torch.device, will be returned unmodified.
        If string, should specify the device e.g. 'cpu' or 'cuda'.
        If None, GPU will be used if available, else CPU.

    Returns
    -------

    device: torch.device
        Initialised device.

    Examples
    --------

    >>> from scorch.utils.cuda import get_device
    >>> print(get_device(None))
    cuda:0

    >>> print(get_device('cpu'))
    cpu

    >>> print(get_device('cuda:0'))
    cuda:0

    >>> import torch
    >>> device = torch.device('cpu')
    >>> print(get_device(device))
    cpu
    """

    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    elif isinstance(device, str):
        device = torch.device(device)

    return device


def move_module_to_device(module, device=None):
    """
    Moves module and all its submodules to specified device.

    Parameters
    ----------

    module: torch.nn.Module
        Module.

    device: torch.device or str, optional (default=None)
        If torch.device, will be returned unmodified.
        If string, should specify the device e.g. 'cpu' or 'cuda'.
        If None, GPU will be used if available, else CPU.

    Examples
    --------

    >>> from scorch.utils.cuda import move_module_to_device
    >>> from scorch.nn import MultiDense
    >>> dense = MultiDense(in_features=6,
    >>>                    hidden_features=[8, 8],
    >>>                    batch_norm=True,
    >>>                    dropout_rate=0.5,
    >>>                    device='cpu')
    >>> move_module_to_device(dense, 'cuda')
    >>> print(dense.device)
    cuda
    """

    device = get_device(device)
    module.to(device)
    module.device = device
    for sub_module in module.modules():
        sub_module.device = device


def move_module_to_cuda(module):
    """
    Moves module and all its submodules to GPU.

    Parameters
    ----------

    module: torch.nn.Module
        Module.

    Examples
    --------

    >>> from scorch.utils.cuda import move_module_to_cuda
    >>> from scorch.nn import MultiDense
    >>> dense = MultiDense(in_features=6,
    >>>                    hidden_features=[8, 8],
    >>>                    batch_norm=True,
    >>>                    dropout_rate=0.5,
    >>>                    device='cpu')
    >>> move_module_to_cuda(dense)
    >>> print(dense.device)
    cuda
    """
    move_module_to_device(module, device='cuda')


def move_module_to_cpu(module):
    """
    Moves module and all its submodules to CPU.

    Parameters
    ----------

    module: torch.nn.Module
        Module.

    Examples
    --------

    >>> from scorch.utils.cuda import move_module_to_cpu
    >>> from scorch.nn import MultiDense
    >>> dense = MultiDense(in_features=6,
    >>>                    hidden_features=[8, 8],
    >>>                    batch_norm=True,
    >>>                    dropout_rate=0.5,
    >>>                    device='cuda')
    >>> move_module_to_cpu(dense)
    >>> print(dense.device)
    cpu
    """
    move_module_to_device(module, device='cpu')
