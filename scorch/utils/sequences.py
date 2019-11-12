import torch
import numpy as np

from scorch.utils.cuda import get_device


def pad_sequences(sequences,
                  lengths=None,
                  padding_val=-1,
                  device=None,
                  dtype=torch.long):
    """
    Pads list of variable length sequences.

    Parameters
    ----------

    sequences: list
        Contains variable length lists.

    lengths: list, optional (default=None)
        The length of each sequence.
        Has the same length as sequences, where lengths[i] = len(sequences[i]).

    padding_val: int (default=-1)
        Value to use for padding sequences.

    device: torch.device or string, optional (default=None)
        Device on which padded sequences will be placed.
        If string, must specify the device e.g. 'cpu' or 'cuda'.
        If None, GPU will be used if available, else CPU.

    dtype: torch data type, optional (default=torch.long)
        Data type of padded sequences.

    Returns
    -------

    padded_sequences: torch.LongTensor, shape (len(sequences), max(lengths))
        Padded sequences arranged in rows, where padded_sequences[i]
        is the padded version of sequences[i].

    Examples
    --------

    >>> from scorch.utils.sequences import pad_sequences
    >>> sequences = [[1, 5, 2, 6],
    >>>              [1, 4],
    >>>              [7],
    >>>              [4, 6, 2, 5, 9, 3]]
    >>> lengths = [len(x) for x in sequences]
    >>> padded_sequences = pad_sequences(sequences)
    >>> print(padded_sequences.shape)
    torch.Size([4, 6])
    """

    # get device
    device = get_device(device)

    if lengths is None:
        # get length of each sequence
        lengths = list(map(len, sequences))

    # convert to tensor
    lengths = torch.tensor(lengths, dtype=torch.long, device=device)

    # create array into which we will put the sequences
    num_rows = len(sequences)
    num_cols = int(lengths.max())
    padded_sequences = padding_val * \
        torch.ones((num_rows, num_cols), dtype=dtype, device=device)

    # create mask specifying which positions in this array will take values from the sequences
    mask = lengths[:, None] > torch.arange(
        num_cols, dtype=torch.long, device=device)

    # put values in array
    padded_sequences[mask] = torch.from_numpy(
        np.concatenate(sequences)).type(dtype).to(device)

    return padded_sequences
