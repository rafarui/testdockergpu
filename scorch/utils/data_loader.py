import torch
import torch.nn as nn
import numpy as np
import gc
from abc import ABC

from scorch.utils.sequences import pad_sequences
from scorch.utils.recursion import get_sons
from scorch.utils.cuda import get_device


class DataFrameLoader(object):
    """
    Retrieves batches of data from Pandas DataFrame.

    The DataFrame should include features and optionally one or more
    targets and sample weights.

    The feature columns can be either numeric (floats) or categorical (integers).


    Parameters
    ----------

    df: Pandas DataFrame
        Each row is a data sample.

    num_cols: list of strings, optional (default=None)
        Specifies names of numeric columns in df.
        These columns should contain floats.

    cat_cols: list of strings, optional (default=None)
        Specifies names of categorical columns in df.
        These columns should contain integers.

    target_cols: string or list of strings, optional (default=None)
        Specifies names of target column(s) in df.

    target_types: torch data type or list of torch types, optional (default=torch.float)
        Specifies the type of the target column(s).
        If target_cols is a single string then must be a single torch data type.
        If target_cols is a list then must have len(target_types) = len(target_cols)
        where target_types[i] is the data type of target_cols[i].

    sample_weight_cols: string or list of strings, optional (default=None)
        Specifies name of sample weight column(s) in df.
        If target_cols is a single string then must be also be a single string.
        If target_cols is a list then must have len(sample_weight_cols) = len(target_cols)
        where sample_weight_cols[i] gives the sample weights for target_cols[i].

    batch_size: int, optional (default=32)
        Number of samples in each batch.

    shuffle: boolean, optional (default=True)
        Whether or not to shuffle samples at the start of each epoch.

    random_seed: integer, optional (default=42)
        Numpy random seed to use when shuffling.

    device: torch.device or string, optional (default=None)
        Device on which data will be placed.
        If string, must specify the device e.g. 'cpu' or 'cuda'.
        If None, GPU will be used if available, else CPU.

    Examples
    --------

    >>> import numpy as np
    >>> import pandas as pd
    >>> import torch
    >>> from scorch.utils.data_loader import DataFrameLoader
    >>> num_rows = 5
    >>> df = pd.DataFrame({
    >>>     'num1': np.random.rand(num_rows),
    >>>     'num2': np.random.rand(num_rows),
    >>>     'num3': np.random.rand(num_rows),
    >>>     'cat1': np.random.randint(0, 10, num_rows),
    >>>     'cat2': np.random.randint(0, 10, num_rows),
    >>>     'y1': np.random.rand(num_rows),
    >>>     'y2': (np.random.rand(num_rows)).astype(int) > 0.5,
    >>>     'w1': np.random.rand(num_rows),
    >>>     'w2': np.random.rand(num_rows),
    >>> })
    >>> data_loader = DataFrameLoader(df,
    >>>                               num_cols=['num1', 'num2', 'num3'],
    >>>                               cat_cols=['cat1', 'cat2'],
    >>>                               target_cols=['y1', 'y2'],
    >>>                               target_types=[torch.float, torch.long],
    >>>                               sample_weight_cols=['w1', 'w2'])
    >>> data = data_loader.next_batch()
    >>> print(data.keys())
    dict_keys(['rows', 'X_num', 'X_cat', 'y', 'sample_weight'])
    """

    def __init__(self,
                 df,
                 num_cols=None,
                 cat_cols=None,
                 target_cols=None,
                 target_types=torch.float,
                 sample_weight_cols=None,
                 batch_size=32,
                 shuffle=True,
                 random_seed=42,
                 device=None):

        # sort the columns by name
        # data will be returned with columns in this order
        if num_cols is not None:
            num_cols = sorted(num_cols)
        if cat_cols is not None:
            cat_cols = sorted(cat_cols)

        # initialise some instance variables
        self.num_cols = num_cols
        self.cat_cols = cat_cols
        self.target_cols = target_cols
        self.target_types = target_types
        self.sample_weight_cols = sample_weight_cols
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.random_seed = random_seed
        self.device = get_device(device)

        # set random seed
        np.random.seed(random_seed)

        if num_cols is not None:
            # convert numeric features to torch tensor
            self.X_num = torch.from_numpy(
                df[num_cols].values).float().to(self.device)
        else:
            self.X_num = None

        if cat_cols is not None:
            # convert categorical features to torch tensor
            self.X_cat = torch.from_numpy(
                df[cat_cols].values).long().to(self.device)
        else:
            self.X_cat = None

        if target_cols is not None:
            if np.isscalar(target_cols):
                # convert target to torch tensor
                self.y = torch.from_numpy(df[target_cols].values).type(
                    target_types).to(self.device)
            else:
                # convert each target to torch tensor and store in a list
                self.y = [torch.from_numpy(df[col].values).type(target_types[i]).to(self.device)
                          for i, col in enumerate(target_cols)]
        else:
            self.y = None

        if sample_weight_cols is not None:
            if np.isscalar(sample_weight_cols):
                # convert sample_weight to torch tensor
                self.sample_weight = torch.from_numpy(
                    df[sample_weight_cols].values).float().to(self.device)
            else:
                # convert each sample weight vector to torch tensor and store in a list
                self.sample_weight = [torch.from_numpy(df[col].values).float().to(self.device)
                                      for col in sample_weight_cols]
        else:
            self.sample_weight = None

        # get number of batches
        self.num_rows = len(df)
        self.num_batches = int(np.floor(self.num_rows / batch_size))

        # initialise row order
        self.row_order = np.arange(self.num_rows)

        # reset
        self.reset()

    def reset(self):
        """
        Resets the data loader, ready to begin a new epoch.
        """

        self.epoch_complete = False
        self.next_batch_index = 0
        if self.shuffle:
            self.row_order = self.row_order[np.random.choice(
                self.num_rows, self.num_rows, replace=False)]

    def next_batch(self):
        """
        Retrieves the next batch of data.

        Returns
        -------

        batch_data: dict
            Has the following fields:

            rows: np.array
                Indices of rows in batch.

            X_num: torch.FloatTensor, shape (self.batch_size, len(self.num_cols))
                Numeric features.
                Columns are ordered according to the alphabetical order of their names.

            X_cat: torch.LongTensor, shape (self.batch_size, len(self.cat_cols))
                Categorical features.
                Columns are ordered according to the alphabetical order of their names.

            y: torch.Tensor or list or torch.Tensor, shape (self.batch_size,)
                Data labels.
                Will only be present if self.target_cols not None.
                If self.target_cols is a string, a single tensor with data type
                self.target_types will be returned.
                If self.target_cols is a list, a list of tensors of length
                len(self.target_cols) will be returned,
                where y[i] has data type self.target_types[i].

            sample_weight: torch.FloatTensor or list or torch.FloatTensor, shape (self.batch_size,)
                Sample weights.
                Will only be present if self.sample_weight_cols not None.
                If self.target_cols is a string, a single tensor will be returned.
                If self.target_cols is a list, a list of tensors of length
                len(self.target_cols) will be returned, where sample_weight[i]
                contains the sample weights corresponding to y[i].
        """

        if self.epoch_complete is False:

            batch_data = {}

            # get rows in this batch
            i_first = self.batch_size * self.next_batch_index
            if self.next_batch_index + 1 < self.num_batches:
                i_last = self.batch_size * (self.next_batch_index + 1)
            else:
                # this is the last batch
                i_last = self.num_rows
                self.epoch_complete = True
            rows = self.row_order[i_first:i_last]
            batch_data['rows'] = rows

            if self.X_num is not None:
                # get numeric data in this batch
                batch_data['X_num'] = self.X_num[rows]
            else:
                batch_data['X_num'] = torch.tensor([]).float().to(self.device)

            if self.X_cat is not None:
                # get categorical data in this batch
                batch_data['X_cat'] = self.X_cat[rows]
            else:
                batch_data['X_cat'] = torch.tensor([]).float().to(self.device)

            if self.y is not None:
                # get labels in this batch
                if isinstance(self.y, list):
                    batch_data['y'] = [y[rows] for y in self.y]
                else:
                    batch_data['y'] = self.y[rows]

            if self.sample_weight is not None:
                # get sample weights in this batch
                if isinstance(self.sample_weight, list):
                    batch_data['sample_weight'] = [w[rows]
                                                   for w in self.sample_weight]
                else:
                    batch_data['sample_weight'] = self.sample_weight[rows]

            # increase batch index
            self.next_batch_index += 1

        else:
            print('Epoch Complete. Reset batch loader to start new epoch.')
            batch_data = {}

        return batch_data


class RelationalDataFrameLoader(object):
    """
    Retrieves batches of data from a relational DataFrame structure.

    - The tables are assumed to be indexed from 0 to k, with table 0 assumed to be the main table.

    - The relations between tables are defined by a directed acyclic graph (DAG).

    - For each pair of related tables we define the relations between their rows.

    - Input features in each table can be numeric, categorical or both.

    - Each batch returns a subset of the main table's rows as well as all related rows from the other tables.

    Parameters
    ----------

    Xs: list of DataFrames
        A DataFrame for each node in dag.
        Xs[i] is the DataFrame corresponding to node represented by i in dag.
        Xs[0] is the main table.

    dag: dict
        Defines relationships between tables in a directed acyclic graph format.
        Each element is of the form {table index (int): [indices of child tables]}.
        e.g. dag = {
                    0: [1, 2, 3],
                    1: [2, 3],
                    2: [3],
                }
        means that tables 1, 2 and 3 are children of table 0,
        tables 2 and 3 are children of table 1
        and table 3 is a child of table 2.

    row_relations: dict
        Keys are tuples of length 2 which represent relations in dag and
        values are lists of lists specifying the relations between the rows
        of the DataFrames specified by the tuple.
        e.g. {(1, 2): [[4, 6, 2], [], [1, 2], ....]} specifies the row relations
        between Xs[1] and Xs[2]. Here the first row of Xs[1] is related to rows
        4, 6 and 2 of Xs[2], the second row of Xs[1] isn't related to any rows of
        Xs[2], the third row of Xs[1] is related to rows 1 and 2 of Xs[2], and so on.

    features: dict
        Specifies the numeric and categorical features to use from each DataFrame.
        Of the form {DataFrame index (int): {'num': [numeric features], 'cat': [categorical features]}}.
        e.g. {0: {'num': ['x1', 'x2'], 'cat': ['x3']}, 1: {'num': [], 'cat': ['x4']}}

    target: string, optional (default=None)
        Name of target column.

    target_type: torch data type or string, optional (default='torch.FloatTensor')
        Data type of target.

    time_cols: list of strings, optional (default=None)
        If not None, must be the same length as Xs and time_cols[i] should be
        the name of the column in Xs[i] which specifies the time between each row of
        Xs[i] and some fixed reference date.
        The reference date must be the same for all tables.
        If given, the time between related rows will be computed and included
        in the numeric features of all tables except the main table.

    sample_weight: string, optional (default=None)
        Name of the column in Xs[0] which contains the sample weights.

    batch_size: int, optional (default=32)
        Number of samples in each batch.

    random_seed: int, optional (default=42)
        Numpy random seed to use when shuffling.

    shuffle_samples: boolean, optional (default=True)
        Whether or not to shuffle samples before creating batches.

    shuffle_batches: boolean, optional (default=True)
        Whether or not to shuffle batch order at the beginning of every new epoch.

     device: torch.device or string, optional (default=None)
        Device on which data will be placed.
        If string, must specify the device e.g. 'cpu' or 'cuda'.
        If None, GPU will be used if available, else CPU.

    Examples
    --------

    >>> import numpy as np
    >>> import pandas as pd
    >>> from scorch.utils.data_loader import RelationalDataFrameLoader
    >>> dag = {0: [1, 2], 1: [2]}
    >>> num_rows = [5, 8, 4]
    >>> X0 = pd.DataFrame({
    >>>     'num1': np.random.rand(num_rows[0]),
    >>>     'num2': np.random.rand(num_rows[0]),
    >>>     'num3': np.random.rand(num_rows[0]),
    >>>     'cat1': np.random.randint(0, 10, num_rows[0]),
    >>>     'cat2': np.random.randint(0, 10, num_rows[0]),
    >>>     'y': (np.random.rand(num_rows[0])).astype(int) > 0.5,
    >>>     't': np.random.rand(num_rows[0]),
    >>>     'w': np.random.rand(num_rows[0]),
    >>> })
    >>> X1 = pd.DataFrame({
    >>>     'num1': np.random.rand(num_rows[1]),
    >>>     'num2': np.random.rand(num_rows[1]),
    >>>     't': np.random.rand(num_rows[1]),
    >>> })
    >>> X2 = pd.DataFrame({
    >>>     'cat1': np.random.randint(0, 10, num_rows[2]),
    >>>      't': np.random.rand(num_rows[2]),
    >>> })
    >>> Xs = [X0, X1, X2]
    >>> row_relations = {}
    >>> row_relations[(0, 1)] = [[], [3, 4], [1, 2, 6], [], [4]]
    >>> row_relations[(0, 2)] = [[1], [3], [], [], [0, 1]]
    >>> row_relations[(1, 2)] = [[2, 3], [1], [2], [], [1], [1, 3], [], [0]]
    >>> features = {}
    >>> features[0] = {'num': ['num1', 'num2', 'num3'], 'cat': ['cat1', 'cat2']}
    >>> features[1] = {'num': ['num1', 'num2'], 'cat': []}
    >>> features[2] = {'num': [], 'cat': ['cat1']}
    >>> data_loader = RelationalDataFrameLoader(Xs,
    >>>                                         dag,
    >>>                                         row_relations,
    >>>                                         features,
    >>>                                         target='y',
    >>>                                         time_cols=['t', 't', 't'],
    >>>                                         sample_weight='w')
    >>> data = data_loader.next_batch()
    >>> print(data.keys())
    dict_keys(['rows', 'maps', 'Xs', 'y', 'sample_weight'])
    """

    def __init__(self,
                 Xs,
                 dag,
                 row_relations,
                 features,
                 target=None,
                 target_type='torch.FloatTensor',
                 time_cols=None,
                 sample_weight=None,
                 batch_size=32,
                 random_seed=42,
                 shuffle_samples=True,
                 shuffle_batches=True,
                 device=None):

        # initialise some instance variables
        self.dag = dag
        self.batch_size = batch_size
        self.random_seed = random_seed
        self.shuffle_samples = shuffle_samples
        self.shuffle_batches = shuffle_batches
        self.num_tables = len(Xs)
        self.num_rows = len(Xs[0])
        self.device = get_device(device)

        # convert DataFrames to torch tensors
        self.Xs_num = []
        self.Xs_cat = []
        for i in range(self.num_tables):

            # get sorted lists of features
            # the columns will be returned in alphabetical order
            num_cols = sorted(features[i]['num'])
            cat_cols = sorted(features[i]['cat'])

            if len(num_cols) > 0:
                # get numeric features
                X_num = torch.from_numpy(
                    Xs[i][num_cols].values.astype(float)).float().to(self.device)
                if len(X_num.size()) == 1:
                    X_num = X_num.view(-1, 1)
            else:
                X_num = torch.tensor([], dtype=torch.float, device=self.device)

            if len(cat_cols) > 0:
                # get categorical features
                X_cat = torch.from_numpy(
                    Xs[i][cat_cols].values.astype(int)).long().to(self.device)
                if len(X_cat.size()) == 1:
                    X_cat = X_cat.view(-1, 1)
            else:
                X_cat = torch.tensor([], dtype=torch.long, device=self.device)

            # append
            self.Xs_num.append(X_num)
            self.Xs_cat.append(X_cat)

        if target is not None:
            # get targets
            self.y = torch.from_numpy(Xs[0][target].values.astype(
                float)).type(target_type).to(self.device)
        else:
            self.y = None

        if sample_weight is not None:
            # get sample weights
            self.sample_weight = torch.from_numpy(
                Xs[0][sample_weight].values.astype(float)).float().to(self.device)
        else:
            self.sample_weight = None

        if time_cols is not None:
            # get times
            ts = [torch.from_numpy(Xs[i][col].values.astype(float)).float().to(self.device).reshape((-1, 1))
                  for i, col in enumerate(time_cols)]
        else:
            ts = None

        # clean up
        del Xs
        gc.collect()

        # set random seed
        np.random.seed(random_seed)

        # initialise batches
        self.initialise_nested_batches(row_relations, ts)

        # reset
        self.reset()

    def initialise_nested_batches(self, row_relations, ts=None):
        """
        Initialises the rows and row map for each batch.

        Parameters
        ----------

        row_relations: dict
            Keys are tuples of length 2 which represent relations in dag and
            values are lists of lists specifying the relations between the rows
            of the DataFrames specified by the tuple.
            e.g. {(1, 2): [[4, 6, 2], [], [1, 2], ....]} specifies the row relations
            between Xs[1] and Xs[2]. Here the first row of Xs[1] is related to rows
            4, 6 and 2 of Xs[2], the second row of Xs[1] isn't related to any rows of
            Xs[2], the third row of Xs[1] is related to rows 1 and 2 of Xs[2], and so on.

        ts: list of torch.FloatTensor, optional (default=None)
            If not None, must be a list where ts[i] is a 1D torch.FloatTensor
            the same length as Xs[i], where ts[i][j] is a number specifying the time
            between the j-th row of Xs[i] and some fixed reference date.
            The reference date must be the same for all tables.
        """

        # make sure row relations are np arrays
        row_relations = {key: np.array(val) for key, val in row_relations.items()}

        # initialise row order
        row_order = np.arange(self.num_rows)

        # shuffle data?
        if self.shuffle_samples:
            row_order = row_order[np.random.choice(
                self.num_rows, self.num_rows, replace=False)]

        # get number of batches
        num_batches = int(np.ceil(self.num_rows / self.batch_size))

        # construct batches one-by-one in a loop
        self.batch_rows = {}
        for batch in range(num_batches):

            if batch % 100 == 0:
                print('Initialising batch {} of {}...'.format(batch + 1, num_batches))

            # initialise variables for this batch
            Xs_rows = {i: np.array([], dtype='int32')
                       for i in range(self.num_tables)}
            row_maps = {}
            if ts is not None:
                ts_diffs = {i: torch.tensor([], dtype=torch.float, device=self.device)
                            for i in range(self.num_tables)}
            else:
                ts_diffs = None

            # get rows of principal table in batch
            i_first = self.batch_size * batch
            i_last = min(self.num_rows,
                         self.batch_size * (batch + 1))
            rows = row_order[i_first:i_last]

            # get nested rows
            self.get_nested_rows(row_relations, Xs_rows, row_maps,
                                (0,), rows, ts, ts_diffs)
            self.batch_rows[batch] = {'rows': Xs_rows, 'maps': row_maps}
            if ts is not None:
                self.batch_rows[batch]['ts'] = ts_diffs

        # initialise batch order
        self.batch_order = np.array(list(self.batch_rows.keys()))

    def get_nested_rows(self, row_relations, Xs_rows, row_maps, trace, rows, ts=None, ts_diffs=None):
        """
        Recursively adds rows and row maps to batch data by following the traces
        in self.dag which defines relations between multiple tables.

        Parameters
        ----------

        row_relations: dict
            Keys are tuples of length 2 which represent relations in dag and
            values are lists of lists specifying the relations between the rows
            of the DataFrames specified by the tuple.
            e.g. {(1, 2): [[4, 6, 2], [], [1, 2], ....]} specifies the row relations
            between Xs[1] and Xs[2]. Here the first row of Xs[1] is related to rows
            4, 6 and 2 of Xs[2], the second row of Xs[1] isn't related to any rows of
            Xs[2], the third row of Xs[1] is related to rows 1 and 2 of Xs[2], and so on.

        Xs_rows: dict
            Has the form {i (int): np.array}, where i specifies the table index
            and np.array specifies the rows of X[i] which are in the batch.
            Array specifying the rows starts off empty and rows are recursively added.

        row_maps: dict
            Has the form {tuple: np.array}, where tuple defines a path in self.dag and np.array
            specifies the mapping between the rows of the pair of tables corresponding to the final
            two indices in the tuple.
            Starts of empty and rows are recursively added.
            e.g. {(0, 1, 2, 3): v} specifies the mapping between the rows of Xs[2] and Xs[3]
            in the path 0 --> 1 --> 2 --> 3, where v has the same length as Xs[3] and
            v[i] = j means that row i of Xs[3] is related to row j of Xs[2].
            If v[i] < 0 then row i of Xs[3] is not related to any row in Xs[2].

        trace: tuple
            Specifies the current path in self.dag.
            e.g. (0, 1, 2, 3) specifies the path 0 --> 1 --> 2 --> 3.

        rows: np.array
            Specifies rows of second last table in trace which are in the batch.
            e.g. if trace is (0, 1, 2, 3) then specifies the rows
            of Xs[2] which are in the batch along this trace.

        ts: list of torch.FloatTensor, optional (default=None)
            If not None, must be a list where ts[i] is a 1D torch.FloatTensor
            the same length as Xs[i], where ts[i][j] is a number specifying the time
            between the j-th row of Xs[i] and some fixed reference date.
            The reference date must be the same for all tables.

        ts_diffs: dict, optional (default=None)
            Has the form {i: np.array}, where i specifies the table index
            and np.array specifies the time difference between the rows of
            table i which are in the batch and the related rows of its parent table.
            Arrays starts off empty and rows are recursively added.
        """

        node = trace[-1]
        if trace == (0,):
            # get rows of principal table
            Xs_rows[0] = rows

        else:
            parent = trace[-2]

            if ts is not None:
                # get times corresponding to rows from parent table
                t1 = ts[parent][rows]

            # get rows in child table which are related to rows in parent table,
            # and a map which specifies, for each row in the child table,
            # which row is it related to in the parent table,
            # e.g. row_map[i] = j means that the i-th row in the child table
            # is related to the j-th row in the parent table
            n = len(rows)
            rows, row_map = self.get_related_rows(row_relations[trace[-2:]][rows])

            if ts is not None:
                # compute the time difference between related rows
                t1 = t1[row_map]
                t2 = ts[node][rows]
                ts_diffs[node] = torch.cat((ts_diffs[node], t1 - t2), dim=0)

            # rows from the parent table may already have been added to the batch data
            # for a different relation with the same parent table, so we need to reindex
            # the row map by adding the difference between the total number of rows in
            # the parent table's batch data and the number of rows in the parent table's
            # batch data which are related to rows in this child table
            row_map += len(Xs_rows[parent]) - n

            # the row map must be the same length as the total number of rows in the
            # child table's batch data, but if rows from the child table have already
            # been added to the batch data for different relations involving this same
            # child table then the length of the row map will be less than the total
            # number of rows in the child table's batch data, therefore we need to
            # left-pad the row map with -1s (this tells the nn to ignore them)
            row_map = np.concatenate(
                (-1 * np.ones(len(Xs_rows[node]), dtype='int32'), row_map))

            # update the row map dict
            row_maps[trace] = row_map

            # append rows of child table to batch data
            Xs_rows[node] = np.concatenate((Xs_rows[node], rows))

            # as we have just appended rows to the child table's batch data
            # any existing row maps for this same child table will now be shorter
            # that the total number of rows in the child table's batch data,
            # we therefore need to right-pad such maps with -1s
            for key, val in row_maps.items():
                if (key[-1] == node) & (key != trace):
                    row_maps[key] = np.concatenate(
                        (row_maps[key], -1 * np.ones(len(rows), dtype='int32')))

        # recurse
        [self.get_nested_rows(row_relations, Xs_rows, row_maps, trace + (son,), rows, ts, ts_diffs)
         for son in get_sons(self.dag, node)]

    @staticmethod
    def get_related_rows(relations):
        """
        Gets rows of child table which are related to parent table as well as a row map
        which specifies, for each related row in the child table, which row of the parent
        table it is related to.

        Parameters
        ----------

        relations: array of arrays
            Specifies the relations between two tables.
            Has the same length as the parent table and relations[i] is an array specifying
            which rows in the child table are related to the i-th row of the parent table.
            e.g. [[4, 6, 2], [], [1, 2], ....]} means that the first row of the parent table
            is related to rows 4, 6 and 2 of the child table, the second row of the parent table
            isn't related to any rows of the child table, the third row of the parent table
            is related to rows 1 and 2 of the child table, and so on.

        Returns
        -------

        rows: np.array
            Indices of rows in child table which are related to some row in parent table.

        row_map: np.array
            Has same length as rows. Specifies the mapping between rows of the parent and child.
            row_map[i] = j indicates that rows[i] of the child is related to the j-th row of the parent.
        """

        if len(relations) > 0:
            rows = np.concatenate(relations).astype('int32')
            row_map = np.concatenate(
                [[i] * len(x) for i, x in enumerate(relations)]).astype('int32')
        else:
            rows, row_map = np.array(
                [], dtype='int32'), np.array([], dtype='int32')

        return rows, row_map

    def reset(self):
        """
        Resets the data loader, ready to begin a new epoch.
        """

        self.epoch_complete = False
        self.next_batch_index = 0

        if self.shuffle_batches:
            # shuffle the order of the batches
            num_batches = len(self.batch_order)
            self.batch_order = self.batch_order[np.random.choice(
                num_batches, num_batches, replace=False)]

    def next_batch(self):
        """
        Gets the next batch of data.

        Returns
        -------

        data: dict
            Batch data with the following keys:
                'Xs': dict
                    Data features for all tables.
                    Keys are table indices.
                    data['Xs'][i] is a dictionary containing the numeric
                    and categorical features of the i-th table, of the form
                    {'num': torch.FloatTensor, 'cat': torch.LongTensor}.
                    The columns in data['Xs'][i]['num'] correspond to the
                    column names specified in self.features[i], and the
                    columns are ordered according to the alphabetical order of their names.
                    Similarly for data['Xs'][i]['cat'].
                    If table does not contain any numeric features,
                    data['Xs'][i]['num'] will be an empty torch.FloatTensor.
                    If table does not contain any categorical features,
                    data['Xs'][i]['cat'] will be an empty torch.LongTensor.

                'y': torch.Tensor
                    Data labels.
                    y[j] is the target corresponding to data['Xs'][i]['num'][j]
                    and data['Xs'][i]['cat'][j].
                    Only returned if self.y not None.

                'w': torch.FloatTensor
                    Sample weights.
                    w[j] is the sample weight corresponding to data['Xs'][i]['num'][j]
                    and data['Xs'][i]['cat'][j].
                    Only returned if self.w not None.

                'rows': np.array
                    Indices of rows of Xs[0] which are in the batch.

                'maps': dict
                    Has the form {tuple: np.array}, where tuple defines a path in the DAG and the array
                    specifies the mapping between the rows of the pair of tensors corresponding to the final
                    two indices in the tuple.
                    e.g. {(0, 1, 2, 3): v} specifies the mapping between the batch rows of Xs[2] and Xs[3]
                    in the path 0 --> 1 --> 2 --> 3, where v has the same length as Xs[3] and
                    v[i] = j means that Xs[3][i] is related to Xs[2][j].
                    If v[i] < 0 then Xs[3][i] is not related to any row in Xs[i].
        """

        data = {}

        # get batch rows and row maps
        batch_index = self.batch_order[self.next_batch_index]
        rows = self.batch_rows[batch_index]['rows']
        data['rows'] = rows[0]
        data['maps'] = self.batch_rows[batch_index]['maps']
        data['Xs'] = {}

        # get features from each table
        for i in range(self.num_tables):

            if len(rows[i]) > 0:

                if len(self.Xs_num[i]) > 0:
                    # get numeric features
                    X_num = self.Xs_num[i][rows[i]]
                else:
                    X_num = torch.tensor(
                        [], dtype=torch.float, device=self.device)

                if len(self.Xs_cat[i]) > 0:
                    # get categorical features
                    X_cat = self.Xs_cat[i][rows[i]]
                else:
                    X_cat = torch.tensor(
                        [], dtype=torch.long, device=self.device)

                if (i > 0) & ('ts' in self.batch_rows[batch_index]):
                    # add time difference to numeric features
                    ts = self.batch_rows[batch_index]['ts'][i]
                    X_num = torch.cat((X_num, ts), dim=1)

            else:
                X_num = torch.tensor([], dtype=torch.float, device=self.device)
                X_cat = torch.tensor([], dtype=torch.long, device=self.device)

            data['Xs'][i] = {'num': X_num, 'cat': X_cat}

        if self.y is not None:
            # get targets
            data['y'] = self.y[rows[0]]

        if self.sample_weight is not None:
            # get sample weight
            data['sample_weight'] = self.sample_weight[rows[0]]

        # increase batch index
        self.next_batch_index += 1

        # epoch complete?
        if self.next_batch_index == len(self.batch_order):
            self.epoch_complete = True

        return data

    def to(self, device):
        """
        Moves all tensors in the data loader to the specified device.

        Parameters
        ----------

        device: torch.device
            Tensors will be moved to this device.
        """

        self.Xs_num = [x.to(device) for x in self.Xs_num]
        self.Xs_cat = [x.to(device) for x in self.Xs_cat]

        if 'ts' in self.batch_rows[0]:
            for i in self.batch_rows.keys():
                self.batch_rows[i]['ts'] = {
                    j: x.to(device) for j, x in self.batch_rows[i]['ts'].items()}

        if self.y is not None:
            self.y = self.y.to(device)

        if self.sample_weight is not None:
            self.sample_weight = self.sample_weight.to(device)

        self.device = device

    def cpu(self):
        """
        Moves all tensors in the data loader to the CPU.
        """
        self.to(torch.device("cpu"))

    def cuda(self):
        """
        Moves all tensors in the data loader to the GPU.
        """
        self.to(torch.device("cuda"))


class SequenceLoader(object):
    """
    Retrieves batches of sequential data,
    with option to include structured data as well.

    Parameters
    ----------

    sequences: list of lists
        Each list must be a sequence of integers.

    padding_value: int
        Value used to pad the end of variable length sequences.

    targets: np.array or list of np.arrays, optional (default=None)
        If array, must have the same length as sequences,
        with targets[i] being the target for sequences[i].
        If list of arrays, each array must have the same length as sequences,
        with targets[i][j] being the i-th target for sequences[j] (for multi-task networks).

    target_types: torch type or list of torch types, optional(default=None)
        Specifies the type of the target column(s).
        If targets is a single np.array then must be a single torch data type.
        If targets is a list then must have len(target_types) = len(targets)
        where target_types[i] is the data type of targets[i].

    df: Pandas DataFrame, optional (default=None)
        Complementary structured data for sequences.
        If not None, each row is a data sample,
        with df.loc[i, :] being structured data for sequences[i].

    num_cols: list of strings, optional (default=None)
        Specifies names of numeric features in df.

    cat_cols: list of strings, optional (default=None)
        Specifies names of categorical features in df.

    sample_weight: np.array or list of np.arrays, optional (default=None)
        If array, must have the same length as sequences,
        with sample_weight[i] being the weight for targets[i].
        If list of arrays, each array must have the same length as sequences,
        with sample_weight[i][j] being the weight for targets[i][j] (for multi-task networks).

    batch_size: integer, optional (default=32)
        Number of samples in each batch.

    meta_batch_size: integer, optional (default=10000)
        Number of samples in each meta-batch.
        If group_by_length=True and shuffle=True, examples are shuffled
        within each meta-batch at the start of every epoch.

    group_by_length: boolean, optional (default=True)
        If True, batches will contain sequences of similar lengths.
        This can significantly speed up training for large datasets.

    batch_first: boolean, optional (default=False)
        If True, padded sequences in batch will have shape
        (batch_size, max_length), else (max_length, batch_size).

    shuffle: boolean, optional (default=True)
        Whether or not to shuffle samples at the start of each epoch.

    random_seed: integer (default=42)
        Numpy random seed for shuffling samples.

    device: torch.device or string, optional (default=None)
        Device on which data will be placed.
        If string, must specify the device e.g. 'cpu' or 'cuda'.
        If None, GPU will be used if available, else CPU.

    Examples
    --------

    >>> import numpy as np
    >>> import pandas as pd
    >>> import torch
    >>> from scorch.utils.data_loader import SequenceLoader
    >>> num_rows = 5
    >>> sequences = [list(np.random.randint(1, 10, np.random.randint(10))) for _ in range(num_rows)]
    >>> df = pd.DataFrame({
    >>>         'num1': np.random.rand(num_rows),
    >>>         'num2': np.random.rand(num_rows),
    >>>         'num3': np.random.rand(num_rows),
    >>>         'cat1': np.random.randint(0, 10, num_rows),
    >>>         'cat2': np.random.randint(0, 10, num_rows),
    >>> })
    >>> targets = [np.random.rand(num_rows), (np.random.rand(num_rows) > 0).astype(int)]
    >>> target_types = [torch.float, torch.long]
    >>> sample_weight = [np.random.rand(num_rows), np.random.rand(num_rows)]
    >>> data_loader = SequenceLoader(sequences,
    >>>                              padding_value=0,
    >>>                              targets=targets,
    >>>                              target_types=target_types,
    >>>                              df=df,
    >>>                              num_cols=['num1', 'num2', 'num3'],
    >>>                              cat_cols=['cat1', 'cat2'],
    >>>                              sample_weight=sample_weight)
    >>> data = data_loader.next_batch()
    >>> print(data.keys())
    dict_keys(['batch_first', 'rows', 'S', 'i_end', 'X_num', 'X_cat', 'y', 'sample_weight'])
    """

    def __init__(self,
                 sequences,
                 padding_value=0,
                 targets=None,
                 target_types=None,
                 df=None,
                 num_cols=None,
                 cat_cols=None,
                 sample_weight=None,
                 batch_size=32,
                 meta_batch_size=10000,
                 group_by_length=True,
                 batch_first=False,
                 shuffle=True,
                 random_seed=42,
                 device=None):

        # sort the columns by name
        # data will be returned with columns in this order
        if num_cols is not None:
            num_cols = sorted(num_cols)
        if cat_cols is not None:
            cat_cols = sorted(cat_cols)

        # initialise some instance variables
        self.padding_value = padding_value
        self.target_types = target_types
        self.num_cols = num_cols
        self.cat_cols = cat_cols
        self.batch_size = batch_size
        self.meta_batch_size = meta_batch_size
        self.group_by_length = group_by_length
        self.batch_first = batch_first
        self.shuffle = shuffle
        self.random_seed = random_seed
        self.device = get_device(device)

        # set random seed
        np.random.seed(random_seed)

        # convert sequences to list of torch tensors
        self.sequences = [torch.tensor(
            x, dtype=torch.long, device=self.device) for x in sequences]

        # get sequence lengths
        self.sequence_lengths = np.array(
            [max(len(x), 1) for x in self.sequences])

        if (df is not None) & (num_cols is not None):
            # convert numeric features to torch tensor
            self.X_num = torch.from_numpy(
                df[num_cols].values).float().to(self.device)
        else:
            self.X_num = None

        if (df is not None) & (cat_cols is not None):
            # convert categorical features to torch tensor
            self.X_cat = torch.from_numpy(
                df[cat_cols].values).long().to(self.device)
        else:
            self.X_cat = None

        if targets is not None:
            if isinstance(targets, list):
                # convert each target vector to torch tensor and store in a list
                self.y = [torch.from_numpy(y).type(target_types[i]).to(self.device)
                          for i, y in enumerate(targets)]
            else:
                # convert target vector to torch tensor
                self.y = torch.from_numpy(targets).type(
                    target_types).to(self.device)
        else:
            self.y = None

        if sample_weight is not None:
            if isinstance(sample_weight, list):
                # convert each weight vector to torch tensor and store in a list
                self.sample_weight = [torch.from_numpy(w).float().to(self.device)
                                      for i, w in enumerate(sample_weight)]
            else:
                # convert weight vector to torch tensor
                self.sample_weight = torch.from_numpy(
                    sample_weight).float().to(self.device)
        else:
            self.sample_weight = None

        # get number of batches
        self.num_rows = len(sequences)
        self.num_batches = max(int(np.floor(self.num_rows / batch_size)), 1)

        # get number of meta batches
        self.num_meta_batches = max(int(np.ceil(self.num_rows / meta_batch_size)), 1)

        # initialise row order
        if group_by_length:
            self.row_order = np.argsort(self.sequence_lengths)
        else:
            self.row_order = np.arange(self.num_rows)
        self.batch_row_order = self.row_order

        # initialise batch order
        self.batch_order = np.arange(self.num_batches)

        # reset
        self.reset()

    def reset(self):
        """
        Resets the data loader, ready to begin a new epoch.
        """

        self.epoch_complete = False
        self.next_batch_index = 0
        if self.shuffle:
            # shuffle batch order
            self.batch_order = np.random.choice(
                self.num_batches, self.num_batches, replace=False)

            if self.group_by_length:

                # for each meta batch...
                for n in range(self.num_meta_batches):

                    # shuffle within batch
                    i_first = self.meta_batch_size * n
                    i_last = min(self.num_rows, self.meta_batch_size * (n + 1))
                    i = np.arange(i_first, i_last)
                    self.batch_row_order[i] = self.row_order[i][np.random.choice(
                        len(i), len(i), replace=False)]

            else:
                # shuffle all rows
                self.row_order = self.row_order[np.random.choice(
                    self.num_rows, self.num_rows, replace=False)]

    def next_batch(self):
        """
        Retrieves the next batch of data.

        Returns
        -------

        batch_data: dict
            Has the following fields:

                rows: np.array
                    Indices of rows in batch.

                S: torch.LongTensor, shape (seq_len, batch_size)
                    Padded sequences.

                batch_first: bool
                    Indicates whether the sequences in S are arrange batch first or not.
                    If True, S will have shape (batch_size, seq_len),
                    else (seq_len, batch_size).

                i_end: np.array
                    Has length S.size(1), where i_end[i] is the index of the
                    last non-padded element of the sequence S[:, i].

                X_num: torch.FloatTensor, shape (self.batch_size, len(self.num_cols))
                    Numeric features.
                    Columns are ordered according to the alphabetical order of their names.

                X_cat: torch.LongTensor, shape (self.batch_size, len(self.cat_cols))
                    Categorical features.
                    Columns are ordered according to the alphabetical order of their names.

                y: torch.Tensor or list or torch.Tensor, shape (self.batch_size,)
                    Data labels.
                    Will only be present if self.y not None.
                    If self.y is a single tensor, a single tensor with data type
                    self.target_types will be returned.
                    If self.y is a list, a list of tensors of length
                    len(self.y) will be returned,
                    where y[i] has data type self.target_types[i].

                sample_weight: torch.FloatTensor or list or torch.FloatTensor, shape (self.batch_size,)
                    Sample weights.
                    Will only be present if self.sample_weight not None.
                    If self.sample_weight is a single tensor, a single tensor will be returned.
                    If self.sample_weight is a list, a list of tensors of length
                    len(self.sample_weight) will be returned, where sample_weight[i]
                    contains the sample weights corresponding to y[i].
        """

        if self.epoch_complete is False:

            batch_data = {'batch_first': self.batch_first}

            # get rows in this batch
            batch = self.batch_order[self.next_batch_index]
            i_first = self.batch_size * batch
            if batch == self.num_batches - 1:
                i_last = self.num_rows
            else:
                i_last = self.batch_size * (batch + 1)
            rows = self.batch_row_order[i_first:i_last]
            batch_data['rows'] = rows

            # get sequences
            sequences = [self.sequences[i] for i in rows]

            # pad sequences
            batch_data['S'] = nn.utils.rnn.pad_sequence(sequences,
                                                        padding_value=self.padding_value,
                                                        batch_first=self.batch_first).to(self.device)

            # get index of the end of each sequence
            batch_data['i_end'] = self.sequence_lengths[rows] - 1

            if self.X_num is not None:
                # get numeric data in this batch
                batch_data['X_num'] = self.X_num[rows]
            else:
                batch_data['X_num'] = torch.tensor([]).float().to(self.device)

            if self.X_cat is not None:
                # get categorical data in this batch
                batch_data['X_cat'] = self.X_cat[rows]
            else:
                batch_data['X_cat'] = torch.tensor([]).float().to(self.device)

            if self.y is not None:
                # get labels in this batch
                if isinstance(self.y, list):
                    batch_data['y'] = [y[rows] for y in self.y]
                else:
                    batch_data['y'] = self.y[rows]

            if self.sample_weight is not None:
                # get sample weights in this batch
                if isinstance(self.sample_weight, list):
                    batch_data['sample_weight'] = [w[rows]
                                                   for w in self.sample_weight]
                else:
                    batch_data['sample_weight'] = self.sample_weight[rows]

            # increase batch index
            self.next_batch_index += 1

            if self.next_batch_index == self.num_batches:
                self.epoch_complete = True

        else:
            print('Epoch Complete. Reset batch loader to start new epoch.')
            batch_data = {}

        return batch_data
