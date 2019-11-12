

def get_sons(dag, node):
    """
    Gets the sons of a node in a directed acyclic graph.

    Parameters
    ----------

    dag: dict
        Directed acyclic graph of the form {node: [sons]}.

    node: int
        Node index.
        Must be a key in dag.

    Returns
    -------

    sons: list
        Sons of node.

    Examples
    --------

    >>> from scorch.utils.recursion import get_sons
    >>> dag = {
    >>>         0: [1, 2, 3],
    >>>         1: [2, 3],
    >>>         2: [3]
    >>> }
    >>> print(get_sons(dag, 0))
    >>> [1, 2, 3]

    >>> print(get_sons(dag, 5))
    >>> []
    """

    sons = dag.get(node, [])

    return sons


def get_nodes(dag):
    """
    Returns all nodes in a directed acyclic graph.

    Parameters
    ----------

    dag: dict
        Directed acyclic graph of the form {node: [sons]}.

    Returns
    -------

    node: list
        All nodes in dag.

    Returns
    -------

    >>> from scorch.utils.recursion import get_nodes
    >>> dag = {
    >>>         0: [1, 2, 3],
    >>>         1: [2, 3],
    >>>         2: [3]
    >>> }
    >>> print(get_nodes(dag))
    [0, 1, 2, 3]
    """

    nodes = list(set(list(dag.keys()) + [i for l in dag.values() for i in l]))

    return nodes
