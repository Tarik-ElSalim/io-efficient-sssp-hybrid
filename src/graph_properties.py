# graph_properties.py
"""
Graph utilities for CSR graphs: union-find (WCC detection) and reachability with BFS to select start nodes.

Main routines:
- bfs_reach_count_csr: directed BFS reachability count from a given source.
- best_random_node_in_largest_wcc_by_reach: sample nodes from the largest WCC and pick a start node with high reach.
- largest_wcc_mask: compute the largest weakly connected component (WCC) (returned as node ids).

Internal DSU helpers (Union-Find):
- _find, _union, _init_parent_rank, _union_from_csr, _compress_all, _bincount_roots.
"""

# Standard library
from typing import Tuple

# Third-party
import numpy as np
from numba import njit, prange


@njit(inline="always")
def _find(parent: np.ndarray, x: int) -> int:
    """
    Find parent of x.

    :param parent: Parent array of the DSU.
    :param x: Node id.
    :return: Root representative of x.
    """
    # while not found the root yet
    while parent[x] != x:
        # set parent of x to parent of the parent and save it in x
        parent[x] = parent[parent[x]]
        x = parent[x]
    # return the root of x in DSU
    return x


@njit(inline="always")
def _union(parent: np.ndarray, rank: np.ndarray, a: int, b: int) -> None:
    """
    Union operation of DSU by rank.

    :param parent: Parent array of the DSU.
    :param rank: Rank array.
    :param a: First node.
    :param b: Second node.
    :return: None, its in-place
    """
    # get the representative of both nodes a and b
    ra = _find(parent, a)
    rb = _find(parent, b)
    # if they are the same stop
    if ra == rb:
        return
    # Union, the component with the smaller rank will be the child
    if rank[ra] < rank[rb]:
        parent[ra] = rb
    elif rank[ra] > rank[rb]:
        parent[rb] = ra
    # if they have the same rank, let ra be the parent of rb and increase ra rank by 1
    else:
        parent[rb] = ra
        # check is important here because of limitation to uint8
        if rank[ra] < 255:
            rank[ra] += 1


@njit
def _init_parent_rank(n: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Initialize parent and rank arrays.

    :param n: Number of nodes.
    :return: parent[int32], rank np.uint8
    """
    # init empty parent array and rank array with 0´s of size n
    parent = np.empty(n, dtype=np.int32)
    rank = np.zeros(n, dtype=np.uint8)
    # init parent as the node self
    for i in range(n):
        parent[i] = i
    # return parent and rank array
    return parent, rank


@njit
def _union_from_csr(indptr: np.ndarray, indices: np.ndarray, parent: np.ndarray, rank: np.ndarray) -> None:
    """
    Single pass over CSR to union endpoints (treat directed edges as undirected).

    :param indptr: CSR row pointer.
    :param indices: Edge indices.
    :param parent: DSU parent array.
    :param rank: DSU rank array.
    :return: None, in-place
    """
    # get number of nodes
    n = indptr.size - 1
    # for each node
    for u in range(n):
        # get the nodes edge pointers
        s = indptr[u]
        e = indptr[u+1]
        # for every edge
        for k in range(s, e):
            # get the edge
            v = int(indices[k])
            # and union start + target node, undirected
            _union(parent, rank, u, v)


@njit(parallel=True)
def _compress_all(parent: np.ndarray) -> None:
    """
    Full path compression pass on all nodes (set all nodes to root for all components).

    :param parent: DSU parent array.
    :return: None, in-place
    """
    # get number of nodes
    n = parent.size

    # for each node, we calculate its root with parallel range
    for i in prange(n):
        parent[i] = _find(parent, i)


@njit
def _bincount_roots(parent: np.ndarray, n: int) -> np.ndarray:
    """
    Count nodes per root id ~ count how big each component is. Important, compress first.

    :param parent: roots for each node.
    :param n: Number of nodes.
    :return: counts[root] = size of that component.
    """
    # init array with 0 of size n
    counts = np.zeros(n, dtype=np.int32)
    # for every node
    for i in range(n):
        # increase count if root in parent array
        counts[parent[i]] += 1
    # return counts array, only the roots have counts
    return counts


@njit(cache=True)
def bfs_reach_count_csr(indptr: np.ndarray, indices: np.ndarray, source: int) -> int:
    """
    Count how many other nodes are reachable from source via bfs.

    :param indptr: CSR row pointer.
    :param indices: CSR edge indices.
    :param source: Source node id.
    :return: Number of reachable nodes excluding source.
    """
    # get number of nodes in graph
    n = indptr.size - 1

    # initialize visited array and queue for bfs
    visited = np.zeros(n, dtype=np.uint8)
    queue = np.empty(n, dtype=np.int32)

    # pointers for the queue because we use an array
    head = 0
    tail = 0

    # set visited at source position to 1
    visited[source] = 1
    # set queue at tail position to source and increase tail counter
    queue[tail] = source
    tail += 1

    # count var for counting reachable nodes
    count = 0

    # while there are entries in the queue that can be popped
    while head < tail:
        # pop u
        u = queue[head]
        # move head pointer
        head += 1

        # get the edge positions from CSR
        start = indptr[u]
        end = indptr[u + 1]

        # iterate over all edges
        for k in range(start, end):
            # get the edge target
            v = indices[k]
            # if not visited, then set to visited
            if visited[v] == 0:
                visited[v] = 1
                # and add to queue and increase tail pointer
                queue[tail] = v
                tail += 1
                # add 1 to count if we find not visited node
                count += 1

    # finally return count
    return count


def best_random_node_in_largest_wcc_by_reach(indptr: np.ndarray, indices: np.ndarray, m: int = 1000, *,
                                             early: bool = True, seed: int = 5) -> Tuple[int, int]:
    """
    Sample m nodes from the largest WCC and return the one with maximal
    (directed) BFS reach.

    Optionally early exit if a threshold of reachable nodes is exceeded.

    :param indptr: CSR row pointers
    :param indices: CSR edge indices
    :param m: Number of candidates to sample from the largest WCC. 1000 typically enough
    to find a good starting point, but this might be different for some graph kinds and sizes.
    :param early: If True, enable early exit.
    :param seed: Optional RNG seed for reproducibility.
    :return: best_node ~ the found node and the best_reach_excl_source ~ reach radius
    """
    # get number of nodes
    n = indptr.size - 1

    # in this part we get the wcc
    # each node is initialized as its own set
    parent, rank = _init_parent_rank(n)
    # now read all edges and build the wcc; u->v --> u<->v
    _union_from_csr(indptr, indices, parent, rank)
    # now for each wcc we set one root as representative
    _compress_all(parent)
    # calculate size of each wcc by its root
    counts = _bincount_roots(parent, n)
    # get the largest of the roots
    largest_root = int(np.argmax(counts))
    # create mask that shows what node belongs to the wcc
    mask = (parent == largest_root)
    # now get the node ids of all nodes in the wcc
    idx_in_wcc = np.flatnonzero(mask).astype(np.int32, copy=False)
    # finally get the size of the wcc
    wcc_size = int(idx_in_wcc.size)

    # now calculate thresholds for stopping bfs
    # edge to node ratio of whole graph; if to slow use wcc size instead of complete graph size
    mu = indices.size / float(n)
    # easy heuristic for stopping based on edge/node ratio ~ values between 0.1 and 0.4
    early_ratio = 0.1 + 0.3 * (1.0 - np.exp(-mu))
    # get threshold in total nodes if early else we have no threshold
    threshold = int(np.floor(early_ratio * n)) if early else -1

    # get random nodes and do bfs
    # get seed for choosing random node candidates
    rng = np.random.default_rng(seed)
    # if wcc is to small we set sample amount to wcc_size (if there are e.g. less than 1000 nodes in the wcc)
    m_eff = m if m <= wcc_size else wcc_size
    # now choose m_eff random ids from the wcc
    sel = rng.choice(wcc_size, size=m_eff, replace=False)
    # choose the appropriate candidates
    candidates = idx_in_wcc[sel]

    # set variables for return
    best_node = -1
    best_reach = -1

    # for all chosen nodes
    for u in candidates:
        # calculate how many nodes can be reached with bfs
        reach = bfs_reach_count_csr(indptr, indices, int(u))

        # if threshold is set and reach is bigger than threshold return the node and the number of reachable nodes
        if 0 <= threshold < reach:
            return int(u), int(reach)

        # else always update the best_reach and the corresponding node
        if reach > best_reach:
            best_reach = int(reach)
            best_node = int(u)

    # at the end if no early exit was done, return the best_node and the corresponding best reach
    return best_node, int(best_reach)


@njit(cache=True)
def largest_wcc_mask(indptr: np.ndarray, indices: np.ndarray) -> np.ndarray:
    """
    Compute the largest weakly connected component of a graph given in CSR form.
    Directed edges are treated as undirected when forming components.
    Returns the node ids (int32) of the largest weakly connected component.

    :param indptr: CSR row pointer array
    :param indices: CSR edge pointers
    :returns: int32 array of node ids in the largest WCC.
    """
    # get number of nodes
    n = indptr.size - 1

    # each node is initialized as its own set
    parent, rank = _init_parent_rank(n)

    # now read all edges and build the wcc; u->v --> u<->v
    _union_from_csr(indptr, indices, parent, rank)

    # now for each wcc we set one root as representative
    _compress_all(parent)

    # count what parent got the biggest component and sets root to this parent
    counts = _bincount_roots(parent, n)
    root = int(np.argmax(counts))

    # initialize n-size array
    mask = np.zeros(n, dtype=np.uint8)

    # now for each node check if it is in the component, if yes then set the mask = 1 else 0
    for i in range(n):
        if parent[i] == root:
            mask[i] = 1

    # return the mask
    return np.flatnonzero(mask).astype(np.int32)
