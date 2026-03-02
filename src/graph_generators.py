# graph_generators.py
"""Graph generators and CSR helpers using rustworkx and NumPy.

Main routines:
- gnp_directed_csr: directed Erdős–Rényi G(n, p) generator (CSR output).
- barabasi_albert_csr: directed Barabási–Albert generator (CSR output).
- random_geometric_csr: random geometric graph, symmetrized to directed (CSR output).
- directed_scale_free_csr: fixed-size directed scale-free generator (CSR output).
- create_graph_by_gen: creates graph by given parameters and generator type

CSR / edge helpers:
- _pack_struct_edges: pack parallel (u, v, w) arrays into a structured edge array.
- edges_to_csr_io: convert structured edges (u, v, w) to CSR arrays (not sorted).
- _csr_from_edges: build CSR from (rows, cols, weights) via SciPy sparse matrices.
- _weights: weight sampler (uniform[0,1) or Exp(1)).
"""


# Standard library
from typing import cast, Tuple

# Third-party
import numpy as np
from numpy.typing import NDArray
import rustworkx as rx
import scipy.sparse as sp
from scipy.spatial import cKDTree
from numba import njit


def _pack_struct_edges(u: np.ndarray, v: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Pack parallel arrays (u, v, w) into a structured edge array [(u,v,w)...]

    :param u: Source node IDs - int32
    :param v: Target node IDs - int32
    :param w: Edge weights - float 32
    :return: Structured array with dtype [(u, int32), (v, int32), (w, float32)].
    """
    m = u.shape[0]
    # initialize empty edge array
    edges = np.empty(m, dtype=np.dtype([('u', np.int32), ('v', np.int32), ('w', np.float32)]))
    # pack the 3 arrays into it
    edges['u'] = u.astype(np.int32, copy=False)
    edges['v'] = v.astype(np.int32, copy=False)
    edges['w'] = w.astype(np.float32, copy=False)
    # return the edges array
    return edges


def edges_to_csr_io(edges: np.ndarray, indices_dtype=np.int32,
                    indptr_dtype=np.int32, weight_dtype=np.float32) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert structured edges (u,v,w) to CSR (indptr, indices, weights). - NOT sorted

    :param edges: Structured edge array with fields u, v, w.
    :param indices_dtype: dtype for CSR indices array.
    :param indptr_dtype: dtype for CSR indptr array.
    :param weight_dtype: dtype for CSR weight array.
    :return: indptr[n+1], indices[m], weights[m]
    """
    # get start,target and source as single arrays
    u = edges['u']
    v = edges['v']
    w = edges['w']

    # get #nodes by max node id
    max_u = int(u.max())
    max_v = int(v.max())
    n = max(max_u, max_v) + 1

    # get edges by counting number of u entries
    m = u.shape[0]
    # count outgoing edges by node
    counts = np.bincount(u, minlength=n).astype(indptr_dtype, copy=False)

    # initialize index pointer and fill via prefix sums
    indptr = np.empty(n + 1, dtype=indptr_dtype)
    indptr[0] = 0
    np.cumsum(counts, out=indptr[1:])

    # initialize empty indices and weights arrays
    indices = np.empty(m, dtype=indices_dtype)
    weights = np.empty(m, dtype=weight_dtype)

    # copy of index pointer for in-place writing
    write_pos = indptr[:-1].copy()

    # for every edge
    for src, dst, ww in zip(u, v, w):
        # get the position of the edge out of write_pos
        p = write_pos[src]
        # fill the indices arr with the dest node at the right position
        indices[p] = dst
        # also fill the weights arr
        weights[p] = ww
        # increment write_pos by 1 for this node, so we will get the correct index for the next time we see this node
        write_pos[src] = p + 1

    # return the csr arrays
    return indptr, indices, weights


def _weights(m: int, *, mode: str, seed: int, dtype) -> np.ndarray:
    """Easy weight function that draw random weights in either exponential(rate=1) or uniform[0,1).

    :param m: Number of weights.
    :param mode: exp for exponential, uniform for uniform.
    :param seed: RNG seed for reproducibility.
    :param dtype: Target dtype
    :return: Array of weights with shape and given dtype.
    """
    # get seed
    rng = np.random.default_rng(seed)

    # if exp mode
    if mode == "exp":
        # create random number between 0 and 1
        u = rng.random(m)
        # if one of the rnd numbers is 0 replace it with the next greater one
        u[u == 0.0] = np.nextafter(0.0, 1.0)
        # return negative log of the number; for float 32 this is between 0 and 744
        return (-np.log(u)).astype(dtype, copy=False)

    # if mode uniform create an array of length m with random numbers between 0 and 1 and return it (0 not included)
    elif mode == "uniform":
        x = rng.random(m).astype(dtype, copy=False)
        x[x == 0] = np.nextafter(dtype(0), dtype(1))
        return x
    # else error
    else:
        raise ValueError("weight mode must be 'exp' or 'uniform'")


def _csr_from_edges(n: int, rows: np.ndarray, cols: np.ndarray, w: np.ndarray,
                    *, indptr_dtype=np.int32, indices_dtype=np.int32, weight_dtype=np.float32)\
                    -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build CSR from source array (rows), target array (cols), weighted array (w).

    :param n: Number of nodes
    :param rows: Row indices (sources).
    :param cols: Column indices (targets).
    :param w: Edge weights.
    :return: (indptr[n+1], indices[m], weights[m])
    """
    # create a sparse adjacency matrix (only non 0 entries)
    coo = sp.coo_matrix((w.astype(weight_dtype, copy=False),
                         (rows.astype(np.int64, copy=False),
                          cols.astype(np.int64, copy=False))),
                        shape=(n, n))
    # converts sparse matrix to csr formatter
    csr = coo.tocsr()

    # returns the csr
    return (csr.indptr.astype(indptr_dtype, copy=False),
            csr.indices.astype(indices_dtype, copy=False),
            csr.data.astype(weight_dtype, copy=False))


def gnp_directed_csr(n: int, p: float, *, seed: int = -1, exponential_weights: bool = False,
                     ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate a directed G(n, p) graph and return CSR arrays.

    :param n: Number of nodes.
    :param p: Edge probability.
    :param seed: RNG seed; -1 for default RNG graph_generators
    :param exponential_weights: If True use Exp(1) weights, else uniform[0,1).
    :return: (indptr[n+1], indices[m], weights[m]) ~ CSR
    """
    # create the random graph with rustwork
    g = rx.directed_gnp_random_graph(n, p, seed=None if seed < 0 else seed)
    # gets all edges (u,v) out of the rustwork graph
    elist: list[tuple[int, int]] = g.edge_list()
    # get number of edges
    m = len(elist)

    # out of the edge list create 2 numpy arrays with src (rows) -> target (cols) at position i in both arrays
    rows = np.fromiter((u for u, _ in elist), dtype=np.int64, count=m)
    cols = np.fromiter((v for _, v in elist), dtype=np.int64, count=m)

    # finally based on the weight mode calculate the weight array
    w = _weights(m, mode="exp" if exponential_weights else "uniform", seed=seed, dtype=np.float32)

    # take the 3 arrays and convert them to csr with csr from edges function
    return _csr_from_edges(n, rows, cols, w)


def barabasi_albert_csr(n: int, m_attach: int, *, seed: int = -1, exponential_weights: bool = False
                        ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate a directed Barabasi–Albert graph and return CSR arrays.

    :param n: Number of nodes
    :param m_attach: New node attaches to m_attach existing nodes.
    :param seed: RNG seed; -1 if you want standard library seed
    :param exponential_weights: If True use Exp(1) weights, else uniform[0,1).
    :return: (indptr[n+1], indices[m], weights[m])
    """
    # create directed Barabasi–Albert graph in rustworkx and get the edge list
    g = rx.directed_barabasi_albert_graph(n, m_attach, seed=None if seed < 0 else seed)
    elist: list[tuple[int, int]] = g.edge_list()
    m = len(elist)

    # create source, target arrays and calculate weights
    rows = np.fromiter((u for u, _ in elist), dtype=np.int64, count=m)
    cols = np.fromiter((v for _, v in elist), dtype=np.int64, count=m)
    w = _weights(m, mode="exp" if exponential_weights else "uniform", seed=seed, dtype=np.float32)

    # return as csr
    return _csr_from_edges(n, rows, cols, w)


def random_geometric_csr(n: int, radius: float, *, seed: int = -1, exponential_weights: bool = True,
                         ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate an undirected random geometric graph and symmetrize to directed.

    Each undirected edge (u, v) is converted to two directed edges u→v and v→u
    with the same weight. We use SciPy because it is a lot faster than rustworkx here.

    :param n: Number of nodes.
    :param radius: Connection radius
    :param seed: RNG seed; -1 if you want standard library seed
    :param exponential_weights: If True use Exp(1) weights, else uniform[0,1).
    :return: (indptr[n+1], indices[m], weights[m])
    """
    # create rng based on seed
    rng = np.random.default_rng(None if seed is None or seed < 0 else seed)

    # create random points in [0,1]^2
    x = rng.random((n, 2), dtype=np.float32)

    # build k-d-tree and find neighbors within radius (edges)
    tree = cKDTree(x, leafsize=32, compact_nodes=True, balanced_tree=True)
    coo = cast(
        sp.coo_matrix,
        tree.sparse_distance_matrix(tree, max_distance=radius, output_type="coo_matrix")
    )

    # extract only one direction of the neighbour edges
    u = coo.row
    v = coo.col
    mask = u < v
    # save edges as arrays and get #undirected edges
    u_arr = u[mask]
    v_arr = v[mask]
    m_und = u_arr.size

    # set weights for each undirected edge
    w_und = _weights(m_und, mode="exp" if exponential_weights else "uniform", seed=seed, dtype=np.float32)

    # undirected -> directed; duplicate each edge; create empty array of double size and duplicate rows,cols reverse
    rows = np.empty(2 * m_und, dtype=np.int64)
    cols = np.empty(2 * m_und, dtype=np.int64)
    w = np.empty(2 * m_und, dtype=np.float32)

    rows[0::2] = u_arr.astype(np.int64, copy=False)
    cols[0::2] = v_arr.astype(np.int64, copy=False)
    w[0::2] = w_und

    rows[1::2] = v_arr.astype(np.int64, copy=False)
    cols[1::2] = u_arr.astype(np.int64, copy=False)
    w[1::2] = w_und

    # build CSR and return it
    return _csr_from_edges(n, rows, cols, w)


@njit(cache=True)
def _directed_scale_free_uv_until_n(alpha: float, beta: float, gamma: float, delta_in: float,
                                    delta_out: float, seed: int, exp_weights: int, target_n: int):
    """Directed scale-free model with fixed number of nodes target_n.

    Adds exactly one edge per iteration; node count grows only on alpha/gamma events
    and stops once cur_n == target_n. The total number of edges is not fixed. Please
    take care to set alpha, beta, gamma correctly (alpha + beta + gamma = 1)


    :param alpha: Probability of adding a new target node.
    :param beta: Probability of adding an internal edge.
    :param gamma: Probability of adding a new source node.
    :param delta_in: Initial attractiveness for in-degree.
    :param delta_out: Initial attractiveness for out-degree.
    :param seed: RNG seed; -1 for default RNG
    :param exp_weights: 1 for Exp(1) weights, 0 for uniform[0,1).
    :param target_n: Desired number of nodes.
    :return: (u[m], v[m], w[m], n) with n == target_n.
    """

    # get the random seed if set
    if seed >= 0:
        np.random.seed(seed)

    # set target number of nodes and a tiny epsilon
    n = target_n
    eps = np.float64(1e-12)

    # init draw array for drawing (new) source,target nodes fast
    # binary indexed tree for fast update and fast prefix sum calculation for drawing random nodes based on degree
    # bit i contains sum of [i - lowbit(i) +1, i]; with lowbit(i) = i & -i ~ most right bit that is set to 1 in i
    bit_in = np.zeros(n + 1, np.float64)
    bit_out = np.zeros(n + 1, np.float64)

    # easy intern help function to increase the bit_in/bit_out array vals fast O(log(n))
    def bit_add(bit, i, delta):
        # because we work with start index = 1, we start with i+1
        i += 1
        # get correct size of the bit array
        m = bit.size - 1
        # update all following affected entries
        while i <= m:
            # add delta to i val
            bit[i] += delta
            # go the next i val, that is affected by the change of this i val
            i += i & -i

    # second easy help function for fast (O(logn)) finding a node to a given prefix sum in the bit array
    def bit_find_prefix(bit, prefsum):
        # get array size without 0 index
        m = bit.size - 1
        # stores biggest id so that the prefix sum is less than x
        idx = 0
        # find the highest power of two smaller than m
        step = 1
        # stop when step is bigger
        while step <= m:
            step <<= 1
        # and go back one stop, so we have exactly the highest power of two smaller than m
        step >>= 1
        # binary search like search for correct prefix index
        while step != 0:
            # check if we can go step to the right without surpassing our prefix sum prefsum
            t = idx + step
            # if we do not surpass our prefix sum and are in bound, we move idx to the right and halve our step
            if t <= m and bit[t] < prefsum:
                prefsum -= bit[t]
                idx = t
            step >>= 1
        # return the biggest id, so that the prefix sum is just smaller tan x
        return idx

    # init first node
    cur_n = 1
    # set bitin and bitout of first node to delta_in, delta_out ~ no edge only one node
    bit_add(bit_in,  0, delta_in)
    bit_add(bit_out, 0, delta_out)
    # set total_in and total_out to delta_in/delta_out
    total_in = delta_in
    total_out = delta_out

    # calculation of expected edges to choose appropriate array size
    # prob of getting a new node in each iteration
    denom = alpha + gamma
    # expected edges is total nodes/prob of new node per round
    expect_edges = (n - 1) / denom
    # set generous cap
    cap = int(2.0 * expect_edges + 8.0 * n + 64.0)

    # create empty u,v,w arrays
    u = np.empty(cap, np.int64)
    v = np.empty(cap, np.int64)
    w = np.empty(cap, np.float32)
    e = 0

    # now as long as we did not reach our desired #nodes
    while cur_n < n:
        # if we have more edges than cap, we need to increase our cap size; create copy of all arrays and set new cap
        if e >= cap:
            new_cap = int(cap * 1.6) + 32
            uu = np.empty(new_cap, np.int64)
            uu[:cap] = u
            vv = np.empty(new_cap, np.int64)
            vv[:cap] = v
            ww = np.empty(new_cap, np.float32)
            ww[:cap] = w
            u, v, w = uu, vv, ww
            cap = new_cap
        # get random float64 between [0,1)
        r = np.random.random()

        # now based on r choose what draw event happens
        if r < alpha:
            # alpha event, create new node and exactly attach one edge to it
            # draw random float between 0 and total_out
            x = np.random.random() * total_out
            # choose source node that will be connected with the new target node based on x with eps as fallback if x=0
            uu = bit_find_prefix(bit_out, x if x > 0.0 else eps)
            # new node get next available node id
            vv = cur_n
            # the new node now gets the initial delta_in/delta_out val, so it can be drawn, also increase total_in/out
            bit_add(bit_in,  vv, delta_in)
            total_in += delta_in
            bit_add(bit_out, vv, delta_out)
            total_out += delta_out
            # increase cur counter
            cur_n += 1

        # beta event, no new node we will create an internal edge
        elif r < alpha + beta:
            # choose two random values x,y and find the two corresponding nodes via prefix find
            x = np.random.random() * total_out
            uu = bit_find_prefix(bit_out, x if x > 0.0 else eps)
            y = np.random.random() * total_in
            vv = bit_find_prefix(bit_in,  y if y > 0.0 else eps)
            # no self loops, change to 0 or to next node; create minimal bias
            if uu == vv:
                vv = 0 if (vv + 1 >= cur_n) else (vv + 1)

        else:
            # gamma event, new node created with outgoing edge
            # get random val between 0 and total_in
            y = np.random.random() * total_in
            # prefix find the corresponding node
            vv = bit_find_prefix(bit_in, y if y > 0.0 else eps)
            # give the new node the next available index
            uu = cur_n
            # init bit_in/bit_out vals for the new node and increase total_in/out
            bit_add(bit_in,  uu, delta_in)
            total_in += delta_in
            bit_add(bit_out, uu, delta_out)
            total_out += delta_out
            # increase cur available node id counter
            cur_n += 1

        # here the new created edge is written into the edge array
        u[e] = uu
        v[e] = vv
        # update the corresponding bit_in/bit_out values of the affected nodes and total_in/out
        bit_add(bit_out, uu, 1.0)
        total_out += 1.0
        bit_add(bit_in,  vv, 1.0)
        total_in += 1.0

        # set weight based on random val and increase edge counter
        ran = np.random.random()
        w[e] = np.float32(ran if (exp_weights == 0) else -np.log(ran if ran > eps else eps))
        e += 1

    # return u,v,w arrays up to number of edges and n
    return u[:e], v[:e], w[:e], n


def directed_scale_free_csr(alpha: float = 1/16, beta: float = 14/16, gamma: float = 1/16, delta_in: float = 4/3,
                            delta_out: float = 4/3, *, seed: int = -1, exponential_weights: bool = False,
                            nodes: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Directed scale-free generator (fixed nodes) -->  CSR output.

    Parameters are identical to the Numba kernel above.
    :return: (indptr[n+1], indices[m], weights[m]) CSR arrays
    """

    # use the intern function above to create the source,target and weight arrays
    us, vs, ws, n = _directed_scale_free_uv_until_n(
        alpha, beta, gamma, delta_in, delta_out, seed,
        1 if exponential_weights else 0,
        target_n=nodes
    )

    # remove multi-edges (keep min weight per (u,v))
    key = (us.astype(np.uint64) << 32) | vs.astype(np.uint64)
    o = np.argsort(key)
    key = key[o]
    us = us[o]
    vs = vs[o]
    ws = ws[o]
    s = np.r_[True, key[1:] != key[:-1]]
    i = np.flatnonzero(s)
    us = us[i]
    vs = vs[i]
    ws = np.minimum.reduceat(ws, i)

    # pack (u,v,w) into numpy array
    edges = _pack_struct_edges(us, vs, ws)

    # convert the edges array to csr
    return edges_to_csr_io(edges)


def create_graph_by_gen(gen: str, num_nodes: int, seed: int, exp_weights: bool, prob: float, radius: float,
                        m_attach: int) -> Tuple[NDArray[np.int32], NDArray[np.int32], NDArray[np.float32]]:
    """
    Build a CSR graph using the selected generator and parameters.
    :param gen: "ScaleFree" | "Barabasi" | "Gnp" | "Geometric".
    :param num_nodes: Number of nodes.
    :param seed: RNG seed.
    :param exp_weights: True → exponential weights, False → uniform
    :param prob: Edge probability for Gnp.
    :param radius: Connection radius for Geometric.
    :param m_attach: m parameter for Barabasi–Albert.
    :return: (indptr[int32 n+1], indices[int32 m], weights[float32 m]).
    """
    # If queries and call up the appropriate generator
    if gen == "ScaleFree":
        indptr, indices, weights = directed_scale_free_csr(seed=seed, exponential_weights=exp_weights, nodes=num_nodes)
    elif gen == "Barabasi":
        indptr, indices, weights = barabasi_albert_csr(n=num_nodes, m_attach=m_attach, seed=seed,
                                                       exponential_weights=exp_weights)
    elif gen == "Gnp":
        indptr, indices, weights = gnp_directed_csr(n=num_nodes, p=prob, seed=seed,
                                                    exponential_weights=exp_weights)
    elif gen == "Geometric":
        indptr, indices, weights = random_geometric_csr(n=num_nodes, radius=radius, seed=seed,
                                                        exponential_weights=exp_weights)
    else:
        print("Unknown graph generator")
        return (np.array([], dtype=np.int32), np.array([], dtype=np.int32),
                np.array([], dtype=np.float32))

    # Returns graph in csr format
    return indptr, indices, weights
