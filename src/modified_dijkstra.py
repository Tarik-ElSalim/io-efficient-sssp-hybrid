# modified_dijkstra.py
"""
Numba-accelerated shortest-path utilities for CSR graphs (directed, non-negative weights).

This module contains a fast Dijkstra variant (lazy binary heap), helpers to materialize and analyze the
resulting shortest-path tree (SPT),as well as a simple SEM-style I/O cost model.

Main routines:
- core_dijkstra_with_edges_tree / dijkstra_with_edges_tree: Dijkstra on CSR with a lazy min-heap. Tracks statistics.

- build_tree_csr_from_pred: Builds SPT based on core_dijkstra_with_edges_tree results

- level_stats_from_settled: Calculates the stats of each SPT level based on build_tree_csr_from_pred results

- calc_real_dijkstra_cost: Calculates the I/O-Cost of Dijkstra based on more a realistic SEM approach
    (Not used in the experiments, included for the sake of completeness)

Notes:
- All algorithms assume non-negative edge weights (Dijkstra correctness).
- The SEM I/O accounting is intentionally simplified: Dijkstra pop ≈ 1 I/O, BF round ≈ e_b I/Os.
"""

# Third party
import numpy as np
from numba import njit
from numpy.typing import NDArray

# Local application
from .heap_numba import ensure_push, heap_pop3

# inf definitions for float 64 and int 32 -- for placeholders in arrays
INF = np.float64(np.inf)
U32INF = np.uint32(0xFFFFFFFF)


@njit(cache=True, fastmath=True)
def core_dijkstra_with_edges_tree(indptr: NDArray[np.int32], indices: NDArray[np.int32], weights: NDArray[np.float32],
                                  source: int, limit: int, use_bf: np.uint8, bf_dist: NDArray[np.float64],
                                  bf_settled: NDArray[np.uint8], bf_active_ids: NDArray[np.int32], e_b_border=-1
                                  ) -> tuple[NDArray[np.float64], NDArray[np.uint32], NDArray[np.int32],
                                             NDArray[np.uint8], NDArray[np.int32], NDArray[np.int32]]:
    """
    Dijkstra on CSR with a lazy binary heap. Optional, starts from a given Bellman Ford State, needs the distances
    from Bellman Ford, the settled array from the last dijkstra iteration and all active ids. If you want to start
    from given BF state use_bf needs to be 1 and you need to provide dist,settled and active_ids from BF.

    Tracks:
      - dist[v]: shortest distance (float64)
      - edges[v]: hop count on best-known path (0 for source, U32INF if unreachable)
      - pred[v]: parent node in the shortest-path tree (−1 if unknown)
      - settled_flag[v]: 1 if v was popped validly (finalized)
      - pop_order: order of finalized nodes (truncated to settled count, not settled nodes are not taken into account)
      - heap_size_log[i]: Size of the heap at performed Dijkstra relaxation step i
    If limit > 0, stop after limit finalized nodes and return the partial state. If started from given
    Bellman Ford state, only returns the statistics for the partial dijkstra run.

    :param indptr: CSR row pointer of shape (n+1).
    :param indices: CSR column indices of shape (m).
    :param weights: Non-negative edge weights aligned with indices.
    :param source: Start node id for Dijkstra.
    :param limit: If > 0, stop after this many finalized nodes.
    :param use_bf: 1 if we want to run dijkstra from given bellman ford state 0 else
    :param bf_dist: last distance array from bellman ford, contains all current distances
    :param bf_settled: contains all settled nodes (from last dijkstra run)
    :param bf_active_ids: contains all active nodes; can be all nodes not settled and finite distance OR all nodes from
    Violation Frontier.
    :param e_b_border: if set, dijkstra will stop when the heap got at least e_b finite entries (used in multi switch)

    :returns: dist, edges, pred, settled_flag, pop_order[:settled], heap_size_log[:settled]
    """

    # Get number of nodes and initialize all used arrays
    n = indptr.shape[0] - 1
    dist = np.full(n, INF, dtype=np.float64)
    edges = np.full(n, U32INF, dtype=np.uint32)
    pred = np.full(n, -1, dtype=np.int32)
    settled_flag = np.zeros(n, dtype=np.uint8)
    pop_order = np.zeros(n, dtype=np.int32)

    # Initialize the heap structures used for Dijkstra
    hk = np.empty(max(1, n), dtype=np.float64)
    hn = np.empty(max(1, n), dtype=np.int32)
    hs = np.empty(max(1, n), dtype=np.uint32)
    size = 0

    # init unique heap entry count
    in_heap = np.zeros(n, dtype=np.uint8)
    unique_heap_size = np.int32(0)

    # array to count heap sizes at each step
    heap_size_log = np.zeros(n, dtype=np.int32)

    # start from scratch
    if use_bf == 0:
        # For the source node initialize its values in the corresponding arrays and heap dist = 0, edges = 0...
        dist[source] = 0.0
        edges[source] = np.uint32(0)
        hk, hn, hs, size = ensure_push(hk, hn, hs, size, 0.0, np.int32(source), np.uint32(0))
        # set source for our unique heap counter
        in_heap[source] = 1
        unique_heap_size = np.int32(1)

    # else we start from given BF state
    else:
        # copy the distances from bf and the settled flags
        for i in range(n):
            dist[i] = bf_dist[i]
            settled_flag[i] = bf_settled[i]
        # Fill heap with all not settled nodes with finite distance
        for i in range(bf_active_ids.size):
            v = bf_active_ids[i]
            if settled_flag[v] == 0 and dist[v] < np.float64(np.inf):
                hk, hn, hs, size = ensure_push(hk, hn, hs, size, dist[v], v, np.uint32(0))
                if in_heap[v] == 0:
                    in_heap[v] = 1
                    unique_heap_size += np.int32(1)

    # Initialize settled count and set limited if limit > 0
    settled = 0
    limited = (limit > 0)

    # while the heap still contains any nodes
    while size > 0:

        # we pop the node with the current best (shortest) dist
        du, u, eu, size = heap_pop3(hk, hn, hs, size)

        # We check if the dist is worse than the dist in the result array, we need to this because of lazy heap
        if du > dist[u]:
            # if it is we already settled this node, we continue
            continue

        # if a node is popped and was in heap marked before, set its marker to 0 and reduce count
        if in_heap[u] == 1:
            in_heap[u] = 0
            unique_heap_size -= np.int32(1)

        # update the statistics for the pop order and settled nodes, increase by 1 and add node to pop order
        pop_order[settled] = u
        settled_flag[u] = 1
        settled += 1

        # relaxation round here get the edge positions of the node that was popped
        a = indptr[u]
        b = indptr[u + 1]

        # iterate over all edges in the indices array
        for e in range(a, b):
            # get the current edge and the weight of current edge + cur distance to node u
            v = indices[e]
            alt = du + weights[e]
            # here we can add continue if v already settled
            # if the distance is better we update the heap via lazy insert
            if alt < dist[v]:
                # update the dist array, so it always contains the current best dist to each node
                dist[v] = alt
                # update the edge count -- increase it by 1
                ev = eu + np.uint32(1)
                edges[v] = ev
                # set the new parent node in the pred array
                pred[v] = u
                # finaly push the new distance for v in the heap
                hk, hn, hs, size = ensure_push(hk, hn, hs, size, alt, v, ev)
                # mark v as in heap and increase in_heap counter
                if in_heap[v] == 0:
                    in_heap[v] = 1
                    unique_heap_size += np.int32(1)

        # stat for current heap size gets saved
        heap_size_log[settled-1] = unique_heap_size

        # limit check to break when limit is reached
        if limited and settled >= limit:
            break

        # break when our heap size is bigger than e_b; only if e_b_border is set
        if e_b_border > 0:
            if unique_heap_size >= e_b_border:
                break

    # returns the final results
    return dist, edges, pred, settled_flag, pop_order[:settled], heap_size_log[:settled]


def dijkstra_with_edges_tree(indptr, indices, weights, source, limit=-1,
                             bf_dist=None, bf_settled=None, bf_active_ids=None, e_b_border=-1
                             ) -> tuple[NDArray[np.float64], NDArray[np.uint32], NDArray[np.int32],
                                        NDArray[np.uint8], NDArray[np.int32], NDArray[np.int32]]:
    """
    Wrapper to start the core dijkstra with edges, with and without given Bellman Ford State.
    See parameters and returns in the core function above.
    """
    # get number of nodes for dummies
    n = indptr.shape[0] - 1

    # if we want to start naive dijkstra
    if bf_dist is None:
        return core_dijkstra_with_edges_tree(indptr, indices, weights, int(source), int(limit), np.uint8(0),
                                             np.empty(n, dtype=np.float64), np.zeros(n, dtype=np.uint8),
                                             np.empty(0, dtype=np.int32), e_b_border=e_b_border)

    # else we start from given Bellman Ford State
    else:
        return core_dijkstra_with_edges_tree(indptr, indices, weights, int(source), int(limit), np.uint8(1),
                                             bf_dist.astype(np.float64, copy=False),
                                             bf_settled.astype(np.uint8, copy=False),
                                             bf_active_ids.astype(np.int32, copy=False), e_b_border=e_b_border)


@njit(cache=True)
def build_tree_csr_from_pred(pred: NDArray[np.int32], settled_flag: NDArray[np.uint8]) \
                            -> tuple[NDArray[np.int32], NDArray[np.int32]]:
    """
    Build the CSR of the shortest-path tree from a predecessor/parent array.
    Only edges p -> v are included where v is finalized (settled_flag[v] == 1)
    and pred[v] == p.

    :param pred: Predecessor array of shape from dijkstra function. Pred of node i at position i.
    :param settled_flag: Finalization flags where 1 marks finalization of a node i at position i.
    :returns: indptr, indices CSR of the SPT.
    """

    # Get size of the original graph and initialize the child count array.
    n = pred.shape[0]
    child_count = np.zeros(n, dtype=np.int32)

    # for each node in the graph
    for v in range(n):
        # check if it was settled during dijkstra run
        if settled_flag[v] == 1:
            # if it was, get its parent node for the tree
            p = pred[v]
            # if it has a parent node (is not the root) increase the child count for the parent node by one
            if p != -1:
                child_count[p] += 1

    # initialize empty index pointer for CSR with size of nodes in original graph
    indptr = np.empty(n + 1, dtype=np.int32)
    s = 0

    # now fill the index pointer array by iterating over all graph nodes
    for i in range(n):
        # set the index pointers based on the child of each node
        indptr[i] = s
        s += child_count[i]
    indptr[n] = s

    # create indices arr and fill help array
    indices = np.empty(s, dtype=np.int32)
    fill = np.zeros(n, dtype=np.int32)

    # again for each node in the original graph fill the indices array
    for v in range(n):
        # if the node was settled in dijkstra
        if settled_flag[v] == 1:
            # we get the corresponding parent node
            p = pred[v]
            # if the parent node exists
            if p != -1:
                # Compute write index: parents start in indices (indptr[p]) plus number of placed children (fill[p])
                pos = indptr[p] + fill[p]
                # now fill the indices arr at the computed position
                indices[pos] = v
                # increase child count placed for the parent node by 1
                fill[p] += 1

    # return spt graph as CSR
    return indptr, indices


@njit(cache=True)
def level_stats_from_settled(indptr: NDArray[np.int32], edges: NDArray[np.uint32],
                             settled_flag: NDArray[np.uint8]) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    """
    Compute per level statistics over the finalized (settled) vertices.

    Levels are defined by hop count edges[v] where the source has level 0.
    Only vertices with settled_flag[v] == 1 are counted, this keeps results
    consistent under early stopping.

    :param indptr: CSR row pointer array of shape (n+1), indptr of the ORIGINAL GRAPH
    :param edges: Hop counts per vertex (0 for source, U32INF if unreachable)
    :param settled_flag: Finalization flags (1 if node is finalized).
    :returns: level_counts, level_edge_work nodes at level d
              level_edge_work[d] = sum of outdegrees in the original graph of those nodes
    """

    # get number of nodes in original graph and counter for max depth
    n = edges.shape[0]
    maxd = 0

    # calculate the max depth by iterating over the whole edges array
    for v in range(n):
        if settled_flag[v] == 1:
            d = int(edges[v])
            # always safe the highest value in maxd
            if d > maxd and edges[v] != U32INF:
                maxd = d
    # final Level is maxd + 1
    final_l = maxd + 1

    # now initialize level_counts and level_edge_work
    level_counts = np.zeros(final_l, dtype=np.int64)
    level_edge_work = np.zeros(final_l, dtype=np.int64)

    # finally fill them by iterating over all nodes
    for u in range(n):
        # if a node has been settled
        if settled_flag[u] == 1:
            # its depth can be found in the edges array
            d = int(edges[u])
            # increas level count at depth d by 1 and level_edge_work by the number of edges of outoging edges of u
            level_counts[d] += 1
            level_edge_work[d] += (indptr[u+1] - indptr[u])

    # return results
    return level_counts, level_edge_work


@njit(cache=True)
def calc_real_dijkstra_cost(indptr: NDArray[np.int32], pop_order: NDArray[np.int32], edges_per_block: int,
                            mem_edges_capacity: int = -1) -> int:
    """
    Estimate Dijkstra I/O by counting block loads while scanning CSR rows in pop order.
    Note: Uses a stochastic eviction when the cache is full; results are non-deterministic.
    Not used in the experiments but for the sake of completeness we included it here.

    :param indptr: CSR row pointer
    :param pop_order: Finalization order of nodes during Dijkstra.
    :param edges_per_block: Block size in number of edges.
    :param mem_edges_capacity: Cache capacity in edges ; if <= 0 - defaults to n (≈ floor(n/edges_per_block) blocks).

    :returns: Total number of block loads (int) observed during the simulated scan.
    """

    # get #nodes and #edges
    n = indptr.size - 1
    m = int(indptr[n])

    # set ram capacity ~ default is n = number of nodes
    if mem_edges_capacity <= 0:
        mem_edges_capacity = n

    # calculates how many block the ram can hold in total
    cap_blocks = mem_edges_capacity // edges_per_block
    if cap_blocks < 1:
        cap_blocks = 1

    # number of blocks to store the whole graph is calculated
    num_blocks = (m + edges_per_block - 1) // edges_per_block

    # init in_cache flag (if block is in cache 1 else 0; cache list: contains all cache ids that are currently in ram)
    in_cache = np.zeros(num_blocks, dtype=np.uint8)
    cache_list = np.empty(cap_blocks, dtype=np.int64)
    cached = 0

    # statistics for total loads...
    total_loads = 0
    cache_hits = 0

    # now for each node in pop_order
    for i in range(pop_order.size):
        # get the node and the edge positions in indptr
        u = pop_order[i]
        a = indptr[u]
        b = indptr[u + 1]

        # get degree of the node; if 0 continue no edges will be load
        deg = b - a
        if deg <= 0:
            continue

        # first block and last block that is involved in loading edges of u
        first_blk = a // edges_per_block
        last_blk = (b - 1) // edges_per_block

        # for each block in between
        for blk in range(first_blk, last_blk + 1):
            # if the block is already in cache --> good no I/O needed, increase cache hits by 1 and continue
            if in_cache[blk] != 0:
                cache_hits += 1
                continue

            # if not, if our cache is full we drop a random block
            if cached >= cap_blocks:
                evict_idx = np.random.randint(0, cached)
                evict_blk = cache_list[evict_idx]
                in_cache[evict_blk] = 0

                last_idx = cached - 1
                if evict_idx != last_idx:
                    cache_list[evict_idx] = cache_list[last_idx]
                cached -= 1

            # now our loaded block is in cache, we increase cached and total loads by 1
            in_cache[blk] = 1
            cache_list[cached] = blk
            cached += 1
            total_loads += 1

    # return the total amount of loads
    return total_loads
