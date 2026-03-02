# bellman_ford.py
"""
Numba-accelerated Bellman–Ford (BF) frontier routines on CSR graphs with positive weights.

- bf_csr_frontier_complete: Run BF from a given source for up to n−1 rounds.
- bf_csr_frontier_continue: Continue BF from an existing Dijkstra state.

Both return distance arrays and per-round statistics (relaxations, last-improved histograms and more).
"""

# Standard library
from typing import Tuple

# Third-party
import numpy as np
from numba import njit


@njit(cache=True)
def bf_csr_frontier_complete(indptr: np.ndarray, indices: np.ndarray, weights: np.ndarray, source: int,
                             max_rounds: int = -1,) \
                                -> Tuple[np.ndarray, np.ndarray, int, np.ndarray, np.ndarray, np.ndarray]:
    """
       Bellman–Ford algorithm for shortest path from given source. Returns the shortest paths (dist array) and
       additional statistics like number of relaxes per round (relax_counts), BF relaxation rounds in total (iters_done)
       as well as how many node distances where improved in each level (count_once) and how many nodes where last
       improved in each level (settled per level). Works with the concept of SPFA violation frontier and also returns
       the violation frontier sizes (frontier_sizes) after each BF Round.

       :param indptr: CSR row pointer (n+1).
       :param indices: CSR column indices (m).
       :param weights: Edge weights (m).
       :param source: Source node in [0, n-1]. Ensure that node is a real node in the graph and not out of range.
       :param max_rounds: Iteration cap, if < 0 or >n --> n-1

       :returns: (dist_out, relax_counts, iters_done, count_once, settled_per_level, frontier_sizes)
       """

    # Sets node size and zero val
    n = indptr.size - 1
    zero = np.float64(0.0)

    # Set maximum number of rounds conducted; n-1 normally or max_rounds if max_rounds makes sense
    rounds_default = n - 1
    rounds = rounds_default if (max_rounds <= 0 or max_rounds > rounds_default) else max_rounds

    # initialize all used arrays with zeros/inf and corresponding datatypes
    relax_counts = np.zeros(rounds, dtype=np.int64)
    count_once = np.zeros(rounds + 1, dtype=np.uint32)
    settled_per_level = np.zeros(rounds + 1, dtype=np.uint32)
    dist_out = np.full(n, np.inf, dtype=np.float64)
    dist_out[source] = zero
    last_found_level = np.full(n, -1, dtype=np.int64)
    seen_in_level_mark = np.full(n, -1, dtype=np.int64)
    frontier = np.empty(n, dtype=np.int32)
    next_frontier = np.empty(n, dtype=np.int32)
    in_next = np.zeros(n, dtype=np.uint8)

    # fsz is frontier size and frontier contain all nodes from which the graph will be relaxed in the next round
    fsz = 1

    # at the start of BF we will only consider the source node in our next relaxation round
    frontier[0] = np.int32(source)

    # array to save the sizes of the violation frontier
    frontier_sizes = np.zeros(rounds, dtype=np.int64)

    # initialize statistics for first level (source based)
    level = 0
    count_once[0] = np.uint32(1)
    last_found_level[source] = 0
    seen_in_level_mark[source] = 0

    # iters_done important, so we know how many rounds we really had at the end of the algorithm
    iters_done = 0

    # array to prevent cascading
    frontier_du = np.empty(n, dtype=np.float64)

    # at most n-1 times we do the BF relaxations
    for r in range(rounds):

        # save frontier size for this round
        frontier_sizes[r] = fsz

        # initialize relax count for this round and size of the next frontier (every node we relax this round)
        cnt_relax = 0
        nsz = 0

        for i in range(fsz):
            frontier_du[i] = dist_out[frontier[i]]

        # now iterate over all candidates in the frontier
        for i in range(fsz):

            # get the corresponding node u out of the frontier and its current distance
            #u = frontier[i]
            #du = dist_out[u]
            u = frontier[i]
            du = frontier_du[i]

            # get all edges of the current node u
            start = indptr[u]
            end = indptr[u + 1]

            # now iterate over all edges of node u
            for e in range(start, end):

                # get the target node v of the current edge and its distance
                v = indices[e]
                nd = du + weights[e]

                # if the new distance is better than the current, relax it and replace it in the dist out array
                if nd < dist_out[v]:
                    dist_out[v] = nd

                    # add 1 to the relax count in for this relaxation round
                    cnt_relax += 1

                    # checks if this node was already improved in this relaxation round
                    if seen_in_level_mark[v] != level:

                        # if not add +1 to the relaxation count (count_once) and set the seen_in_level to cur level
                        count_once[level] += 1  # level 0 also includes +1 for the initial start node
                        seen_in_level_mark[v] = level

                    # check if this node is already added to the next frontier if not add it and increase the nsz count
                    if in_next[v] == 0:
                        in_next[v] = 1
                        next_frontier[nsz] = v
                        nsz += 1

                    # update the last level found count for this node
                    last_found_level[v] = level

        # when all nodes of the frontier are finished we increase iters by 1 and save the relax count for this round
        relax_counts[r] = cnt_relax
        iters_done = r + 1

        # if frontier empty we can break, nothing to update
        if nsz == 0:
            break

        # else we can set frontier to next frontier, so we got the right frontier for the next relaxation round
        fsz = nsz
        for i in range(nsz):
            in_next[next_frontier[i]] = 0
            frontier[i] = next_frontier[i]

        # advance to the next BF round.
        level += 1

    # final number of levels
    levels_used = level + 1

    # Build the last improved per level histogram by iterating of the last_found_level array for each node
    for v in range(n):
        # get value of the last found level of node v
        lf = last_found_level[v]
        # increase last improved/settled per level in this level by 1
        if 0 <= lf < levels_used:
            settled_per_level[lf] += 1

    # returns all the data
    return (dist_out,
            relax_counts[:iters_done],
            iters_done,
            count_once[:levels_used],
            settled_per_level[:levels_used],
            frontier_sizes[:iters_done])


@njit(cache=True)
def bf_csr_frontier_continue(indptr: np.ndarray, indices: np.ndarray, weights: np.ndarray, dist_in: np.ndarray,
                             settled_in: np.ndarray, max_rounds: int = -1) \
                                -> Tuple[np.ndarray, np.ndarray, int, np.ndarray, np.ndarray, np.ndarray, np.ndarray,
                                         np.ndarray]:
    """
    Also Bellman–Ford algorithm for shortest path but starts from existing dijkstra state.
    Returns the shortest paths (dist) and additional statistics like number of relaxes per round (relax_counts),
    BF relaxation rounds in total (iters_done) as well as how many node distances were improved in
    each level (count_once) and how many nodes were last improved in each level (settled per level).
    Also returns active_ids: all nodes that are not settled and have finite distances, changed_ids: the
    violation frontier from the last round that produced relaxations (i.e., nodes whose distance improved
    and will be processed next), as well as settled_in: settled flags from dijkstra.

    This is very similar to the Bellman Ford function above, but we have included it here specifically
    to better test changes while still being able to use a naive BF implementation.

    :param indptr: CSR row pointer (n+1).
    :param indices: CSR column indices (m).
    :param weights: Edge weights (m).
    :param dist_in: Dijkstra distances (float64).
    :param settled_in: Dijkstra finalized flags (uint8, 1=finalized).
    :param max_rounds: Iteration cap, if < 0 or >n --> n-1

    :returns: (dist_out, relax_counts, iters_done, count_once, settled_per_level, active_ids, changed_ids, settled_in)
    """

    # get number of nodes
    n = indptr.size - 1

    # set maximum number of rounds conducted; n-1 normally or max_rounds if max_rounds makes sense
    rounds_default = n - 1
    rounds = rounds_default if (max_rounds <= 0 or max_rounds > rounds_default) else max_rounds

    # initialize statistics arrays
    relax_counts = np.zeros(rounds,     dtype=np.int64)
    count_once = np.zeros(rounds + 1, dtype=np.uint32)
    settled_per_level = np.zeros(rounds + 1, dtype=np.uint32)

    # copy the dijkstra inputs to our bellman ford data structures
    dist_out = dist_in.copy()

    # initialize last found level of nodes and seen in level with -1 for all entries
    last_found_level = np.full(n, -1, dtype=np.int64)
    seen_in_level_mark = np.full(n, -1, dtype=np.int64)

    # initialize frontier and next frontier as well as in_next round array
    frontier = np.empty(n, dtype=np.int32)
    next_frontier = np.empty(n, dtype=np.int32)
    in_next = np.zeros(n, dtype=np.uint8)

    # creates frontier based on dijkstra distances
    fsz = 0
    prev_nsz = 0
    for v in range(n):

        # if nodes was not settled in dijkstra and has a finite distance value we add it to the frontier
        if np.isfinite(dist_in[v]) and settled_in[v] == 0:
            frontier[fsz] = np.int32(v)
            fsz += 1

    # initialize level vals
    level = 0
    count_once[0] = np.uint32(0)
    iters_done = 0

    # if frontier is empty we break return our results; can happen when dijkstra finished everything by its own
    if fsz == 0:
        return (dist_out,
                relax_counts[:0],
                np.int64(0),
                count_once[:1],
                settled_per_level[:1],
                np.empty(0, dtype=np.int32),
                np.empty(0, dtype=np.int32),
                settled_in)

    # frontier for prevent cascading
    frontier_du = np.empty(n, dtype=np.float64)

    # if frontier not empty then we will iterate until at most rounds (max n+1) times
    for r in range(rounds):

        # initialize relax count for this round and size of the next frontier (every node we relax this round)
        cnt_relax = 0
        nsz = 0

        # prevent cascading
        for i in range(fsz):
            u = frontier[i]
            frontier_du[i] = dist_out[u]

        # now iterate over all candidates in the frontier
        for i in range(fsz):

            # set cascading to dist_out
            u = frontier[i]
            du = frontier_du[i]

            # get the target node v of the current edge and its distance
            #u = frontier[i]
            #du = dist_out[u]

            # get all edges of the current node u
            start = indptr[u]
            end = indptr[u + 1]

            # now iterate over all edges of node u
            for e in range(start, end):

                # get the target node v of the current edge and its distance
                v = indices[e]
                nd = du + weights[e]

                # if the new distance is better than the current, relax it and replace it in the dist out array
                if nd < dist_out[v]:
                    dist_out[v] = nd
                    cnt_relax += 1

                    # checks if this node was already improved in this relaxation round
                    if seen_in_level_mark[v] != level:
                        # if not add +1 to the relaxation count (count_once) and set the seen_in_level to cur level
                        count_once[level] += 1
                        seen_in_level_mark[v] = level

                    # check if this node is already added to the next frontier if not add it and increase the nsz count
                    if in_next[v] == 0:
                        in_next[v] = 1
                        next_frontier[nsz] = v
                        nsz += 1

                    # advance to the next BF round
                    last_found_level[v] = level

        # when all nodes of the frontier are finished we increase iters by 1 and save the relax count for this round
        relax_counts[r] = cnt_relax
        iters_done = r + 1

        # if frontier empty we can break, nothing to update
        if nsz == 0:
            break

        # else we can set frontier to next frontier, so we got the right frontier for the next relaxation round
        fsz = nsz
        prev_nsz = nsz
        for i in range(nsz):
            in_next[next_frontier[i]] = 0
            frontier[i] = next_frontier[i]

        # increase current relaxation level by 1 (we can replace level with r)
        level += 1

    # final number of levels
    levels_used = level + 1

    # Build the settled per level histogram by iterating of the last_found_level array for each node
    for v in range(n):
        lf = last_found_level[v]
        if 0 <= lf < levels_used and not settled_in[v]:
            settled_per_level[lf] += 1

    # build active ids set, all unsettled nodes with non-infinite dist
    tmp = np.empty(n, dtype=np.int32)
    sz = 0
    for v in range(n):
        if settled_in[v] == 0 and np.isfinite(dist_out[v]):
            tmp[sz] = np.int32(v)
            sz += 1
    active_ids = tmp[:sz]
    # we only want to return the violation frontier; nodes that changed in the last successful relaxation round
    changed_ids = next_frontier[:prev_nsz]

    # returns dist array and all statistics
    return (dist_out,
            relax_counts[:iters_done],
            np.int64(iters_done),
            count_once[:levels_used],
            settled_per_level[:levels_used],
            active_ids,
            changed_ids,
            settled_in)
