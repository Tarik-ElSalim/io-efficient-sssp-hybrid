# spdag.py
"""
Shortest-Path DAG (SPDAG) construction and schedule simulation utilities.

Main routines:
- spdag_from_dist: build the shortest-path DAG (SPDAG) from a completed distance array (CSR input/output).
- (fast_)online_frontier_set_cover_core: greedy online set-cover heuristic on the SPDAG that schedules BF-rounds
  vs. Dijkstra pops under a simple cost model (based on semi-external-memory-model).
- apply_schedule_get_dist / apply_schedule_core: execute a given schedule on the original CSR graph and
  return the resulting distance array together with the accumulated I/O cost.

Notes:
- The schedule simulator is intended as a correctness/sanity check for schedules produced on the SPDAG.
"""

# Standard library
from typing import Any, Tuple

# Third-party
import numpy as np
from numba import njit


@njit(cache=True)
def spdag_from_dist(indptr: np.ndarray, indices: np.ndarray, weights: np.ndarray,
                    dist: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.bool_]:

    """
    Build the Shortest-Path DAG (SPDAG) of a CSR graph with help of a completed sssp distance array.
    Keeps exactly those edges (u→v) with dist[u] + weight(u,v) == dist[v].

    :param indptr: CSR row pointer of the graph
    :param indices: CSR edge pointer of the graph
    :param weights: CSR Edge weights of the graph
    :param dist: Distance array of shape with dtype=float64; computed by dijkstra in modified_dijkstra.py
    :returns: sp_indptr, sp_indices, sp_weights and the SPDAG in CSR and a bool has_multiple_parents to check if
              we have a real spdag
    """

    # get number nodes of the graph
    n = indptr.size - 1

    # init array that contains outgoing edges for each node u in the SPDAG
    deg_sp = np.zeros(n, dtype=np.int32)
    total = 0
    # for each node
    for u in range(n):
        # get its dist
        du = dist[u]
        # if it is not finite; so the node cant be reached from source -> continue
        if not np.isfinite(du):
            continue
        # if it is reachable get its edges
        a = indptr[u]
        b = indptr[u + 1]
        cnt = 0
        # now for each edge
        for e in range(a, b):
            # get the target node of each edge
            v = indices[e]
            # get the target distance
            dv = dist[v]
            # if dist from source to u + edge to v is exactly the dist from source to v --> we found shortest path edge
            if du + weights[e] == dv:
                # increase count for this node by 1
                cnt += 1
        # now set the degree for u to number of sssp edges found
        deg_sp[u] = cnt
        # increase total edge count
        total += cnt

    # build SPDAG row pointer by prefix sum of counted degrees
    sp_indptr = np.empty(n + 1, dtype=np.int32)
    s = 0
    # for each node u set its start pointer to s and the end pointer to s+deg of u
    for u in range(n):
        sp_indptr[u] = s
        s += deg_sp[u]
    sp_indptr[n] = s

    # now fill the weights and indices CSR array
    sp_indices = np.empty(total, dtype=np.int32)
    sp_weights = np.empty(total, dtype=np.float64)

    # tie flag, just to check if we realy have a spdag or a spt
    indeg = np.zeros(n, dtype=np.int32)
    has_multiple_parents = np.bool_(False)

    # for each mode
    for u in range(n):
        # get its dist
        du = dist[u]
        # if it is not reachable continue
        if not np.isfinite(du):
            continue
        # else get the pointer to the edges
        a = indptr[u]
        b = indptr[u + 1]
        base = sp_indptr[u]
        wptr = 0
        # now for each edge
        for e in range(a, b):
            # get the target node and target dist
            v = indices[e]
            dv = dist[v]
            # again if we find a shortest path edge
            if du + weights[e] == dv:
                # get the pos for the CSR edge array
                pos = base + wptr
                # the corresponding edge and edge weight
                sp_indices[pos] = v
                sp_weights[pos] = weights[e]
                # increase counter for this node by 1
                wptr += 1

                indeg[v] += 1
                if indeg[v] >= 2:
                    has_multiple_parents = np.bool_(True)

    # finally return the SPDAG as CSR
    return sp_indptr, sp_indices, sp_weights, has_multiple_parents


@njit(cache=True)
def apply_schedule_core(indptr: np.ndarray, indices: np.ndarray, weights: np.ndarray,
                         src: int, act_kind: np.ndarray, act_arg: np.ndarray,
                         e_b: float) -> Tuple[np.ndarray, float]:
    """
        Execute a schedule of BF-rounds or/and Dijkstra-pops on a CSR graph.

        :param indptr: CSR row pointer of the original graph
        :param indices: CSR edge indices of the original graph
        :param weights: Edge weights in dytpe float64
        :param src: Source node id
        :param act_kind: Action codes dtype=int8; 0 = BF-round, 1 = DJ-pop.
        :param act_arg: Action arguments dtype=int32; Node that needs to be popped by dijkstra, for BF Rounds no matter
        :param e_b: Scan Cost ~ I/O cost for BF Round
        :returns: dist, io_cost where
                  - dist: float64 array with distances after the schedule operations
                  - io_cost: total I/O cost after performing the schedule (1 dijk. pop -> 1 I/O; 1 BF Round -> e_B I/O)
        """

    # get number of nodes
    n = indptr.shape[0] - 1

    # init dist array with inf for all nodes and 0 for source
    dist = np.full(n, np.inf, dtype=np.float64)
    dist[src] = 0.0

    # init popped array with 0
    popped = np.zeros(n, dtype=np.uint8)
    # init var for final i/o cost
    io_cost = 0.0

    # now for each action
    for i in range(act_kind.size):
        # get the kind of the action 0 = bf; 1 = dijkstra
        k = act_kind[i]

        # if bf action
        if k == 0:
            # add e_b to io_cost
            io_cost += e_b
            # then for each node
            for u in range(n):
                # if it was already relaxed by a dijkstra pop we continue
                if popped[u] == 1:
                    continue
                # if not we get its current dist
                du = dist[u]
                # if the dist is infinite -> Node was not reached yet also continue
                if not np.isfinite(du):
                    continue
                # else get the edge pointers
                a = indptr[u]
                b = indptr[u+1]
                # now for each edge
                for e in range(a, b):
                    # get target node and new distance to target node
                    v = indices[e]
                    nd = du + weights[e]
                    # if we can improve the distance to target node in the dist array we do it
                    if nd < dist[v]:
                        dist[v] = nd

        # else if our action is dijkstra
        elif k == 1:
            # we add 1 to i_o cost
            io_cost += 1.0
            # get the node we need to relax
            u = act_arg[i]
            # if we already popped the node, we output an error --> in this case the schedule was incorrect
            if popped[u] == 1:
                raise RuntimeError("invalid DJ pop")
            # else we get the current dist of u and its edge pointers
            du = dist[u]
            a = indptr[u]
            b = indptr[u+1]
            # now for each edge
            for e in range(a, b):
                # get target node and its new distance
                v = indices[e]
                # check if current dist is infinite, if yes continue
                nd = du + weights[e]
                # if the new distance is better update the dist array
                if nd < dist[v]:
                    dist[v] = nd
            # and finally set u to popped
            popped[u] = 1

    # return the resulting dist array and i_o cost
    return dist, io_cost


def apply_schedule_get_dist(indptr: np.ndarray, indices: np.ndarray, src: int,
                            schedule: Any, e_b: float,
                            weights: np.ndarray) -> Tuple[np.ndarray, float]:
    """
     Convert a Python schedule into typed arrays and execute it on a CSR graph via the Numba core function.

        :param indptr: CSR row pointer of the original graph
        :param indices: CSR edge indices of the original graph
        :param src: Source node id
        :param schedule: The schedule with the optimal actioned determined by the greedy set cover algorithm
        :param e_b: Scan Cost ~ I/O cost for BF Round
        :param weights: Edge weights in dytpe float64
        :returns: see core function above
     """
    # convert weights to dtype float64
    weights = weights.astype(np.float64, copy=False)

    # translate schedule for numba
    len_sched = len(schedule)  # get schedule size
    # build empty kind and arg node array
    act_kind = np.empty(len_sched, dtype=np.int8)
    act_arg = np.zeros(len_sched, dtype=np.int32)

    # now for each schedule entry
    for i, act in enumerate(schedule):
        # get current action
        k = act[0]
        # if action is BF, set action kind to 0
        if k == "BF":
            act_kind[i] = 0
        # if action is dijkstra, set act kind = 1 and set act arg = corresponding node to be popped
        elif k == "DJ":
            act_kind[i] = 1
            act_arg[i] = int(act[1])

    # now apply the schedule and return the results (dist, io)
    return apply_schedule_core(indptr.astype(np.int32, copy=False),
                                indices.astype(np.int32, copy=False),
                                weights, int(src), act_kind, act_arg, float(e_b))


@njit(cache=True)
def _online_frontier_set_cover_core(indptr: np.ndarray, indices: np.ndarray, src: int,
                                    e_b: float) -> Tuple[np.ndarray, np.ndarray]:
    """
        Run a greedy online set cover procedure on a SPDAG (Shortest Path Graph of an original graph).
        In each round the algorithm decides between performing Bellman Ford on the current frontier or relaxing
        a single node with Dijkstra. Its chooses always the option which has the larger gain under our cost model
        (Dijkstra Cost is 1 per pop and Bellman-Ford Cost is e_b per round)

        :param indptr: CSR node pointer of the SPDAG (NOT the original graph)
        :param indices: CSR edge indices of the SPDAG
        :param src: Source node
        :param e_b: Cost of a single BF step (SCAN Cost)

        :return: sched_kind, sched_node
                 - sched_kind: 0 = BF step in this round, 1 = DJ step in this round
                 - sched_node: sched_node[t] = -1 for BF steps, sched_node[t] = u for DJ steps (node to be popped)
        """
    # get number of nodes and source as int
    n = indptr.shape[0] - 1
    src_c = int(src)

    # init covered array, frontier and first time covered array
    covered = np.zeros(n, dtype=np.bool_)
    frontier = np.zeros(n, dtype=np.bool_)
    frontier_size = 0

    # start with source covered
    covered[src_c] = True

    # init first frontier -> all neighbour nodes of the source
    # get edge pointer
    a0 = indptr[src_c]
    b0 = indptr[src_c + 1]
    # for each edge
    for e in range(a0, b0):
        # get target node and activate it frontier state
        v = indices[e]
        frontier[v] = True
        frontier_size += 1
    # set max round T to set cover
    tlim = n

    # create empty result array of length n
    sched_kind = np.empty(tlim, dtype=np.int8)   # 0 = BF, 1 = DJ
    sched_node = np.empty(tlim, dtype=np.int32)  # DJ: node, BF: -1

    # init round counter
    t = 0
    # while we below round T limit
    while t < tlim:

        # if our frontier is empty we are finished no node can be relaxed any more
        if frontier_size <= 0:
            break

        # if not compute our possible gain with one BF relaxation
        bf_gain = frontier_size / e_b

        # Look for best Dijkstra pop candidate
        best_u = -1
        best_gain = 0.0
        # for each node
        for u in range(n):
            # if it is not covered yet, we cant relax it with dijkstra, so continue
            if not covered[u]:
                continue
            # if it is covered get its outgoing edges
            g = 0
            a = indptr[u]
            b = indptr[u + 1]
            # for each edge
            for e in range(a, b):
                # get the target node
                v = indices[e]
                # if our target node is not covered yet; increase cover counter for this node by 1
                if not covered[v]:
                    g += 1
            # check if we actually found a better node for dijkstra pop; if yes updates best_gain and best_node
            if g > best_gain:
                best_gain = float(g)
                best_u = u

        # if bf_gain better than best dijkstra pop gain: choose bf; else choose best dijkstra pop
        choose_bf = (bf_gain >= best_gain)

        # if we choose bf
        if choose_bf:

            # sched kind is set to 0 and node is set to -1
            sched_kind[t] = 0
            sched_node[t] = -1

            # init empty gain array
            gained_count = frontier_size
            gained = np.empty(gained_count, dtype=np.int32)
            idx = 0
            # for each node
            for v in range(n):
                # if it is in the frontier
                if frontier[v]:
                    # we add it to the gained count array
                    gained[idx] = v
                    # just a counter to fill gained; increased after each fill
                    idx += 1

            # Now remove each former node from frontier and set cover it
            for i in range(gained_count):
                v = gained[i]
                frontier[v] = False
                frontier_size -= 1
                covered[v] = True
                # Build new frontier out of covered nodes, get edge pointers
                a = indptr[v]
                b = indptr[v + 1]
                # for each edge
                for e in range(a, b):
                    # get the target node
                    w = indices[e]
                    # if it was not covered until yet, add it to the frontier --> candidate for the next round
                    if not covered[w]:
                        frontier[w] = True
                        frontier_size += 1
        # else if no bf is performed we pop the best win dijkstra node
        else:
            # get node
            u = best_u
            # set output array correct to 1 = Dijkstra pop and u = Node that is popped
            sched_kind[t] = 1
            sched_node[t] = u

            # now get neighbours of u
            a = indptr[u]
            b = indptr[u + 1]
            for e in range(a, b):
                # get neighbour
                v = indices[e]
                # if the neighbour was already covered continue
                if covered[v]:
                    continue
                # else if not; its now so remove it out of the frontier
                frontier[v] = False
                frontier_size -= 1
                # cover it
                covered[v] = True
                # now build new frontier based on our just covered node
                a2 = indptr[v]
                b2 = indptr[v + 1]
                # for each outgoing edge
                for e2 in range(a2, b2):
                    # get the target node
                    w = indices[e2]
                    # and if it was not covered yet; add it to the frontier and increase frontier size
                    if not covered[w]:
                        frontier[w] = True
                        frontier_size += 1

        # increase round count by 1
        t += 1

    # finally return sched_kind and sched_node up to our round count
    return sched_kind[:t], sched_node[:t]


@njit(cache=True)
def fast_online_frontier_set_cover_core(indptr: np.ndarray, indices: np.ndarray, src: int,
                                        e_b: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Same as the online version before, but faster with using gain array, that holds the possible gain of each node.
    Additionally logs per step statistics for BF and Dijkstra Gain for Visualisation.

    :param indptr: CSR row pointer of the SPDAG (NOT the original graph)
    :param indices: CSR edge indices of the SPDAG
    :param src: Source node id
    :param e_b: Cost of a single BF round (scan cost)
    :returns: sched_kind, sched_node where
              - sched_kind[t] = 0 for BF round, 1 for DJ pop
              - sched_node[t] = -1 for BF, u for DJ pop
              - best_gain_log[t]: Gain for the best Dijkstra candidate.
              - frontier_sizes[t]: SPDAG frontier size after step t, corresponds to gain of one Bellman-Ford round.
    """

    # basic graph info
    n = indptr.shape[0] - 1
    src_c = int(src)
    m = indices.shape[0]

    # build reverse CSR (incoming edges) in O(m)
    deg_in = np.zeros(n, dtype=indptr.dtype)
    for u in range(n):
        a = indptr[u]
        b = indptr[u + 1]
        for e in range(a, b):
            v = indices[e]
            deg_in[v] += 1

    in_indptr = np.zeros(n + 1, dtype=indptr.dtype)
    in_indptr[0] = 0
    for v in range(n):
        in_indptr[v + 1] = in_indptr[v] + deg_in[v]

    in_indices = np.empty(m, dtype=indices.dtype)
    next_pos = np.empty(n, dtype=indptr.dtype)
    for v in range(n):
        next_pos[v] = in_indptr[v]

    for u in range(n):
        a = indptr[u]
        b = indptr[u + 1]
        for e in range(a, b):
            v = indices[e]
            pos = next_pos[v]
            in_indices[pos] = u
            next_pos[v] = pos + 1

    # state arrays
    covered = np.zeros(n, dtype=np.bool_)
    frontier = np.zeros(n, dtype=np.bool_)

    # dense representation for frontier (nodes + positions)
    frontier_nodes = np.empty(n, dtype=np.int32)
    frontier_pos = np.empty(n, dtype=np.int32)
    for i in range(n):
        frontier_pos[i] = -1
    frontier_len = 0

    # gain[u] = number of uncovered neighbors of u
    gain = np.zeros(n, dtype=np.int32)

    # dense candidate set for Dijkstra (nodes with gain[u] > 0)
    candidate_nodes = np.empty(n, dtype=np.int32)
    candidate_pos = np.empty(n, dtype=np.int32)
    for i in range(n):
        candidate_pos[i] = -1
    candidate_len = 0

    # output schedule arrays
    tlim = n
    sched_kind = np.empty(tlim, dtype=np.int8)    # 0 = BF, 1 = DJ
    sched_node = np.empty(tlim, dtype=np.int32)   # DJ: node, BF: -1

    # logs (one entry per executed step)
    best_gain_log = np.empty(tlim, dtype=np.int32)
    frontier_sizes = np.empty(tlim, dtype=np.int32)

    # reusable buffer for BF step (snapshot of frontier)
    gained = np.empty(n, dtype=np.int32)

    # initialize with source covered
    covered[src_c] = True

    # initial frontier = outgoing neighbors of source
    a0 = indptr[src_c]
    b0 = indptr[src_c + 1]
    for e in range(a0, b0):
        v = indices[e]
        if (not covered[v]) and (not frontier[v]):
            frontier[v] = True
            frontier_pos[v] = frontier_len
            frontier_nodes[frontier_len] = v
            frontier_len += 1

    # initial gain for source
    g0 = 0
    for e in range(a0, b0):
        v = indices[e]
        if not covered[v]:
            g0 += 1
    gain[src_c] = g0
    if g0 > 0:
        candidate_pos[src_c] = candidate_len
        candidate_nodes[candidate_len] = src_c
        candidate_len += 1

    t = 0

    # main greedy loop
    while t < tlim and frontier_len > 0:
        # BF gain uses current frontier size
        bf_gain = frontier_len / e_b

        # best Dijkstra candidate only over candidate_nodes
        best_u = -1
        best_gain = 0.0
        for i in range(candidate_len):
            u = candidate_nodes[i]
            g = gain[u]
            if g > best_gain:
                best_gain = float(g)
                best_u = u

        # save frontier size for visualisation
        frontier_sizes[t] = frontier_len
        # choose BF vs DJ according to cost model
        choose_bf = (bf_gain >= best_gain)

        # Bellman-Ford-like round: cover whole frontier
        if choose_bf:
            sched_kind[t] = 0
            sched_node[t] = -1

            # snapshot current frontier
            gained_count = frontier_len
            for i in range(gained_count):
                gained[i] = frontier_nodes[i]

            # cover all frontier nodes
            for i in range(gained_count):
                v = gained[i]

                # node may have been removed meanwhile
                if not frontier[v]:
                    continue

                # remove v from frontier in O(1) (swap with last)
                idx = frontier_pos[v]
                last_idx = frontier_len - 1
                last_v = frontier_nodes[last_idx]
                frontier_nodes[idx] = last_v
                frontier_pos[last_v] = idx

                frontier_pos[v] = -1
                frontier[v] = False
                frontier_len -= 1

                # cover v
                if not covered[v]:
                    covered[v] = True

                    # compute initial gain(v)
                    gv = 0
                    a = indptr[v]
                    b = indptr[v + 1]
                    for e in range(a, b):
                        w = indices[e]
                        if not covered[w]:
                            gv += 1
                    gain[v] = gv
                    if gv > 0 and candidate_pos[v] == -1:
                        candidate_pos[v] = candidate_len
                        candidate_nodes[candidate_len] = v
                        candidate_len += 1

                    # update gains of predecessors of v
                    a_in = in_indptr[v]
                    b_in = in_indptr[v + 1]
                    for e in range(a_in, b_in):
                        p = in_indices[e]
                        if not covered[p]:
                            continue
                        if gain[p] > 0:
                            gain[p] -= 1
                            if gain[p] == 0 and candidate_pos[p] != -1:
                                cidx = candidate_pos[p]
                                clast = candidate_len - 1
                                plast = candidate_nodes[clast]
                                candidate_nodes[cidx] = plast
                                candidate_pos[plast] = cidx
                                candidate_pos[p] = -1
                                candidate_len -= 1

                    # add uncovered neighbors of v to next frontier
                    for e in range(a, b):
                        w = indices[e]
                        if (not covered[w]) and (not frontier[w]):
                            frontier[w] = True
                            frontier_pos[w] = frontier_len
                            frontier_nodes[frontier_len] = w
                            frontier_len += 1

        # Dijkstra pop: cover neighbors of best_u
        else:
            u = best_u
            sched_kind[t] = 1
            sched_node[t] = u

            a = indptr[u]
            b = indptr[u + 1]
            for e in range(a, b):
                v = indices[e]
                if covered[v]:
                    continue

                # remove v from frontier if present
                if frontier[v]:
                    idx = frontier_pos[v]
                    last_idx = frontier_len - 1
                    last_v = frontier_nodes[last_idx]
                    frontier_nodes[idx] = last_v
                    frontier_pos[last_v] = idx
                    frontier_pos[v] = -1
                    frontier[v] = False
                    frontier_len -= 1

                # cover v
                covered[v] = True

                # compute initial gain(v)
                gv = 0
                a2 = indptr[v]
                b2 = indptr[v + 1]
                for e2 in range(a2, b2):
                    w = indices[e2]
                    if not covered[w]:
                        gv += 1
                gain[v] = gv
                if gv > 0 and candidate_pos[v] == -1:
                    candidate_pos[v] = candidate_len
                    candidate_nodes[candidate_len] = v
                    candidate_len += 1

                # update gains of predecessors of v
                a_in = in_indptr[v]
                b_in = in_indptr[v + 1]
                for e2 in range(a_in, b_in):
                    p = in_indices[e2]
                    if not covered[p]:
                        continue
                    if gain[p] > 0:
                        gain[p] -= 1
                        if gain[p] == 0 and candidate_pos[p] != -1:
                            cidx = candidate_pos[p]
                            clast = candidate_len - 1
                            plast = candidate_nodes[clast]
                            candidate_nodes[cidx] = plast
                            candidate_pos[plast] = cidx
                            candidate_pos[p] = -1
                            candidate_len -= 1

                # add uncovered neighbors of v to frontier
                for e2 in range(a2, b2):
                    w = indices[e2]
                    if (not covered[w]) and (not frontier[w]):
                        frontier[w] = True
                        frontier_pos[w] = frontier_len
                        frontier_nodes[frontier_len] = w
                        frontier_len += 1

        # log sizes AFTER this step
        best_gain_log[t] = np.int32(best_gain)

        t += 1

    return sched_kind[:t], sched_node[:t], best_gain_log[:t], frontier_sizes[:t]
