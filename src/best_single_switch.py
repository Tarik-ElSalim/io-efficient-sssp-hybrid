# best_single_switch.py
"""
Implements the Best-Single-Switch SSSP approach on CSR graphs and needed utilities.

This module implements the "Best Single Switch" analysis used in the project:
given a shortest-path tree (SPT) and an SEM-style I/O cost model, it estimates
for every valid Dijkstra pop how many Bellman–Ford (BF) rounds would still be
required, and selects the cost-minimizing switch point.

Main routines:
- choose_switch_k: A simple level-based baseline using SPT level counts
  (not used in the finalevaluation; kept for completeness).

- calc_best_switch:
  Exact best-switch computation on a known SPT using residual-height simulation.
  Returns a BF-round log (per Dijkstra pop), the best SEM cost, and the 1-based
  Dijkstra limit at which to switch (-1 means "no switch").
"""

# Standard library
from typing import Tuple

# Third party
import numpy as np
from numba import njit
from numpy.typing import NDArray

# Local application
from .heap_numba import ensure_push, heap_pop3

# inf definitions for float 64 and int 32 -- for placeholders in arrays
INF = np.float64(np.inf)
U32INF = np.uint32(0xFFFFFFFF)


def choose_switch_k(level_counts: NDArray[np.integer], e_b: float) -> tuple[int, float]:
    """
    Choose the best SPT level k to switch from Dijkstra to Bellman–Ford. Dijkstra I/O Cost per relaxed node is 1.
    Bellman Ford Scan Cost is e_b (Scan of indices + weights). This is a simplified calculation based on the nodes
    in each SPT-Level. We do not use this version for the experiments and evaluation.
    The exact version for the best switch calculation is calc_best_switch().

    :param level_counts: Number of finalized nodes per level d in the SPT.
    :param e_b: Estimated BF scan cost per remaining level.
    :returns: limit, cost: we return the optimal depth limit to switch and the corresponding cost for a whole run
    """

    # create work as float 64 array of level counts
    work = np.asarray(level_counts, dtype=np.float64)
    n = work.size

    # set Level to size -1
    costl = n - 1
    # Calculate the prefix sum for the work array --> corresponds to dijkstra cost up to level i
    pref = np.cumsum(work)
    # Index array filled with 1...n
    idx = np.arange(n, dtype=np.float64)
    # Calculate the result Cost Array, it contains cost for switch from dijkstra to bellman ford at level index
    cost = pref + (costl - idx) * float(e_b)   # prefix sum: dijkstra cost up to index + scan cost for remaining rounds

    # get the min cost index out of the cost array and its value and return them
    limit = int(cost.argmin())
    cost_out = float(cost[limit])

    return limit, cost_out


@njit(cache=True)
def _init_residual_height_and_maxchild(n: int, t_indptr: NDArray[np.int32], t_indices: NDArray[np.int32],
                                       parent: NDArray[np.int32]) -> Tuple[NDArray[np.int32], NDArray[np.int32]]:
    """
    Initialize residual heights bottom-up on the SPT forest and cache each nodes
    maximum child height. All nodes are considered unsettled initially.

    :param n: Number of nodes.
    :param t_indptr: CSR row pointer of the SPT children graph.
    :param t_indices: CSR column indices of the SPT forest.
    :param parent: Parent array (SPT predecessor per node; -1 if root).

    :returns: (res_height, max_child_h)
              - res_height[u]: height to deepest (unsettled) descendant in the SPT;
                               leaves get 0; roots get the tree height.
              - max_child_h[u]: max(res_height[child]) for u; -1 if u has no children.
    """

    # get out degree for each node in the tree
    outdeg = np.zeros(n, dtype=np.int32)
    for u in range(n):
        outdeg[u] = t_indptr[u + 1] - t_indptr[u]

    # init res height array for all nodes and an empty queue; res height contains height to deepest unsettled node
    res_height = np.zeros(n, dtype=np.int32)
    q = np.empty(n, dtype=np.int32)
    head = 0
    tail = 0
    # for each node if it is a leaf, add it to the queue, so leaf will get processed first
    for u in range(n):
        if outdeg[u] == 0:
            q[tail] = u
            tail += 1

    # while queue is not empty
    while head < tail:
        # get next child out of queue and move head pointer
        x = q[head]
        head += 1

        # get its parent node; -1 if root
        p = parent[x]

        # if it is not the root
        if p != -1:
            # get its height
            hx = res_height[x] + 1
            # if it is higher than the existing one for p, update the existing one
            if hx > res_height[p]:
                res_height[p] = hx
            # mark that we have finished 1 child
            outdeg[p] -= 1
            # if p is now finished : all its children are checked -> we can add p itself to the queue
            if outdeg[p] == 0:
                q[tail] = p
                tail += 1

    # init array that contains for each node its max child height
    max_child_h = np.full(n, -1, dtype=np.int32)

    # now for every node
    for u in range(n):
        # get its edges in the tree
        a = t_indptr[u]
        b = t_indptr[u + 1]
        # set local max to -1; if u has no children
        m = -1
        # for every edge that is outgoing from u
        for e in range(a, b):
            # save its max res height
            h = res_height[t_indices[e]]
            # if we find bigger height than max before update it
            if h > m:
                m = h
        # finally update the max_child_h array and return it
        max_child_h[u] = m

    return res_height, max_child_h


@njit(cache=True)
def _propagate_up_after_settle_fast(u: np.int32, parent: NDArray[np.int32], t_indptr: NDArray[np.int32],
                                    t_indices: NDArray[np.int32], res_height: NDArray[np.int32],
                                    in_heap: NDArray[np.int32], buckets: NDArray[np.int64], h_of: NDArray[np.int32],
                                    max_child_h: NDArray[np.int32], hmax_cur: np.int32) -> np.int32:
    """
    Update residual heights upward after settling node u.
    Recompute a parents max child height only if u carried that max,
    and maintain the frontier histogram and the current max height.

    :param u: Node that was just settled by Dijkstra.
    :param parent: SPT parent array (-1 at roots).
    :param t_indptr: CSR row pointer of SPT children.
    :param t_indices: CSR column indices (children) of SPT.
    :param res_height: Current residual heights (>=0 for unsettled, -1 if settled).
    :param in_heap: Per node: number of active heap entries (0 ⇒ not in frontier).
    :param buckets: Histogram of residual heights over active frontier nodes.
    :param h_of: Per node residual height as counted in `buckets` (=-1 if not active).
    :param max_child_h: Cached max child height per node.
    :param hmax_cur: Current maximum residual height among active nodes.

    :returns: Updated hmax_cur (the new max residual height after settling u).
    """

    # saves old residual height of node u
    old_h_u = res_height[u]
    # marks u as settled; so height -> -1
    res_height[u] = -1
    # gets parent of u
    p = parent[u]
    # now propagate up until there is no parent node anymore (so we are at the root)
    while p != -1:
        # save the old res_height of p
        old_hp = res_height[p]
        # only if u was the deepest child of p the max res height can change for p
        if max_child_h[p] == old_h_u:
            # then we need to rescan all child of p and get the new largest children height
            m = -1
            a = t_indptr[p]
            b = t_indptr[p + 1]
            for e in range(a, b):
                w = t_indices[e]
                hw = res_height[w]
                if hw > m:
                    m = hw
            # update the new max child height for p
            max_child_h[p] = m

        # calculate new residual height of p
        new_hp = np.int32(0 if max_child_h[p] < 0 else (max_child_h[p] + 1))

        # if the height did not change we can break now
        if new_hp == old_hp:
            break

        # else only when p is active in the heap we need to update buckets and hmax_cur
        if in_heap[p] > 0:
            # remove old height from buckets and lower hmax_cur if we emptied that level
            if old_hp >= 0:
                buckets[old_hp] -= 1
                if old_hp == hmax_cur:
                    while hmax_cur > 0 and buckets[hmax_cur] == 0:
                        hmax_cur -= 1
            # add new height to buckets and raise hmax_cur if needed
            if new_hp >= 0:
                buckets[new_hp] += 1
                if new_hp > hmax_cur:
                    hmax_cur = new_hp
            # record the parents height as counted in the buckets
            h_of[p] = new_hp

        # commit the new residual height of the parent
        res_height[p] = new_hp
        # for the next step upward: the "child height" seen by the grandparent is the old height of p
        old_h_u = old_hp
        # move one level up in the SPT
        p = parent[p]

    # return the updated maximum residual height over all active frontier nodes
    return hmax_cur


@njit(cache=True, fastmath=True)
def calc_best_switch(tree_indptr: NDArray[np.int32], tree_indices: NDArray[np.int32], e_b: float, source: int,
                     indptr: NDArray[np.int32], indices: NDArray[np.int32], weights: NDArray[np.float32],
                     pred_orig: NDArray[np.int32]) -> Tuple[NDArray[np.int32], float, int]:
    """
    Simulates Dijkstra and calculates for each pop how many BF-rounds are needed via residual SPT heights. Exact because
    we already have the complete spt tree. BF Rounds log contains for every pop the number of needed BF rounds after
    this dijkstra pop. Out of this we compute the best cost and the exact limit where we should switch to Bellman-Ford.

    :param tree_indptr: Children-CSR (SPT forest) row pointer.
    :param tree_indices: Children-CSR indices.
    :param e_b: Estimated BF cost per round corresponds to Scan Cost in SEM
    :param source: Source vertex id.
    :param indptr: CSR row pointer of the original graph.
    :param indices: CSR column indices of the original graph.
    :param weights: Non-negative edge weights of the original graph.
    :param pred_orig: The parent array from the dijkstra_with_edges function
    :returns: (bf_rounds_log, best_cost_plus_one, limit) where limit is 1-based or -1.
    """

    # Initialize number of nodes as n
    n = indptr.shape[0] - 1

    # Initialize Dijkstra arrays
    dist = np.full(n, INF, dtype=np.float64)
    edges = np.full(n, U32INF, dtype=np.uint32)
    settled_flag = np.zeros(n, dtype=np.uint8)

    # Initialize Heap arrays for dijkstra
    hk = np.empty(max(1, n), dtype=np.float64)
    hn = np.empty(max(1, n), dtype=np.int32)
    hs = np.empty(max(1, n), dtype=np.uint32)
    size = 0

    # Initialize parent array for each node with original pred array from full dijkstra_with_edges run
    parent = pred_orig

    # For each node we calculate res height and the max height of one of its children
    # Res height is -1 if settled else it is the greatest height to an unsettled neighbour
    res_height, max_child_h = _init_residual_height_and_maxchild(
        n, tree_indptr, tree_indices, parent
    )

    # Gets the max of all residual Heights and saves it in hmax, if -1 then hmax = 0
    hmax = np.int32(0)
    for i in range(n):
        if res_height[i] > hmax:
            hmax = res_height[i]
    if hmax < 0:
        hmax = 0

    # Create help arrays, inheap: how many heap entries for each node, h_of: residual height for each heap node
    in_heap = np.zeros(n, dtype=np.int32)
    h_of = np.full(n, -1, dtype=np.int32)
    # Buckets: For each possible residual height value, how many nodes in the frontier currently have which res. height
    buckets = np.zeros(hmax + 1, dtype=np.int64)
    # Current max height
    hmax_cur = np.int32(0)
    # active distinct nodes in heap frontier
    active_total = np.int64(0)

    # init source (heap, arrays, counters...)
    dist[source] = 0.0
    edges[source] = np.uint32(0)
    hk, hn, hs, size = ensure_push(hk, hn, hs, size, 0.0, np.int32(source), np.uint32(0))
    in_heap[source] = 1
    h_of[source] = res_height[source]
    # update buckets count at res_height[source] and update h_max cour
    if h_of[source] >= 0:
        buckets[h_of[source]] += 1
        if h_of[source] > hmax_cur:
            hmax_cur = h_of[source]
    active_total += 1

    # initialize settled and the result bf_rounds_log to n (because at most it can contain a value for every node)
    settled = 0
    bf_rounds_log = np.full(n, -1, dtype=np.int32)

    # Main loop that simulates Dijkstra and calculates cost; while heap is not empty
    while size > 0:
        # pop the min node
        du, u, eu, size = heap_pop3(hk, hn, hs, size)

        # if there is another heap entry for the node u
        if in_heap[u] > 0:
            # we update the in_heap value by subtract 1
            in_heap[u] -= 1
            # if there are now no more entries for this node
            if in_heap[u] == 0:
                # we get its current res height.
                old_h = h_of[u]
                # if the res height is bigger than 0 we update the bucket arr that contains counts for each res height
                if old_h >= 0:
                    buckets[old_h] -= 1
                # update the new res height of u to -1, because u is now settled; in BF no need to consider it anymore
                h_of[u] = -1
                # update the total active nodes in the heap
                active_total -= 1
                # Now update the current max res height by checking the bucket array entries at hmax_cur position
                while hmax_cur > 0 and buckets[hmax_cur] == 0:
                    hmax_cur -= 1

        # if it was a lazy pop, we continue
        if du > dist[u]:
            continue

        # else we finalize u and set it to settled
        settled_flag[u] = 1
        settled += 1

        # Now because of lazy heap, if there are any more entries of u in the heap delete them and update res heights
        if in_heap[u] > 0:
            # get its res height
            old_h = h_of[u]
            if old_h >= 0:
                # update the buckets array
                buckets[old_h] -= 1
            # update active total
            active_total -= 1
            in_heap[u] = 0
            h_of[u] = -1
            while hmax_cur > 0 and buckets[hmax_cur] == 0:
                hmax_cur -= 1

        # update all residual heights (with early stop)
        hmax_cur = _propagate_up_after_settle_fast(u, parent, tree_indptr, tree_indices, res_height, in_heap,
                                                   buckets, h_of, max_child_h, hmax_cur)

        # Now relax the outgoing edges of u; get the edge pointers from indptr
        a = indptr[u]
        b = indptr[u + 1]
        # iterate over all edges
        for e in range(a, b):
            # get the edge
            v = indices[e]
            # if v is already settled: continue
            if settled_flag[v] != 0:
                continue
            # if not check whether the new dist is better than the old
            alt = du + weights[e]
            if alt < dist[v]:
                # if it is better update the distance and the edge count and the parent
                dist[v] = alt
                ev = eu + np.uint32(1)
                edges[v] = ev
                # lazy push the new entry into the heap
                hk, hn, hs, size = ensure_push(hk, hn, hs, size, alt, v, ev)

                # now if the node was not in the heap before, set in_heap[v] = 1
                if in_heap[v] == 0:
                    in_heap[v] = 1
                    # Get the res height of new added node v
                    h = res_height[v]
                    # Update its value in the res height array for each node
                    h_of[v] = h
                    # if not really needed just for safety
                    if h >= 0:
                        # Update the corresponding bucket value of the new res height by 1
                        buckets[h] += 1
                        # if it is a new max, update the max val
                        if h > hmax_cur:
                            hmax_cur = h
                    # and add 1 to active total because we now have 1 more active node
                    active_total += 1
                # else we know we have a duplicate entry, just increase the in heap counter; no bucket update needed
                else:
                    in_heap[v] += 1

        # estimate BF rounds after a valid pop, by setting it to the current max res height hmax
        bf_rounds_log[settled - 1] = 0 if active_total == 0 else (hmax_cur + 1)

    # Here the best switch is calculated; initialize cost_l as number of settled Nodes
    cost_l = settled
    best_cost = cost_l
    best_i = cost_l
    # for all settled nodes
    for i in range(cost_l):
        # calculate the cost until i (dijkstra pop --> 1 I/O) + the remaining BF Round * scan Cost e_b
        c = i + bf_rounds_log[i] * e_b
        # if this calculated cost is better than the current best cost, we update the vars
        if c < best_cost:
            best_cost = c
            best_i = i

    # Set limit too -1 if the best switch point is at the end, else its best_i+1 due to index shifting
    limit = -1 if best_i >= cost_l else (best_i + 1)

    # return the round_log, best cost and limit
    return bf_rounds_log[:cost_l], best_cost + 1.0, limit
