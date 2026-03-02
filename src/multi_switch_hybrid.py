# multi_switch_hybrid.py
"""
Implements the multi-switch hybrid SSSP approach on CSR graphs.

The algorithm alternates between (warm-start) Dijkstra and frontier-based
Bellman–Ford to calculate SSSP. A single threshold e_b controls switching: Dijkstra runs until
the heap grows too large, then BF runs until the violation frontier shrinks,
then Dijkstra continues. The driver also tracks a simple SEM-style I/O counter
and records per-phase / per-iteration statistics (pop order, heap sizes,
frontier sizes, relax counts, phase log).

Main routine:
- hybrid_dijkstra_bf(): Executes Multi-Switch on given CSR-Graph

"""

# Third party
import numpy as np
from numpy.typing import NDArray

# Local application
from .bellman_ford import bf_csr_frontier_continue
from .modified_dijkstra import dijkstra_with_edges_tree


def hybrid_dijkstra_bf(indptr, indices, weights, source: int, e_b: int) \
                                                    -> tuple[NDArray[np.float64], NDArray[np.uint8], NDArray[np.int32],
                                                             int, NDArray[np.int64], NDArray[np.int32],
                                                             NDArray[np.int32], NDArray[np.int32]]:
    """
    Hybrid SSSP driver: alternates Dijkstra with bf_rounds BF rounds. Uses Greedy mechanic, switches from Dijkstra to
    Bellman-Ford once our heap got more than e_b active nodes, switches back once our violation frontier has less than
    e_b nodes.

    Phase layout (repeats):
      1) Dijkstra settles nodes until the unique heap size is bigger than e_b (lazy heap).
      2) Bellman–Ford runs bf rounds until the violation frontier is smaller than e_b, then warm start dijkstra again

    :param indptr: CSR row pointer
    :param indices: CSR edge indices
    :param weights: CSR edge weights
    :param source: Start node id
    :param e_b: Scan Cost, we switch from Dijkstra to BF once the heap got more than e_b entries and back once our
    violation Frontier is smaller than e_b

    :returns: (dist, settled_flag, pop_order_all, phases_done, bf_relax_per_phase) where:
        - dist: Final distances, SSSP result.
        - settled_flag: 1 if node was validly popped in any Dijkstra phase.
        - pop_order_all: Concatenated pop order across all Dijkstra phases.
        - hybrid_io: IO Cost of the approach in our sem-model
        - bf_relax_per_phase: Relax counts of the BF rounds in each BF phase.
        - heap_sizes_all: heap sizes during each dijkstra pop
        - bf_frontier_sizes: frontier sizes during each bellman ford round
        - phase_log: actual schedule of our algorithm (0 for dijk, 1 for bf)
        """

    # get number of nodes
    n = indptr.shape[0] - 1

    # Initial Dijkstra
    dist, _, _, settled_flag, pop_part, heap_part = (
        dijkstra_with_edges_tree(indptr, indices, weights, int(source), -1, e_b_border=e_b))

    # Global logs (upper bounds)
    pop_order_all = np.empty(n, dtype=np.int32)
    heap_sizes_all = np.empty(n, dtype=np.int32)
    bf_relax_log = np.empty(2 * n + 2, dtype=np.int64)
    bf_frontier_log = np.empty(2 * n + 2, dtype=np.int32)
    phase_log = np.empty((2 * n + 2, 2), dtype=np.int32)

    # Write pointers
    wptr = 0
    hptr = 0
    bptr = 0
    pptr = 0

    # Copy initial phase outputs
    if pop_part.size > 0:
        take = pop_part.size if pop_part.size <= (n - wptr) else (n - wptr)
        pop_order_all[wptr:wptr + take] = pop_part[:take]
        wptr += take

    if heap_part.size > 0:
        take = heap_part.size if heap_part.size <= (n - hptr) else (n - hptr)
        heap_sizes_all[hptr:hptr + take] = heap_part[:take]
        hptr += take

    # Phase log: initial Dijkstra
    phase_log[pptr, 0] = 0
    phase_log[pptr, 1] = np.int32(pop_part.size)
    pptr += 1

    # Hybrid accounting
    hybrid_io = pop_part.size
    frontier_size = e_b
    active_ids = np.empty(0, dtype=np.int32)

    # Hybrid loop BF -> dijkstra -> BF...
    while True:

        # Stop if all nodes are settled
        if settled_flag.sum() == n:
            break

        # var to count bf rounds
        bf_rounds = np.int32(0)

        # if our violation frontier got more nodes than e_b we do one bf round again
        while frontier_size >= e_b:
            # execute 1 bf round
            (dist_bf, relax_counts, iters_done, _count_once, _settled_per_level,
             active_ids, changed_ids, settled_flag) = (bf_csr_frontier_continue(indptr, indices, weights,
                                                                                dist, settled_flag, 1))
            bf_rounds += 1
            # Log BF round stats (always, also when changed_ids.size == 0)
            bf_relax = np.int64(np.sum(relax_counts[:iters_done]))
            bf_relax_log[bptr] = bf_relax
            bf_frontier_log[bptr] = np.int32(changed_ids.size)
            bptr += 1

            # if BF did not relax anything, we can stop ~ dist is optimal
            if bf_relax == 0:

                # update stats before return
                phase_log[pptr, 0] = 1
                phase_log[pptr, 1] = bf_rounds
                pptr += 1

                return (dist,
                        settled_flag,
                        pop_order_all[:wptr],
                        hybrid_io,
                        bf_relax_log[:bptr],
                        heap_sizes_all[:hptr],
                        bf_frontier_log[:bptr],
                        phase_log[:pptr])

            # Replace distances with BF-updated ones
            dist = dist_bf

            # Update violation frontier for warm-start Dijkstra
            if changed_ids.size > 0:
                frontier_size = changed_ids.size
                active_ids = changed_ids
            else:
                frontier_size = 0  # frontier empty => BF phase ends

            # update our i_o
            hybrid_io += e_b

        # Log BF phase
        phase_log[pptr, 0] = 1
        phase_log[pptr, 1] = bf_rounds
        pptr += 1

        # (3) Warm-start Dijkstra for next pops_per_phase
        dist, _, _, settled_flag, pop_part, heap_part = (
            dijkstra_with_edges_tree(indptr, indices, weights, int(source), -1, dist.astype(np.float64, copy=False),
                                     settled_flag.astype(np.uint8, copy=False), active_ids.astype(np.int32, copy=False),
                                     e_b_border=e_b))

        # Append newly settled nodes of this phase
        if pop_part.size > 0:
            take = pop_part.size if pop_part.size <= (n - wptr) else (n - wptr)
            pop_order_all[wptr:wptr + take] = pop_part[:take]
            wptr += take

            take = heap_part.size if heap_part.size <= (n - hptr) else (n - hptr)
            heap_sizes_all[hptr:hptr + take] = heap_part[:take]
            hptr += take

            phase_log[pptr, 0] = 0
            phase_log[pptr, 1] = np.int32(pop_part.size)
            pptr += 1

            hybrid_io += pop_part.size
            frontier_size = e_b  # force BF check next
        # break if no pops happened during dijkstra -> heap is empty we are finished
        else:
            break

    return (dist,
            settled_flag,
            pop_order_all[:wptr],
            hybrid_io,
            bf_relax_log[:bptr],
            heap_sizes_all[:hptr],
            bf_frontier_log[:bptr],
            phase_log[:pptr])
