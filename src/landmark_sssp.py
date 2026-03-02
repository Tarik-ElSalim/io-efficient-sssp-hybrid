# landmark_sssp.py
"""
Landmark-assisted SSSP for CSR graphs (directed, nonnegative weights).

This module implements a landmark pipeline that trades exactness for a fast
distance estimate and then restores exactness via a warm-start Dijkstra pass.

Pipeline:
1) Choose s landmarks inside the largest weakly connected component (WCC).
2) Run R rounds of multi-source relaxation from the landmarks and optionally
   extract a landmark-to-landmark graph H. (prevents cascading BF-relaxations)
3) Compute landmark distances on H (Dijkstra on the compressed graph).
4) Back-project the landmark distances to all nodes via R2 rounds of
   multi-source relaxation.
5) Finalize exact distances with a warm-start Dijkstra seeded from a violation frontier.

Main Routine:
- landmark_sssp(): Executes the landmark pipeline described above

"""

# Standard library
from typing import Tuple


# Third-party
import numpy as np
from numba import njit

# Local application
from . import graph_properties as gp
from . import modified_dijkstra as md


def choose_landmarks_in_wcc(indptr: np.ndarray, indices: np.ndarray, s: int, *,
                            seed: int = 42, force_source: int) -> np.ndarray:
    """
    Sample up to s landmarks from the largest WCC. Half highest out-degree, half uniform. Always include source.

    Nodes are sampled from the largest weakly connected component. The RNG is controlled via seed for reproducible
    selections. If the WCC has fewer than s nodes, all nodes of the WCC are returned.

    :param indptr: CSR row pointer array
    :param indices: CSR edge pointers
    :param s: Target number of landmarks to draw from the WCC
    :param seed: Random seed for deterministic sampling.
    :param force_source: Node IDs of the source node, needs to be included in the landmarks always
    :returns: Array of landmark node IDs.
    """

    # get random seed
    rng = np.random.default_rng(seed)

    # get outdegree of nodes
    deg_out = np.diff(indptr)

    # get all nodes in the WCC
    wcc_nodes = gp.largest_wcc_mask(indptr, indices)

    # draw pool, all WCC nodes except the source
    pool = wcc_nodes[wcc_nodes != force_source]

    # we always force the source landmark separately, so we need at most (s-1) additional landmarks
    # also limit to [0, pool.size] so we never request negative or more nodes than available
    need = min(max(s - 1, 0), pool.size)

    # split the remaining landmark budget: half degree-based, half uniform; this can be changed if needed
    k_out = need // 2
    k_uni = need - k_out

    # degree-based part: take the k_out nodes with highest outdegree (if k_out > 0)
    top_idx = np.empty(0, dtype=np.int32)
    if k_out > 0:
        # degrees restricted to candidate pool
        cdeg = deg_out[pool]
        # argpartition gives top-k in O(n) time (not fully sorted)
        part = np.argpartition(cdeg, -k_out)[-k_out:]
        top_idx = pool[part].astype(np.int32, copy=False)

    # uniform part: sample k_uni nodes from remaining pool (excluding top_idx)
    # if we did not pick any top nodes, remain is just the full pool (avoid setdiff overhead)
    remain = pool if top_idx.size == 0 else np.setdiff1d(pool, top_idx, assume_unique=False)

    uni_idx = np.empty(0, dtype=np.int32)
    if k_uni > 0:
        # uniform sampling without replacement; size=0 is handled above
        uni_idx = rng.choice(remain, size=k_uni, replace=False).astype(np.int32, copy=False)

    # combine both draws uncomment all and code above to draw based on outdegree; optional
    take = np.concatenate([top_idx, uni_idx])
    #  draw random; optional
    # take = rng.choice(pool, size=need, replace=False).astype(np.int32, copy=False)

    # initialize empty return array of size s
    lm = np.empty(take.size + 1, dtype=np.int32)

    # source = first node in lm
    lm[0] = force_source

    # rest is filled with sampled nodes
    lm[1:] = take

    # return the array with the sampled nodes + source
    return lm


@njit(cache=True)
def multisource_bf_R(indptr: np.ndarray, indices: np.ndarray, weights: np.ndarray, landmarks: np.ndarray, R: int,
                     use_init: np.uint8, init_dist: np.ndarray,
                     build_H: np.uint8) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Run R rounds of multi-source Bellman–Ford from a set of landmarks on a CSR graph.
    Optionally warm-start per-landmark distances and optionally record a landmark
    matrix H (edge weights between landmarks discovered during relaxations).
    Corresponds to step 2 of the pipeline (and partly step 4): 2. run R-round multi-source Bellman–Ford to build H

    :param indptr: CSR row pointer array
    :param indices: CSR edge pointers
    :param weights: CSR edge weights
    :param landmarks: Landmark node IDs
    :param R: Number of Bellman–Ford rounds to execute (R ≥ 0).
    :param use_init: If 0, initialize each landmark with distance 0. If 1, use init_dist.
    :param init_dist: Initial distances for each landmark j; only needed, if use_init = 1.
    :param build_H: If 1, update the s×s matrix Hmat on landmark hits; if 0, leave it at INF.
    :returns: Tuple (dist, owner, Hmat, last_frontier, violation_frontier) where:
              - dist : current distances to all initial landmarks
              - owner: for each node the current landmark with the smallest dist to it, -1 if not reached yet
              - Hmat : matrix for landmark to landmark distances (sqrt(n) * sqrt(n) size)
              - last_frontier : nodes processed in the last nonempty round,
              - violation_frontier : nodes u with an outgoing edge that can be used to relax another node
    """

    # get number of nodes and number of landmarks
    n = indptr.size - 1
    s = landmarks.size

    # create array for is landmark check, -1 if node is not a landmark ID (0 - j)
    is_landmark = np.full(n, -1, dtype=np.int32)
    for j in range(s):
        is_landmark[int(landmarks[j])] = j

    # init INF and dist/owner array
    infi = np.float64(np.inf)
    dist = np.full(n, infi, dtype=np.float64)
    owner = np.full(n, -1, dtype=np.int32)

    # initialize frontier and next frontier as well as in_next round array
    frontier = np.empty(n, dtype=np.int32)
    next_frontier = np.empty(n, dtype=np.int32)
    in_next = np.zeros(n, dtype=np.uint8)

    # in this part we initialize the frontier with landmarks (either with 0 or given distances
    fsz = 0  # init frontier size
    # Iterate over all landmarks -> j = landmark index in [0..s-1]
    for j in range(s):
        # v = original node ID of the j-th landmark
        v = int(landmarks[j])
        # set its distance to 0 OR if we want to start from given distance get the init_dist val of j
        d0 = 0.0 if use_init == 0 else np.float64(init_dist[j])
        # if landmark is reachable or set to 0
        if d0 < infi:
            # refresh the dist array
            dist[v] = d0
            # set owner to itself
            owner[v] = j
            # add landmark to frontier
            frontier[fsz] = v
            # increase frontier size by 1
            fsz += 1

    # Init empty matrix for the landmark relations
    Hmat = np.full((s, s), infi, dtype=np.float64)

    # ensure number of rounds R > 0
    rounds = R if R > 0 else 0

    # prevents cascading
    frontier_du = np.empty(n, dtype=np.float64)
    frontier_ou = np.empty(n, dtype=np.int32)

    # now start the relaxation rounds
    for _ in range(rounds):
        # if frontier is empty we are finished
        if fsz == 0:
            break

        # prevents cascading
        for i in range(fsz):
            u = frontier[i]
            frontier_du[i] = dist[u]
            frontier_ou[i] = owner[u]

        # else initialize next frontier size var
        nsz = 0
        # now iterate over the frontier
        for i in range(fsz):
            # get the current frontier node, its current owner and dist
            #u = frontier[i]
            #ou = owner[u]
            #du = dist[u]
            u = frontier[i]
            du = frontier_du[i]
            ou = frontier_ou[i]

            # get the corresponding edges for the frontier node
            a = indptr[u]
            b = indptr[u+1]
            # now for each edge
            for e in range(a, b):
                # get the corresponding target node
                v = indices[e]
                # calculate the new weight
                nd = du + weights[e]

                # now if we have the build landmark x landmark matrix var active
                if build_H == 1:
                    # check if our target is a landmark
                    j = is_landmark[v]
                    # if u has an owner node, v is a landmark and its no self loop
                    if ou != j and ou >= 0 and j >= 0:

                        # get the current best dist between owner landmark and target landmark
                        hcur = Hmat[ou, j]

                        # only update when new distance is strict better than the current entry
                        if nd < hcur:
                            Hmat[ou, j] = nd

                # and we also perform the ordinary BF relaxation
                if nd < dist[v]:
                    # if we get better dist refresh dist and owner of target node v
                    dist[v] = nd
                    owner[v] = ou
                    # and set v as node in the next frontier, increase next frontier count
                    if in_next[v] == 0:
                        in_next[v] = 1
                        next_frontier[nsz] = v
                        nsz += 1

        # after full relaxation round switch set new frontier to current frontier
        fsz = nsz
        for k in range(nsz):
            v = next_frontier[k]
            in_next[v] = 0
            frontier[k] = v

    # saves the last frontier for continue with dijkstra
    last_frontier = frontier[:fsz].copy()

    # builds violation frontier - nodes that can still be relaxed by an edge -> Here 1 BF Round, but we could
    # also include it in the last BF round so no additional I/O Cost
    mark_u = np.zeros(n, dtype=np.uint8)  # Tail-Knoten u

    # for every node
    for u in range(n):

        # get its current dist
        du = dist[u]

        # if not reachable continue
        if not np.isfinite(du):
            continue

        # else get the edge pointers
        a = indptr[u]
        b = indptr[u + 1]

        # iterate over the edges
        for e in range(a, b):
            # get target node and target dist
            v = int(indices[e])
            dv = dist[v]
            nd = du + float(weights[e])

            # if our new distance is better, mark the node
            if nd < dv:
                mark_u[u] = 1

    # count 1 occurences in marks
    cnt = 0
    for i in range(n):
        if mark_u[i] != 0:
            cnt += 1

    # create violation frontier array with correct size and fill it
    violation_frontier = np.empty(cnt, dtype=np.int32)
    w = 0
    for i in range(n):
        if mark_u[i] != 0:
            violation_frontier[w] = np.int32(i)
            w += 1

    # return dist, owner, the landmark matrix, last frontier and violation frontier
    return dist, owner, Hmat, last_frontier, violation_frontier


def dense_to_csr(Hmat: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert a dense adjacency matrix to CSR format.

    Finite entries in Hmat[i, j] are interpreted as edges i -> j with weight Hmat[i, j].
    Non-finite entries are treated as no edge and are skipped.
    Part of step 2 of the pipeline: 2. run R-round multi-source Bellman–Ford to build H

    :param Hmat: Dense s×s weight matrix, where finite entries denote edges.
    :returns: Hmat graph as CSR
    """
    # get row/column size of landmark matrix
    s = Hmat.shape[0]
    # init count array; will be filled with count of how many edges each row have
    counts = np.zeros(s, dtype=np.int32)
    # now for each row
    for i in range(s):
        row = Hmat[i]
        c = 0
        # count the number of finite vals ~ edges between landmarks
        for j in range(s):
            if np.isfinite(row[j]):
                c += 1
        counts[i] = c
    # init empty index pointer array
    indptr = np.empty(s+1, dtype=np.int32)
    indptr[0] = 0
    # build prefix sum by using count of each row
    np.cumsum(counts, out=indptr[1:])
    # number of all edges (indptr last position)
    mH = int(indptr[-1])
    # init weights and indices list
    indices = np.empty(mH, dtype=np.int32)
    weights = np.empty(mH, dtype=np.float64)
    wp = indptr[:-1].copy()
    # now for each landmark
    for i in range(s):
        # get its row of the matrix
        row = Hmat[i]
        for j in range(s):
            # get the corresponding target node
            w = row[j]
            # if weight is finite -> edge exists
            if np.isfinite(w):
                # set weight and indices at correct position
                p = wp[i]
                indices[p] = j
                weights[p] = np.float64(w)
                wp[i] = p + 1

    # return the csr structure
    return indptr, indices, weights


def dijkstra_on_H_via_modified(H_indptr: np.ndarray, H_indices: np.ndarray, H_weights: np.ndarray,
                               source_lm: int) -> np.ndarray:
    """
    Run Dijkstra from modified_dijkstra.py on the landmark graph H (in CSR form) from a given source landmark.
    Corresponds to pipeline step 3: 3. SSSP on H with Dijkstra from source
    :param H_indptr: CSR row pointer of the landmark graph H
    :param H_indices: CSR edges of the landmark graph H
    :param H_weights: CSR edge weights for the landmark graph H
    :param source_lm: Source landmark ID in [0, s-1] to start Dijkstra from.
    :returns: Distance array where dist[j] is the shortest distance from source_lm to landmark j.
    """
    # ensure correct datatypes
    H_indptr = H_indptr.astype(np.int32,  copy=False)
    H_indices = H_indices.astype(np.int32, copy=False)
    H_weights = H_weights.astype(np.float64, copy=False)

    # perform dijkstra
    dist, edges, pred, settled_flag, pop_order, _ = md.dijkstra_with_edges_tree(
        H_indptr, H_indices, H_weights, int(source_lm), limit=-1
    )
    # return the corresponding distances
    return dist


def finalize_with_dijkstra_from_existing_frontier(indptr: np.ndarray, indices: np.ndarray, weights: np.ndarray,
                                                  source_node: int, dist: np.ndarray,
                                                  violation_frontier: np.ndarray | None = None) \
        -> tuple[np.ndarray, np.ndarray]:
    """
    Finalize exact SSSP distances
    of the landmark approach via warm start Dijkstra from a preselected violation frontier.
    Corresponds to step 5 of our algorithm pipeline: 5. finalize exactly via warm-start Dijkstra from a frontier.

    :param indptr: CSR row pointer
    :param indices: CSR edge indices
    :param weights: CSR edge weights
    :param source_node: Source node ID for Dijkstra
    :param dist: Initial distance guess per node (from source; may not be correct yet)
    :param violation_frontier: Array of node IDs to seed Dijkstra; Nodes that still can be relaxed
    :returns: dist, pop_order:
              - dist: exact shortest-path distances from source_node.
              - pop_order: order in which nodes were settled by Dijkstra; just for counting I/Os
    """
    # get number of nodes and ensure src is int
    n = indptr.shape[0] - 1
    src = int(source_node)

    # Set source distance to 0 and init empty bf_settled array
    dist[src] = 0.0
    bf_settled = np.zeros(n, dtype=np.uint8)

    # If violation frontier exists as input and is not empty: set start nodes for dijkstra to violation frontier nodes
    starts = np.array([], dtype=np.int32) \
        if violation_frontier is None else violation_frontier.astype(np.int32, copy=False)

    # ensure source is in frontier and frontier is not empty
    if starts.size == 0 or not np.any(starts == src):
        starts = np.concatenate([starts, np.array([src], dtype=np.int32)])
    starts = np.unique(starts)

    # Now use dijkstra on the given violation frontier to fix wrong distances
    dist, edges, pred, settled_flag, pop_order, _ = md.dijkstra_with_edges_tree(
        indptr, indices, weights, src, limit=-1,
        bf_dist=dist, bf_settled=bf_settled, bf_active_ids=starts
    )

    # return the final dists and pop orders
    return dist, pop_order


def landmark_sssp(indptr: np.ndarray, indices: np.ndarray, weights: np.ndarray, source_node: int, e_b, R) -> dict:
    """

    Compute exact SSSP distances via a landmark-assisted pipeline on a CSR graph.

    The routine first builds an approximate distance potential using a small set of
    landmarks and a bounded number of Bellman–Ford (BF) relaxation rounds, and then
    restores exactness with a warm-start Dijkstra run seeded from a violation frontier.

    :param indptr: CSR row pointer of the original graph (shape n+1).
    :param indices: CSR edge targets of the original graph (shape m).
    :param weights: CSR edge weights.
    :param source_node: Source node id.
    :param e_b: Cost of one BF scan/round under the chosen I/O model.
    :param R: Number of multi-source BF rounds used for building and back-projecting (R >= 0).

    :returns: Dictionary with keys:
              - "dist_landmark": Exact shortest-path distances from source_node.
              - "dist_pre": Distances after the back-projection step (before final Dijkstra).
              - "io_cost_landmark": Accumulated proxy I/O cost.
              - "s": Number of landmarks used.
              - "landmarks": Array of landmark node ids (landmarks[0] == source_node).
              - "src_lm": Index of the source landmark in the landmark array.
              - "R": The chosen number of BF rounds.
              - "seed": RNG seed used for landmark selection.
    """
    # get number of nodes
    n = int(indptr.shape[0] - 1)

    # val for total I/O Cost
    io_cost_landmark = 0

    # 0. optional: budget-matched warmup (e_b pops)
    # dist_warm, _, _, settled_warm, pop_order_warm = md.dijkstra_with_edges_tree(
    # indptr, indices, weights, int(source_node), limit=e_b)
    # io_cost_landmark += e_b

    # 1. pick landmarks in the largest WCC
    s = int(n**0.5)
    seed = 42
    landmarks = choose_landmarks_in_wcc(indptr, indices, s, seed=seed, force_source=source_node)
    # I/O Cost --> 1 SCAN so e_b
    io_cost_landmark += e_b

    # get source position in landmarks
    pos = np.where(landmarks == int(source_node))[0]
    src_lm = int(pos[0])

    # Set R; in our experiments we choose R based one the expected diameter of our graph
    # R = int((math.log2(n))/2)

    # 2. run R-round multi-source Bellman–Ford to build H
    dist, _, Hmat, _, _ = multisource_bf_R(indptr, indices, weights, landmarks, R=R, use_init=np.uint8(0),
                                           init_dist=np.empty(0, np.float64), build_H=np.uint8(1))
    # Now Build H
    H_indptr, H_indices, H_weights = dense_to_csr(Hmat)
    # I/O cost is R * scan; build H is free because the H Graph is O(n) size
    io_cost_landmark += (e_b*R)

    # 3. sssp on H with Dijkstra from source
    dist_on_h = dijkstra_on_H_via_modified(H_indptr, H_indices, H_weights, src_lm)
    # No I/O Cost because we assume RAM size O(n)

    # optional choose R_2; we use R = R2 for our experiments
    R2 = R

    # 4. back-project with R rounds Bellman-Ford from landmark dists
    dist_to_source, owner, _, last_frontier, violation_frontier = multisource_bf_R(
        indptr, indices, weights,
        landmarks, R2,
        np.uint8(1), dist_on_h.astype(np.float64, copy=False),
        np.uint8(0)
    )
    # I/O cost is R * scan
    io_cost_landmark += (e_b*R2)
    dist_pre = dist_to_source.copy()

    # 5. finalize exactly via warm-start Dijkstra from a frontier
    dist_exact, pop_order = finalize_with_dijkstra_from_existing_frontier(
        indptr, indices, weights, source_node, dist=dist_to_source, violation_frontier=violation_frontier)
    # I/O Cost here is simplified the number of dijkstra pops -> 1 pop 1 scan
    io_cost_landmark += pop_order.size

    return {
        "dist_landmark": dist_exact,
        "dist_pre": dist_pre,
        "io_cost_landmark": io_cost_landmark,
        "s": s,
        "landmarks": landmarks,
        "src_lm": src_lm,
        "R": R,
        "seed": seed
    }
