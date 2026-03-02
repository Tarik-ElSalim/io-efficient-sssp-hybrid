# opt_hybrid_with_set_cover.py
"""
Exact hybrid Dijkstra/Bellman-Ford scheduling on the SPDAG using an IP model.

This module formulates and solves an integer program to compute
a cost-optimal sequence of Dijkstra pops and Bellman-Ford rounds in
the semi-external memory model, based on a shortest-path DAG in CSR format.
Intended for small SPDAG instances, only efficient solving for instance with less than ~ 10000 Nodes + Edges.

Main routine:
- solve_exact_spdag_schedule_ip: Takes SPDAG, source and cost parameters as input and computes optimal schedule with IP

"""

# Standard library
from typing import Tuple, List

# Third-party
import numpy as np
import pulp


def solve_exact_spdag_schedule_ip(sp_indptr: np.ndarray, sp_indices: np.ndarray, src: int, e_b: float,
                                  org_dist: np.ndarray,
                                  time_limit: float = None) -> Tuple[List[Tuple[str, int]], float]:
    """
    Resolve the exact IP for the optimal Dijkstra/BF schedule on the SPDAG.

    :param sp_indptr: spdag node pointer (CSR)
    :param sp_indices: spdag edge pointer (CSR)
    :param src: Source node as int
    :param e_b: Scan IO Cost -> Cost for 1 BF round
    :param org_dist: the original dist array to determine wheter a node can be reached in the spdag
    :param time_limit: set time limit for solver; if set solution may not be optimal; we can also set gap

    :returns schedule: list of optimal BF/Dijkstra action to minimize IO in SEM
             io_cost: optimal cost of the optimal schedule in SEM
    """

    # ensure datatypes of CSR
    sp_indptr = np.asarray(sp_indptr, dtype=np.int32)
    sp_indices = np.asarray(sp_indices, dtype=np.int32)

    # get number of nodes and source as int
    num_nodes = sp_indptr.shape[0] - 1
    src = int(src)

    # set T as number of max timesteps for our ip
    T = num_nodes
    # Create corresponding list
    steps = list(range(1, T + 1))

    # out of org dist create reachable array; ignore edges that are not reachable
    reachable = np.isfinite(org_dist)
    # set all reachable nodes that are not the source
    V = [v for v in range(num_nodes) if v != src and reachable[v]]

    # max capazity for one bellman ford relaxation
    M = len(V)

    # get all edges (u,v) out of the spdag that are reachable
    edges = []
    for u in range(num_nodes):
        if not reachable[u]:
            continue
        a = sp_indptr[u]
        b = sp_indptr[u + 1]
        for e in range(a, b):
            v = int(sp_indices[e])
            if reachable[v]:
                edges.append((u, v))

    # build preds list for all nodes
    preds = [[] for _ in range(num_nodes)]
    for (u, v) in edges:
        preds[v].append(u)

    # now create IP model, minimize target function
    prob = pulp.LpProblem("SPDAG_SSSP_Schedule", pulp.LpMinimize)

    # For each time step 1...T we define a var x and y --> x_t = 1 for dijkstra step at t and y_t = 1 for BF step a t
    x = pulp.LpVariable.dicts("x", steps, lowBound=0, upBound=1, cat="Binary")
    y = pulp.LpVariable.dicts("y", steps, lowBound=0, upBound=1, cat="Binary")

    # for each node create a var z_node_step, z_v_t = 1 if node v is used a relax node in dijkstra or bf at time t
    z = pulp.LpVariable.dicts("z", (V, steps), lowBound=0, upBound=1, cat="Binary")

    # target funktion to be minimized, x_t + e_b * y_t; so minimize the total cost of all relaxations
    prob += pulp.lpSum([x[t] + e_b * y[t] for t in steps])

    # Constraints
    # at most 1 relaxation per step so x_t + y_t <= 1, we cant BF relax and Dijkstra Relax at the same time
    for t in steps:
        prob += x[t] + y[t] <= 1, f"at_most_one_op_step_{t}"

    # Each node is used at most once in BF or Dijkstra at most one of the z_v_t var for each v is 1
    for v in V:
        prob += pulp.lpSum([z[v][t] for t in steps]) <= 1, f"at_most_one_time_node_{v}"

    # force ip to perform all actions at the beginning rounds
    for t in range(1, T):
        prob += x[t] + y[t] >= x[t + 1] + y[t + 1], f"monotone_steps_{t}"

    # ensures capazity for relaxations, we cannot relax more than 1 Dijkstra node per step and N BF nodes per step
    for t in steps:
        # sum the number of nodes that are used at step t (sum z_v_t) and ensure it is smaller/equal than 1 for dijk
        # and smaller equal than num nodes for bellman ford
        prob += pulp.lpSum([z[v][t] for v in V]) <= x[t] + M * y[t], f"capacity_step_{t}"

    # ensures topological order for covering, a node can only be covered if one of its predecessor was already covered
    for (u, v) in edges:
        # source we can continue, its always covered
        if u == src:
            continue
        # also continue if one of the nodes is not reachable
        if v not in V or u not in V:
            continue
        # for each edge that can be reached and is not adjacent to the source
        # sv and su is exactly the t at which v/u is in frontier and gets relaxed
        sv = pulp.lpSum([t * z[v][t] for t in steps])
        su = pulp.lpSum([t * z[u][t] for t in steps])
        # sum_z_v is 1 if at any time v is an active frontier node that is relaxed either by dijkstra or bf
        sum_z_v = pulp.lpSum([z[v][t] for t in steps])
        # If v is scheduled as an active relaxation node (sum_z_v = 1), enforce topological order:
        # t(v) >= t(u) + 1 --> v can only be processed after its predecessor u.
        # If v is never selected (sum_z_v = 0), the constraint is relaxed via the big-M term and imposes no ordering.
        prob += sv + T * (1 - sum_z_v) >= su + 1, f"topo_{u}_to_{v}"

    # now for each node
    for v in V:
        # check if source is its predecessor and create variables for each predecessor of v that is not source
        terms = []
        has_src_pred = False
        # for each predecessor
        for u in preds[v]:
            # if its source set has _src _ pred as true
            if u == src:
                has_src_pred = True
            # if not create var for each step and predecessor
            else:
                for t in steps:
                    terms.append(z[u][t])
        # ensure each node is covered
        lhs = pulp.lpSum(terms)
        if has_src_pred:
            lhs += 1
        prob += lhs >= 1, f"cover_{v}"

    # solve model, set time limit or gap if you want
    solver = pulp.GUROBI(msg=False, timeLimit=time_limit)

    # if no gurobi is available use pulp instead
    # solver = pulp.PULP_CBC_CMD(msg=False, timeLimit=time_limit)

    prob.solve(solver)

    # check if solver find an optimal solution for our IP
    status = pulp.LpStatus[prob.status]
    if status != "Optimal":
        print("No optimal solution found.")
        return [], -1.0

    # Finally build the schedule of dijkstra pops/bf rounds and return it
    schedule = [("DJ", src)]  # schedule list with source node initialized
    io_cost = 1.0  # io return variable

    # now for each step
    for t in steps:
        # get x_val and y_val
        x_val = pulp.value(x[t])
        y_val = pulp.value(y[t])

        # ensure they exist
        if x_val is None:
            x_val = 0.0
        if y_val is None:
            y_val = 0.0

        # case for if dijkstra is chosen (x_val > 0.5 due to small deviations from 1)
        if x_val > 0.5:
            # scan the z var for the node v that was chosen at time t
            chosen_v = None
            for v in V:
                zv = pulp.value(z[v][t])
                # again if z_v > 0.5 we take it as chosen and break
                if zv is not None and zv > 0.5:
                    chosen_v = int(v)
                    break
            # error if we used dijkstra relax at t but no z_v_t was set at this t
            if chosen_v is None:
                raise RuntimeError(f"No node assigned to DJ step t={t} in optimal solution.")
            # append to schedule and increase i/o cost
            schedule.append(("DJ", chosen_v))
            io_cost += 1.0

        # now same for BF var; if it is larger than 0.5 -> BF relax at t
        elif y_val > 0.5:
            # add BF and -1 to schedule
            schedule.append(("BF", -1))
            # increase i_o cost by e_b
            io_cost += float(e_b)

    # return schedule and io_cost
    return schedule, io_cost
