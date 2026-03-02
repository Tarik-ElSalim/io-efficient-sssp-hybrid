# graph_io.py
"""
Experiment driver and I/O helpers.

Loads experiment instances from TOML, generates CSR graphs, runs the different SSSP
approaches (Dijkstra/SPT, Best-Single-Switch, Bellman–Ford, Multi-Switch,
Greedy-Hybrid, Hybrid-Opt and Landmark), checks distance correctness, and exports
compact JSON summaries plus PDF plots to results/.

Main routines:
- single_experiments: run the single-instance pipeline and export plots/summaries.
- asym_experiments: run the asymptotic series pipeline and export aggregated results.
"""

# Standard library
import json
import math
import tomllib
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

# Third-party
import numpy as np
from numpy.typing import NDArray

# Local application
from . import bellman_ford
from . import best_single_switch
from . import graph_generators
from . import graph_plots
from . import graph_properties
from . import landmark_sssp
from . import modified_dijkstra
from . import multi_switch_hybrid
from . import opt_hybrid_with_set_cover
from . import spdag


def load() -> NDArray[Any]:
    """
    This is a simple test function to load a graph as .edges or .txt. We do not use this in the experiments.

    :return: Adjacency list of the graph, you need to convert it to csr with graph_generators.edges_to_csr()
    """
    edges = []
    # change input file here
    with open("test_bf.edges", "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            s, d, w = line.split()
            edges.append((int(s), int(d), float(w)))
            # If you want to load an unweighted graph, use this part it will create weights for each edge with w=1
            # s, d= line.split()
            # edges.append((int(s), int(d), float(1)))
    print("edges loaded")
    edges = np.asarray(list(edges), dtype=np.dtype([('u', np.int32),
                                                    ('v', np.int32),
                                                    ('w', np.float32)]))
    return edges


def all_dists_equal(reference: NDArray[np.float64], *dists: Any) -> bool:
    """
    Check whether all provided distance arrays match the reference dist array.
    Dist inputs equal to -1 (or None) are ignored.

    :param reference: Reference dist array (baseline), shape (n,).
    :param dists: Other dist arrays to compare against reference (or -1/None to ignore).
    :return: True iff all non-ignored dist arrays are elementwise equal to reference.
    """
    ref = reference
    for dist in dists:
        if dist is None:
            continue
        if isinstance(dist, (int, np.integer)) and int(dist) == -1:
            continue
        if ref.shape != dist.shape:
            return False
        if int(np.size(ref) - np.sum(ref == dist)) != 0:
            return False
    return True


def calculate_best_switch_results(indptr: NDArray[np.int32], indices: NDArray[np.int32], weights: NDArray[np.float32],
                      e_b: int) -> Dict[str, Any]:
    """
    Compute Dijkstra, the corresponding shortest path tree and the best Dijkstra to Bellman–Ford switch for a CSR graph.

    :param indptr: CSR row pointers (n+1).
    :param indices: CSR column indices (m).
    :param weights: Edge weights (float32, size m).
    :param e_b: Blocks per full edge + weight scan (I/O cost factor).
    :return: {
        n_nodes,n_edges,start_node,reach_ratio,
        tree_indptr,tree_indices,level_counts,level_edge_work,
        limit_fast,costs_fast,bf_rounds_log,cost_min,best_switch}
    """

    # Pick a good start: best random node in the largest WCC by reachability and print the result
    print("Calculate optimal start location")
    # change seed here if you want different starting nodes
    node, reaches = graph_properties.best_random_node_in_largest_wcc_by_reach(indptr, indices, seed=2)

    # prints can be deleted, only activated because they are good for control purposes
    print("Node "+str(node)+" found with connection to "+str(reaches/(len(indptr)-1)*100) + " % of the graph nodes")
    print("Graph got in total " + str(len(indices)) + " edges")

    # Calculate dijkstra results
    print("Start Dijkstra run")
    dist, edges, pred, settled_flag, pop_order, _ = modified_dijkstra.dijkstra_with_edges_tree(indptr,
                                                                                               indices, weights, node)

    # Out of dijkstra results calculate the shortest path tree
    print("Calculate SPT by Dijkstra Results")
    tree_indptr, tree_indices = modified_dijkstra.build_tree_csr_from_pred(pred, settled_flag)

    # Calculate the stats of the tree - nodes/edges per tree level
    print("Calculate levels stats of the tree")
    level_counts, level_edge_work = modified_dijkstra.level_stats_from_settled(indptr, edges, settled_flag)

    # Calculates our Best-Single-Switch Results and additional round based statistics
    print("Calculate the optimal switching point from Dijkstra to BF (Best-Single-Switch)")
    bf_rounds_log, cost_min, limit = best_single_switch.calc_best_switch(tree_indptr, tree_indices,
                                                                         e_b, node, indptr, indices, weights, pred)

    # Save all important results and return them as dict
    return {
        # Metadata
        "n_nodes": len(indptr) - 1, "n_edges": len(indices), "start_node": node,
        "reach_ratio": reaches / (len(indptr) - 1),

        # Dijkstra results without dist array
        "dist": dist, "edges": edges, "pred": pred, "settled_flag": settled_flag, "pop_order": pop_order,

        # Tree statistics
        "tree_indptr": tree_indptr, "tree_indices": tree_indices, "level_counts": level_counts,
        "level_edge_work": level_edge_work,

        # Best-Sinlge-Switch Results
        "bf_rounds_log": bf_rounds_log, "cost_min": cost_min, "best_switch": limit,
    }


def hybrid_results(indptr: NDArray[np.int32], indices: NDArray[np.int32], weights: NDArray[np.float32], node: int,
                   e_b: int) -> Dict[str, Any]:
    """
    Compute results for the Multi-Switch variant that alternates Dijkstra relaxation rounds with Bellman–Ford rounds.

    :param indptr: CSR row pointers (n+1).
    :param indices: CSR col indices (m).
    :param weights: Edge weights (m).
    :param node: source node id
    :param e_b: i/os needed to scan all edges + weights of the graph --> reference point for greedy switch point
    :return: Dictionary with hybrid run results:
        - dist: float64[n] final distances.
        - settled_flag: nodes finalized by Dijkstra (1 = settled).
        - pop_order: concatenated Dijkstra pop order.
        - bf_relax_per_phase:  relax counts per BF phase.
        - hybrid_io: total i/o count with this method
    """

    print("Now calculate Multi-Switch approach.")
    # execute alternating hybrid (Multi-Switch) algorithm
    dist, settled_flag, pop_order_all, hybrid_io, bf_relax_per_phase, heap_sizes, frontier_sizes, schedule_order =\
        multi_switch_hybrid.hybrid_dijkstra_bf(indptr, indices, weights, node, int(e_b))

    return {
        "dist": dist,
        "settled_flag": settled_flag,
        "pop_order": pop_order_all,
        "bf_relax_per_phase": np.asarray(bf_relax_per_phase, dtype=np.int64),
        "hybrid_io": hybrid_io,
        "hybrid_heap_sizes": heap_sizes,
        "hybrid_frontier_sizes": frontier_sizes,
        "hybrid_schedule_order": schedule_order
    }


def load_toml(path: Union[str, Path]) -> List[Dict[str, Any]]:
    """
    Load experiment instances from a TOML file.
    :param path: Path to the TOML file.
    :return list[dict]: List of instance dictionaries from cfg["instance"] (or empty list).
    """
    # normalize to Path
    path = Path(path)

    # open as binary for tomllib
    with path.open("rb") as f:
        # parse TOML into a dict
        cfg = tomllib.load(f)

    # pick list of instances
    instances = cfg.get("instance", [])
    return instances


def save_run_summary(instance: Dict[str, Any], results: Dict[str, Any], bellman_results: Tuple[Any, ...],
                     results3: Dict[str, Any], schedule_kind: Any, io_greedy: int,
                     io_ip: int, results_landmark: Dict[str, Any], dir_name: str, blocksize: int, e_b: int,
                     is_spdag: bool) -> Path:
    """
    Save a compact JSON summary for one single-experiment run.
    Stores only lightweight metrics plus schedules (no dist/pred/pop_order arrays). Can also store larger metrics, just
    add them below, if needed.

    :param instance: Input instance config from TOML.
    :param results: Output of calculate_results().
    :param bellman_results: Output tuple of bf_csr_frontier_complete().
    :param results3: Output of hybrid_results().
    :param schedule_kind: Greedy schedule kinds (or -1).
    :param io_greedy: I/O cost of greedy schedule.
    :param io_ip: I/O cost of IP schedule (-1 if skipped).
    :param results_landmark: Output of landmark_sssp().
    :param dir_name: Results subdirectory name.
    :param blocksize: Block size used for this run.
    :param e_b: Blocks per full edge+weight scan.
    :param is_spdag: True if shortest path structure is real spdag if it is a spt False
    :return: Path to the written JSON file.
    """

    # helper function for syntax typo
    def to_native(x: Any) -> Any:
        if x is None:
            return None
        if isinstance(x, (int, np.integer)) and int(x) == -1:
            return -1
        if hasattr(x, "item"):
            try:
                return x.item()
            except (ValueError, TypeError):
                pass
        if hasattr(x, "tolist"):
            try:
                return x.tolist()
            except (ValueError, TypeError):
                pass
        if isinstance(x, Path):
            return str(x)
        return x

    payload = {
        "meta": {"dirname": dir_name},
        "instance": {
            "generator": instance.get("generator_to_use"),
            "node_count": instance.get("node_count"),
            "seed": instance.get("seed"),
            "start_node_seed": 5,  # fixed because we will not change this during our experiments
            "distribution": instance.get("distribution"),
            "prob": instance.get("prob"),
            "radius": instance.get("radius"),
            "barabasi_m": instance.get("barabasi_m"),
            "limit": instance.get("limit"),
        },
        "graph": {
            "n_nodes": results.get("n_nodes"),
            "n_edges": results.get("n_edges"),
            "start_node": results.get("start_node"),
            "reach_ratio": results.get("reach_ratio"),
            "reachable_n": int(np.isfinite(np.asarray(results["dist"], dtype=np.float64)).sum()),
            "blocksize": blocksize,
            "e_b": e_b,
        },
        "baselines": {
            "io_dijkstra": int(np.isfinite(np.asarray(results["dist"], dtype=np.float64)).sum()),
            "io_bellman_full": int(to_native(bellman_results[2])) * int(e_b),
        },
        "dijkstra_best_switch": {
            "switch_point_best_single_switch": results.get("best_switch"),
            "io_best_single_switch": results.get("cost_min"),
        },
        "bellman_ford": {
            "iters_done": to_native(bellman_results[2]) if len(bellman_results) > 2 else None,
            "io_full": int(to_native(bellman_results[2])) * int(e_b),
        },
        "multi_switch": {
            "io_multi_switch": results3.get("hybrid_io"),
            "num_phases": int(np.asarray(results3["hybrid_schedule_order"]).shape[0]),
        },
        "greedy_hybrid": {
            "io_greedy_hybrid": to_native(io_greedy),
            "len": int(np.asarray(schedule_kind).size),
            "num_bf": int(np.sum(np.asarray(schedule_kind, dtype=np.int32) == 0)),
            "num_dj": int(np.sum(np.asarray(schedule_kind, dtype=np.int32) == 1)),
        },
        "hybrid_opt_with_ip": {
            "io_hybrid_opt": to_native(io_ip),
            "is_spdag?": is_spdag
        },
        "landmark": {
            "io_landmark": to_native(results_landmark.get("io_cost_landmark")),
            "landmark_R_R2": (len(results["level_counts"]))//2,
            "landmark_seed": 42,  # also fixed due to our experiment size, seed got low influence on algorithm outcome
        },
        "stats": {
            "level_counts": to_native(results.get("level_counts")),
        },
    }

    out_dir = Path("results", "results_single_experiments", dir_name)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{dir_name}.json"

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    return out_path


def single_experiments() -> int:
    """
    Runs the full single experiment pipeline for a set block size (Einzelbetrachtung). It loads graph instances
    from a TOML file, builds each graph, computes e_b (blocks per full edge+weight scan),
    and evaluates all approaches and produces different graphics and statistics.
    Finally, it save the results as PDFs/json to the results/ folder and returns 0 on success.
    """

    # set the blocksize for the experiments here
    blocksize = 16  # Simplified how many edges+weights fit in one block; values for our experiments 512,4096,65536

    # set the input toml file with parameters here and call the load toml function
    config = Path("asym_input") / "einzelbetrachtung_IP_seed_23.toml"
    toml_content = load_toml(config)

    # now iterate over all instances of the toml file
    for instance in toml_content:

        # print the instance and its parameters; not necessary but nice to check the progress
        print(instance)

        # generate the corresponding graph
        indptr, indices, weights = graph_generators.create_graph_by_gen(instance["generator_to_use"],
                                                                        instance["node_count"], instance["seed"],
                                                                        True if instance["distribution"] == "Exp"
                                                                        else False, instance["prob"],
                                                                        instance["radius"], instance["barabasi_m"])

        # Calculate how many block transfers are needed for one complete scan of all graph edges + weights (e_b)
        e_b = math.ceil(len(indices) / blocksize)

        # Calculates results for SPT, SEM-Dijkstra and Best-Single-Switch
        results = calculate_best_switch_results(indptr, indices, weights, e_b)

        # Calculates results for Bellman--Ford
        print("Start Bellman-Ford Run")
        bellman_results = bellman_ford.bf_csr_frontier_complete(indptr, indices, weights, results["start_node"], -1)

        # Calculate the results for Multi-Switch Approach
        results3 = hybrid_results(indptr, indices, weights, results["start_node"], e_b)

        # Results for Greedy-Hybrid
        print("Calculate Greedy-Hybrid Set Cover Results now")
        sp_indptr, sp_indices, sp_weights, is_spdag = spdag.spdag_from_dist(indptr, indices, weights, results["dist"])
        schedule_kind, schedule_pop, greedy_dij_gain, greedy_bf_gain = (
            spdag.fast_online_frontier_set_cover_core(sp_indptr, sp_indices, results["start_node"], e_b))
        dist_greedy, io_greedy = spdag.apply_schedule_core(indptr, indices, weights, results["start_node"],
                                                           schedule_kind, schedule_pop, e_b)

        # Optional for smaller graphs with up to 500 edges the Hybrid-Opt IP Approach
        if len(indices) < 500:
            print("Calculate Hybrid-Opt IP results now")
            sched_ip, io_ip = opt_hybrid_with_set_cover.solve_exact_spdag_schedule_ip(sp_indptr, sp_indices,
                                                                                      results["start_node"], e_b,
                                                                                      results["dist"], time_limit=180)
            dist_ip, io_ip = spdag.apply_schedule_get_dist(indptr, indices, results["start_node"], sched_ip, e_b,
                                                           weights)
        else:
            sched_ip, io_ip, dist_ip = -1, -1, -1

        # Results for Landmark
        print("Calculate Landmark results now")
        results_landmark = landmark_sssp.landmark_sssp(indptr, indices, weights, results["start_node"], e_b,
                                                       (len(results["level_counts"])//2))

        # check if all approaches return correct distance arrays
        dists_ok = all_dists_equal(results["dist"], bellman_results[0], results3["dist"], dist_greedy, dist_ip,
                                   results_landmark["dist_landmark"])

        if not dists_ok:
            print("One approach returned incorrect distances.")
            break

        # Create directory for pdf results
        dir_name = (f"{instance["generator_to_use"]}_nodes_{results["n_nodes"]}_edges_{len(indices)}_"
                    f"weights_{instance["distribution"]}_seed_{instance["seed"]}")
        Path("results", "results_single_experiments", dir_name).mkdir(parents=True, exist_ok=True)

        # export results to json; if you want to add or delete results, do it directly in save_run_summary()
        save_run_summary(instance, results, bellman_results, results3, int(io_greedy),
                         sched_ip, io_ip, results_landmark, dir_name, blocksize, e_b, is_spdag)

        # -------------- Plot functions from here ----------------------------------------------------------------
        graph_plots.plot_level_counts(results["level_counts"], instance["generator_to_use"], results["n_nodes"],
                                      len(indices), dir_name)

        graph_plots.plot_bf_switch_points(results["bf_rounds_log"], instance["generator_to_use"], results["n_nodes"],
                                          int(np.isfinite(np.asarray(results["dist"], dtype=np.float64)).sum()),
                                          results["n_edges"], results["cost_min"], results["best_switch"], dir_name)

        graph_plots.plot_bf_convergence(results["level_counts"], bellman_results[5], instance["generator_to_use"],
                                        results["n_nodes"], len(indices), dir_name)

        graph_plots.plot_hybrid_schedule_sizes(results3["hybrid_heap_sizes"], results3["hybrid_frontier_sizes"],
                                               results3["hybrid_schedule_order"], e_b, instance["generator_to_use"],
                                               results["n_nodes"], len(indices), dir_name)
        graph_plots.plot_greedy_schedule_gains(schedule_kind, greedy_dij_gain, greedy_bf_gain,
                                               instance["generator_to_use"], results["n_nodes"], len(indices),
                                               dir_name)

        graph_plots.plot_landmark_level_pipeline(results["edges"], (len(results["level_counts"])//2), results["dist"],
                                                 results_landmark["dist_pre"], instance["generator_to_use"],
                                                 results["n_nodes"], len(indices), dir_name)
        # Hybrid-Opt IP Visualisation
        if len(indices) < 500:
            graph_plots.plot_ip_schedule_relaxation_gains(
                indptr, indices, weights,
                results["start_node"], sched_ip, results["dist"],
                instance["generator_to_use"], results["n_nodes"], len(indices),
                dir_name
            )

    return 0


def asym_experiments() -> int:
    """
    Runs the experiments pipeline for the asymptotic analysis of the different approaches (Reihenbetrachtung).
    Plots the results and returns 0 on success.
    """

    # set the input toml file with parameters here and call the load toml function
    config = Path("asym_input") / "reihenbetrachtung_gnp_seed_23.TOML"
    toml_content = load_toml(config)

    # set the blocksize for the experiments here
    blocksize = 128  # Simplified how many edges with weights fit in one block; standard values 512,1024,8192,32768

    # collect results here
    rows = []

    # iterate over all instances
    for inst in toml_content:

        print(inst)

        # get all instance information
        graph_type = inst["generator_to_use"]
        n = int(inst["node_count"])
        seed = int(inst.get("seed", 0))
        distribution = inst.get("distribution", "Uniform")
        prob = inst.get("prob", 0.0)
        radius = inst.get("radius", 0.0)
        barabasi_m = inst.get("barabasi_m", 0)


        # generate the corresponding graph
        indptr, indices, weights = graph_generators.create_graph_by_gen(graph_type, n, seed,
                                                                        True if distribution == "Exp" else False,
                                                                    prob, radius, barabasi_m)

        # Now calculate how many block transfers are needed for one complete scan of all graph edges + weights
        m = len(indices)
        e_b = math.ceil(m / blocksize)

        # calc results for spt, sem-dijkstra and best-single-switch
        results = calculate_best_switch_results(indptr, indices, weights, e_b)

        # Calculates results for Bellman--Ford
        print("Start Bellman-Ford Run")
        bellman_results = bellman_ford.bf_csr_frontier_complete(indptr, indices, weights, results["start_node"], -1)

        # calc results for landmark
        print("Calculate Landmark results now")
        results_landmark = landmark_sssp.landmark_sssp(indptr, indices, weights, results["start_node"], e_b,
                                                       (len(results["level_counts"]) // 2))

        # calc results for the greedy set cover approach on the SPDAG
        print("Calculate Greedy-Hybrid Set Cover Results now")
        sp_indptr, sp_indices, sp_weights, is_spdag = spdag.spdag_from_dist(indptr, indices, weights, results["dist"])
        schedule_kind, schedule_pop, greedy_dij_gain, greedy_bf_gain = spdag.fast_online_frontier_set_cover_core(
                                                                    sp_indptr, sp_indices, results["start_node"], e_b)
        dist_greedy, io_greedy = spdag.apply_schedule_core(indptr, indices, weights, results["start_node"],
                                                           schedule_kind, schedule_pop, e_b)

        # calc Multi-Switch result
        results_alternating = hybrid_results(indptr, indices, weights, results["start_node"], e_b)

        # calc ip solution again only for small instances with up to 500 edges
        if len(indices) < 500:
            print("Calculate exact IP approach now")
            sched, io_ip = opt_hybrid_with_set_cover.solve_exact_spdag_schedule_ip(sp_indptr, sp_indices,
                                                                                   results["start_node"],
                                                                                   e_b, results["dist"], time_limit=30)
            dist_ip, io_ip = spdag.apply_schedule_get_dist(indptr, indices, results["start_node"], sched, e_b, weights)
        else:
            dist_ip, io_ip = -1, -1

        # now check if all approaches return correct dist arrays
        dists_ok = all_dists_equal(results["dist"], bellman_results[0], results_alternating["dist"],
                                   dist_greedy, dist_ip, results_landmark["dist_landmark"])

        if not dists_ok:
            print("One approach returned incorrect distances.")
            break

        # I/O metrics
        dijkstra_io = int(np.isfinite(results["dist"]).sum())
        bellman_io = bellman_ford.bf_csr_frontier_complete(indptr, indices, weights, results["start_node"], -1)[2] * e_b
        best_switch_io = results["cost_min"]
        hybrid_alt_io = results_alternating["hybrid_io"]  # multi switch
        io_cost_landmark = results_landmark["io_cost_landmark"]

        # add results to return list
        rows.append({
            "graph_type": graph_type,
            "n": n,
            "reachable_n": int(results["reach_ratio"] * n),
            "m": m,
            "avg_deg": (m / n) if n > 0 else 0.0,
            "blocksize": blocksize,
            "e_b": e_b,
            "seed": seed,
            "distribution": distribution,
            "barabasi_m": barabasi_m,
            "io_dijkstra_simplified": dijkstra_io,
            "io_bellman_full": bellman_io,
            "io_dj_to_bf_best_switch": best_switch_io,
            "io_hybrid_multi": hybrid_alt_io,
            "io_cost_landmark": io_cost_landmark,
            "io_best_switches_overall": io_greedy,
            "io_Ip": io_ip
        })

    # Create dir for pdf results
    dir_name = f"{graph_type}_asymptotic_experiments_{blocksize}_{inst["distribution"]}"
    out_dir = Path("results") / "results_asym" / dir_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # compute and add gradient results to rows
    stats = graph_plots.compute_asym_slopes_single(rows)
    rows.append({
        "summary": "asymptotic_stats",
        "alpha_r2": stats,
    })

    # write results to json
    json_path = out_dir / "results.json"  # oder f"{dir_name}.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)

    # save the stats results central in one data
    stats_path = Path("results") / "asym_stats.json"
    # if the json already exits open it and load it as all stats
    if stats_path.exists():
        with stats_path.open("r", encoding="utf-8") as f:
            all_stats = json.load(f)
    # else just create empty all_stats
    else:
        all_stats = {}
    # unique id for each experiment
    exp_id = dir_name
    # fill the corresponding entry
    all_stats[exp_id] = stats
    # finally save the results again
    with stats_path.open("w", encoding="utf-8") as f:
        json.dump(all_stats, f, ensure_ascii=False, indent=2)

    # create plot with plot function
    graph_plots.plot_asym_ios_single(rows, dir_name)

    return 0
