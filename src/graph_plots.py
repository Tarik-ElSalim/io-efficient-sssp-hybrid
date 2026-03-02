# graph_plots.py
"""
Plotting helpers for approach and a quick Graph (CSR) visualizer.
"""

# Standard library
from pathlib import Path
from typing import Optional, Sequence, Tuple, Union, Dict, Any


# Third-party
import re
import json
import matplotlib
matplotlib.use("Agg")  # important: Only uncomment this line if you want to export pdf
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator
from matplotlib.colors import TwoSlopeNorm

def _set_plot_defaults(scale: float = 1.25) -> None:
    """
    Set global matplotlib rcParams for consistent, publication-ready plots.

    :param scale: Multiplicative scale factor for the base font size (default 1.25).
    :return: None
    """
    base = 10 * scale
    matplotlib.rcParams.update({
        # font size
        "font.size": base,

        # axes, titles
        "axes.titlesize": base + 2,
        "axes.labelsize": base + 1,
        "figure.titlesize": base + 3,

        # tick-labels
        "xtick.labelsize": base,
        "ytick.labelsize": base,

        # legend
        "legend.fontsize": base,
        "legend.title_fontsize": base,

        # tight layout automatically (helps against overlaps)
        "figure.autolayout": True,

        # never cut off text/legends when saving
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.10,

        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

# set plot defaults to scale = 1.5 (best readability without overlapping)
_set_plot_defaults(scale=1.5)


def plot_level_counts(level_counts: Sequence[int], gen_name: str, num_nodes: int,num_edges:int, dirname: str) -> Path:
    """
     Bar plot of nodes per Shortest Path Tree level. Saves the plot as PDF.

     :param level_counts: Counts per level (index = level).
     :param gen_name: Generator name used in title/filename.
     :param num_nodes: Number of nodes for labeling.
     :param num_edges: Number of Edges for labeling
     :param dirname: Subdirectory inside "results/" to save to.
     :returns: Path to the created PDF.
     """

    # Gets the data series for visualisation: SPT level and nodes in each level
    counts = np.asarray(level_counts)
    indices = np.arange(len(counts))

    # Sets output path based on gen_name and num_nodes
    safe_name = f"{''.join(c if c.isalnum() or c in '-_.' else '_' for c in gen_name)}{num_nodes}_{num_edges}"
    output_path = Path(f"results/results_single_experiments/{dirname}/Level_structure_{safe_name}.pdf")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Creates plot and plot descriptions
    plt.figure(figsize=(10, 6))
    plt.bar(indices, counts, color="tab:blue")
    plt.xlabel("Shortest path tree level")
    plt.ylabel("Nodes in each shortest path tree level")
    plt.title(f"SPT level distribution: {gen_name} with (n={num_nodes:,}, m={num_edges:,})")
    plt.tight_layout()

    # Save as pdf and close plot
    plt.savefig(output_path, format="pdf")
    plt.close("all")

    # returns out path
    return output_path

def plot_bf_convergence(level_counts: Sequence[int], frontier_sizes: Sequence[int], gen_name: str,
                        num_nodes: int, num_edges: int, dirname: str) -> Path:
    """
    Plot Bellman–Ford convergence curves (nodes per SPT level vs. frontier (Violation Frontier) size per BF round).
    Saves the plot as pdf.

    :param level_counts: Counts per SPT level (index = level).
    :param frontier_sizes: Frontier size (active nodes) per BF round.
    :param gen_name: Generator name used in title/filename.
    :param num_nodes: Number of nodes for labeling.
    :param num_edges: Number of Edges for labeling
    :param dirname: Subdirectory inside "results_single_experiments/" to save to.
    :returns: Path to the created PDF.
    """

    # normalize inputs
    y_bar = np.asarray(level_counts, dtype=np.int64)
    y_line = np.asarray(frontier_sizes, dtype=np.int64)

    # align to common length (levels vs BF rounds)
    t = int(min(y_bar.size, y_line.size))

    # x-axis for shared index
    x = np.arange(t, dtype=np.int64)

    # sanitize generator name for filesystem-safe filenames
    safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in gen_name)
    out = Path(f"results/results_single_experiments/{dirname}/BF_frontier_vs_levels_{safe}_{num_nodes}_{num_edges}.pdf")
    out.parent.mkdir(parents=True, exist_ok=True)

    # choose scaling so y-axis stays readable for large counts
    max_y = int(max(y_bar[:t].max(), y_line[:t].max()))
    scale = 1
    scale_label = "Count"
    if max_y >= 1_000_000:
        scale = 1_000_000
        scale_label = r"Count ($\times 10^{6}$)"
    elif max_y >= 1_000:
        scale = 1_000
        scale_label = r"Count ($\times 10^{3}$)"

    # apply scaling consistently to both series
    yb = y_bar[:t] / scale
    yl = y_line[:t] / scale

    # set up figure/axes
    plt.figure(figsize=(10, 6))
    ax = plt.gca()

    # bars: nodes per SPT level
    ax.bar(x, yb, color="tab:blue", zorder=1, label="Nodes per SPT level")
    # line: BF violation frontier per round
    ax.plot(x, yl, color="tab:red", linewidth=1.8, marker="o", markersize=4,
            zorder=3, label="Violation frontier size (per BF round)")

    # labels/title
    ax.set_xlabel("Shortest path tree level/Bellman Ford Relaxation Round")
    ax.set_ylabel(scale_label)
    ax.set_title(f"SPT level distribution vs. BF violation frontier: {gen_name} (n={num_nodes:,}, m={num_edges:,})")

    # set style
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(True, axis="y", linestyle="--", linewidth=0.6, alpha=0.35)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False)

    # export PDF
    plt.tight_layout()
    plt.savefig(out, format="pdf")
    plt.close("all")
    return out

def plot_bf_switch_points(bf_rounds_log: Sequence[int], gen_name: str, num_nodes: int, nodes_reachable: int,
                          num_edges:int, cost_min: float,best_switch: int, dirname: str) -> Path:
    """
    Bar-plot of needed BF rounds after each Dijkstra pop and the optimal switch point to BF. Saves the plot as PDF.

    :param bf_rounds_log: BF-round left per settled node with dijkstra (length = #pops).
    :param gen_name: Generator name used in title/filename.
    :param num_nodes: Number of nodes for labeling.
    :param num_edges: Number of Edges for labeling
    :param nodes_reachable: Number of nodes reachable.
    :param cost_min: Best combined cost for the chosen switch - cost with Dijkstra + optimal Switch to BF (label value).
    :param best_switch: 1-based best switch point marked as a vertical line.
    :param dirname: Subdirectory inside "results/" to save to.

    :returns: Path to the created PDF.
    """

    # normalize BF-round log to int array (remaining rounds per pop k)
    vals_full = np.asarray(bf_rounds_log, dtype=np.int64)
    n = len(vals_full)

    # x axis corresponds to Dijkstra pops k (1-based)
    x_full = np.arange(n, dtype=np.int64) + 1

    # downsample for very long curves (keeps PDF size/runtime reasonable)
    if n > 1000:
        idx = np.linspace(0, n - 1, num=1000, dtype=np.int64)
        # ensure endpoints are included
        idx = np.unique(np.concatenate((np.asarray([0], dtype=np.int64), idx, np.asarray([n - 1], dtype=np.int64))))
        x = x_full[idx]
        vals = vals_full[idx]
    else:
        x = x_full
        vals = vals_full

    # sanitize generator name for filesystem safe filenames
    safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in gen_name)
    out = Path(
        f"results/results_single_experiments/{dirname}/Best_Single_Switch_costs_{safe}_{num_nodes}_{num_edges}.pdf")
    out.parent.mkdir(parents=True, exist_ok=True)

    # set up figure/axes
    plt.figure(figsize=(10, 6))
    ax = plt.gca()

    # bar plot containing remaining BF rounds after switching at pop k
    ax.bar(x, vals, color="tab:blue", zorder=1)

    # mark best switch point
    ax.axvline(best_switch, color="black", linestyle="--", linewidth=2.2, zorder=5,
               ymin=0.0, ymax=0.8, label=f"Best switch $k={best_switch}$")

    # labels/title
    ax.set_xlabel("Dijkstra pops (settled vertices) $k$")
    ax.set_ylabel("Remaining BF rounds to convergence")
    ax.set_title(f"Single-switch cost curve (Dijkstra \u2192 BF): {gen_name} (n={num_nodes:,}, m={num_edges:,})")

    # annotate key costs (minimum total cost and baseline Dijkstra cost proxy)
    txt = (
        f"$C_\\min$ = {int(cost_min)}  (pops until switch + remaining scans)\n"
        f"Dijkstra cost = {nodes_reachable}  (reachable vertices)"
    )
    ax.text(
        0.98, 0.95, txt,
        ha="right", va="top",
        transform=ax.transAxes,
        bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"),
    )

    # set style
    ax.set_axisbelow(True)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(True, axis="y", linestyle="--", linewidth=0.6, alpha=0.35)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, loc="upper right", bbox_to_anchor=(0.95, 0.80))

    # export PDF
    plt.tight_layout()
    plt.savefig(out, format="pdf")
    plt.close("all")
    return out

def plot_landmark_level_pipeline(levels: Sequence[int], r: int,dist_ref: Sequence[float],dist_pre: Sequence[float],
                                 gen_name: str,num_nodes: int,num_edges: int,dirname: str) -> Path:
    """
    Stacked bar plot per SPT level: reachable nodes per level (blue) and nodes already correct after the Landmark
    pre-labeling (red). Unreachable nodes (dist_ref = inf) are ignored. Saves the plot as PDF.

    :param levels: SPT level per node
    :param r: R,R2 Value for Landmark corresponds to (SPT depth//2)
    :param dist_ref: Reference SSSP distances (Dijkstra)
    :param dist_pre: Landmark pre-distance labels before finalization via warm-start dijkstra
    :param gen_name: Generator name used in title/filename.
    :param num_nodes: Number of nodes for labeling.
    :param num_edges: Number of edges for labeling.
    :param dirname: Subdirectory inside results_single_experiments/ to save to.
    :returns: Path to the created PDF.
    """

    # Ensure arrays are np.array
    lvl = np.asarray(levels, dtype=np.int64)
    d0 = np.asarray(dist_ref, dtype=np.float64)
    d1 = np.asarray(dist_pre, dtype=np.float64)

    # Only consider nodes reachable in the reference solution
    mask = np.isfinite(d0)
    lvl_m = lvl[mask]
    d0_m = d0[mask]
    d1_m = d1[mask]

    # Robust correctness check for float distances
    ok = np.isclose(d1_m, d0_m, rtol=1e-12, atol=1e-12)

    # get total, correct and incorrect nodes per level
    l = int(np.max(lvl_m)) + 1
    total = np.bincount(lvl_m, minlength=l).astype(np.int64)
    correct = np.bincount(lvl_m[ok], minlength=l).astype(np.int64)
    incorrect = total - correct

    # set x axis
    x = np.arange(l, dtype=np.int64)

    # set output path
    safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in gen_name)
    out = Path(
        f"results/results_single_experiments/{dirname}/"
        f"Landmark_efficiency_{safe}_{num_nodes}_{num_edges}.pdf"
    )
    out.parent.mkdir(parents=True, exist_ok=True)

    # plot figure
    plt.figure(figsize=(10, 6))
    ax = plt.gca()

    # Create bars
    ax.bar(x, correct, color="tab:red", zorder=2, label="Correct after landmark pre-labeling")
    ax.bar(x, incorrect, bottom=correct, color="tab:blue", zorder=1, label="Not correct yet")

    # Mark R (and R2) on the level axis
    ax.axvline(r, color="black", linestyle="--", linewidth=2.0, zorder=5, label=f"R, R2 value: {r}")

    # set axis and labels
    ax.set_xlabel("Shortest-path level (hop distance) $\\ell$")
    ax.set_ylabel("Vertices (count)")
    ax.set_title(
        f"Landmark accuracy pre Dijkstra warm-start by level: {gen_name} "
        f"(n={num_nodes:,}, m={num_edges:,})"
    )

    # Set style
    ax.set_axisbelow(True)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(True, axis="y", linestyle="--", linewidth=0.6, alpha=0.35)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, loc="upper right", bbox_to_anchor=(1.10, 1.0))

    # Save as pdf
    plt.tight_layout()
    plt.savefig(out, format="pdf")
    plt.close("all")
    return out


def plot_hybrid_schedule_sizes(heap_sizes: Sequence[int], frontier_sizes: Sequence[int], schedule: np.ndarray,
                              e_b: int, gen_name: str, num_nodes: int, num_edges: int, dirname: str) -> Path:
    """
    Plot the hybrid Multi-Switch schedule. Includes also curves for Dijkstra unique heap size after
    each pop-round (blue) and Bellman–Ford violation frontier size after each BF round (red).
    Phases are shaded using the schedule. The switching threshold e_b is shown as a horizontal line.
    Saves the figure as pdf.

    :param heap_sizes: Unique heap size logged after each Dijkstra pop-round.
    :param frontier_sizes: Violation frontier size logged after each BF round.
    :param schedule: Phase schedule as int32 array of shape (P,2) with rows [kind,length], kind 0=Dijkstra, 1=BF.
    :param e_b: Switching threshold (scan cost) used by the hybrid algorithm.
    :param gen_name: Generator name used in title/filename.
    :param num_nodes: Number of nodes for labeling.
    :param num_edges: Number of edges for labeling.
    :param dirname: Subdirectory inside "results_single_experiments/" to save to.
    :returns: Path to the created PDF.
    """

    # Ensure input got right dtype and is np.array
    hs = np.asarray(heap_sizes, dtype=np.int64)
    fs = np.asarray(frontier_sizes, dtype=np.int64)
    sched = np.asarray(schedule, dtype=np.int32)

    # Set out path
    safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in gen_name)
    out = Path(f"results/results_single_experiments/{dirname}/Multi_Switch_Schedule_{safe}_{num_nodes}_{num_edges}.pdf")
    out.parent.mkdir(parents=True, exist_ok=True)

    # Build step-wise time series (t = hybrid step index)
    x, y, k = [], [], []  # x: step index, y: normalized size, k: phase kind (0=Dijkstra, 1=BF)
    i_h, i_f, t = 0, 0, 0

    # Expand schedule phases into per-step points
    for i in range(sched.shape[0]):
        kind = int(sched[i, 0])   # 0 = Dijkstra phase, 1 = BF phase
        ln = int(sched[i, 1])
        if ln <= 0:
            continue

        if kind == 0:
            # Dijkstra: consume heap sizes
            take = ln
            if i_h + take > hs.size:
                take = hs.size - i_h
            for r in range(take):
                x.append(t)
                y.append(float(hs[i_h + r]) / float(e_b))  # normalize by e_b
                k.append(0)
                t += 1
            i_h += take
        else:
            # BF: consume frontier sizes
            take = ln
            if i_f + take > fs.size:
                take = fs.size - i_f
            for r in range(take):
                x.append(t)
                y.append(float(fs[i_f + r]) / float(e_b))  # normalize by e_b
                k.append(1)
                t += 1
            i_f += take

    # drop trailing BF point (keeps phase transitions consistent for plotting)
    if len(k) > 0 and k[-1] == 1:
        x.pop()
        y.pop()
        k.pop()

    # convert to numpy arrays for plotting
    x = np.asarray(x, dtype=np.int64)
    y = np.asarray(y, dtype=np.float64)
    k = np.asarray(k, dtype=np.int32)

    # avoid log(0): mask non-positive values
    y_plot = y.copy()
    y_plot[y_plot <= 0] = np.nan

    # set up figure/axes
    plt.figure(figsize=(10, 6))
    ax = plt.gca()

    # draw transition connectors between phase types
    trans = np.where(k[1:] != k[:-1])[0]
    for i in trans:
        ax.plot(x[i:i + 2], y_plot[i:i + 2], color="gray", linewidth=1.1, alpha=0.6, zorder=1)

    # draw continuous segments per phase type (blue=Dijkstra, red=BF)
    start = 0
    for i in range(1, x.size + 1):
        if i == x.size or k[i] != k[i - 1]:
            seg = slice(start, i)
            ax.plot(
                x[seg], y_plot[seg],
                color=("tab:blue" if k[start] == 0 else "tab:red"),
                linewidth=1.6,
                zorder=3,
            )
            start = i

    # overlay markers for each phase
    ax.plot(x[k == 0], y_plot[k == 0], linestyle="None", marker="o", markersize=2.2, color="tab:blue", zorder=4)
    ax.plot(x[k == 1], y_plot[k == 1], linestyle="None", marker="o", markersize=2.2, color="tab:red", zorder=4)

    # reference line: normalized switch threshold (1.0 corresponds to e_b)
    ax.axhline(1.0, color="black", linestyle="--", linewidth=1.6, zorder=2)

    # log-scale to show wide range of sizes
    ax.set_yscale("log")
    if np.any(np.isfinite(y_plot)):
        ymin = float(np.nanmin(y_plot[np.isfinite(y_plot)]))
        ax.set_ylim(bottom=max(1e-6, ymin * 0.8))

    # labels/title
    ax.set_xlabel(r"Hybrid step index $t$ (one Dijkstra pop or one BF round)")
    ax.set_ylabel(r"Normalized violation frontier/heap size")
    ax.set_title(f"Multi-switch hybrid schedule (normalized by $e_b$): {gen_name} (n={num_nodes:,}, m={num_edges:,})")

    # Set style and legend
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(True, axis="y", linestyle="--", linewidth=0.6, alpha=0.30)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.plot([], [], color="tab:blue", linewidth=1.6, marker="o", markersize=2.2,
            label=r"Dijkstra phase with Heap Size $|H_t|/e_b$ at step $t$")
    ax.plot([], [], color="tab:red", linewidth=1.6, marker="o", markersize=2.2,
            label=r"BF phase with Frontier Size $|F_t|/e_b$ at step $t$")
    ax.plot([], [], color="gray", linewidth=1.1, alpha=0.6,
            label="Phase transition")
    ax.plot([], [], color="black", linestyle="--", linewidth=1.6,
            label=r"Switch threshold: $e_b$ = "+ str(int(e_b)) +" (normalized: 1.0)")

    ax.legend(frameon=False)

    # export PDF
    plt.tight_layout()
    plt.savefig(out, format="pdf")
    plt.close("all")

    # delete when pdf is larger than 500 kb (alternative is downsampling but for this visualization unsuitable)
    if out.stat().st_size > 500*1024:
        out.unlink(missing_ok=True)

    return out

def plot_greedy_schedule_gains(schedule_kind: Sequence[int], greedy_dij_gain: Sequence[float],
                               greedy_bf_gain: Sequence[float], gen_name: str, num_nodes: int, num_edges: int,
                               dirname: str) -> Path:
    """
    Plot Greedy-Hybrid schedule in the same style as the Multi-Switch schedule plot, but with gains:
    DJ steps show best DJ gain, BF steps show BF gain. Values are normalized by e_b and the threshold is drawn at 1.0.
    Saves plot as pdf.

    :param schedule_kind: Action codes per step; 0 = BF, 1 = DJ.
    :param greedy_dij_gain: Best available DJ gain per step.
    :param greedy_bf_gain: BF gain per step (typically frontier_size / e_b).
    :param gen_name: Generator name used in title/filename.
    :param num_nodes: Number of nodes for labeling.
    :param num_edges: Number of edges for labeling.
    :param dirname: Subdirectory inside "results_single_experiments/" to save to.
    :returns: Path to the created PDF.
    """

    # Ensure input types
    k = np.asarray(schedule_kind, dtype=np.int32)
    dj = np.asarray(greedy_dij_gain, dtype=np.float64)
    bf = np.asarray(greedy_bf_gain, dtype=np.float64)

    # align to common length
    t = min(k.size, dj.size, bf.size)
    k = k[:t]
    dj = dj[:t]
    bf = bf[:t]

    # Set out path
    safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in gen_name)
    out = Path(f"results/results_single_experiments/{dirname}/Greedy_Hybrid_schedule_{safe}_{num_nodes}_{num_edges}.pdf")
    out.parent.mkdir(parents=True, exist_ok=True)

    # shift BF gains to express "before" frontier gain if needed for analysis
    bf_before = bf.copy()
    if bf_before.size > 1:
        bf_before[1:] = bf[:-1]  # before(t) = after(t-1)

    # build step-wise series: y is gain selected by schedule kind
    x = np.arange(t, dtype=np.int64)
    y = np.where(k == 1, dj, bf)

    # drop last BF point, same rule as multi-switch plot
    if k.size > 0 and k[-1] == 0:
        x = x[:-1]
        y = y[:-1]
        k = k[:-1]

    # avoid log(0): mask non-positive values
    y_plot = y.copy()
    y_plot[y_plot <= 0] = np.nan

    # set up figure/axes
    plt.figure(figsize=(10, 6))
    ax = plt.gca()

    # draw transition connectors between step types
    trans = np.where(k[1:] != k[:-1])[0]
    for i in trans:
        ax.plot(x[i:i + 2], y_plot[i:i + 2], color="gray", linewidth=1.1, alpha=0.6, zorder=1)

    # draw continuous segments per step type (blue=DJ, red=BF)
    start = 0
    for i in range(1, x.size + 1):
        if i == x.size or k[i] != k[i - 1]:
            seg = slice(start, i)
            ax.plot(
                x[seg], y_plot[seg],
                color=("tab:blue" if k[start] == 1 else "tab:red"),
                linewidth=1.6,
                zorder=3,
            )
            start = i

    # overlay markers for each step type
    ax.plot(x[k == 1], y_plot[k == 1], linestyle="None", marker="o", markersize=2.2, color="tab:blue", zorder=4)
    ax.plot(x[k == 0], y_plot[k == 0], linestyle="None", marker="o", markersize=2.2, color="tab:red", zorder=4)

    # log-scale to show wide range of gains
    ax.set_yscale("log")
    if np.any(np.isfinite(y_plot)):
        ymin = float(np.nanmin(y_plot[np.isfinite(y_plot)]))
        ax.set_ylim(bottom=max(1e-6, ymin * 0.8))

    # labels/title
    ax.set_xlabel(r"Hybrid step index $t$ (one Dijkstra pop or one BF round)")
    ax.set_ylabel(r"Gain $g_t$ (nodes covered on SPDAG)")
    ax.set_title(f"Greedy hybrid schedule with gains for each step: {gen_name} (n={num_nodes:,}, m={num_edges:,})")

    # set style
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(True, axis="y", linestyle="--", linewidth=0.6, alpha=0.30)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # set legend
    ax.plot([], [], color="tab:blue",
            linewidth=1.6, marker="o", markersize=2.2, label="DJ step at $t$ with best gain")
    ax.plot([], [], color="tab:red", linewidth=1.6, marker="o", markersize=2.2, label="BF step at $t$ with gain")
    ax.plot([], [], color="gray", linewidth=1.1, alpha=0.6, label="Phase transition")
    ax.legend(frameon=False)

    # export PDF
    plt.tight_layout()
    plt.savefig(out, format="pdf")
    plt.close("all")

    # delete when pdf is larger than 500 kb
    if out.stat().st_size > 500*1024:
        out.unlink(missing_ok=True)

    return out

def plot_ip_schedule_relaxation_gains(indptr: np.ndarray, indices: np.ndarray, weights: np.ndarray,
                                      src: int, sched_ip: Sequence[Tuple[Any, ...]], dist_ref: Sequence[float],
                                      gen_name: str, num_nodes: int, num_edges: int,
                                      dirname: str) -> Path:
    """
    Plot per-step relaxation gains of an exact IP schedule.

    Simulates the given schedule (BF rounds and chosen DJ pops) on the input CSR graph and measures the gain g_t
    as the number of newly correct nodes after each step compared to dist_ref. Exports a PDF plot in the same
    style as the greedy schedule gain plot.

    :param indptr: CSR row pointer array of shape.
    :param indices: CSR column index array of shape.
    :param weights: Edge weight array of shape.
    :param src: Source node id.
    :param sched_ip: IP schedule; ("BF") or ("DJ", u).
    :param dist_ref: Reference distance array used to define correctness.
    :param gen_name: Generator name used for title/filename.
    :param num_nodes: Number of nodes for labeling.
    :param num_edges: Number of edges for labeling.
    :param dirname: Results subdirectory name.
    :return: Path to the written PDF file.
    """

    # translate schedule to arrays (0 = BF, 1 = DJ)
    t = len(sched_ip)
    k = np.empty(t, dtype=np.int32)
    arg = np.zeros(t, dtype=np.int32)
    for i, act in enumerate(sched_ip):
        if act[0] == "BF":
            k[i] = 0
        else:
            k[i] = 1
            arg[i] = int(act[1])

    # simulate schedule + compute per-step gains
    n = indptr.shape[0] - 1
    dist = np.full(n, np.inf, dtype=np.float64)
    dist[int(src)] = 0.0
    popped = np.zeros(n, dtype=np.uint8)

    # reference distances define which nodes are reachable and what "correct" means
    dref = np.asarray(dist_ref, dtype=np.float64)
    reachable = np.isfinite(dref)
    eps = 1e-12

    # store gains per step separated by kind (for plotting convenience)
    dj_gain = np.zeros(t, dtype=np.float64)
    bf_gain = np.zeros(t, dtype=np.float64)

    correct_prev = int(np.sum(reachable & np.isfinite(dist) & (np.abs(dist - dref) <= eps)))

    for i in range(t):
        if k[i] == 0:
            # BF round: relax out of all non-popped, already reached nodes
            for u in range(n):
                if popped[u] == 1:
                    continue
                du = dist[u]
                if not np.isfinite(du):
                    continue
                a = indptr[u]
                b = indptr[u + 1]
                for e in range(a, b):
                    v = indices[e]
                    nd = du + weights[e]
                    if nd < dist[v]:
                        dist[v] = nd
        else:
            # DJ pop: relax from chosen node u, then mark u as popped
            u = int(arg[i])
            du = dist[u]
            a = indptr[u]
            b = indptr[u + 1]
            for e in range(a, b):
                v = indices[e]
                nd = du + weights[e]
                if nd < dist[v]:
                    dist[v] = nd
            popped[u] = 1

        # gain = newly correct nodes after this step
        correct_now = int(np.sum(reachable & np.isfinite(dist) & (np.abs(dist - dref) <= eps)))
        gain = float(correct_now - correct_prev)
        correct_prev = correct_now

        if k[i] == 0:
            bf_gain[i] = gain
        else:
            dj_gain[i] = gain

    # plotting: same style as greedy schedule plot (no threshold, no normalization)
    safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in gen_name)
    out = Path(f"results/results_single_experiments/{dirname}/IP_schedule_relax_gains_{safe}_{num_nodes}_{num_edges}.pdf")
    out.parent.mkdir(parents=True, exist_ok=True)

    # x-axis is step index, y-axis is gain for the chosen kind at that step
    x = np.arange(t, dtype=np.int64)
    y = np.where(k == 1, dj_gain, bf_gain)

    # drop last BF point (validation-only), same rule as your schedule plots
    if k.size > 0 and k[-1] == 0:
        x = x[:-1]
        y = y[:-1]
        k = k[:-1]

    # avoid log(0): mask non-positive gains
    y_plot = y.copy()
    y_plot[y_plot <= 0] = np.nan

    plt.figure(figsize=(10, 6))
    ax = plt.gca()

    # draw transition connectors between BF and DJ segments
    trans = np.where(k[1:] != k[:-1])[0]
    for i in trans:
        ax.plot(x[i:i + 2], y_plot[i:i + 2], color="gray", linewidth=1.1, alpha=0.6, zorder=1)

    # draw continuous segments per kind (blue=DJ, red=BF)
    start = 0
    for i in range(1, x.size + 1):
        if i == x.size or k[i] != k[i - 1]:
            seg = slice(start, i)
            ax.plot(
                x[seg], y_plot[seg],
                color=("tab:blue" if k[start] == 1 else "tab:red"),
                linewidth=1.6,
                zorder=3,
            )
            start = i

    # overlay markers for each kind
    ax.plot(x[k == 1], y_plot[k == 1], linestyle="None", marker="o", markersize=2.2, color="tab:blue", zorder=4)
    ax.plot(x[k == 0], y_plot[k == 0], linestyle="None", marker="o", markersize=2.2, color="tab:red", zorder=4)

    # log-scale to show wide range of gains
    ax.set_yscale("log")
    if np.any(np.isfinite(y_plot)):
        ymin = float(np.nanmin(y_plot[np.isfinite(y_plot)]))
        ax.set_ylim(bottom=max(1e-6, ymin * 0.8))

    # labels/title
    ax.set_xlabel(r"Hybrid step index $t$ (one Dijkstra pop or one BF round)")
    ax.set_ylabel(r"Gain $g_t$ (newly correct nodes after step $t$)")
    ax.set_title(f"Exact IP schedule with per-step relaxation gains: {gen_name} (n={num_nodes:,}, m={num_edges:,})")

    # set style
    ax.grid(True, axis="y", linestyle="--", linewidth=0.6, alpha=0.30)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.plot([], [], color="tab:blue", linewidth=1.6, marker="o", markersize=2.2, label="DJ step at $t$ with gain")
    ax.plot([], [], color="tab:red", linewidth=1.6, marker="o", markersize=2.2, label="BF step at $t$ with gain")
    ax.plot([], [], color="gray", linewidth=1.1, alpha=0.6, label="Phase transition")
    ax.legend(frameon=False)

    # export pdf
    plt.tight_layout()
    plt.savefig(out, format="pdf")
    plt.close("all")
    return out


def plot_asym_ios_single(rows: list[Any], dirname: str):
    """
    Plot I/Os vs graph size for a single graph_type contained in rows.
    Assumes rows is the list of dicts returned by asym_experiments() and
    that all entries share the same graph_type.

    - x-axis: graph size n (log scale)
    - y-axis: I/Os (log scale) for clear asymptotic behavior (power-laws appear linear)

    :param rows: list of dicts returned by asym_experiments() with I/O-Data for the approaches.
    :param dirname: Subdirectory inside "results_single_experiments/" to save to.

    :returns: path to created pdf
    """

    # Names of the approaches displayed in the legend
    _ASYM_ALGO_KEYS = [("io_dijkstra_simplified", "Dijkstra"), ("io_bellman_full", "Bellman-Ford"),
        ("io_dj_to_bf_best_switch", "Best-Single-Switch"),
                       ("io_hybrid_multi", "Multi-Switch"),
                       ("io_cost_landmark","Landmark"),
                       ("io_best_switches_overall","Greedy-Hybrid"),
                       ("io_Ip","Hybrid-Opt-Ip")]

    # Extract the (single) graph type for the title
    graph_type = rows[0].get("graph_type", "unknown")

    # Build series per algorithm: (n, io) pairs
    series_by_algo = {k: [] for k, _ in _ASYM_ALGO_KEYS}
    for r in rows:
        # skip rows that are not relevant
        if "n" not in r:
            continue
        n = int(r["n"])
        for key, _label in _ASYM_ALGO_KEYS:
            if key in r and r[key] is not None:
                series_by_algo[key].append((n, float(r[key])))

    # Sort each series by n
    for key in series_by_algo:
        series_by_algo[key].sort(key=lambda t: t[0])

    # Prepare figure
    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    fig.suptitle("I/Os vs #Nodes of Graph in log-log scale", fontsize=13, y=0.98)

    marker_cycle = ["o", "s", "^", "D", "v", "P"]

    # Plot each algorithm curve
    algo_idx = 0
    for key, label in _ASYM_ALGO_KEYS:
        pts = series_by_algo.get(key, [])
        if not pts:
            continue

        # Filter to positive values for log-scale
        xs = [n for (n, io) in pts if n > 0 and io > 0]
        ys = [io for (n, io) in pts if n > 0 and io > 0]

        if xs and ys:
            ax.plot(
                xs, ys,
                marker=marker_cycle[algo_idx % len(marker_cycle)],
                linestyle='-',
                linewidth=1.2,
                markersize=4.0,
                label=label
            )
            algo_idx += 1

    # Log–log scales
    ax.set_xscale("log", base=10)
    ax.set_yscale("log", base=10)

    # set labels and titles
    ax.set_xlabel("Graph size #nodes (log scale)")
    ax.set_ylabel("I/Os (log scale)")
    ax.set_title(f"Graph Type = {graph_type}")
    ax.grid(True, which="both", alpha=0.3, linestyle="--")
    ax.legend(fontsize=11, loc="best")

    fig.tight_layout(rect=[0, 0.02, 1, 0.95])

    # set output path and export fig as pdf
    output_path = Path(f"results/results_asym/{dirname}/{dirname}.pdf")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)

    return output_path

def compute_asym_slopes_single(rows: list[dict]) -> dict[str, tuple[float, float]]:
    """
    Computes the log–log slope α and R² for each algorithm.
    Assumes T(n) ≈ c·n^α and fits log10(T) = a + α·log10(n) to get α
    and R² for each algorithm.

    :param rows: takes the rows from asym_experiments() out of graph.io with all the experiment results
    :returns: stats: dictionary with a gradient and R² value for each approach (asym_stats.json)
    """

    # set keys of approaches that should be analyzed
    _ASYM_ALGO_KEYS = [
        "io_dijkstra_simplified",
        "io_bellman_full",
        "io_dj_to_bf_best_switch",
        "io_hybrid_multi",
        "io_cost_landmark",
        "io_best_switches_overall",
        "io_Ip",
    ]

    # iterate of the rows and get the corresponding io values for each approach
    series_by_algo: dict[str, list[tuple[int, float]]] = {k: [] for k in _ASYM_ALGO_KEYS}
    for r in rows:
        n = int(r["reachable_n"])
        for key in _ASYM_ALGO_KEYS:
            if key in r and r[key] is not None:
                series_by_algo[key].append((n, float(r[key])))

    # create empty return dict
    stats: dict[str, tuple[float, float]] = {}

    # now for each approach
    for key in _ASYM_ALGO_KEYS:
        # get all the ios and sort them by n
        pts = sorted(series_by_algo.get(key, []), key=lambda t: t[0])
        # if we have less than 2 measure points skip
        if len(pts) < 2:
            continue

        # create #nodes array and ios array; delete all negative values and again skip if < 2 measure points
        ns = np.array([n for (n, io) in pts], dtype=float)
        ios = np.array([io for (n, io) in pts], dtype=float)

        mask = (ns > 0) & (ios > 0)
        ns = ns[mask]
        ios = ios[mask]
        if len(ns) < 2:
            continue

        # convert the values to log-log area
        x = np.log10(ns)
        y = np.log10(ios)

        # get the mean values for ios and number of nodes
        x_mean = x.mean()
        y_mean = y.mean()

        # gradient calculation in linear regression
        sxx = ((x - x_mean) ** 2).sum()
        sxy = ((x - x_mean) * (y - y_mean)).sum()

        # get grow exponent alpha
        alpha = sxy / sxx

        # now calculate R² value: how far away are the points from the functional line
        y_pred = y_mean + alpha * (x - x_mean)
        ss_res = ((y - y_pred) ** 2).sum()
        ss_tot = ((y - y_mean) ** 2).sum()
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0

        # add the results to the stats dictionary and go on with the next approach
        stats[key] = (float(alpha), float(r2))

    # return the results
    return stats

def plot_asym_slope_matrices(json_path: Union[str, Path], dirname: str) -> int:
    """
    Visualizes the gradient stats for all approaches and graph types as matrix.
    Creates 1 file for each graph type with blocksize as approach as column and blocksize as
    row. Saves matrices as pdfs.

    color codes:
        blue for values < 1
        red for values > 1
        white for values around 1

    :param json_path: path to .json with gradient stats (asym_stats.json)
    :param dirname: folder where the results should be saved
    :returns: 0 on success
    """

    # Legend names for approaches
    names = {
        "io_dijkstra_simplified": "Dijkstra",
        "io_bellman_full": "Bellman–Ford",
        "io_dj_to_bf_best_switch": "Best Single Switch",
        "io_best_switches_overall": "Greedy Hybrid",
        "io_hybrid_multi": "Multi Switch",
        "io_cost_landmark": "Landmark",
        "io_Ip": "Hybrid-Opt IP",
    }

    # open json and load stats
    stats: Dict[str, Any] = json.loads(Path(json_path).read_text(encoding="utf-8"))

    # regex to get the keys out of the json file
    rx = re.compile(r"^(?P<gt>.+?)_asymptotic_experiments_(?P<bs>\d+)(?:_(?P<dist>.+))?$")

    # collect metadata + all slope values (for global normalization)
    keys, algos, vals = [], set(), []
    for k, e in stats.items():
        m = rx.match(k)
        if not m or not isinstance(e, dict):
            continue

        gt, bs, dist = m["gt"], int(m["bs"]), (m["dist"] or "NA")
        keys.append((gt, bs, dist))
        algos |= set(e)

        for v in e.values():
            if isinstance(v, (list, tuple)) and v and v[0] is not None:
                vals.append(float(v[0]))


    # nothing to plot -> hard fail (can happen if json is empty)
    if not vals:
        raise ValueError("No slope values in JSON (expected lists/tuples with first element = slope).")

    # keep only finite slope values
    vals = [v for v in vals if np.isfinite(v)]
    if not vals:
        raise ValueError("No finite slope values in JSON.")

    dev = max(abs(min(vals) - 1.0), abs(max(vals) - 1.0))

    # TwoSlopeNorm needs vmin < vcenter < vmax
    if not np.isfinite(dev) or dev <= 0.0:
        dev = 1e-3
    else:
        dev = max(dev, 1e-3)
    vmin, vcenter, vmax = 1.0 - dev, 1.0, 1.0 + dev
    norm, cmap = TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax), plt.get_cmap("bwr")

    # sort axes order
    gts  = sorted({gt for gt, _, _ in keys})
    bss  = sorted({bs for _, bs, _ in keys})
    dss  = sorted({d  for _, _, d  in keys})
    algs = sorted(algos)

    # build a direct (gt, bs, dist) -> entry mapping
    data = {
        (gt, bs, d): stats[f"{gt}_asymptotic_experiments_{bs}" + ("" if d == "NA" else f"_{d}")]
        for gt, bs, d in keys
    }

    # create output dir
    out = Path("results") / "results_asym" / dirname
    out.mkdir(parents=True, exist_ok=True)

    # now for each graph type and distribution
    for gt in gts:
        for d in dss:
            # create matrix filled with NaN (missing entries stay empty)
            m = np.full((len(bss), len(algs)), np.nan)

            # fills the matrix with rows=blocksize, cols=approach
            for i, bs in enumerate(bss):
                e = data.get((gt, bs, d), {})
                for j, a in enumerate(algs):
                    v = e.get(a)
                    if isinstance(v, (list, tuple)) and v and v[0] is not None:
                        m[i, j] = float(v[0])

            # skip empty matrices (no finite values)
            if not np.isfinite(m).any():
                continue

            # figure size depends on matrix size
            fig, ax = plt.subplots(figsize=(0.55 * len(algs) + 3.2, 0.42 * len(bss) + 2.2))
            im = ax.imshow(m, cmap=cmap, norm=norm, aspect="auto")

            # set axis ticks/labels
            ax.set_xticks(range(len(algs)))
            labels = [names.get(a, a.replace("io_", "").replace("_", " ").title()) for a in algs]

            ax.set_xticklabels(
                labels,
                rotation=35,
                ha="right",
                rotation_mode="anchor",
                fontsize=11
            )
            ax.set_yticks(range(len(bss)))
            ax.set_yticklabels(list(map(str, bss)))

            ax.set_xlabel("Approach")
            ax.set_ylabel("Blocksize")
            ax.set_title(f"Asymptotic gradient\n{gt} | {d}")

            # write slope values into each cell
            for i in range(len(bss)):
                for j in range(len(algs)):
                    if np.isfinite(m[i, j]):
                        ax.text(j, i, f"{m[i, j]:.3f}", ha="center", va="center", fontsize=7)

            # add colorbar legend
            fig.colorbar(im, ax=ax).set_label("Gradient ≈ exponent")
            fig.tight_layout()

            # safe file names
            safe = lambda s: "".join(c if c.isalnum() or c in "-._" else "_" for c in s)
            fig.savefig(out / f"asym_slopes_{safe(gt)}_{safe(d)}.pdf", format="pdf")
            plt.close(fig)

    return 0

def visualize_csr(indptr: np.ndarray, indices: np.ndarray, *,
    sample_edges: Optional[int] = 300_000, sample_nodes: Optional[int] = 50_000,
    node_size: float = 2, edge_lw: float = 0.5, edge_alpha: float = 0.1, figsize: Tuple[int, int] = (8, 8),
    seed: int = 42) -> Figure:
    """
    Fast CSR visualization with optional downsampling. Not relevant for experiment pipeline. Can be used to check
    how a generated graph or loaded graph looks like. If you want to use, deactivate matplotlib.use("Agg") at the top.

    :param indptr: CSR row pointer array
    :param indices: CSR column indices
    :param sample_edges: Max edges to draw; if None, draw all.
    :param sample_nodes: Max nodes to scatter; if None, draw all.
    :param node_size: Scatter marker size.
    :param edge_lw: Edge line width.
    :param edge_alpha: Edge transparency (alpha).
    :param figsize: Figure size in inches.
    :param seed: RNG seed for sampling/positions.
    :returns: The created matplotlib Figure.
    """

    # Get row pointers and column indices as np array and get number nodes and edges
    indptr  = np.asarray(indptr)
    indices = np.asarray(indices)
    n = indptr.size - 1
    m = indices.size

    # Sets rng seed for visualisation and generates positions for each node
    rng = np.random.default_rng(seed)
    pos = rng.random((n, 2), dtype=np.float64)

    # Get source node for each edge
    deg = indptr[1:] - indptr[:-1]
    sources = np.repeat(np.arange(n, dtype=np.int64), deg)
    targets = indices.astype(np.int64, copy=False)

    # Optional edge-sampling for big graphs
    if sample_edges is not None and m > sample_edges:
        take = rng.choice(m, size=sample_edges, replace=False)
        sources = sources[take]
        targets = targets[take]

    # Line segments (m_samp, 2, 2)
    segs = np.stack((pos[sources], pos[targets]), axis=1)

    # Create fig
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_aspect('equal', adjustable='box')
    ax.axis('off')

    # Create edges with line collection
    lc = LineCollection(segs, linewidths=edge_lw, alpha=edge_alpha)
    ax.add_collection(lc)

    # Optional node-sampling for big graphs
    if sample_nodes is not None and n > sample_nodes:
        nodes_draw = rng.choice(n, size=sample_nodes, replace=False)
    else:
        nodes_draw = np.arange(n)

    # Draw nodes and add edges
    ax.scatter(pos[nodes_draw, 0], pos[nodes_draw, 1],
               s=node_size, alpha=0.9)

    # set boundaries so everything is visible and not cutted
    ax.set_xlim(pos[:,0].min()-0.01, pos[:,0].max()+0.01)
    ax.set_ylim(pos[:,1].min()-0.01, pos[:,1].max()+0.01)
    plt.tight_layout()
    plt.show()

    # returns the fig
    return fig