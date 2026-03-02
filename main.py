# main.py
"""
main project file executes complete experiment pipeline
"""

from src import graph_io
from src import graph_plots


def main() -> int:
    """
    Main function that is executed when this file is started.
    Here, it is possible to call up various given experiment pipelines or write your own experiments or functions.
    """
    # executes the experiments for the single instances; "Einzelbetrachtung"
    graph_io.single_experiments()

    # executes the experiments for the asymptotical evaluation, saves them as pdf and json; "Reihenbetrachtung"
    #graph_io.asym_experiments()

    # build matrix graphic from asym_stats json
    #graph_plots.plot_asym_slope_matrices("results/asym_stats.json", dirname="gradient_matrices")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
