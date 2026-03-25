"""
Network Topology Analysis Module.

Computes and tracks key graph metrics for the EV charging station network
across yearly snapshots:
    - Density (edge saturation)
    - Average edge distance (km)
    - Diameter (worst-case shortest path)
    - Clustering coefficient (local triangle density)

Metrics are weighted by connected component size to fairly represent
disconnected graphs.

Usage:
    python -m src.analysis
"""

import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd

from .config import NETWORKS_BASELINE, FIGURES_DIR
from .utils import calculate_weighted_metrics

logger = logging.getLogger(__name__)


def load_graphs(
    years: list[int],
    basedir: Path = NETWORKS_BASELINE,
    max_dist: int = 100,
) -> dict[int, nx.Graph]:
    """Load pre-computed GraphML network files for the given years."""
    graphs = {}
    for year in years:
        path = basedir / f"network_{year}_{max_dist}.graphml"
        logger.info("Loading %s", path.name)
        graphs[year] = nx.read_graphml(str(path))
    return graphs


def compute_metrics(graphs: dict[int, nx.Graph]) -> pd.DataFrame:
    """
    Calculate weighted topology metrics for each yearly network.

    Returns:
        DataFrame with columns: Year, Total Nodes, Density,
        Avg. Distance (km), Diameter, Avg. Clustering.
    """
    rows = []
    for year, graph in sorted(graphs.items()):
        logger.info("Computing metrics for %d…", year)
        m = calculate_weighted_metrics(graph, year)
        rows.append(m)

    df = pd.DataFrame(rows)
    df = df[["year", "total_nodes", "density", "average_distance",
             "diameter", "average_clustering"]]
    df.columns = [
        "Year", "Total Nodes", "Density",
        "Avg. Distance (km)", "Diameter", "Avg. Clustering",
    ]
    return df


def plot_metric_trends(df: pd.DataFrame, save: bool = True) -> None:
    """Plot the evolution of each network metric over time."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    metrics = [
        ("Density", "Network Density"),
        ("Avg. Distance (km)", "Average Edge Distance (km)"),
        ("Diameter", "Graph Diameter"),
        ("Avg. Clustering", "Clustering Coefficient"),
    ]
    for ax, (col, title) in zip(axes.flat, metrics):
        ax.plot(df["Year"], df[col], "o-", linewidth=2, markersize=5)
        ax.set_title(title, fontsize=14)
        ax.set_xlabel("Year")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save:
        out = FIGURES_DIR / "network_metrics_trends.png"
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(out), dpi=150, bbox_inches="tight")
        logger.info("Saved → %s", out)
    plt.show()


def run(years: list[int] | None = None) -> pd.DataFrame:
    """Load graphs, compute metrics, plot trends, and return results."""
    if years is None:
        years = list(range(2009, 2023))
    graphs = load_graphs(years)
    df = compute_metrics(graphs)
    print(df.to_string(index=False))
    plot_metric_trends(df)
    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(description="Analyze EV network topology")
    parser.add_argument("--years", type=int, nargs="+", default=list(range(2009, 2023)))
    args = parser.parse_args()
    run(args.years)
