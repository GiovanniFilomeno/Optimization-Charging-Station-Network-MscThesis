"""
Visualization & Evaluation Module.

Compares baseline (real-world) vs. GA-optimized networks using the
multi-objective fitness function, and generates overlay plots on Germany's map.

Usage:
    python -m src.visualization
    python -m src.visualization --years 2011 2012 2013
"""

import argparse
import logging
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
from shapely.geometry import MultiPoint

from .config import (
    GERMANY_GEOJSON,
    NETWORKS_BASELINE,
    NETWORKS_OPTIMIZED,
    FIGURES_DIR,
    NORM_BOUNDS,
    FITNESS_WEIGHTS,
)
from .utils import draw_graph, calculate_weighted_metrics, weighted_mean

logger = logging.getLogger(__name__)


def _load_polygon():
    boundary = gpd.read_file(GERMANY_GEOJSON)
    return boundary.geometry.unary_union


def coverage_ratio(graph: nx.Graph, country_polygon) -> float:
    """Compute network convex hull area / country area."""
    coords = [
        (graph.nodes[n]["longitude"], graph.nodes[n]["latitude"])
        for n in graph.nodes()
    ]
    hull = MultiPoint(coords).convex_hull
    return hull.area / country_polygon.area


def fitness_function(graph: nx.Graph, year: int, country_polygon) -> float:
    """Evaluate a network using the multi-objective fitness (lower = better)."""
    m = calculate_weighted_metrics(graph, year)
    cov = coverage_ratio(graph, country_polygon)

    b = NORM_BOUNDS
    vals = [
        (m["average_distance"] - b["average_distance"][0])
        / (b["average_distance"][1] - b["average_distance"][0]),
        (m["diameter"] - b["diameter"][0])
        / (b["diameter"][1] - b["diameter"][0]),
        (m["average_clustering"] - b["clustering"][0])
        / (b["clustering"][1] - b["clustering"][0]),
        1 - (m["density"] - b["density"][0])
        / (b["density"][1] - b["density"][0]),
        1 - cov,
    ]
    w = list(FITNESS_WEIGHTS.values())
    return weighted_mean(vals, w)


def compare_networks(years: list[int]) -> pd.DataFrame:
    """
    Compare baseline and optimized networks for the given years.

    Returns:
        DataFrame with fitness scores and key metrics for both versions.
    """
    polygon = _load_polygon()
    rows = []

    for yr in years:
        # Baseline
        base_path = NETWORKS_BASELINE / f"network_{yr}_100.graphml"
        if base_path.exists():
            g_base = nx.read_graphml(str(base_path))
            m_base = calculate_weighted_metrics(g_base, yr)
            f_base = fitness_function(g_base, yr, polygon)
            rows.append({
                "Year": yr,
                "Type": "Baseline",
                "Fitness": f_base,
                "Nodes": m_base["total_nodes"],
                "Density": m_base["density"],
                "Avg. Dist (km)": m_base["average_distance"],
                "Diameter": m_base["diameter"],
                "Clustering": m_base["average_clustering"],
            })

        # Optimized
        opt_path = NETWORKS_OPTIMIZED / f"network_{yr}_optimized.graphml"
        if opt_path.exists():
            g_opt = nx.read_graphml(str(opt_path))
            m_opt = calculate_weighted_metrics(g_opt, yr)
            f_opt = fitness_function(g_opt, yr, polygon)
            rows.append({
                "Year": yr,
                "Type": "Optimized",
                "Fitness": f_opt,
                "Nodes": m_opt["total_nodes"],
                "Density": m_opt["density"],
                "Avg. Dist (km)": m_opt["average_distance"],
                "Diameter": m_opt["diameter"],
                "Clustering": m_opt["average_clustering"],
            })

    return pd.DataFrame(rows)


def plot_comparison(years: list[int]) -> None:
    """Generate side-by-side network visualizations for baseline vs. optimized."""
    for yr in years:
        base_path = NETWORKS_BASELINE / f"network_{yr}_100.graphml"
        opt_path = NETWORKS_OPTIMIZED / f"network_{yr}_optimized.graphml"

        if base_path.exists():
            g = nx.read_graphml(str(base_path))
            draw_graph(g, title=f"Baseline Network ({yr})")

        if opt_path.exists():
            g = nx.read_graphml(str(opt_path))
            draw_graph(g, title=f"Optimized Network ({yr})")


def run(years: list[int] | None = None) -> None:
    """Run the full evaluation pipeline."""
    if years is None:
        years = list(range(2011, 2018))

    df = compare_networks(years)
    print(df.to_string(index=False))
    plot_comparison(years)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(description="Evaluate network optimization")
    parser.add_argument("--years", type=int, nargs="+", default=list(range(2011, 2018)))
    args = parser.parse_args()
    run(args.years)
