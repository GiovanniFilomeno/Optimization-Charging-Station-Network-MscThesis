"""
Utility functions for EV Charging Station Network Optimization.

This module provides core helper functions used across the project:
- Geographic distance calculations (Haversine formula and OSRM road distances)
- Network graph visualization overlaid on Germany's boundaries
- Network topology metrics computation (density, diameter, clustering, etc.)
- Weighted metrics aggregation across connected components
"""

import logging
import math
import sqlite3
from pathlib import Path

import networkx as nx
import numpy as np
import requests

logger = logging.getLogger(__name__)


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate the great-circle distance between two points on Earth
    using the Haversine formula.

    This provides an approximation of the shortest distance over the Earth's
    surface, ignoring elevation changes and road networks.

    Args:
        lat1: Latitude of the first point in decimal degrees.
        lon1: Longitude of the first point in decimal degrees.
        lat2: Latitude of the second point in decimal degrees.
        lon2: Longitude of the second point in decimal degrees.

    Returns:
        Distance in kilometers between the two geographic points.
    """
    R = 6371  # Earth's radius in kilometers

    dLat = math.radians(lat2 - lat1)
    dLon = math.radians(lon2 - lon1)
    lat1 = math.radians(lat1)
    lat2 = math.radians(lat2)

    a = math.sin(dLat / 2) * math.sin(dLat / 2) + math.sin(dLon / 2) * math.sin(
        dLon / 2
    ) * math.cos(lat1) * math.cos(lat2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c


def get_osrm_distance(
    lat1: float,
    lon1: float,
    lat2: float,
    lon2: float,
    *,
    max_osrm_distance_km: float = 100.0,
    cache_path: str | Path = "distances.db",
    request_timeout_s: float = 10.0,
    session=None,
) -> float | None:
    """
    Return a routed distance for nearby points and a prefilter distance otherwise.

    Uses the OSRM (Open Source Routing Machine) API for points whose Haversine
    separation is within ``max_osrm_distance_km``. For points farther apart,
    returns the Haversine distance in meters as a fast pre-filter result.

    Results are cached in a local SQLite database ('distances.db') to avoid
    redundant API calls and improve performance on repeated computations.

    Args:
        lat1: Latitude of the first point in decimal degrees.
        lon1: Longitude of the first point in decimal degrees.
        lat2: Latitude of the second point in decimal degrees.
        lon2: Longitude of the second point in decimal degrees.
        max_osrm_distance_km: Maximum straight-line separation for an OSRM request.
        cache_path: SQLite file used to cache successful routed distances.
        request_timeout_s: Per-request timeout in seconds.
        session: Optional requests-compatible client, primarily for testing.

    Returns:
        Routed distance in meters for nearby points, or approximate Haversine
        distance in meters when the prefilter threshold is exceeded.
        Returns None if the OSRM API call fails.
    """
    approx_distance_km = haversine_distance(lat1, lon1, lat2, lon2)
    approx_distance_m = 1000 * approx_distance_km

    if approx_distance_km > max_osrm_distance_km:
        return approx_distance_m

    cache_path = Path(cache_path)
    with sqlite3.connect(cache_path) as conn:
        conn.execute(
            """CREATE TABLE IF NOT EXISTS distances (
                lat1 REAL,
                lon1 REAL,
                lat2 REAL,
                lon2 REAL,
                distance REAL,
                UNIQUE(lat1, lon1, lat2, lon2)
            )"""
        )
        result = conn.execute(
            "SELECT distance FROM distances WHERE lat1=? AND lon1=? AND lat2=? AND lon2=?",
            (lat1, lon1, lat2, lon2),
        ).fetchone()
        if result is not None:
            return float(result[0])

        client = session or requests
        url = (
            "https://router.project-osrm.org/route/v1/driving/"
            f"{lon1},{lat1};{lon2},{lat2}?overview=false"
        )
        try:
            response = client.get(url, timeout=request_timeout_s)
            response.raise_for_status()
            routes = response.json().get("routes", [])
            if not routes:
                return None
            distance = float(routes[0]["distance"])
        except (requests.RequestException, ValueError, KeyError, TypeError) as exc:
            logger.warning("OSRM distance lookup failed: %s", exc)
            return None

        conn.execute(
            """INSERT OR REPLACE INTO distances
               (lat1, lon1, lat2, lon2, distance)
               VALUES (?, ?, ?, ?, ?)""",
            (lat1, lon1, lat2, lon2, distance),
        )
        return distance


def draw_graph(graph_to_draw: nx.Graph, title: str = "") -> None:
    """
    Visualize a network graph overlaid on a map of Germany.

    Converts node coordinates from WGS84 (lat/lon) to UTM Zone 32N for
    proper 2D visualization, and plots Germany's federal state boundaries
    as a geographic reference.

    Args:
        graph_to_draw: A NetworkX graph where each node has 'latitude' and
                       'longitude' attributes in decimal degrees.
        title: Optional title string displayed above the plot.
    """
    import geopandas as gpd
    import matplotlib.pyplot as plt
    from pyproj import Transformer

    options = {
        "node_color": "lavender",
        "node_size": 10,
        "width": 1,
        "arrowsize": 1,
    }

    # Load Germany's federal state boundaries from GeoJSON
    germany_boundary = gpd.read_file("data/boundaries/4_niedrig.geo.json")

    # Convert geographic coordinates (WGS84) to UTM Zone 32N for 2D plotting
    germany_boundary = germany_boundary.to_crs("epsg:32632")

    fig, ax = plt.subplots(figsize=(12, 8))

    # Build position dictionary for nodes by transforming lat/lon -> UTM
    nodes_id = list(graph_to_draw.nodes)
    transformer = Transformer.from_crs("epsg:4326", "epsg:32632", always_xy=True)
    positions = [
        (graph_to_draw.nodes[nid]["latitude"], graph_to_draw.nodes[nid]["longitude"])
        for nid in nodes_id
    ]
    positions_utm = np.array([transformer.transform(x[1], x[0]) for x in positions])
    pos = {nodes_id[i]: positions_utm[i] for i in range(len(nodes_id))}

    # Draw the network graph
    nx.draw_networkx(graph_to_draw, pos=pos, ax=ax, **options)

    # Overlay Germany's boundaries in red
    germany_boundary.boundary.plot(ax=ax, linewidth=2, color="red", zorder=3)

    plt.title(title, fontsize=15)
    plt.show()


def calculate_metrics(graph: nx.Graph, seed: int = 42) -> dict:
    """
    Compute key network topology metrics for a connected graph component.

    Metrics calculated:
    - Density: Ratio of actual edges to possible edges (0 to 1).
    - Average Distance: Mean edge weight across all edges (converted to km).
    - Diameter: Maximum eccentricity (longest shortest path) in the graph.
    - Average Clustering: Mean local clustering coefficient across nodes.

    Uses approximation algorithms for diameter and clustering to handle
    large graphs efficiently.

    Args:
        graph: A connected NetworkX graph with edge 'weight' attributes
               representing distances in meters.
        seed: Random seed supplied to NetworkX approximation routines.

    Returns:
        Dictionary with keys: 'density', 'average_distance', 'diameter',
        'average_clustering'.
    """
    metrics = {}

    metrics["density"] = nx.density(graph)
    edge_weights = [edge[2]["weight"] for edge in graph.edges(data=True)]
    metrics["average_distance"] = float(np.mean(edge_weights)) / 1000 if edge_weights else 0.0
    metrics["diameter"] = (
        nx.approximation.diameter(graph, seed=seed) if graph.number_of_nodes() > 1 else 0
    )
    metrics["average_clustering"] = (
        nx.approximation.average_clustering(graph, seed=seed)
        if graph.number_of_nodes() > 1
        else 0.0
    )

    return metrics


def weighted_mean(values: list, weights: list) -> float:
    """
    Compute the weighted arithmetic mean.

    Args:
        values: List of numeric values.
        weights: List of corresponding weights (must be same length as values).

    Returns:
        Weighted mean as a float.
    """
    if len(values) != len(weights):
        raise ValueError("values and weights must have the same length")
    total_weight = sum(weights)
    if total_weight == 0:
        raise ValueError("weights must not sum to zero")
    return sum(value * weight for value, weight in zip(values, weights, strict=True)) / total_weight


def calculate_weighted_metrics(graph: nx.Graph, year: int, seed: int = 42) -> dict:
    """
    Calculate network metrics weighted by connected component size.

    For disconnected graphs, this function:
    1. Identifies all connected components with >= 2 nodes.
    2. Computes topology metrics for each component.
    3. Aggregates metrics using node-count-weighted averages.

    This approach ensures that larger subnetworks contribute more to the
    overall metric values, providing a fair representation of the network's
    characteristics.

    Args:
        graph: A NetworkX graph (may be disconnected).
        year: The year associated with this network snapshot.
        seed: Random seed supplied to NetworkX approximation routines.

    Returns:
        Dictionary containing:
        - Weighted averages of density, average_distance, diameter, clustering
        - 'year': The input year
        - 'total_nodes': Total number of nodes in the graph
        - 'isolated_nodes': Number of nodes excluded as isolates
        - 'subnetwork_sizes': List of node counts per connected component
    """
    # Extract connected components with at least 2 nodes (isolated nodes excluded)
    components = [graph.subgraph(cc) for cc in nx.connected_components(graph) if len(cc) >= 2]

    metric_sums = {
        "density": 0.0,
        "average_distance": 0.0,
        "diameter": 0,
        "average_clustering": 0.0,
    }

    total_weight = 0
    subnetwork_sizes = []

    for component in components:
        metrics = calculate_metrics(component, seed=seed)
        weight = component.number_of_nodes()
        total_weight += weight
        for metric, value in metrics.items():
            metric_sums[metric] += value * weight
        subnetwork_sizes.append(weight)

    # Compute weighted averages
    weighted_metrics = {
        metric: (value / total_weight if total_weight else 0.0)
        for metric, value in metric_sums.items()
    }
    weighted_metrics["year"] = year
    weighted_metrics["total_nodes"] = graph.number_of_nodes()
    weighted_metrics["subnetwork_sizes"] = subnetwork_sizes
    weighted_metrics["isolated_nodes"] = nx.number_of_isolates(graph)

    return weighted_metrics
