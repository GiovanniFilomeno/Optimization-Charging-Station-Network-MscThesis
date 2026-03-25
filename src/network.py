"""
Spatial Network Construction Module.

Builds proximity-based graphs where:
    - Nodes = EV charging stations (with lat/lon attributes)
    - Edges = pairs of stations within MAX_EDGE_DISTANCE_KM (road distance)

Edge weights represent actual road distances from the OSRM routing API,
with a local SQLite cache to avoid redundant API calls.

Usage:
    python -m src.network --year 2015
    python -m src.network --year-range 2009 2022
"""

import argparse
import logging
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from pyproj import Transformer
from shapely.geometry import Point
from tqdm import tqdm

from .config import (
    CLEANED_CSV,
    GERMANY_GEOJSON,
    MAX_EDGE_DISTANCE_KM,
    NETWORKS_BASELINE,
    FIGURES_DIR,
    STATE_NAME_CORRECTIONS,
)
from .utils import get_osrm_distance

logger = logging.getLogger(__name__)


def load_stations(csv_path: Path = CLEANED_CSV) -> pd.DataFrame:
    """Load cleaned station data, fix state encodings, and filter to Germany."""
    data = pd.read_csv(csv_path, encoding="utf-8")
    data["year"] = pd.to_datetime(data["commissioning_date"]).dt.year
    data["federal_state"] = data["federal_state"].replace(STATE_NAME_CORRECTIONS)

    # Create geometry and filter to German boundaries
    data["geometry"] = data.apply(
        lambda r: Point(r["longitude_[dg]"], r["latitude_[dg]"]), axis=1
    )
    gdf = gpd.GeoDataFrame(data, geometry="geometry")
    germany = gpd.read_file(GERMANY_GEOJSON)
    polygon = germany.geometry.unary_union
    gdf = gdf[gdf.geometry.within(polygon)]

    logger.info("Loaded %d stations within Germany", len(gdf))
    return pd.DataFrame(gdf)


def create_network(
    data: pd.DataFrame,
    year: int,
    max_distance_km: int = MAX_EDGE_DISTANCE_KM,
) -> nx.Graph:
    """
    Build a spatial network graph for all stations up to a given year.

    Args:
        data: DataFrame with columns latitude_[dg], longitude_[dg], year, federal_state.
        year: Include all stations commissioned on or before this year (cumulative).
        max_distance_km: Maximum road distance (km) to create an edge.

    Returns:
        NetworkX Graph with node attributes (latitude, longitude, federal_state)
        and edge weights (distance in meters).
    """
    data_year = data[data["year"] <= year]
    n = len(data_year)
    positions = data_year[["latitude_[dg]", "longitude_[dg]"]].to_numpy()
    states = data_year["federal_state"].values

    network = nx.Graph()

    # Add nodes with geographic attributes
    for i in tqdm(range(n), desc=f"Nodes ({year})"):
        network.add_node(
            i + 1,
            latitude=positions[i, 0],
            longitude=positions[i, 1],
            federal_state=states[i],
        )

    # Add edges: O(n²) pairwise distance computation
    max_dist_m = max_distance_km * 1000
    for i in tqdm(range(n), desc=f"Edges ({year})"):
        for j in range(i + 1, n):
            distance = get_osrm_distance(
                positions[i, 0], positions[i, 1],
                positions[j, 0], positions[j, 1],
            )
            if distance is not None and distance < max_dist_m:
                network.add_edge(i + 1, j + 1, weight=distance)

    logger.info(
        "Year %d: %d nodes, %d edges",
        year, network.number_of_nodes(), network.number_of_edges(),
    )
    return network


def save_network(network: nx.Graph, year: int, max_distance_km: int) -> None:
    """Save a network graph to GraphML format."""
    NETWORKS_BASELINE.mkdir(parents=True, exist_ok=True)
    path = NETWORKS_BASELINE / f"network_{year}_{max_distance_km}.graphml"
    nx.write_graphml(network, str(path))
    logger.info("Saved → %s", path)


def plot_network(
    network: nx.Graph,
    year: int,
    max_distance_km: int,
    save: bool = True,
) -> None:
    """Visualize a network overlaid on Germany's boundaries."""
    transformer = Transformer.from_crs("epsg:4326", "epsg:32632", always_xy=True)
    germany_utm = gpd.read_file(GERMANY_GEOJSON).to_crs("epsg:32632")

    nodes = list(network.nodes)
    positions = [
        (network.nodes[n]["latitude"], network.nodes[n]["longitude"]) for n in nodes
    ]
    utm = np.array([transformer.transform(p[1], p[0]) for p in positions])
    pos = {nodes[i]: utm[i] for i in range(len(nodes))}

    fig, ax = plt.subplots(figsize=(12, 8))
    nx.draw_networkx(
        network, pos=pos, ax=ax,
        node_color="lavender", node_size=10, width=1, arrowsize=1,
    )
    germany_utm.boundary.plot(ax=ax, linewidth=2, color="red", zorder=3)
    plt.title(
        f"EV Charging Station Network ({year}) — {max_distance_km} km Range",
        fontsize=15,
    )

    if save:
        out = FIGURES_DIR / "network_graphs"
        out.mkdir(parents=True, exist_ok=True)
        fig.savefig(
            out / f"network_{year}_{max_distance_km}.png",
            dpi=300, bbox_inches="tight",
        )
    plt.show()


def run(years: list[int], max_distance_km: int = MAX_EDGE_DISTANCE_KM) -> None:
    """Build, save, and plot networks for the given years."""
    data = load_stations()
    for year in years:
        network = create_network(data, year, max_distance_km)
        save_network(network, year, max_distance_km)
        plot_network(network, year, max_distance_km)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(description="Build EV charging station networks")
    parser.add_argument("--year", type=int, help="Single year to build")
    parser.add_argument("--year-range", type=int, nargs=2, metavar=("START", "END"),
                        help="Range of years (inclusive)")
    args = parser.parse_args()

    if args.year:
        run([args.year])
    elif args.year_range:
        run(list(range(args.year_range[0], args.year_range[1] + 1)))
    else:
        run(list(range(2009, 2023)))
