"""
Genetic Algorithm Optimization Module.

Optimizes placement of new EV charging stations using a multi-objective
Genetic Algorithm that maximizes a weighted fitness function:

    Fitness = 0.3 × avg_distance + 0.1 × diameter + 0.1 × clustering
            + 0.1 × density + 0.4 × coverage

GA components:
    - Initial population: RF-predicted locations, state-proportional
    - Selection: Tournament (size 2)
    - Crossover: Single-point with geographic validity
    - Mutation: Random perturbation (±1°) within German borders
    - Elitism: Top 5% preserved
    - Stopping: No improvement for 10 generations

Usage:
    python -m src.optimization --year 2011 --num-stations 174
"""

import argparse
import logging
import os
import pickle
import random
from pathlib import Path

import geopandas as gpd
import joblib
import networkx as nx
import pandas as pd
from shapely.geometry import MultiPoint, Point
from sklearn import preprocessing
from tqdm import tqdm

from .config import (
    CLEANED_CSV,
    GA_ELITISM_FRACTION,
    GA_MUTATION_RATE,
    GA_NO_IMPROVEMENT_THRESHOLD,
    GA_NUM_GENERATIONS,
    GA_POPULATION_SIZE,
    GA_TOURNAMENT_SIZE,
    GERMANY_GEOJSON,
    MODELS_DIR,
    NETWORKS_BASELINE,
    NETWORKS_OPTIMIZED,
    NORM_BOUNDS,
    FITNESS_WEIGHTS,
    STATE_NAME_CORRECTIONS,
)
from .utils import get_osrm_distance, calculate_weighted_metrics, weighted_mean

logger = logging.getLogger(__name__)


def _load_data_and_boundaries():
    """Load station data and Germany boundary polygon."""
    data = pd.read_csv(CLEANED_CSV, encoding="utf-8")
    data["federal_state"] = data["federal_state"].replace(STATE_NAME_CORRECTIONS)
    data["geometry"] = data.apply(
        lambda r: Point(r["longitude_[dg]"], r["latitude_[dg]"]), axis=1
    )
    boundary = gpd.read_file(GERMANY_GEOJSON)
    polygon = boundary.geometry.unary_union
    return data, polygon


def genetic_algorithm(
    graph: nx.Graph,
    data: pd.DataFrame,
    germany_polygon,
    population_size: int = GA_POPULATION_SIZE,
    num_generations: int = GA_NUM_GENERATIONS,
    mutation_rate: float = GA_MUTATION_RATE,
    num_new_stations: int = 50,
    year: int = 2022,
) -> nx.Graph:
    """
    Optimize EV charging station placement using a Genetic Algorithm.

    Args:
        graph: Existing baseline network (NetworkX Graph).
        data: Full station DataFrame (for state distribution).
        germany_polygon: Shapely polygon of Germany's borders.
        population_size: Candidate solutions per generation.
        num_generations: Maximum iterations.
        mutation_rate: Initial mutation probability per gene.
        num_new_stations: Number of new stations to place.
        year: Target year for RF prediction models.

    Returns:
        Optimized NetworkX Graph with new stations added.
    """
    tournament_size = GA_TOURNAMENT_SIZE
    no_improvement_threshold = GA_NO_IMPROVEMENT_THRESHOLD

    # ── Initial population generation ──────────────────────────────────
    def generate_initial_population():
        """Generate candidate configurations using RF-predicted locations."""
        cache_path = NETWORKS_OPTIMIZED / f"population_{year}.pkl"

        if cache_path.exists():
            with open(cache_path, "rb") as f:
                pop = pickle.load(f)
            logger.info("Loaded cached population for %d", year)
            return pop

        rf_lat = joblib.load(MODELS_DIR / str(year) / f"rf_model_{year}_lat.pkl")
        rf_lon = joblib.load(MODELS_DIR / str(year) / f"rf_model_{year}_lon.pkl")

        le = preprocessing.LabelEncoder()
        le.fit(data["federal_state"].unique())

        # Distribute stations proportionally to existing state distribution
        counts = data["federal_state"].value_counts()
        proportions = counts / counts.sum()
        per_state = {
            s: int(round(num_new_stations * p)) for s, p in proportions.items()
        }
        total = sum(per_state.values())
        if total < num_new_stations:
            top = max(per_state, key=per_state.get)
            per_state[top] += num_new_stations - total

        population = []
        for _ in tqdm(range(population_size), desc="Initial population"):
            stations = []
            for state, n in per_state.items():
                enc = le.transform([state])[0]
                for _ in range(n):
                    features = pd.DataFrame(
                        [[year, enc]], columns=["year", "federal_state_encoded"]
                    )
                    lat = rf_lat.predict(features)[0]
                    lon = rf_lon.predict(features)[0]
                    pt = Point(lon, lat)
                    while not germany_polygon.contains(pt):
                        lat = rf_lat.predict(features)[0]
                        lon = rf_lon.predict(features)[0]
                        pt = Point(lon, lat)
                    stations.append((lat, lon))
            population.append(stations)

        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "wb") as f:
            pickle.dump(population, f)
        return population

    # ── Node insertion ─────────────────────────────────────────────────
    def add_node(g, coords, max_dist_km=100):
        """Add a station node and connect to nearby nodes."""
        nid = len(g) + 1
        lat, lon = coords
        g.add_node(nid, latitude=lat, longitude=lon)
        for node in g.nodes:
            if node != nid:
                d = get_osrm_distance(
                    lat, lon,
                    g.nodes[node]["latitude"], g.nodes[node]["longitude"],
                )
                if d is not None and d < max_dist_km * 1000:
                    g.add_edge(nid, node, weight=d)
        return g

    # ── Coverage metric ────────────────────────────────────────────────
    def coverage_ratio(g):
        """Network convex hull area / country area."""
        coords = [
            (g.nodes[n]["longitude"], g.nodes[n]["latitude"]) for n in g.nodes()
        ]
        hull = MultiPoint(coords).convex_hull
        return hull.area / germany_polygon.area

    # ── Fitness function ───────────────────────────────────────────────
    def fitness(solution):
        """Multi-objective fitness (lower = better)."""
        g = graph.copy()
        for c in solution:
            g = add_node(g, c)

        m = calculate_weighted_metrics(g, year)
        cov = coverage_ratio(g)

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

    # ── Genetic operators ──────────────────────────────────────────────
    def crossover(p1, p2):
        """Single-point crossover with geographic validity."""
        pt = random.randint(1, len(p1) - 1)
        c1 = p1[:pt] + p2[pt:]
        c2 = p2[:pt] + p1[pt:]
        for i, (lat, lon) in enumerate(c1):
            if not germany_polygon.contains(Point(lon, lat)):
                c1[i] = p1[i]
        for i, (lat, lon) in enumerate(c2):
            if not germany_polygon.contains(Point(lon, lat)):
                c2[i] = p2[i]
        return c1, c2

    def mutate(sol, rate):
        """Random perturbation (±1°) constrained to Germany."""
        for i in range(len(sol)):
            if random.random() < rate:
                lat, lon = sol[i]
                pt = Point(lon + random.uniform(-1, 1), lat + random.uniform(-1, 1))
                while not germany_polygon.contains(pt):
                    pt = Point(lon + random.uniform(-1, 1), lat + random.uniform(-1, 1))
                sol[i] = (pt.y, pt.x)
        return sol

    def tournament(pop, fits):
        """Tournament selection (min fitness wins)."""
        idx = random.sample(range(len(pop)), tournament_size)
        best = min(idx, key=lambda i: fits[i])
        return pop[best]

    # ── Main GA loop ───────────────────────────────────────────────────
    population = generate_initial_population()
    best_fitness = float("inf")
    best_solution = None
    stagnation = 0

    for gen in range(num_generations):
        fits = [fitness(s) for s in population]
        gen_best = min(fits)

        if gen_best < best_fitness:
            best_fitness = gen_best
            best_solution = population[fits.index(gen_best)]
            stagnation = 0
        else:
            stagnation += 1

        logger.info("Gen %3d | Fitness: %.6f", gen, best_fitness)

        if stagnation >= no_improvement_threshold:
            logger.info("Converged after %d generations", gen + 1)
            break

        # Elitism
        n_elite = max(1, int(GA_ELITISM_FRACTION * population_size))
        elite_idx = sorted(range(len(fits)), key=lambda i: fits[i])[:n_elite]
        elites = [population[i] for i in elite_idx]

        # Selection + reproduction
        parents = [tournament(population, fits) for _ in range(population_size - n_elite)]
        if len(parents) % 2:
            parents.append(parents[-1])

        offspring = []
        for i in range(0, len(parents), 2):
            c1, c2 = crossover(parents[i], parents[i + 1])
            offspring.append(mutate(c1, mutation_rate))
            offspring.append(mutate(c2, mutation_rate))

        population = elites + offspring

    # Build the final optimized graph
    best_solution = min(population, key=fitness)
    result = graph.copy()
    for coords in best_solution:
        result = add_node(result, coords)

    logger.info(
        "Optimized: %d nodes, %d edges",
        result.number_of_nodes(), result.number_of_edges(),
    )
    return result


def run(year: int, num_stations: int) -> nx.Graph:
    """Execute the GA optimization pipeline for a given year."""
    data, polygon = _load_data_and_boundaries()

    graph_path = NETWORKS_BASELINE / f"network_{year}_100.graphml"
    graph = nx.read_graphml(str(graph_path))
    logger.info("Loaded baseline network for %d", year)

    optimized = genetic_algorithm(
        graph, data, polygon,
        num_new_stations=num_stations,
        year=year,
    )

    NETWORKS_OPTIMIZED.mkdir(parents=True, exist_ok=True)
    out = NETWORKS_OPTIMIZED / f"network_{year + 1}_optimized.graphml"
    nx.write_graphml(optimized, str(out))
    logger.info("Saved → %s", out)

    return optimized


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(description="Optimize EV station placement via GA")
    parser.add_argument("--year", type=int, required=True, help="Base year")
    parser.add_argument("--num-stations", type=int, required=True,
                        help="Number of new stations to place")
    args = parser.parse_args()
    run(args.year, args.num_stations)
