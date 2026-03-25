"""
Configuration constants and shared paths for the project.

Centralizes all file paths, hyperparameters, and constants to avoid
hardcoding values across modules.
"""

from pathlib import Path

# ──────────────────────────────────────────────────────────────────────
# Directory paths (relative to project root)
# ──────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
DATA_BOUNDARIES = PROJECT_ROOT / "data" / "boundaries"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
NETWORKS_BASELINE = RESULTS_DIR / "networks" / "baseline"
NETWORKS_OPTIMIZED = RESULTS_DIR / "networks" / "optimized"
FIGURES_DIR = RESULTS_DIR / "figures"

# ──────────────────────────────────────────────────────────────────────
# Data files
# ──────────────────────────────────────────────────────────────────────
RAW_CSV = DATA_RAW / "Ladesaeulenregister_CSV.csv"
CLEANED_CSV = DATA_PROCESSED / "ChargingStationCleaned.csv"
GERMANY_GEOJSON = DATA_BOUNDARIES / "4_niedrig.geo.json"

# ──────────────────────────────────────────────────────────────────────
# Network parameters
# ──────────────────────────────────────────────────────────────────────
MAX_EDGE_DISTANCE_KM = 100       # BEV range threshold for edge creation
EARTH_RADIUS_KM = 6371           # For Haversine distance formula

# ──────────────────────────────────────────────────────────────────────
# Encoding corrections (German UTF-8 artifacts)
# ──────────────────────────────────────────────────────────────────────
STATE_NAME_CORRECTIONS = {
    "Baden-Wï¿½rttemberg": "Baden-Württemberg",
    "Thï¿½ringen": "Thüringen",
}

# ──────────────────────────────────────────────────────────────────────
# Genetic algorithm defaults
# ──────────────────────────────────────────────────────────────────────
GA_POPULATION_SIZE = 60
GA_NUM_GENERATIONS = 30
GA_MUTATION_RATE = 0.8
GA_TOURNAMENT_SIZE = 2
GA_ELITISM_FRACTION = 0.05
GA_NO_IMPROVEMENT_THRESHOLD = 10

# ──────────────────────────────────────────────────────────────────────
# Fitness function weights (must sum to 1.0)
# ──────────────────────────────────────────────────────────────────────
FITNESS_WEIGHTS = {
    "average_distance": 0.3,
    "diameter": 0.1,
    "clustering": 0.1,
    "density": 0.1,
    "coverage": 0.4,
}

# Normalization bounds (observed min/max from historical data 2009-2022)
NORM_BOUNDS = {
    "average_distance": (15.10, 58.43),
    "diameter": (1, 11),
    "clustering": (0.72, 0.93),
    "density": (0.08, 1.0),
}
