<div align="center">

# Optimizing Germany's EV Charging Station Network

### A Graph-Theoretic and Genetic Algorithm Approach to Infrastructure Planning

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)](https://python.org)
[![NetworkX](https://img.shields.io/badge/NetworkX-Graph%20Analysis-4B8BBE)](https://networkx.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-ML%20Models-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

</div>

---

## Abstract

This project tackles the **optimal placement of electric vehicle (EV) charging stations** across Germany using a combination of **spatial network analysis**, **machine learning prediction**, and **evolutionary optimization**. Starting from the official *Bundesnetzagentur* dataset of 65,000+ public EV charging points, the system:

1. **Models** the charging infrastructure as a spatial graph (nodes = stations, edges = road connectivity within EV range)
2. **Analyzes** network topology evolution from 2009–2022 using graph metrics (density, diameter, clustering)
3. **Predicts** plausible future station locations using Random Forest models trained on historical spatial patterns
4. **Optimizes** new station placement via a custom Genetic Algorithm with a multi-objective fitness function balancing coverage, connectivity, and distance efficiency

The GA-optimized networks demonstrate measurable improvements in geographic coverage and network connectivity compared to the actual historical station deployment.

---

## Key Results

| Metric | Baseline (2012) | GA-Optimized (2012) | Improvement |
|--------|:---------------:|:-------------------:|:-----------:|
| Coverage Area Ratio | 0.67 | 0.82 | +22% |
| Network Density | 0.12 | 0.18 | +50% |
| Avg. Station Distance | 42.3 km | 35.1 km | -17% |

---

## Technical Architecture

```
┌──────────────────┐     ┌──────────────────┐     ┌──────────────────┐
│  preprocessing.py │────▶│   network.py      │────▶│   analysis.py    │
│                   │     │                   │     │                  │
│  • CSV parsing    │     │  • Haversine dist │     │  • Density       │
│  • Geocoding fix  │     │  • OSRM road dist │     │  • Diameter      │
│  • State normali- │     │  • 100km edge     │     │  • Clustering    │
│    zation         │     │    threshold      │     │  • Weighted avg  │
└──────────────────┘     └──────────────────┘     └──────────────────┘
         │                                                  │
         ▼                                                  ▼
┌──────────────────┐     ┌──────────────────┐     ┌──────────────────┐
│  prediction.py    │────▶│  optimization.py  │────▶│ visualization.py │
│                   │     │                   │     │                  │
│  • Random Forest  │     │  • Tournament sel │     │  • Fitness comp  │
│  • Per-year model │     │  • 1-pt crossover │     │  • Baseline vs   │
│  • State-weighted │     │  • Geo-constrained│     │    optimized     │
│    sampling       │     │    mutation       │     │  • Geographic    │
│                   │     │  • Elitism (5%)   │     │    overlay plots │
└──────────────────┘     └──────────────────┘     └──────────────────┘
```

---

## Repository Structure

```
.
├── main.py                    # Pipeline orchestrator (CLI entry point)
├── environment.yml            # Conda environment specification
├── requirements.txt           # pip dependencies
│
├── src/                       # Core Python modules
│   ├── __init__.py
│   ├── config.py              # Centralized paths, constants, hyperparameters
│   ├── utils.py               # Distance calculations, graph drawing, metrics
│   ├── preprocessing.py       # Data cleaning & normalization pipeline
│   ├── statistics.py          # Descriptive analysis & visualizations
│   ├── network.py             # Spatial graph construction (OSRM + caching)
│   ├── analysis.py            # Topology metrics computation
│   ├── prediction.py          # Random Forest location prediction
│   ├── optimization.py        # Genetic Algorithm for station placement
│   └── visualization.py       # Results comparison & evaluation
│
├── notebooks/                 # Interactive walkthroughs (import from src/)
│   ├── 01_preprocessing.ipynb
│   ├── 02_statistics.ipynb
│   ├── 03_network_generation.ipynb
│   ├── 04_network_analysis.ipynb
│   ├── 05_prediction.ipynb
│   ├── 06_optimization.ipynb
│   └── 07_results_visualization.ipynb
│
├── data/
│   ├── raw/                   # Original Bundesnetzagentur CSV
│   ├── processed/             # Cleaned dataset
│   └── boundaries/            # Germany federal state GeoJSON
│
├── models/                    # Trained Random Forest models (per year)
│   ├── 2010/
│   │   ├── rf_model_2010_lat.pkl
│   │   └── rf_model_2010_lon.pkl
│   └── .../
│
├── results/
│   ├── figures/
│   │   ├── network_graphs/            # Baseline network visualizations
│   │   ├── network_graphs_optimized/  # GA-optimized visualizations
│   │   └── statistics/                # Statistical analysis plots & Excel
│   ├── fitness/                       # GA convergence logs
│   └── networks/
│       ├── baseline/                  # GraphML files (real-world networks)
│       └── optimized/                 # GraphML files (GA-optimized networks)
│
└── docs/references/            # Academic papers
```

---

## Getting Started

### Prerequisites
- Python 3.10+ or Conda/Miniconda
- ~4 GB disk space for data and models

### Installation

**Option A — Conda (recommended):**
```bash
git clone https://github.com/<your-username>/ev-charging-network-optimization.git
cd ev-charging-network-optimization
conda env create -f environment.yml
conda activate ev-charging-optimization
```

**Option B — pip:**
```bash
git clone https://github.com/<your-username>/ev-charging-network-optimization.git
cd ev-charging-network-optimization
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### Running the Pipeline

**Full pipeline:**
```bash
python main.py
```

**Individual steps:**
```bash
python main.py --step preprocess
python main.py --step statistics
python main.py --step network --year-range 2009 2022
python main.py --step analysis
python main.py --step prediction
python main.py --step optimize --year 2011 --stations 174
python main.py --step visualize
```

**Run modules directly:**
```bash
python -m src.preprocessing
python -m src.prediction --compare-only
python -m src.optimization --year 2012 --num-stations 121
```

> **Note**: Network generation (`--step network`) is computationally intensive — O(n²) pairwise distance calculations per year. Road distances are cached in a local SQLite database to avoid redundant OSRM API calls.

---

## Technical Skills Demonstrated

| Category | Technologies & Concepts |
|----------|------------------------|
| **Data Engineering** | Pandas ETL pipeline, geospatial data processing, CSV/Excel/GraphML I/O, SQLite caching layer |
| **Graph Theory** | NetworkX, spatial graph construction, connected components, diameter, clustering coefficient, density |
| **Machine Learning** | Random Forest regression, KNN, scikit-learn, model evaluation (MAE, MSE, R²), per-year incremental training |
| **Optimization** | Custom Genetic Algorithm: multi-objective fitness, tournament selection, crossover, mutation, elitism, early stopping |
| **Geospatial** | GeoPandas, Shapely, PyProj (CRS transformations), OSRM API routing, Haversine formula, convex hull computation |
| **Visualization** | Matplotlib, Folium, Bokeh, geospatial overlays, statistical plots |
| **Software Engineering** | Modular package architecture, CLI entry point, centralized configuration, type hints, docstrings, logging |

---

## Methodology

### 1. Network Construction
Charging stations are modeled as nodes in an undirected weighted graph. An edge connects two stations if the road distance between them is ≤ 100 km (typical range of early BEVs). Edge weights represent actual road distances obtained via the OSRM routing API.

### 2. Topology Analysis
For each year (2009–2022), four key metrics are computed and aggregated via node-count-weighted averaging across connected components:
- **Density**: Edge saturation level (actual / possible edges)
- **Average Distance**: Mean edge weight in km
- **Diameter**: Longest shortest path (worst-case reachability)
- **Clustering Coefficient**: Local triangle density

### 3. Location Prediction
Random Forest regressors are trained per-year on cumulative historical data to predict station coordinates given the year and federal state. These models capture the spatial deployment patterns specific to each state's geography and urban density.

### 4. Genetic Algorithm Optimization
A custom GA optimizes new station placement using:

| Component | Implementation |
|-----------|---------------|
| **Fitness Function** | Weighted multi-objective: coverage (40%), distance (30%), diameter / clustering / density (10% each) |
| **Initialization** | RF-predicted locations, distributed proportionally across federal states |
| **Selection** | Tournament selection (k=2) |
| **Crossover** | Single-point with geographic validity enforcement |
| **Mutation** | Random ±1° perturbation, constrained to German borders via Shapely |
| **Elitism** | Top 5% preserved across generations |
| **Stopping** | Early stopping after 10 generations without improvement |

---

## Data Source

**Bundesnetzagentur — Ladesäulenregister (Charging Station Registry)**
- Official German federal registry of publicly accessible EV charging infrastructure
- ~65,000 charging points across 16 federal states
- Attributes: operator, location, power output, connector types, commissioning date

---

## License

This project is part of a Master's Thesis. See individual data source licenses for restrictions on the Bundesnetzagentur dataset.
