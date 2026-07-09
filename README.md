<div align="center">

# Graph-Based Planning for Germany's EV Charging Network

**Retrospective MSc research on spatial graphs, predictive candidate generation, and evolutionary optimization**

[![CI](https://github.com/GiovanniFilomeno/Optimization-Charging-Station-Network-MscThesis/actions/workflows/ci.yml/badge.svg)](https://github.com/GiovanniFilomeno/Optimization-Charging-Station-Network-MscThesis/actions/workflows/ci.yml)
[![Python 3.10](https://img.shields.io/badge/Python-3.10-3776AB?logo=python&logoColor=white)](environment.yml)
[![Research status](https://img.shields.io/badge/results-historical%20thesis%20artifacts-5B5BD6)](docs/REPRODUCIBILITY.md)
[![Code license: MIT](https://img.shields.io/badge/code%20license-MIT-2EA44F.svg)](LICENSE)

</div>

![Project overview](docs/assets/project-overview.svg)

> [!IMPORTANT]
> This is the public portfolio edition of a thesis submitted in 2023. The research narrative, code documentation, privacy controls, and focused tests were improved in 2026; the models were **not retrained** and the historical experiments were **not rerun**. Archived results are presented as research evidence, not as newly reproduced benchmarks.

## The problem

Public charging infrastructure is not only a facility-location problem. Once stations are represented as a spatial graph, every placement decision changes geographic reach, local redundancy, and network connectivity at the same time.

This thesis asks a retrospective planning question:

> Given the observed charging network up to year *t* and a reported fixed budget of new stations, can an evolutionary search identify an alternative placement for year *t + 1* with a better value under an explicit multi-objective network score?

The work studies Germany's public charging-station registry from 2009 to 2022 and compares selected historical deployments with simulated alternatives. It is a counterfactual research prototype—not a causal evaluation, a globally optimal solution, or an operational siting system.

## Research contribution

| Layer | What was implemented | Why it matters |
| --- | --- | --- |
| Data engineering | Cleaning and normalization of the Bundesnetzagentur registry | Converts a public administrative dataset into an analysis-ready station history |
| Spatial graph model | Cumulative yearly NetworkX graphs with distance-threshold edges | Makes network structure measurable over time |
| Network diagnostics | Component-weighted density, edge distance, diameter, and clustering | Exposes structural trade-offs hidden by station counts alone |
| Candidate generation | Per-year Random Forest coordinate models using year and federal state | Provides geographically conditioned starting points for search |
| Evolutionary search | A custom genetic algorithm with selection, crossover, constrained mutation, elitism, and early stopping | Searches a large, non-convex placement space under a configurable objective |
| Retrospective comparison | Thesis-reported matched-count simulated and historical case studies | Tests the formulation against observed deployment years while keeping the interpretation explicit |

The central insight is methodological: **the objective function is a planning argument**. A lower composite score does not mean that every network property improved, nor that the result is automatically preferable in practice.

## Method

1. Clean the charging-station registry and reconstruct cumulative yearly station sets.
2. Represent stations as nodes; connect pairs whose calculated road distance is below 100 km.
3. Measure the connected portions of each yearly graph using topology and distance metrics.
4. Generate candidate coordinates from historical year/state patterns.
5. Search alternative station placements with a genetic algorithm while retaining the existing network.
6. Compare the simulated and observed networks at the same reported station count.

The 100 km routing semantics above describe the maintained implementation. Archived graph and optimization artifacts predate the corrected kilometre/metre prefilter and were not rebuilt.

The implemented objective minimizes:

```text
0.30 · normalized average edge distance
+ 0.10 · normalized diameter
+ 0.10 · normalized clustering
+ 0.10 · (1 − normalized density)
+ 0.40 · (1 − convex-hull coverage ratio)
```

Lower fitness is better. The weights encode one particular preference over spread and topology; they are not a universal measure of charging-network quality. Full definitions and threats to validity are documented in [Research Notes](docs/RESEARCH_NOTES.md).

## What the historical results show

The thesis reports a lower composite score for the simulated network in both selected cases, while several component metrics move in opposite directions:

| Case | Stations in each network | Historical fitness | Simulated fitness | Evidence-based interpretation |
| --- | ---: | ---: | ---: | --- |
| 2012 | 494 | 0.5303 | 0.4210 | Lower composite score and graph diameter; also lower density and a longer mean edge distance |
| 2015 | 1,393 | 0.4459 | 0.3981 | Lower composite score and graph diameter; also a slightly lower density and a longer mean edge distance |

These values are transcribed from the submitted thesis and have not been regenerated. The [Research Notes](docs/RESEARCH_NOTES.md#historical-thesis-results) preserve the complete table, units, inconsistencies, and interpretation. The data-only convergence histories are retained under [`results/fitness/`](results/fitness/).

## Reproducibility status

| Scope | Status |
| --- | --- |
| Source inspection | Available |
| Focused offline tests and linting | Available in CI |
| Historical result inspection | Available as archived figures, tables, and the public thesis |
| Clean-clone execution of the full experiment | Not currently possible |
| Independent reproduction of thesis results | Not claimed |

The exact November 2022 registry snapshot, processed data, trained models, routing cache, GraphML networks, run seeds, and a locked historical environment are not available in this repository. The public OSRM demo endpoint is also unsuitable for an unthrottled quadratic full-scale rebuild. See [Reproducibility](docs/REPRODUCIBILITY.md) for the precise artifact matrix and a responsible rerun plan.

## Repository guide

```text
.
├── main.py                       # Pipeline entry point
├── src/                          # Maintained research modules
├── tests/                        # Focused offline regression tests
├── notebooks/archive/            # Original, unexecuted thesis notebooks
├── data/boundaries/              # Federal-state boundary geometry
├── results/                       # Historical figures, tables, and convergence data
├── output/pdf/                    # Privacy-reviewed public thesis
└── docs/
    ├── assets/project-overview.svg
    ├── DATA.md                    # Sources, licenses, and data-quality caveats
    ├── REPRODUCIBILITY.md         # Exact clean-clone execution status
    └── RESEARCH_NOTES.md          # Research framing, evidence, and limitations
```

Start with:

- [Public thesis — portfolio edition](output/pdf/msc-thesis-public.pdf)
- [Research notes](docs/RESEARCH_NOTES.md)
- [Reproducibility statement](docs/REPRODUCIBILITY.md)
- [Data documentation](docs/DATA.md)
- [Public edition changelog](docs/PUBLIC_EDITION_CHANGELOG.md)
- [Historical result inventory](results/README.md)

## Local inspection

The maintained environment targets Python 3.10:

```bash
conda env create -f environment.yml
conda activate ev-charging-optimization
python -m pip install pytest ruff
pytest -q
ruff format --check .
ruff check .
```

Or use a Python 3.10 virtual environment and `requirements.txt`. The CLI surface can be inspected without starting a research run:

```bash
python main.py --help
```

Pipeline commands require locally acquired source data and, for later stages, generated intermediate artifacts. `python main.py` runs preprocessing through prediction; optimization and visualization require explicit steps and parameters. Follow [Reproducibility](docs/REPRODUCIBILITY.md) before execution.

## Data and responsible use

The historical input is attributed to the German Federal Network Agency's public charging-station registry. The exact 2022 snapshot is absent; newly downloaded data will not reproduce the historical study. The [current official page](https://www.bundesnetzagentur.de/DE/Fachthemen/ElektrizitaetundGas/E-Mobilitaet/Ladesaeulenkarte/artikel_2.html) publishes registry data under CC BY 4.0 with attribution to `Bundesnetzagentur.de`.

The model omits demand, traffic, capacity, charging speed, queueing, grid constraints, construction cost, land availability, accessibility, and equity. Convex-hull area is only a coarse geographic-spread proxy. No result in this repository should be used directly for infrastructure investment or public-policy decisions.

## Citation and licensing

Citation metadata is available in [`CITATION.cff`](CITATION.cff).

Original source code and repository documentation are licensed under the [MIT License](LICENSE). The thesis, historical result artifacts, third-party data, boundary geometry, institutional marks, and external services have separate rights or attribution conditions described in [Third-Party Notices](THIRD_PARTY_NOTICES.md).

## Author

**Giovanni Filomeno** — MSc thesis, Technical University of Munich, 2023

Research interests represented here: graph analytics, geospatial systems, machine learning, evolutionary optimization, and evidence-driven infrastructure planning.
