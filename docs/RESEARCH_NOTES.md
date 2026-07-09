# Research Notes

## Status and scope

This repository is a public portfolio edition of the Master's thesis *Analysis and Optimization of the Charging Station Network in Germany*, submitted by Giovanni Filomeno to the TUM School of Management on 29 November 2023.

The 2026 portfolio work reorganizes and documents the original research. It does not claim that the models were retrained, that the optimization was rerun, or that the historical numerical results were independently reproduced. See [REPRODUCIBILITY.md](REPRODUCIBILITY.md) for the current execution status.

## Research framing

The study asks a retrospective planning question:

> Given the observed charging-station network up to year *t* and a fixed budget of new stations, can an evolutionary search identify an alternative placement for year *t + 1* with a better value under a stated, multi-objective network score?

The reported experimental design is budget-matched: the simulated and historical networks contain the same number of new stations. The experiment is therefore intended to compare alternative placements, not alternative investment levels. The current implementation allocates new stations to states with independent rounding and only corrects an under-allocation; it can exceed the requested count when rounding overshoots. The station totals in the thesis tables should be treated as the record for the reported cases.

The implemented research pipeline has four layers:

1. Represent public charging stations as nodes in cumulative yearly spatial graphs.
2. Connect station pairs when their calculated distance is below a 100 km threshold.
3. Generate candidate coordinates from models using commissioning year and federal state.
4. Use a genetic algorithm to search over candidate placements while keeping existing stations fixed.

This is a retrospective counterfactual experiment. It is not a causal estimate of what infrastructure operators would have built, and it is not a production siting recommendation.

## Technical contributions

The repository demonstrates the integration of several methods in one infrastructure-planning workflow:

- A documented preprocessing layer for the Bundesnetzagentur charging-station registry.
- Cumulative yearly graph construction for the period 2009-2022.
- Component-weighted summaries of density, average edge distance, diameter, and clustering.
- Random Forest and K-Nearest Neighbors comparison for coordinate prediction, followed by yearly Random Forest model generation.
- A custom genetic algorithm with tournament selection, single-point crossover, geographic mutation constraints, elitism, and early stopping.
- Retrospective, matched-station-count comparisons between simulated and observed yearly networks.

The strongest portfolio contribution is the end-to-end problem formulation: public infrastructure data is transformed into a graph, evaluated through an explicit objective, and searched with a configurable evolutionary algorithm. The results should be read as a research prototype and a basis for further validation.

## Objective function

The current implementation minimizes a weighted composite score:

```text
fitness = 0.30 * normalized average edge distance
        + 0.10 * normalized diameter
        + 0.10 * normalized clustering
        + 0.10 * (1 - normalized density)
        + 0.40 * (1 - convex-hull coverage ratio)
```

Lower fitness is better. The normalization bounds in `src/config.py` are described as observed bounds from 2009-2022. Values are not clipped when they fall outside those bounds.

This score encodes a particular planning preference rather than a universal definition of network quality. In particular, the implementation minimizes clustering while rewarding density and geographic spread. Any public interpretation should therefore discuss the composite trade-off instead of describing every metric as improved.

## Historical thesis results

The following values are transcribed from Tables 2 and 4 of the submitted thesis. They are included for historical traceability, not as a new benchmark.

| Case | Metric | Simulated | Historical |
| --- | --- | ---: | ---: |
| 2012 | Density | 0.14 | 0.18 |
| 2012 | Average distance (km) | 50.001 | 43.65 |
| 2012 | Diameter | 9.0 | 11 |
| 2012 | Average clustering | 0.768 | 0.85 |
| 2012 | Stations | 494 | 494 |
| 2012 | Composite fitness | 0.421 | 0.5303 |
| 2015 | Density | 0.1089 | 0.11 |
| 2015 | Average distance (km) | 58.406 | 50.75 |
| 2015 | Diameter | 10.0 | 11.0 |
| 2015 | Average clustering | 0.709 | 0.80 |
| 2015 | Stations | 1,393 | 1,393 |
| 2015 | Composite fitness | 0.3981 | 0.4459 |

The retained data-only convergence histories are consistent with the reported final simulated fitness to the displayed precision: `results/fitness/fitness-history-2012.csv` ends at approximately 0.42097 and `results/fitness/fitness-history-2015.csv` at approximately 0.39806.

Two interpretive cautions matter:

- The historical experiments report a lower composite fitness, but not uniformly better component metrics. For example, simulated average distance is higher in both tables above.
- The thesis prose describes decreased average distance in one summary passage, while its result tables show the opposite. This public edition treats the tabulated values as the traceable record and does not repeat the prose claim.

## Known limitations and threats to validity

### Data provenance and availability

The raw November 2022 registry snapshot and the processed CSV are not tracked. The exact source URL, download date, checksum, and historical terms were not archived with the repository. The Bundesnetzagentur's current page identifies the published register data as CC BY 4.0, but the repository does not preserve the original 2022 download record. The federal-state boundary file is tracked; a strong upstream match has been identified retrospectively, rather than from contemporaneous provenance metadata. See [DATA.md](DATA.md).

### Distance semantics

The implementation originally published with the repository converted Haversine kilometres to metres and compared the result with `100`, so OSRM was used only within 100 metres rather than the intended 100 km. The 2026 portfolio code corrects the prefilter to compare kilometres with 100 km and adds focused unit tests. The historical graphs and optimization results were not regenerated after that correction, and no run manifest identifies the exact code that produced each artifact. Historical results must therefore still be described as geographic-distance research outputs, not newly validated road-network results.

### Graph abstraction

An edge represents proximity under a distance threshold, not a route with charging demand, travel time, capacity, reliability, or power-grid constraints. Diameter is computed in hops by NetworkX's approximation routine, while average distance is the mean weight of direct edges. Those metrics answer different questions and should not be conflated with end-to-end driving distance.

### Component and coverage mismatch

Topology metrics exclude isolated nodes because only connected components with at least two nodes are aggregated. Convex-hull coverage includes every node. A candidate can therefore enlarge coverage without contributing to the topology metrics, creating an incentive that may not correspond to useful connectivity.

Coverage is calculated from longitude/latitude coordinates without an equal-area projection. A planar convex hull can include territory outside Germany, its area ratio is not geodesic, and the unbounded ratio may exceed one. It is therefore a coarse search proxy rather than a validated measure of service coverage.

### Candidate generation and temporal leakage

The location models use only year and encoded federal state as features. Repeated inputs for the same state and year can produce repeated point predictions, limiting initial-population diversity. In addition, the optimizer derives state proportions from the complete cleaned dataset rather than filtering it to the base year, which introduces future information into retrospective experiments.

The Random Forest/K-Nearest Neighbors comparison uses a random row split across all years. It is neither a temporal holdout nor a spatial holdout, so it does not demonstrate generalization to future years or unseen regions.

### Stochastic reproducibility

The Random Forest comparison is seeded, and the 2026 portfolio code now passes a default seed to the NetworkX approximation routines. The genetic operators still use Python's unseeded `random` module, and the historical artifacts do not record a seed. A single historical run therefore does not establish the stability of the reported score.

### Objective sensitivity

The five weights and normalization bounds are fixed manually. The repository contains no weight sensitivity analysis, ablation study, or comparison with random, greedy, or mathematical-programming baselines. The clustering direction also needs a domain justification because lower clustering can reduce local redundancy.

### Operational scope

The model does not currently include demand, traffic flows, station capacity, charging speed, queueing, grid connection, construction cost, land availability, accessibility, or implementation constraints. The convex hull is a coarse proxy for geographic spread and does not measure population or road-network coverage.

## Responsible portfolio language

The repository can support the following claim:

> This research prototype formulates EV charging-station placement as a spatial graph optimization problem and demonstrates an end-to-end implementation combining data preparation, graph metrics, coordinate prediction, and evolutionary search.

It does not yet support claims that the algorithm finds globally optimal sites, reduces real driving distance, improves every network metric, or is ready for operational deployment.

## Future validation plan

No item in this section is presented as completed work. A rigorous follow-up should:

1. Archive the exact source datasets, terms, checksums, and acquisition dates.
2. Rebuild the historical graphs after the corrected OSRM prefilter has been validated against known routes.
3. Freeze dependency versions and record the OSRM/OSM data or service version used.
4. Remove temporal leakage by fitting all year-*t* inputs exclusively on information available by year *t*.
5. Add explicit seeds and run multiple independent genetic-algorithm trials.
6. Report distributions and confidence intervals, not only the best run.
7. Compare against random placement, historical-proportion sampling, greedy coverage, and a facility-location baseline.
8. Run sensitivity analyses over objective weights, normalization bounds, and the 100 km edge threshold.
9. Evaluate demand-weighted access, route coverage, grid feasibility, cost, and regional equity.
10. Publish one machine-readable experiment manifest that maps each figure and table to its inputs, code revision, configuration, seed, and checksum.
