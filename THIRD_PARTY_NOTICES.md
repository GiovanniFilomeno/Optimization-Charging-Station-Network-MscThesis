# Third-Party Notices and License Scope

The repository combines original research software with an academic thesis, historical outputs, external data provenance, geographic geometry, and references to third-party services. The root MIT License is intentionally limited in scope.

| Material | Location | Rights and conditions |
| --- | --- | --- |
| Original source code and repository documentation | `src/`, `tests/`, root Markdown, `docs/` | MIT License, copyright 2023-2026 Giovanni Filomeno |
| Public thesis | `output/pdf/msc-thesis-public.pdf` | Copyright © 2023 Giovanni Filomeno. All rights reserved; made available for portfolio reading and citation, not relicensed under MIT |
| Historical figures, CSV, and spreadsheets | `results/` | Research artifacts retained for inspection; not explicitly licensed for reuse under MIT. Underlying data and third-party marks retain their own terms |
| Charging-station registry | Not distributed | Exact 2022 terms were not preserved; current Bundesnetzagentur data is CC BY 4.0 with the attribution specified below |
| Federal-state boundary geometry | `data/boundaries/4_niedrig.geo.json` | Strongly matched to `isellsoap/deutschlandGeoJSON`; upstream repository declares The Unlicense and credits another source. See the provenance caveat below |
| Institutional names, logos, and template elements visible in the thesis | Public thesis only | Rights remain with the respective institutions and trademark owners; no endorsement is implied |
| Python dependencies and external services | Referenced, not vendored | Governed by their respective licenses, policies, and terms |

## Bundesnetzagentur charging-station data

The historical research uses a public charging-station registry attributed to the Bundesnetzagentur. Neither the raw nor processed CSV is distributed. The exact November 2022 acquisition URL, checksum, and attached terms were not archived with the project.

The [current official data page](https://www.bundesnetzagentur.de/DE/Fachthemen/ElektrizitaetundGas/E-Mobilitaet/Ladesaeulenkarte/artikel_2.html), accessed on 9 July 2026, publishes the register data under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) and specifies the attribution `Bundesnetzagentur.de`. Preserve the terms attached to any newly acquired snapshot. No data rights should be inferred from this repository's MIT License.

## Germany federal-state boundary GeoJSON

The file `data/boundaries/4_niedrig.geo.json` contains no embedded source or license metadata. Its strongest upstream match is [`isellsoap/deutschlandGeoJSON`](https://github.com/isellsoap/deutschlandGeoJSON), path `2_bundeslaender/4_niedrig.geo.json`. That repository declares [The Unlicense](https://github.com/isellsoap/deutschlandGeoJSON/blob/main/LICENSE.md) and credits `GIS-DATA` as its source.

The matching evidence is recorded in [`docs/DATA.md`](docs/DATA.md). Because the upstream credit does not identify a precise source dataset or its terms, verify the full provenance chain before treating the geometry as cleared for formal redistribution. Otherwise replace it with a boundary dataset whose origin, version, and license are explicit.

## OpenStreetMap and OSRM

The code can send coordinates to the public OSRM demonstration endpoint at `router.project-osrm.org`. No OSRM code or response cache is distributed. Use of the public endpoint remains subject to the project's [demo-server policy](https://github.com/Project-OSRM/osrm-backend/wiki/Demo-server), including its request-rate and non-commercial-use limits.

The thesis describes OSRM routing as based on OpenStreetMap data. OpenStreetMap data and produced works are governed by OpenStreetMap's own [copyright and attribution framework](https://www.openstreetmap.org/copyright). Verify the requirements for any routing database, map, or derived material that is published.

## Academic publications and university material

Duplicate local copies of six cited academic papers were removed from the publishable tree because redistribution permission was not documented. Their bibliographic citations remain in the thesis. The original editable thesis, university template, presentation, and temporary office files were also removed from the publishable tree; only the privacy-reviewed public PDF is retained.

The thesis may display institutional names, marks, and formatting elements. They remain the property of their respective owners. Reference to the Technical University of Munich describes the academic context and does not imply institutional endorsement of this repository or its portfolio edition.

## Python dependencies

The project declares external dependencies in `requirements.txt` and `environment.yml`. They are installed from their own distributions and are not vendored. Each package remains governed by its own license and notice requirements.

## No endorsement

Bundesnetzagentur, OpenStreetMap, OSRM, the Technical University of Munich, publishers, data-source maintainers, and dependency authors have not endorsed this repository merely because their names, data, services, publications, or software are referenced.
