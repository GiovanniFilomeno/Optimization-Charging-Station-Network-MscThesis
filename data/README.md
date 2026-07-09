# Data Directory

Only the federal-state boundary geometry is distributed with this repository. The historical charging-station registry and its processed derivative are absent.

The maintained code expects:

```text
data/
├── raw/
│   └── Ladesaeulenregister_CSV.csv
├── processed/
│   └── ChargingStationCleaned.csv
└── boundaries/
    └── 4_niedrig.geo.json
```

Do not assume that a current registry download reproduces the November 2022 snapshot used in the thesis. Record the source URL, release or snapshot date, governing terms, and SHA-256 checksum for every acquired input. See [`docs/DATA.md`](../docs/DATA.md) for provenance, licensing, schema expectations, and known quality limitations.
