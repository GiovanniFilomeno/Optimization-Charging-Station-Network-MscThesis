# Public Portfolio Edition Changelog

## 2026 portfolio edition

This edition improves the public presentation and inspectability of the 2023 thesis project without retraining models or rerunning historical experiments.

### Thesis publication

- Added a clearly labelled public portfolio cover.
- Removed the home address and matriculation number from the public PDF.
- Corrected title, author, subject, and creator metadata.
- Embedded all fonts and verified the final 65-page A4 rendering.
- Preserved the submitted thesis body, pagination, tables, and historical claims unchanged.

### Research communication

- Replaced unsupported headline metrics with values traceable to the thesis tables.
- Separated historical evidence from maintained-code improvements.
- Added explicit research framing, data provenance, reproducibility status, threats to validity, citation metadata, and third-party notices.
- Added a conceptual project overview and documented archived result artifacts.

### Repository hygiene

- Excluded the original editable thesis, university template, presentation, temporary office files, and duplicated third-party papers from the publishable tree.
- Excluded obsolete population pickles and converted code-bearing convergence snippets into data-only CSV files.
- Moved the seven original, unexecuted notebooks into an explicitly labelled archive.
- Replaced opaque `point_*` result filenames with descriptive names and added a historical artifact inventory.
- Preserved removed local material in an ignored private quarantine for reversible recovery on the working machine.

### Maintained code

- Corrected the Haversine kilometre/metre comparison used before OSRM routing.
- Added HTTPS, timeouts, error handling, and configurable local caching to OSRM requests.
- Made approximate graph metrics deterministic by default and defined behavior for empty or isolate-only graphs.
- Prevented missing connector types from being counted as the literal plug category `0`.
- Added focused offline regression tests, Ruff checks, and GitHub Actions CI.

### Explicit non-changes

- No model was retrained.
- No genetic-algorithm experiment was rerun.
- No historical figure, table, or score is claimed as independently reproduced.
- No new empirical performance claim was introduced.

Working-tree cleanup does not remove files from earlier Git commits. The published repository history must be replaced or carefully rewritten before this edition can be considered fully sanitized; that separate operation is intentionally not performed by the documentation refresh.

See [`RESEARCH_NOTES.md`](RESEARCH_NOTES.md) and [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md) for the precise scientific scope.
