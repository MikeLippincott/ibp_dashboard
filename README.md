# Featurization Estimations Dashboard

This repository contains a small Streamlit dashboard and a static export for exploring rough compute estimates for featurization workflows.

The dashboard is intentionally lightweight. It is meant to help with planning, not to provide production-grade benchmarks. The IBP section in particular uses coarse assumptions and should be treated as an estimate only.

## What is in the repo

- `justfile` for common dashboard commands.
- `tools/featurization_dashboard/app.py` for the Streamlit dashboard.
- `tools/featurization_dashboard/dashboard_math.py` for the reusable estimation logic.
- `tools/featurization_dashboard/export_static.py` for generating the static export files.
- `tools/featurization_dashboard/static/index.html` for the standalone static site.
- `tests/` for unit coverage of the shared math helpers.
- `.pre-commit-config.yaml` for repository hygiene hooks.
- `LICENSE` for the BSD-3-Clause license.

## Requirements

- Python 3.9 or newer.
- `pip`.
- `just` if you want to use the task shortcuts.

Runtime dependencies live in `tools/featurization_dashboard/requirements.txt`.
Development dependencies live in `requirements-dev.txt`.

## Install

Install the dashboard dependencies:

```bash
just dashboard-install
```

For development tooling and tests:

```bash
python -m pip install -r tools/featurization_dashboard/requirements.txt -r requirements-dev.txt
```

## Run the dashboard

Start the Streamlit app:

```bash
just dashboard-run
```

The app exposes controls for dataset size, compute limits, chart selection, and IBP estimates.

## Export the static site

Generate the standalone export:

```bash
just dashboard-export-static
```

The export script copies `tools/featurization_dashboard/static/index.html` to the dashboard directory root as `index.html` and `static_site_export.html` so the page can be hosted as a default document.

## Serve the static site locally

You can serve the exported static assets with:

```bash
just dashboard-serve-static
```

This uses Python's built-in HTTP server on port 8000.

## Run tests

The unit tests cover the shared estimation helpers and the IBP formulas.

```bash
python -m pytest
```

If you prefer, `pre-commit` will also run the test suite as its final local hook.

## Pre-commit

Install the hooks and run them on all tracked files:

```bash
pre-commit install
pre-commit run --all-files
```

The current hook set checks for common formatting issues, YAML/TOML validity, mixed line endings, large files, and runs `pytest`.

## Repository structure

```text
justfile
README.md
LICENSE
requirements-dev.txt
tests/
tools/
  featurization_dashboard/
    app.py
    dashboard_math.py
    export_static.py
    requirements.txt
    static/
      index.html
```

## Notes on the estimates

- The total compute curves are based on the number of image sets and the number of cores sampled over a broad range.
- The “Time per Plate” chart is logarithmic so long-tail scaling is easier to see.
- The IBP estimates assume the number of plates can be processed in parallel up to available resources.
- The feature-count control in the sidebar now feeds the IBP estimate directly.

## License

This project is licensed under the BSD-3-Clause license. See `LICENSE` for the full text.
