set shell := ["bash", "-eu", "-o", "pipefail", "-c"]

default:
    @just all

help:
    @just --list

dashboard-install:
    #!/bin/bash
    cd "{{justfile_directory()}}" && python -m pip install --upgrade pip
    cd "{{justfile_directory()}}" && python -m pip install -r tools/featurization_dashboard/requirements.txt

dashboard-uninstall:
    #!/bin/bash
    cd "{{justfile_directory()}}" && python -m pip uninstall -r tools/featurization_dashboard/requirements.txt -y

dashboard-run:
    cd "{{justfile_directory()}}" && streamlit run tools/featurization_dashboard/app.py

dashboard-export-static:
    cd "{{justfile_directory()}}" && python tools/featurization_dashboard/export_static.py

dashboard-serve-static:
    #!/bin/bash
    @echo "Serving static site at http://localhost:8000 (CTRL-C to stop)"
    (cd "{{justfile_directory()}}/tools/featurization_dashboard" && python -m http.server 8000)

test:
    cd "{{justfile_directory()}}" && python -m pytest

pre-commit-run:
    cd "{{justfile_directory()}}" && pre-commit run --all-files

all: dashboard-install dashboard-run dashboard-export-static dashboard-serve-static
