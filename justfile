set shell := ["bash", "-eu", "-o", "pipefail", "-c"]

default:
	just --list

sync:
	uv sync

run:
	uv run streamlit run streamlit_app.py


all: sync run
