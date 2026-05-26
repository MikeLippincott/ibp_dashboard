from pathlib import Path
import runpy
import sys

import streamlit as st


ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

try:
    runpy.run_path(str(SRC / "featurization_dashboard" / "app.py"), run_name="__main__")
except Exception as exc:
    st.set_page_config(layout="wide", page_title="Featurization Dashboard")
    st.error("The dashboard failed to start in this deployment.")
    st.exception(exc)
