from pathlib import Path
import sys

import streamlit as st


ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

try:
    import featurization_dashboard.app  # noqa: F401
except Exception as exc:
    st.set_page_config(layout="wide", page_title="Featurization Dashboard")
    st.error("The dashboard failed to start in this deployment.")
    st.exception(exc)
