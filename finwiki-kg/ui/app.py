"""ui/app.py — FinWiki Knowledge Graph main Streamlit application."""
import os

import requests
import streamlit as st

from ui.tab_chat      import render as render_chat
from ui.tab_conflicts import render as render_conflicts
from ui.tab_graph     import render as render_graph

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(
    page_title="FinWiki Knowledge Graph",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("📊 FinWiki Knowledge Graph")
st.caption(
    "Enterprise document intelligence over 2,500 Wikipedia financial services articles "
    "— pipeline correctness · reasoning quality · conflict intelligence · logical inference"
)

# ── API health check ──────────────────────────────────────────────────────────
try:
    health_resp = requests.get(f"{API_URL}/health", timeout=5)
    if health_resp.status_code == 200:
        st.success(f"✅ API connected: {API_URL}")
    else:
        st.error(f"❌ API returned HTTP {health_resp.status_code}")
except Exception:
    st.warning(f"⚠️ Cannot reach API at {API_URL} — some features will be unavailable.")

# ── Navigation ────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["💬 Chat", "⚖️ Conflicts", "🕸️ Graph"])

with tab1:
    render_chat(API_URL)

with tab2:
    render_conflicts(API_URL)

with tab3:
    render_graph(API_URL)
