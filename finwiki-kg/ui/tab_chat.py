"""ui/tab_chat.py — Chat tab: graph-grounded Q&A with full reasoning trace."""
from typing import List

import requests
import streamlit as st

from ui.components import (
    render_causal_context, render_reasoning_trace, render_source_list,
)


def render(api_url: str) -> None:
    st.header("💬 Financial Knowledge Q&A")

    # ── Sidebar filters ───────────────────────────────────────────────────────
    with st.sidebar:
        st.subheader("Query Settings")
        domain_options = ["All", "banking", "securities", "insurance", "derivatives", "regulatory", "accounting"]
        domain_filter       = st.selectbox("Domain", domain_options)
        confidence_threshold = st.slider("Min. Confidence", 0.5, 1.0, 0.7, 0.05)
        max_hops             = st.slider("Max Reasoning Hops", 1, 6, 3)
        show_contested       = st.toggle("Include Contested Assertions", value=True)
        show_derived         = st.toggle("Show Derived Conclusions", value=True)

    # ── Message history ───────────────────────────────────────────────────────
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
            if msg.get("contested_warnings"):
                for w in msg["contested_warnings"]:
                    st.warning(w)
            if msg.get("consistency_status") == "contradictory":
                st.error("⚠️ Consistency conflict detected in reasoning chain")
            if msg.get("reasoning_trace"):
                with st.expander("🧠 Reasoning Trace"):
                    render_reasoning_trace(msg["reasoning_trace"])
            if msg.get("causal_context"):
                with st.expander("⚡ Causal Context"):
                    render_causal_context(msg["causal_context"])
            if msg.get("sources"):
                render_source_list(msg["sources"])

    # ── Chat input ────────────────────────────────────────────────────────────
    if prompt := st.chat_input("Ask a question about financial regulations, risk, or compliance…"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Traversing knowledge graph…"):
                try:
                    payload = {
                        "question":         prompt,
                        "domain":           None if domain_filter == "All" else domain_filter,
                        "min_confidence":   confidence_threshold,
                        "max_hops":         max_hops,
                        "include_contested": show_contested,
                        "include_derived":  show_derived,
                    }
                    resp = requests.post(f"{api_url}/query", json=payload, timeout=90)
                    resp.raise_for_status()
                    data = resp.json()

                    st.write(data["answer"])

                    if data.get("contested_warnings"):
                        for w in data["contested_warnings"]:
                            st.warning(w)
                    if data.get("consistency_status") == "contradictory":
                        st.error("⚠️ Consistency conflict detected — see reasoning trace")

                    with st.expander("🧠 Reasoning Trace"):
                        render_reasoning_trace(data.get("reasoning_trace", []))
                    with st.expander("⚡ Causal Context"):
                        render_causal_context(data.get("causal_context", []))
                    render_source_list(data.get("sources", []))

                    cost = data.get("total_cost_usd", 0.0)
                    st.caption(f"Total API cost so far: ${cost:.4f}")

                    st.session_state.messages.append({
                        "role":              "assistant",
                        "content":           data["answer"],
                        "reasoning_trace":   data.get("reasoning_trace", []),
                        "causal_context":    data.get("causal_context", []),
                        "sources":           data.get("sources", []),
                        "contested_warnings": data.get("contested_warnings", []),
                        "consistency_status": data.get("consistency_status", "consistent"),
                    })

                except requests.exceptions.ConnectionError:
                    st.error(f"Cannot reach API at {api_url}. Is it running?")
                except requests.exceptions.Timeout:
                    st.error("Request timed out (>90s). The graph traversal may be too large.")
                except Exception as e:
                    st.error(f"API error: {e}")
