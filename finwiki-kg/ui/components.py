"""ui/components.py — Shared Streamlit UI components."""
from typing import Any, Dict, List

import streamlit as st

EPISTEMIC_COLORS: Dict[str, tuple] = {
    "authoritative": ("#1a7a1a", "AUTHORITATIVE"),
    "inferred":      ("#1a4a7a", "INFERRED"),
    "contested":     ("#cc6600", "CONTESTED ⚠️"),
    "deprecated":    ("#666666", "DEPRECATED"),
    "derived":       ("#6600cc", "DERIVED 🔗"),
    "orphaned":      ("#999999", "ORPHANED"),
}


def epistemic_badge(status: str) -> str:
    color, label = EPISTEMIC_COLORS.get(status.lower(), ("#333333", status.upper()))
    return (
        f'<span style="background:{color};color:#fff;padding:2px 7px;'
        f'border-radius:3px;font-size:0.78em;font-weight:600">{label}</span>'
    )


def render_assertion_card(assertion: dict, show_derivation: bool = True) -> None:
    badge = epistemic_badge(assertion.get("epistemic_status", "authoritative"))
    st.markdown(f"{badge} {assertion.get('claim_text', '')}", unsafe_allow_html=True)
    cols = st.columns(3)
    cols[0].caption(f"Source: {(assertion.get('source_document') or '')[:50]}")
    cols[1].caption(f"Confidence: {assertion.get('confidence', 0):.0%}")
    cols[2].caption(f"Domain: {assertion.get('domain', '')}")
    if show_derivation and assertion.get("derivation_chain"):
        with st.expander("🔗 Derivation chain"):
            for step in assertion["derivation_chain"]:
                st.text(f"  → {step.get('assertion_id', '')[:8]}… via {step.get('relation_type', '')}")


def render_reasoning_trace(trace: List[dict]) -> None:
    """Render reasoning trace with hop indentation and epistemic badges."""
    if not trace:
        st.info("No reasoning trace for this answer.")
        return
    for item in trace:
        assertion   = item.get("assertion", {})
        hop         = item.get("hop_distance", 0)
        chain_conf  = item.get("chain_confidence", 0.0)
        rel         = item.get("relation_used") or "direct"
        indent_html = "&nbsp;&nbsp;&nbsp;&nbsp;" * hop

        badge = epistemic_badge(assertion.get("epistemic_status", "authoritative"))
        st.markdown(
            f"{indent_html}{badge} "
            f"<b>[Hop {hop}]</b> via <code>{rel}</code> "
            f"<span style='color:#888;font-size:0.85em'>(chain conf: {chain_conf:.2f})</span>",
            unsafe_allow_html=True,
        )
        claim = assertion.get("claim_text", "")[:200]
        st.markdown(f"{indent_html}→ {claim}", unsafe_allow_html=True)

        # Show derivation chain inline if derived
        if assertion.get("epistemic_status") == "derived" and assertion.get("derivation_chain"):
            with st.expander("Show derivation chain"):
                for step in assertion["derivation_chain"]:
                    st.text(f"    → {step.get('assertion_id','')[:8]}… via {step.get('relation_type','')}")


def render_causal_context(causal_items: List[dict]) -> None:
    """Render causal context. Labeled as non-derivable context."""
    if not causal_items:
        st.info("No causal context found.")
        return
    st.caption("💡 Supporting context (not derived — empirical/causal, cannot produce logical conclusions)")
    for item in causal_items:
        assertion   = item.get("assertion", {})
        rel_type    = item.get("relation_type", "")
        mechanism   = item.get("mechanism") or ""
        strength    = item.get("strength") or ""
        icon = {"CAUSES": "⚡", "INHIBITS": "🚫", "CORRELATES_WITH": "↔️"}.get(rel_type, "•")
        claim = assertion.get("claim_text", "")[:150]
        st.markdown(f"{icon} **{rel_type}**: {claim}", unsafe_allow_html=False)
        if mechanism:
            st.caption(f"  Mechanism: {mechanism}  |  Strength: {strength}")


def render_source_list(sources: List[dict]) -> None:
    """Expandable source list."""
    with st.expander(f"📚 Sources ({len(sources)})"):
        for src in sources:
            badge = epistemic_badge(src.get("epistemic_status", "authoritative"))
            url   = src.get("source_url", "#")
            doc   = src.get("source_document", "unknown")
            st.markdown(f"{badge} [{doc}]({url})", unsafe_allow_html=True)
            st.caption(src.get("claim_text", "")[:150])


def scope_overlap_grid(scope_overlap: dict) -> None:
    """Render a 4-column scope overlap grid."""
    dimensions = ["temporal", "geographic", "organizational", "conditional"]
    icon_map   = {"overlaps": "🟡 overlaps", "no_overlap": "🟢 no overlap", "unknown": "⚪ unknown"}
    cols = st.columns(len(dimensions))
    for col, dim in zip(cols, dimensions):
        status = scope_overlap.get(dim, "unknown")
        label  = icon_map.get(status, f"⚪ {status}")
        col.metric(dim.capitalize(), label)
