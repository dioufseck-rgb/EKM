"""ui/tab_conflicts.py — Conflict review tab with priority queue and action buttons."""
import requests
import streamlit as st

from ui.components import epistemic_badge, scope_overlap_grid


def render(api_url: str) -> None:
    st.header("⚖️ Conflict Review Queue")

    # ── Filters ───────────────────────────────────────────────────────────────
    cols = st.columns(4)
    with cols[0]:
        rel_type = st.selectbox("Type", ["All", "CONTRADICTS", "SUPERSEDES", "SPECIALIZES", "DUPLICATE"])
    with cols[1]:
        priority = st.selectbox("Priority", ["All", 1, 2, 3, 4])
    with cols[2]:
        review_status = st.selectbox("Status", ["All", "pending", "validated", "rejected", "deferred"])
    with cols[3]:
        domain = st.selectbox("Domain", ["All", "banking", "securities", "insurance", "derivatives"])

    params: dict = {"page": st.session_state.get("conflict_page", 1), "page_size": 20}
    if rel_type      != "All": params["relationship_type"] = rel_type
    if priority      != "All": params["priority"]          = priority
    if review_status != "All": params["review_status"]     = review_status
    if domain        != "All": params["domain"]            = domain

    # ── Fetch ─────────────────────────────────────────────────────────────────
    try:
        resp = requests.get(f"{api_url}/conflicts", params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
    except requests.exceptions.ConnectionError:
        st.error(f"Cannot reach API at {api_url}")
        return
    except Exception as e:
        st.error(f"Failed to load conflicts: {e}")
        return

    # ── Summary metrics ───────────────────────────────────────────────────────
    m = st.columns(4)
    m[0].metric("Total Conflicts", data["total"])
    m[1].metric("Page",            data["page"])
    m[2].metric("Per Page",        len(data["items"]))
    m[3].metric("Total Pages",     max(1, (data["total"] + 19) // 20))

    if not data["items"]:
        st.success("No conflicts match the current filters.")
        return

    # ── Conflict cards ────────────────────────────────────────────────────────
    priority_icons = {1: "🔴 Critical", 2: "🟠 High", 3: "🟡 Medium", 4: "🔵 Low"}

    for conflict in data["items"]:
        picon   = priority_icons.get(conflict["priority"], "⚪")
        rtype   = conflict["relationship_type"]
        claim_a = conflict["assertion_a"]["claim_text"][:80]

        with st.expander(f"{picon} [{rtype}] {claim_a}…  (conf: {conflict['confidence']:.0%})"):
            # Side-by-side assertion display
            col_a, col_b = st.columns(2)
            with col_a:
                st.subheader("Assertion A")
                badge = epistemic_badge(conflict["assertion_a"].get("epistemic_status", "authoritative"))
                st.markdown(badge, unsafe_allow_html=True)
                st.write(conflict["assertion_a"]["claim_text"])
                st.caption(f"Source: {conflict['assertion_a']['source_document']}")
            with col_b:
                st.subheader("Assertion B")
                badge = epistemic_badge(conflict["assertion_b"].get("epistemic_status", "authoritative"))
                st.markdown(badge, unsafe_allow_html=True)
                st.write(conflict["assertion_b"]["claim_text"])
                st.caption(f"Source: {conflict['assertion_b']['source_document']}")

            # Scope overlap grid
            st.markdown("**Scope Overlap**")
            scope_overlap_grid(conflict.get("scope_overlap", {}))

            # Details
            if conflict.get("conflicting_text"):
                st.markdown(f"**Conflicting text:** `{conflict['conflicting_text']}`")
            if conflict.get("reviewer_question"):
                st.info(f"❓ {conflict['reviewer_question']}")
            if conflict.get("explanation"):
                st.caption(f"Explanation: {conflict['explanation']}")

            st.caption(f"Status: {conflict['review_status']}")

            # Action buttons
            cid = conflict["conflict_id"]
            btn_cols = st.columns(3)
            with btn_cols[0]:
                if st.button("✅ Validate", key=f"val_{cid}"):
                    _update_conflict(api_url, cid, "validated")
            with btn_cols[1]:
                if st.button("❌ False Positive", key=f"fp_{cid}"):
                    _update_conflict(api_url, cid, "false_positive")
            with btn_cols[2]:
                if st.button("⏸ Defer", key=f"defer_{cid}"):
                    _update_conflict(api_url, cid, "deferred")

    # ── Pagination ────────────────────────────────────────────────────────────
    total_pages  = max(1, (data["total"] + 19) // 20)
    current_page = st.session_state.get("conflict_page", 1)
    pg_cols = st.columns(3)
    with pg_cols[0]:
        if current_page > 1 and st.button("← Previous"):
            st.session_state.conflict_page = current_page - 1
            st.rerun()
    with pg_cols[1]:
        st.caption(f"Page {current_page} of {total_pages}")
    with pg_cols[2]:
        if current_page < total_pages and st.button("Next →"):
            st.session_state.conflict_page = current_page + 1
            st.rerun()


def _update_conflict(api_url: str, conflict_id: str, resolution: str) -> None:
    try:
        resp = requests.patch(
            f"{api_url}/conflicts/{conflict_id}",
            json={"resolution": resolution},
            timeout=15,
        )
        resp.raise_for_status()
        st.success(f"Marked as {resolution}")
        st.rerun()
    except Exception as e:
        st.error(f"Update failed: {e}")
