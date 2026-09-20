"""Cost dashboard — what the router saved, and what it cost in quality.

    streamlit run dashboard/app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.logging.database import Database  # noqa: E402
from src.models.registry import ModelRegistry  # noqa: E402

st.set_page_config(
    page_title="LLM Cost AutoPilot",
    page_icon="💸",
    layout="wide",
)

# Colour-blind-safe palette, consistent across every chart on the page.
TIER_COLORS = {"simple": "#4C9F70", "moderate": "#E1A33A", "complex": "#C2504B"}
ACTUAL_COLOR = "#4C78A8"
BASELINE_COLOR = "#B0B0B0"


@st.cache_resource
def get_database() -> Database:
    return Database()


@st.cache_data(ttl=10)
def load_data() -> dict:
    db = get_database()
    return {
        "stats": db.get_stats(),
        "verification": db.get_verification_stats(),
        "daily": db.get_daily_costs(days=30),
        "weekly": db.get_weekly_costs(weeks=12),
        "quality": db.get_quality_distribution(),
        "escalation_trend": db.get_escalation_rate_over_time(days=30),
        "recent": db.get_recent_logs(limit=100),
        "escalations": db.get_escalation_events(limit=50),
    }


def money(value: float) -> str:
    """Sub-cent costs need more precision than currency formatting gives."""
    if value == 0:
        return "$0.00"
    if abs(value) < 0.01:
        return f"${value:.6f}"
    return f"${value:,.2f}"


def md_money(value: float) -> str:
    """money() for markdown contexts.

    Streamlit renders captions as markdown, where a pair of dollar signs on
    one line is parsed as LaTeX math and the amounts vanish. Escape them.
    """
    return money(value).replace("$", r"\$")


data = load_data()
stats = data["stats"]

st.title("💸 LLM Cost AutoPilot")
st.caption(
    "Every prompt is routed to the cheapest model capable of handling it, "
    "then verified against a high-tier reference."
)

if stats["total_requests"] == 0:
    st.info(
        "No requests logged yet. Generate some traffic with "
        "`python scripts/demo_phase3.py`, or start the API and POST to "
        "`/v1/completions`."
    )
    st.stop()


# ── The money shot ──────────────────────────────────────────────────────

reduction = stats["savings_percentage"]

st.markdown(
    f"""
    <div style="
        padding: 2rem 1rem; margin: 1rem 0 2rem 0; border-radius: 12px;
        background: linear-gradient(135deg, #1b5e3f 0%, #2e8b57 100%);
        text-align: center; color: white;">
        <div style="font-size: 1rem; opacity: .85; letter-spacing: .12em;
                    text-transform: uppercase;">Cost reduction vs all-{'GPT-4o'}</div>
        <div style="font-size: 5rem; font-weight: 700; line-height: 1.1;">
            {reduction:.1f}%
        </div>
        <div style="font-size: 1.15rem; opacity: .95;">
            {money(stats['total_savings'])} saved across
            {stats['total_requests']:,} requests
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric(
    "Total spend",
    money(stats["total_cost"]),
    help="Routed-model spend plus the cost of re-running escalated requests",
)
col2.metric(
    "— of which escalations",
    money(stats["escalation_cost"]),
    help=(
        "Extra spend from re-running failed requests on a higher tier. "
        "This is counted against the savings, not excluded from them."
    ),
)
col3.metric(
    "Baseline (all top-tier)",
    money(stats["total_hypothetical_cost"]),
    help="What these same requests would have cost on the most expensive model",
)
col4.metric("Avg latency", f"{stats['avg_latency_ms']:.0f} ms")
col5.metric(
    "Avg quality",
    f"{stats['avg_quality_score']:.2f} / 5" if stats["avg_quality_score"] else "—",
    help="Mean verifier score across sampled requests",
)

if stats["escalation_cost"] > 0:
    overhead = stats["escalation_cost"] / stats["routed_cost"] * 100
    st.caption(
        f"Escalations added {overhead:.0f}% on top of the routed-model spend "
        f"({md_money(stats['routed_cost'])} → {md_money(stats['total_cost'])}). "
        f"The headline figure above is net of that overhead."
    )


# ── Cost over time ──────────────────────────────────────────────────────

st.subheader("Cost over time")
granularity = st.radio(
    "Granularity", ["Daily", "Weekly"], horizontal=True, label_visibility="collapsed"
)

if granularity == "Daily":
    frame = pd.DataFrame(data["daily"])
    x_field = "date"
else:
    frame = pd.DataFrame(data["weekly"])
    x_field = "week"

if frame.empty:
    st.info("Not enough history yet.")
else:
    frame = frame.sort_values(x_field)
    figure = go.Figure()
    figure.add_bar(
        x=frame[x_field],
        y=frame["hypothetical_cost"],
        name="If everything used the top-tier model",
        marker_color=BASELINE_COLOR,
    )
    figure.add_bar(
        x=frame[x_field],
        y=frame["actual_cost"],
        name="Actual cost",
        marker_color=ACTUAL_COLOR,
    )
    figure.update_layout(
        barmode="overlay",
        height=380,
        yaxis_title="Cost (USD)",
        xaxis_title="",
        legend=dict(orientation="h", y=1.12, x=0),
        margin=dict(t=40, b=10, l=10, r=10),
    )
    figure.update_traces(opacity=0.85)
    st.plotly_chart(figure, use_container_width=True)

    total_saved = frame["hypothetical_cost"].sum() - frame["actual_cost"].sum()
    st.caption(
        f"The grey bars are the counterfactual. The gap between them and the "
        f"blue bars is the saving: {md_money(total_saved)} over this window."
    )


# ── Distributions ───────────────────────────────────────────────────────

left, right = st.columns(2)

with left:
    st.subheader("Which models handled the traffic")
    dist = stats["model_distribution"]
    if dist:
        registry = ModelRegistry()
        frame = pd.DataFrame(
            [{"model": k, "requests": v} for k, v in dist.items()]
        ).sort_values("requests", ascending=False)

        pie = px.pie(
            frame,
            names="model",
            values="requests",
            hole=0.45,
            color_discrete_sequence=px.colors.qualitative.Safe,
        )
        pie.update_traces(textposition="inside", textinfo="percent+label")
        pie.update_layout(height=340, margin=dict(t=10, b=10, l=10, r=10))
        st.plotly_chart(pie, use_container_width=True)

        # The share of traffic that never touched a premium model.
        cheap = sum(
            count
            for name, count in dist.items()
            if name in registry and registry.get(name).quality_tier != "high"
        )
        share = cheap / sum(dist.values()) * 100
        st.caption(f"{share:.0f}% of requests were served without a top-tier model.")

with right:
    st.subheader("Quality score distribution")
    scores = data["quality"]
    if not scores:
        st.info("No requests have been verified yet.")
    else:
        hist = px.histogram(
            pd.DataFrame({"score": scores}),
            x="score",
            nbins=20,
            color_discrete_sequence=[ACTUAL_COLOR],
        )
        hist.add_vline(
            x=3.5,
            line_dash="dash",
            line_color="#C2504B",
            annotation_text="default bar",
            annotation_position="top",
        )
        hist.update_layout(
            height=340,
            xaxis_title="Verifier score (1-5)",
            yaxis_title="Requests",
            margin=dict(t=10, b=10, l=10, r=10),
        )
        st.plotly_chart(hist, use_container_width=True)
        below = sum(1 for s in scores if s < 3.5)
        st.caption(
            f"{below} of {len(scores)} verified requests fell below the default "
            f"3.5 bar. Per-use-case bars differ — extraction is held to 4.5."
        )


# ── Escalation rate over time ───────────────────────────────────────────

st.subheader("Escalation rate over time")
trend = pd.DataFrame(data["escalation_trend"])

if trend.empty or trend["verified_count"].sum() == 0:
    st.info("No verified requests yet, so there is no escalation rate to plot.")
else:
    line = go.Figure()
    line.add_scatter(
        x=trend["date"],
        y=trend["escalation_rate"],
        mode="lines+markers",
        name="Escalation rate",
        line=dict(color="#C2504B", width=2),
    )
    line.update_layout(
        height=320,
        yaxis_title="% of verified requests escalated",
        xaxis_title="",
        margin=dict(t=20, b=10, l=10, r=10),
    )
    st.plotly_chart(line, use_container_width=True)
    st.caption(
        "Measured against verified requests, not all requests — unsampled "
        "ones were never given the chance to escalate, and including them "
        "would understate the true rate. A falling line means the classifier "
        "is learning from its mistakes."
    )

verification = data["verification"]
m1, m2, m3, m4 = st.columns(4)
m1.metric("Verified", f"{verification['verified_count']:,}")
m2.metric("Passed", f"{verification['pass_rate']:.0f}%")
m3.metric("Escalations", f"{verification['escalation_count']:,}")
m4.metric(
    "Escalation spend",
    money(verification["escalation_cost_delta"]),
    help="Extra cost incurred re-running failed requests on a higher tier",
)


# ── Audit trail ─────────────────────────────────────────────────────────

st.subheader("Audit trail")
tab_requests, tab_escalations = st.tabs(["Recent requests", "Escalations"])

with tab_requests:
    recent = pd.DataFrame(data["recent"])
    if recent.empty:
        st.info("No requests logged.")
    else:
        columns = [
            "timestamp", "prompt_preview", "use_case", "complexity_tier",
            "routed_model", "cost", "latency_ms", "quality_score",
            "was_escalated",
        ]
        st.dataframe(
            recent[[c for c in columns if c in recent.columns]],
            use_container_width=True,
            hide_index=True,
        )

with tab_escalations:
    escalations = pd.DataFrame(data["escalations"])
    if escalations.empty:
        st.info("No escalations recorded.")
    else:
        columns = [
            "timestamp", "use_case", "original_model", "escalated_model",
            "cost_delta", "quality_gap", "returned_to_user", "trigger_reason",
        ]
        st.dataframe(
            escalations[[c for c in columns if c in escalations.columns]],
            use_container_width=True,
            hide_index=True,
        )

st.caption("Auto-refreshes every 10 seconds.")
