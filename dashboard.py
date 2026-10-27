"""
dashboard.py
-------------
Interactive Streamlit Dashboard — Adaptive AI-Driven CPU Scheduler.

New in this version
───────────────────
• True Runtime Adaptation  via TrueAdaptiveScheduler (policy switches mid-sim).
• Adaptation Timeline Log  showing every policy-transition event.
• Policy-coloured Gantt    (SRTF=indigo, RR=pink, Priority=amber).
• Workload Input Modes:
    1. Random Generator   – sliders (original behaviour)
    2. Manual Table Input – st.data_editor (add/edit/delete rows)
    3. Classic OS Presets – pre-built scenario dropdown

Run:
    streamlit run dashboard.py
"""

import sys
import os

_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np

from core.process import Process
from core.simulation_engine import SimulationEngine, generate_processes
from scheduler.adaptive_scheduler import (
    TrueAdaptiveScheduler,
    POLICY_COLORS,
    POLICY_SRTF,
    POLICY_RR,
    POLICY_PRIORITY,
)
from metrics.calculator import (
    processes_to_dataframe,
    comparison_dataframe,
)

# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Adaptive AI CPU Scheduler",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# CSS — dark glassmorphism
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background: #0a0e1a;
    color: #e2e8f0;
}
.hero-header {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    font-size: 2.8rem; font-weight: 800; line-height: 1.2; margin-bottom: 0.2rem;
}
.hero-sub { color: #94a3b8; font-size: 1.05rem; font-weight: 400; margin-bottom: 2rem; }

/* Metric cards */
.metric-card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.10);
    border-radius: 16px; padding: 1.4rem 1.6rem;
    backdrop-filter: blur(10px);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}
.metric-card:hover { transform: translateY(-3px); box-shadow: 0 12px 40px rgba(102,126,234,0.25); }
.metric-label { font-size: 0.78rem; font-weight: 600; color: #64748b;
    text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 0.4rem; }
.metric-value { font-size: 1.9rem; font-weight: 700; line-height: 1; }
.metric-sub   { font-size: 0.82rem; color: #94a3b8; margin-top: 0.3rem; }
.card-purple .metric-value { color: #a78bfa; }
.card-cyan   .metric-value { color: #67e8f9; }
.card-green  .metric-value { color: #6ee7b7; }
.card-orange .metric-value { color: #fbbf24; }
.card-pink   .metric-value { color: #f9a8d4; }

/* Section heading */
.section-heading {
    font-size: 1.25rem; font-weight: 700; color: #e2e8f0;
    border-left: 4px solid #667eea; padding-left: 0.75rem;
    margin: 1.8rem 0 1rem 0;
}

/* Adaptation log row colours */
.adapt-srtf     { background: rgba(102,126,234,0.15); border-left: 3px solid #667eea; }
.adapt-rr       { background: rgba(240,147,251,0.15); border-left: 3px solid #f093fb; }
.adapt-priority { background: rgba(251,191,36,0.15);  border-left: 3px solid #fbbf24; }

/* Justification box */
.justification-box {
    background: rgba(102,126,234,0.12); border: 1px solid rgba(102,126,234,0.30);
    border-radius: 12px; padding: 1rem 1.4rem; font-size: 0.92rem;
    color: #c7d2fe; line-height: 1.7;
}

/* Policy badges */
.badge {
    display: inline-block; border-radius: 6px; padding: 2px 10px;
    font-size: 0.78rem; font-weight: 600; margin: 2px;
}
.badge-srtf     { background: rgba(102,126,234,0.30); color: #a5b4fc; }
.badge-rr       { background: rgba(240,147,251,0.30); color: #f5d0fe; }
.badge-priority { background: rgba(251,191,36,0.30);  color: #fde68a; }

/* Sidebar */
[data-testid="stSidebar"] {
    background: rgba(15,20,40,0.98);
    border-right: 1px solid rgba(255,255,255,0.06);
}
/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #667eea, #764ba2);
    color: white; border: none; border-radius: 12px;
    padding: 0.7rem 2rem; font-size: 1rem; font-weight: 600;
    width: 100%; transition: opacity 0.2s ease, transform 0.1s ease;
}
.stButton > button:hover { opacity: 0.90; transform: translateY(-1px); }

/* Confidence bar */
.conf-bar-bg {
    background: rgba(255,255,255,0.08); border-radius: 99px;
    height: 8px; margin-top: 8px; overflow: hidden;
}
.conf-bar-fill {
    height: 8px; border-radius: 99px;
    background: linear-gradient(90deg, #667eea, #f093fb);
    transition: width 0.8s ease;
}
/* winner-cell highlight for table */
.winner-cell { background: rgba(110,231,183,0.18); color: #6ee7b7; font-weight: 700; }
.composite-badge {
    display:inline-block; background: linear-gradient(135deg,#667eea,#f093fb);
    color:#fff; border-radius:8px; padding:2px 10px;
    font-size:0.78rem; font-weight:700; letter-spacing:.04em;
}
hr { border-color: rgba(255,255,255,0.07); margin: 1.5rem 0; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Constants: Classic OS Presets
# ─────────────────────────────────────────────────────────────────────────────

PRESETS: dict = {
    "Convoy Effect Demo": [
        # One long process blocks many short ones (classic FCFS problem)
        {"PID": 1, "Arrival Time": 0, "Burst Time": 30, "Priority": 5, "I/O Frequency": 0.00},
        {"PID": 2, "Arrival Time": 1, "Burst Time": 2,  "Priority": 5, "I/O Frequency": 0.10},
        {"PID": 3, "Arrival Time": 2, "Burst Time": 2,  "Priority": 5, "I/O Frequency": 0.10},
        {"PID": 4, "Arrival Time": 3, "Burst Time": 2,  "Priority": 5, "I/O Frequency": 0.10},
        {"PID": 5, "Arrival Time": 4, "Burst Time": 2,  "Priority": 5, "I/O Frequency": 0.10},
    ],
    "Mixed Batch & Interactive": [
        # Long CPU jobs interleaved with short I/O-heavy interactive tasks
        {"PID": 1, "Arrival Time": 0, "Burst Time": 20, "Priority": 3, "I/O Frequency": 0.05},
        {"PID": 2, "Arrival Time": 1, "Burst Time": 3,  "Priority": 4, "I/O Frequency": 0.80},
        {"PID": 3, "Arrival Time": 2, "Burst Time": 15, "Priority": 3, "I/O Frequency": 0.05},
        {"PID": 4, "Arrival Time": 3, "Burst Time": 2,  "Priority": 4, "I/O Frequency": 0.90},
        {"PID": 5, "Arrival Time": 4, "Burst Time": 12, "Priority": 3, "I/O Frequency": 0.10},
        {"PID": 6, "Arrival Time": 5, "Burst Time": 4,  "Priority": 4, "I/O Frequency": 0.75},
    ],
    "High-Priority Interruption": [
        # An urgent (priority=1) process arrives mid-execution
        {"PID": 1, "Arrival Time": 0, "Burst Time": 15, "Priority": 5, "I/O Frequency": 0.10},
        {"PID": 2, "Arrival Time": 1, "Burst Time": 10, "Priority": 4, "I/O Frequency": 0.20},
        {"PID": 3, "Arrival Time": 3, "Burst Time": 8,  "Priority": 1, "I/O Frequency": 0.10},
        {"PID": 4, "Arrival Time": 5, "Burst Time": 6,  "Priority": 3, "I/O Frequency": 0.30},
        {"PID": 5, "Arrival Time": 7, "Burst Time": 4,  "Priority": 2, "I/O Frequency": 0.20},
    ],
}

PRESET_DESCRIPTIONS = {
    "Convoy Effect Demo": (
        "P1 (burst=30) monopolises the CPU; P2–P5 (burst=2 each) wait behind it. "
        "FCFS performs worst here. The adaptive scheduler detects the short queue "
        "and switches to SRTF to clear the backlog quickly."
    ),
    "Mixed Batch & Interactive": (
        "Alternating heavy CPU processes and light I/O-intensive interactive tasks. "
        "The adaptive scheduler switches between SRTF (for CPU jobs) and Round Robin "
        "(for I/O jobs) as the queue composition changes."
    ),
    "High-Priority Interruption": (
        "A critical process (priority=1) arrives at t=3 mid-simulation. "
        "The adaptive scheduler detects the urgent task and switches to "
        "Priority Scheduling to preempt lower-priority work immediately."
    ),
}


def _default_manual_df() -> pd.DataFrame:
    return pd.DataFrame([
        {"PID": 1, "Arrival Time": 0,  "Burst Time": 5,  "Priority": 3, "I/O Frequency": 0.10},
        {"PID": 2, "Arrival Time": 2,  "Burst Time": 8,  "Priority": 2, "I/O Frequency": 0.20},
        {"PID": 3, "Arrival Time": 4,  "Burst Time": 3,  "Priority": 5, "I/O Frequency": 0.50},
        {"PID": 4, "Arrival Time": 6,  "Burst Time": 6,  "Priority": 1, "I/O Frequency": 0.30},
        {"PID": 5, "Arrival Time": 8,  "Burst Time": 4,  "Priority": 4, "I/O Frequency": 0.60},
    ])


def _df_to_processes(df: pd.DataFrame) -> list:
    """Convert a user-edited DataFrame into a list of Process objects."""
    processes = []
    seen_pids = set()
    for _, row in df.iterrows():
        try:
            pid  = int(row["PID"])
            arr  = max(0, int(row["Arrival Time"]))
            burst= max(1, int(row["Burst Time"]))
            prio = max(1, min(10, int(row["Priority"])))
            io   = max(0.0, min(1.0, float(row["I/O Frequency"])))
            if pid in seen_pids:
                continue
            seen_pids.add(pid)
            processes.append(Process(pid=pid, arrival_time=arr, burst_time=burst,
                                     priority=prio, io_frequency=io))
        except Exception:
            pass
    return sorted(processes, key=lambda p: (p.arrival_time, p.pid))

# ─────────────────────────────────────────────────────────────────────────────
# Colour helpers
# ─────────────────────────────────────────────────────────────────────────────

PROCESS_COLORS = [
    "#667eea","#f093fb","#4ade80","#fbbf24","#67e8f9",
    "#f87171","#a78bfa","#34d399","#fb923c","#38bdf8",
    "#e879f9","#86efac","#fde68a","#7dd3fc","#fca5a5",
]

def _pid_color(pid: int) -> str:
    return PROCESS_COLORS[(pid - 1) % len(PROCESS_COLORS)]


# ─────────────────────────────────────────────────────────────────────────────
# Gantt chart  (handles both 3-tuple classic and 4-tuple adaptive timelines)
# ─────────────────────────────────────────────────────────────────────────────

def build_gantt(
    timeline,
    processes,
    algorithm_name: str,
    color_by_policy: bool = False,
    adaptation_log: list = None,
) -> go.Figure:
    """
    Build an interactive Plotly horizontal bar (Gantt) chart.

    color_by_policy=True  → bars coloured by policy (SRTF/RR/Priority)
    color_by_policy=False → bars coloured by PID (classic scheduler view)
    """
    if not timeline:
        fig = go.Figure()
        fig.update_layout(title="No execution data", template="plotly_dark")
        return fig

    pid_map = {p.pid: p for p in processes}
    has_policy = len(timeline[0]) == 4

    rows = []
    for entry in timeline:
        if has_policy:
            pid, start, end, pol = entry
        else:
            pid, start, end = entry
            pol = algorithm_name
        proc = pid_map.get(pid)
        rows.append(dict(pid=pid, start=start, end=end, duration=end-start,
                         policy=pol, io_freq=round(proc.io_frequency, 3) if proc else 0))

    df = pd.DataFrame(rows)
    fig = go.Figure()

    # One trace per (pid, policy) combination to keep legend clean
    seen_legend = set()
    for _, row in df.iterrows():
        pid, pol = row["pid"], row["policy"]
        color = POLICY_COLORS.get(pol, _pid_color(pid)) if color_by_policy else _pid_color(pid)
        legend_key = pol if color_by_policy else f"P{pid}"
        show_leg = legend_key not in seen_legend
        if show_leg:
            seen_legend.add(legend_key)

        fig.add_trace(go.Bar(
            x=[row["duration"]],
            y=[f"P{pid}"],
            base=[row["start"]],
            orientation="h",
            marker=dict(color=color, opacity=0.88,
                        line=dict(color="rgba(0,0,0,0.3)", width=1)),
            text=f"  P{pid} [{row['start']}→{row['end']}]" if row["duration"] > 1 else "",
            textposition="inside",
            insidetextanchor="start",
            hovertemplate=(
                f"<b>P{pid}</b><br>Policy: {pol}<br>"
                f"Start: {row['start']} | End: {row['end']}<br>"
                f"Duration: {row['duration']} ticks<br>I/O Freq: {row['io_freq']}"
                "<extra></extra>"
            ),
            name=legend_key,
            showlegend=show_leg,
        ))

    # Policy transition markers (vertical dashed lines)
    if color_by_policy and adaptation_log:
        for ev in adaptation_log[1:]:          # skip first (initial policy)
            fig.add_vline(
                x=ev["time"],
                line_dash="dash",
                line_color="rgba(255,255,255,0.35)",
                line_width=1.5,
                annotation_text=f"→{ev['policy'][:4]}",
                annotation_font=dict(color="rgba(255,255,255,0.65)", size=10),
                annotation_position="top",
            )

    unique_pids = sorted(df["pid"].unique())
    y_labels = [f"P{p}" for p in unique_pids]

    fig.update_layout(
        title=dict(text=f"<b>Gantt Chart — {algorithm_name}</b>",
                   font=dict(size=16, color="#e2e8f0"), x=0.0),
        xaxis=dict(title="Time (ticks)", showgrid=True,
                   gridcolor="rgba(255,255,255,0.07)", color="#94a3b8"),
        yaxis=dict(title="Process", categoryorder="array",
                   categoryarray=y_labels, color="#94a3b8"),
        barmode="stack",
        paper_bgcolor="rgba(10,14,26,0)",
        plot_bgcolor="rgba(10,14,26,0.4)",
        font=dict(family="Inter, sans-serif", color="#e2e8f0"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    xanchor="right", x=1, font=dict(size=11)),
        height=max(300, 70 + len(unique_pids) * 44),
        margin=dict(l=60, r=30, t=70, b=50),
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Comparison bar chart
# ─────────────────────────────────────────────────────────────────────────────

def build_comparison_chart(all_results: dict) -> go.Figure:
    """Three grouped metrics: AWT, ATAT, and Avg Response Time."""
    short_names = {
        "FCFS":         "FCFS",
        "SJF":          "SJF (NP)",
        "SRTF":         "SRTF",
        "RR":           "Round Robin",
        "Priority":     "Priority",
        "TrueAdaptive": "AI Adaptive",
    }
    labels, awt_vals, atat_vals, art_vals = [], [], [], []
    for key, res in all_results.items():
        if "error" in res:
            continue
        labels.append(short_names.get(key, res.get("algorithm", key)))
        awt_vals.append(res.get("avg_waiting_time", 0.0))
        atat_vals.append(res.get("avg_turnaround_time", 0.0))
        art_vals.append(res.get("avg_response_time", 0.0))

    n = len(labels)
    is_adaptive = [l == "AI Adaptive" for l in labels]
    colors_awt  = ["#f093fb" if a else "#667eea" for a in is_adaptive]
    colors_atat = ["#fbbf24" if a else "#4ade80" for a in is_adaptive]
    colors_art  = ["#f87171" if a else "#60a5fa" for a in is_adaptive]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="Avg Waiting Time", x=labels, y=awt_vals,
        marker=dict(color=colors_awt, opacity=0.88, line=dict(width=0)),
        text=[f"{v:.2f}" for v in awt_vals], textposition="outside",
        textfont=dict(size=10, color="#e2e8f0"),
    ))
    fig.add_trace(go.Bar(
        name="Avg Turnaround Time", x=labels, y=atat_vals,
        marker=dict(color=colors_atat, opacity=0.88, line=dict(width=0)),
        text=[f"{v:.2f}" for v in atat_vals], textposition="outside",
        textfont=dict(size=10, color="#e2e8f0"),
    ))
    fig.add_trace(go.Bar(
        name="Avg Response Time", x=labels, y=art_vals,
        marker=dict(color=colors_art, opacity=0.88, line=dict(width=0)),
        text=[f"{v:.2f}" for v in art_vals], textposition="outside",
        textfont=dict(size=10, color="#e2e8f0"),
    ))

    fig.update_layout(
        barmode="group",
        title=dict(
            text="<b>Scheduler Comparison — AWT | ATAT | Response Time</b>",
            font=dict(size=15, color="#e2e8f0"), x=0.0,
        ),
        xaxis=dict(color="#94a3b8", tickfont=dict(size=12)),
        yaxis=dict(
            title="Time (ticks)", color="#94a3b8",
            showgrid=True, gridcolor="rgba(255,255,255,0.07)",
        ),
        paper_bgcolor="rgba(10,14,26,0)",
        plot_bgcolor="rgba(10,14,26,0.4)",
        font=dict(family="Inter, sans-serif", color="#e2e8f0"),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02,
            xanchor="right", x=1, font=dict(size=12),
        ),
        height=440, margin=dict(l=60, r=30, t=80, b=60),
    )
    return fig


def build_composite_chart(all_results: dict, n_processes: int) -> go.Figure:
    """Horizontal bar chart of composite score per algorithm (lower = better)."""
    short_names = {
        "FCFS":         "FCFS",
        "SJF":          "SJF (NP)",
        "SRTF":         "SRTF",
        "RR":           "Round Robin",
        "Priority":     "Priority",
        "TrueAdaptive": "AI Adaptive",
    }
    labels, scores = [], []
    for key, res in all_results.items():
        if "error" in res:
            continue
        procs = res.get("processes", [])
        awt  = res.get("avg_waiting_time", 0.0)
        art  = res.get("avg_response_time", 0.0)
        cs_n = res.get("context_switches", 0) / max(1, n_processes)
        wts  = [p.waiting_time for p in procs] if procs else []
        if len(wts) >= 2:
            mean_wt = sum(wts) / len(wts)
            fstd = (sum((w - mean_wt) ** 2 for w in wts) / len(wts)) ** 0.5
        else:
            fstd = 0.0
        score = round(0.35 * awt + 0.35 * art + 0.15 * cs_n + 0.15 * fstd, 2)
        labels.append(short_names.get(key, res.get("algorithm", key)))
        scores.append(score)

    # sort ascending (lower = better)
    paired = sorted(zip(scores, labels))
    scores, labels = zip(*paired) if paired else ([], [])

    colors = [
        "linear-gradient(90deg,#667eea,#f093fb)" if l == "AI Adaptive"
        else "rgba(100,116,139,0.6)"
        for l in labels
    ]
    bar_colors = ["#f093fb" if l == "AI Adaptive" else "#475569" for l in labels]

    fig = go.Figure(go.Bar(
        x=list(scores), y=list(labels), orientation="h",
        marker=dict(color=bar_colors, opacity=0.9, line=dict(width=0)),
        text=[f"{s:.2f}" for s in scores], textposition="outside",
        textfont=dict(size=12, color="#e2e8f0"),
    ))
    fig.update_layout(
        title=dict(
            text="<b>Composite Score (AWT 35% + ART 35% + CS 15% + Fairness 15%)</b><br>"
                 "<sup style='color:#64748b'>Lower is better — Adaptive AI optimises this combined metric</sup>",
            font=dict(size=14, color="#e2e8f0"), x=0.0,
        ),
        xaxis=dict(title="Score (lower = better)", color="#94a3b8",
                   showgrid=True, gridcolor="rgba(255,255,255,0.07)"),
        yaxis=dict(color="#94a3b8", tickfont=dict(size=12)),
        paper_bgcolor="rgba(10,14,26,0)",
        plot_bgcolor="rgba(10,14,26,0.4)",
        font=dict(family="Inter, sans-serif", color="#e2e8f0"),
        height=320, margin=dict(l=120, r=80, t=90, b=50),
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Adaptation timeline log renderer
# ─────────────────────────────────────────────────────────────────────────────

def render_adaptation_log(adaptation_log: list):
    if not adaptation_log:
        st.info("No policy transitions — a single policy was maintained throughout the simulation.")
        return

    BADGE = {
        POLICY_SRTF:     '<span class="badge badge-srtf">SRTF</span>',
        POLICY_RR:       '<span class="badge badge-rr">Round Robin</span>',
        POLICY_PRIORITY: '<span class="badge badge-priority">Priority</span>',
    }

    rows_html = ""
    for i, ev in enumerate(adaptation_log):
        badge = BADGE.get(ev["policy"], ev["policy"])
        pids  = ", ".join(f"P{p}" for p in ev.get("queue_pids", []))
        bg    = "rgba(102,126,234,0.08)" if ev["policy"] == POLICY_SRTF else \
                "rgba(240,147,251,0.08)" if ev["policy"] == POLICY_RR else \
                "rgba(251,191,36,0.08)"
        rows_html += f"""
        <tr style="border-bottom:1px solid rgba(255,255,255,0.06);background:{bg};">
            <td style="padding:8px 12px;color:#94a3b8;font-variant-numeric:tabular-nums;">
                t = {ev['time']}
            </td>
            <td style="padding:8px 12px;">{badge}</td>
            <td style="padding:8px 12px;color:#94a3b8;">{ev.get('queue_size',0)} proc</td>
            <td style="padding:8px 12px;color:#64748b;font-size:0.82rem;">{pids}</td>
            <td style="padding:8px 12px;color:#c7d2fe;font-size:0.88rem;">{ev['reason']}</td>
        </tr>"""

    st.markdown(f"""
    <table style="width:100%;border-collapse:collapse;font-family:'Inter',sans-serif;">
        <thead>
            <tr style="border-bottom:2px solid rgba(255,255,255,0.12);">
                <th style="padding:8px 12px;text-align:left;color:#64748b;font-size:0.78rem;
                            text-transform:uppercase;letter-spacing:0.08em;">Time</th>
                <th style="padding:8px 12px;text-align:left;color:#64748b;font-size:0.78rem;
                            text-transform:uppercase;letter-spacing:0.08em;">Policy</th>
                <th style="padding:8px 12px;text-align:left;color:#64748b;font-size:0.78rem;
                            text-transform:uppercase;letter-spacing:0.08em;">Queue</th>
                <th style="padding:8px 12px;text-align:left;color:#64748b;font-size:0.78rem;
                            text-transform:uppercase;letter-spacing:0.08em;">Active PIDs</th>
                <th style="padding:8px 12px;text-align:left;color:#64748b;font-size:0.78rem;
                            text-transform:uppercase;letter-spacing:0.08em;">Reason</th>
            </tr>
        </thead>
        <tbody>{rows_html}</tbody>
    </table>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Metric card helpers
# ─────────────────────────────────────────────────────────────────────────────

def metric_card(label, value, sub="", css_class="card-purple"):
    return f"""
    <div class="metric-card {css_class}">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
        {"<div class='metric-sub'>" + sub + "</div>" if sub else ""}
    </div>"""

def confidence_bar(conf: float) -> str:
    return (f'<div class="conf-bar-bg">'
            f'<div class="conf-bar-fill" style="width:{int(conf*100)}%;"></div>'
            f'</div>')

def _highlight_col(col):
    if col.max() == col.min():
        return [""] * len(col)
    norm = (col - col.min()) / (col.max() - col.min())
    return [f"background-color:rgba({int(220*v+40*(1-v))},{int(40*v+180*(1-v))},80,0.30);"
            f"color:#e2e8f0;" for v in norm]

def _highlight_min(s):
    is_min = s == s.min()
    return ["background-color:rgba(110,231,183,0.20);color:#6ee7b7;font-weight:700"
            if v else "" for v in is_min]


# ─────────────────────────────────────────────────────────────────────────────
# ── SIDEBAR ──
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## ⚙️ Simulation Controls")
    st.markdown("---")

    input_mode = st.radio(
        "Workload Generation Mode",
        ["🎲 Random Generator", "✏️ Manual Table Input", "📋 Classic OS Presets"],
        help="Choose how to create the process batch for this simulation.",
    )

    st.markdown("---")

    # ── Mode 1: Random Generator ──────────────────────────────────────────
    if input_mode == "🎲 Random Generator":
        process_count = st.slider("Process Count",    min_value=3,  max_value=30, value=10)
        max_arrival   = st.slider("Max Arrival Time", min_value=0,  max_value=50, value=20)
        burst_min     = st.slider("Min Burst Time",   min_value=1,  max_value=20, value=2)
        burst_max     = st.slider("Max Burst Time",   min_value=2,  max_value=80, value=20)
        io_probability= st.slider("I/O Probability",  min_value=0.0,max_value=1.0,value=0.4, step=0.05)
        random_seed   = st.number_input("Random Seed (0=random)", min_value=0, max_value=9999, value=42)

    # ── Mode 2: Manual Table Input ────────────────────────────────────────
    elif input_mode == "✏️ Manual Table Input":
        st.markdown("**Edit the process table below.**")
        st.caption("Add rows with the ＋ button. All changes are applied on Run.")
        if "manual_df" not in st.session_state:
            st.session_state["manual_df"] = _default_manual_df()
        edited_df = st.data_editor(
            st.session_state["manual_df"],
            key="proc_table_editor",
            num_rows="dynamic",
            use_container_width=True,
            column_config={
                "PID":          st.column_config.NumberColumn("PID",          min_value=1,  max_value=99, step=1),
                "Arrival Time": st.column_config.NumberColumn("Arrival Time", min_value=0,  max_value=200,step=1),
                "Burst Time":   st.column_config.NumberColumn("Burst Time",   min_value=1,  max_value=100,step=1),
                "Priority":     st.column_config.NumberColumn("Priority",     min_value=1,  max_value=10, step=1),
                "I/O Frequency":st.column_config.NumberColumn("I/O Frequency",min_value=0.0,max_value=1.0,step=0.05,format="%.2f"),
            },
        )
        st.session_state["manual_df"] = edited_df

    # ── Mode 3: Classic OS Presets ────────────────────────────────────────
    else:
        preset_name = st.selectbox(
            "Select Scenario",
            options=list(PRESETS.keys()),
        )
        st.markdown(
            f'<div class="justification-box" style="margin-top:8px;">'
            f'📖 {PRESET_DESCRIPTIONS[preset_name]}</div>',
            unsafe_allow_html=True,
        )

    st.markdown("---")
    rr_quantum = st.slider(
        "Round Robin Base Quantum",
        min_value=1, max_value=20, value=4,
        help="Used as the base quantum; adaptive scheduler may adjust it dynamically.",
    )

    st.markdown("---")
    st.markdown("### 🤖 ML Model")
    retrain_btn = st.button("🔄 Retrain ML Model")
    if retrain_btn:
        with st.spinner("Generating dataset & training model…"):
            try:
                from ml.data_generator import generate_dataset
                from ml.train_model import train
                from ml.predictor import reload_model
                data_path  = os.path.join(_ROOT, "data", "workload_dataset.csv")
                model_path = os.path.join(_ROOT, "ml", "model.joblib")
                generate_dataset(save=True, output_path=data_path)
                train(data_path=data_path, model_path=model_path)
                reload_model()
                st.success("Model retrained successfully!")
            except Exception as exc:
                st.error(f"Training failed: {exc}")

    st.markdown("---")
    st.caption("Adaptive AI CPU Scheduler v2.0")
    st.caption("True Runtime Adaptation Edition")


# ─────────────────────────────────────────────────────────────────────────────
# ── HERO HEADER ──
# ─────────────────────────────────────────────────────────────────────────────

st.markdown('<div class="hero-header">🧠 Adaptive AI-Driven CPU Scheduler</div>',
            unsafe_allow_html=True)
st.markdown(
    '<div class="hero-sub">True Runtime Adaptation — scheduling policy switches '
    'dynamically mid-simulation based on live ready-queue characteristics. '
    'Powered by scikit-learn + Streamlit + Plotly.</div>',
    unsafe_allow_html=True,
)
st.markdown("---")


# ─────────────────────────────────────────────────────────────────────────────
# ── RUN BUTTON ──
# ─────────────────────────────────────────────────────────────────────────────

run_col, info_col = st.columns([1, 3])
with run_col:
    run_btn = st.button("▶ Run Simulation & AI Adaptation", use_container_width=True)
with info_col:
    if input_mode == "🎲 Random Generator":
        st.info(f"Ready to simulate **{process_count} random processes** | "
                f"Burst [{burst_min}–{burst_max}] | I/O prob {io_probability:.0%} | Q={rr_quantum}")
    elif input_mode == "✏️ Manual Table Input":
        st.info(f"Ready to simulate **{len(st.session_state.get('manual_df', _default_manual_df()))} "
                f"manually defined processes** | Base Q={rr_quantum}")
    else:
        st.info(f"Preset: **{preset_name}** | "
                f"{len(PRESETS[preset_name])} processes | Base Q={rr_quantum}")


# ─────────────────────────────────────────────────────────────────────────────
# ── SIMULATION ──
# ─────────────────────────────────────────────────────────────────────────────

if run_btn:
    with st.spinner("Running True Adaptive simulation + all classic schedulers…"):
        try:
            # Build process list based on input mode
            if input_mode == "🎲 Random Generator":
                seed = int(random_seed) if random_seed > 0 else None
                processes = generate_processes(
                    count=process_count,
                    max_arrival_time=max_arrival,
                    burst_time_range=(burst_min, burst_max),
                    io_probability=io_probability,
                    seed=seed,
                )
            elif input_mode == "✏️ Manual Table Input":
                processes = _df_to_processes(st.session_state["manual_df"])
                if not processes:
                    st.error("No valid processes in the table.")
                    st.stop()
            else:  # Preset
                rows = PRESETS[preset_name]
                processes = [
                    Process(
                        pid=r["PID"],
                        arrival_time=r["Arrival Time"],
                        burst_time=r["Burst Time"],
                        priority=r["Priority"],
                        io_frequency=r["I/O Frequency"],
                    )
                    for r in rows
                ]

            # True Adaptive scheduler
            scheduler = TrueAdaptiveScheduler(base_quantum=rr_quantum)
            adaptive_result = scheduler.run(processes, base_quantum=rr_quantum)

            # Classic schedulers
            engine = SimulationEngine(processes)
            all_classic = engine.run_all(time_quantum=rr_quantum)

            # Merge all results
            all_results = {**all_classic, "TrueAdaptive": adaptive_result}

        except Exception as exc:
            st.error(f"Simulation failed: {exc}")
            st.exception(exc)
            st.stop()

    st.session_state["adaptive_result"] = adaptive_result
    st.session_state["all_results"]     = all_results
    st.session_state["processes"]       = processes
    st.session_state["input_mode"]      = input_mode


# Gate: require a completed run
if "adaptive_result" not in st.session_state:
    st.markdown("""
    <div style="text-align:center;padding:4rem;color:#475569;">
        <div style="font-size:4rem;">🚀</div>
        <div style="font-size:1.2rem;font-weight:600;margin-top:1rem;">
            Configure parameters in the sidebar and click <b>Run Simulation</b>
        </div>
    </div>""", unsafe_allow_html=True)
    st.stop()

adaptive_result = st.session_state["adaptive_result"]
all_results     = st.session_state["all_results"]
processes       = st.session_state["processes"]
adaptation_log  = adaptive_result.get("adaptation_log", [])


# ─────────────────────────────────────────────────────────────────────────────
# ── SECTION 1: AI INSIGHT CARDS ──
# ─────────────────────────────────────────────────────────────────────────────

st.markdown('<div class="section-heading">🤖 AI Classification & Runtime Adaptation</div>',
            unsafe_allow_html=True)

c1, c2, c3, c4, c5 = st.columns(5)

label     = adaptive_result.get("workload_label", "Mixed")
class_emoji = {"CPU-bound":"⚡","I/O-bound":"💾","Mixed":"🔀"}.get(label, "🧠")
n_switches  = max(0, len(adaptation_log) - 1)

with c1:
    st.markdown(metric_card("Workload Class", f"{class_emoji} {label}",
                            "Initial ML prediction", "card-purple"), unsafe_allow_html=True)
with c2:
    st.markdown(metric_card("Policy Switches", str(n_switches),
                            "during simulation", "card-cyan"), unsafe_allow_html=True)
with c3:
    policies_used = list(dict.fromkeys(e["policy"] for e in adaptation_log))
    pol_str = " → ".join(policies_used) if policies_used else "—"
    st.markdown(metric_card("Policies Used", str(len(policies_used)),
                            pol_str[:30], "card-green"), unsafe_allow_html=True)
with c4:
    awt = adaptive_result.get("avg_waiting_time", 0.0)
    st.markdown(metric_card("Avg Waiting Time", f"{awt:.2f}",
                            "ticks (adaptive)", "card-orange"), unsafe_allow_html=True)
with c5:
    atat = adaptive_result.get("avg_turnaround_time", 0.0)
    st.markdown(metric_card("Avg Turnaround", f"{atat:.2f}",
                            "ticks (adaptive)", "card-pink"), unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)
st.markdown(
    f'<div class="justification-box">💡 <b>Adaptive Logic:</b> '
    f'{adaptive_result.get("justification", "")}</div>',
    unsafe_allow_html=True,
)

st.markdown("---")


# ─────────────────────────────────────────────────────────────────────────────
# ── SECTION 2: GANTT CHART ──
# ─────────────────────────────────────────────────────────────────────────────

st.markdown('<div class="section-heading">📊 Execution Gantt Chart</div>',
            unsafe_allow_html=True)

gantt_tab1, gantt_tab2 = st.tabs(["🧠 True Adaptive AI (by Policy)", "🔁 Compare All Algorithms"])

with gantt_tab1:
    adaptive_fig = build_gantt(
        timeline=adaptive_result["timeline"],
        processes=adaptive_result["processes"],
        algorithm_name="True Adaptive AI Scheduler",
        color_by_policy=True,
        adaptation_log=adaptation_log,
    )
    st.plotly_chart(adaptive_fig, use_container_width=True, key="gantt_adaptive")

    # Policy legend chips
    if adaptation_log:
        pols_used = list(dict.fromkeys(e["policy"] for e in adaptation_log))
        chips = "".join(
            f'<span class="badge badge-{p.lower().replace(" ","")}">'
            f'{p}</span>' for p in pols_used
        )
        st.markdown(f"**Policies active in this run:** {chips}", unsafe_allow_html=True)

with gantt_tab2:
    def _gantt_label(k):
        if k == "TrueAdaptive":
            return "🧠 True Adaptive AI"
        return all_results[k].get("algorithm", k)

    algo_select = st.selectbox(
        "Select Algorithm to View Gantt",
        options=list(all_results.keys()),
        format_func=_gantt_label,
        key="gantt_algo_select",
    )
    sel_res = all_results[algo_select]
    if "error" in sel_res:
        st.error(f"Scheduler error: {sel_res['error']}")
    else:
        is_adaptive = algo_select == "TrueAdaptive"
        fig2 = build_gantt(
            timeline=sel_res["timeline"],
            processes=sel_res["processes"],
            algorithm_name=_gantt_label(algo_select),
            color_by_policy=is_adaptive,
            adaptation_log=sel_res.get("adaptation_log", []) if is_adaptive else [],
        )
        st.plotly_chart(fig2, use_container_width=True, key=f"gantt_{algo_select}")

st.markdown("---")


# ─────────────────────────────────────────────────────────────────────────────
# ── SECTION 3: ADAPTATION TIMELINE LOG ──
# ─────────────────────────────────────────────────────────────────────────────

st.markdown('<div class="section-heading">⚡ Adaptation Timeline Log</div>',
            unsafe_allow_html=True)
st.caption("Every row is a moment the scheduler re-evaluated the ready queue and selected a new policy.")

render_adaptation_log(adaptation_log)

st.markdown("---")


# ─────────────────────────────────────────────────────────────────────────────
# ── SECTION 4: PROCESS METRICS TABLE ──
# ─────────────────────────────────────────────────────────────────────────────

st.markdown('<div class="section-heading">📋 Process-Level Metrics</div>',
            unsafe_allow_html=True)

table_tab1, table_tab2 = st.tabs(["🧠 Adaptive AI", "📊 All Classic Schedulers"])

with table_tab1:
    proc_df = processes_to_dataframe(adaptive_result["processes"])
    st.dataframe(
        proc_df.style
            .format(precision=2)
            .apply(_highlight_col, subset=["Waiting Time", "Turnaround Time"]),
        use_container_width=True,
        height=min(600, 80 + len(proc_df) * 38),
        hide_index=True,
    )

with table_tab2:
    for algo_key, res in all_results.items():
        if "error" in res or algo_key == "TrueAdaptive":
            continue
        label_str = res["algorithm"]
        with st.expander(
            f"**{label_str}** — AWT: {res.get('avg_waiting_time',0):.2f} | "
            f"ATAT: {res.get('avg_turnaround_time',0):.2f}"
        ):
            df_res = processes_to_dataframe(res["processes"])
            st.dataframe(df_res.style.format(precision=2),
                         use_container_width=True, hide_index=True)

st.markdown("---")


# ─────────────────────────────────────────────────────────────────────────────
# ── SECTION 5: COMPARISON CHARTS ──
# ─────────────────────────────────────────────────────────────────────────────

st.markdown('<div class="section-heading">📈 Algorithm Comparison</div>',
            unsafe_allow_html=True)

comp_fig = build_comparison_chart(all_results)
st.plotly_chart(comp_fig, use_container_width=True, key="comparison_chart")

# ── Composite Score chart ────────────────────────────────────────────────────
st.plotly_chart(
    build_composite_chart(all_results, len(processes)),
    use_container_width=True,
    key="composite_chart",
)

# ── Numeric comparison table ─────────────────────────────────────────────────
comp_df = comparison_dataframe(all_results)
selected_algo_name = adaptive_result["selected_algorithm"]
comp_df = comp_df.reset_index()
comp_df["Algorithm"] = comp_df["Algorithm"].apply(
    lambda name: f"🧠 {name}" if name == selected_algo_name else name
)

# Add composite score column to the table
n_procs = len(processes)
def _composite(row):
    awt  = row["Avg Waiting Time"]
    art  = row["Avg Response Time"]
    cs_n = row["Context Switches"] / max(1, n_procs)
    fstd = row.get("Fairness (WT Std)", 0.0)
    return round(0.35 * awt + 0.35 * art + 0.15 * cs_n + 0.15 * fstd, 2)

comp_df["Composite Score"] = comp_df.apply(_composite, axis=1)

def _highlight_min_green(s):
    """Highlight the minimum value in a numeric column."""
    try:
        is_min = s == s.min()
        return [
            "background-color:rgba(110,231,183,0.20);color:#6ee7b7;font-weight:700"
            if v else ""
            for v in is_min
        ]
    except Exception:
        return [""] * len(s)

def _highlight_max_green(s):
    """Highlight the maximum value (Throughput: higher = better)."""
    try:
        is_max = s == s.max()
        return [
            "background-color:rgba(110,231,183,0.20);color:#6ee7b7;font-weight:700"
            if v else ""
            for v in is_max
        ]
    except Exception:
        return [""] * len(s)

st.dataframe(
    comp_df.style
        .format(precision=3, subset=[
            "Avg Waiting Time", "Avg Turnaround Time",
            "Avg Response Time", "Throughput",
            "Fairness (WT Std)", "Composite Score",
        ])
        .apply(_highlight_min_green, subset=[
            "Avg Waiting Time", "Avg Turnaround Time",
            "Avg Response Time", "Composite Score", "Fairness (WT Std)",
        ])
        .apply(_highlight_max_green, subset=["Throughput"]),
    use_container_width=True,
    hide_index=True,
)

# ── AI Advantage insight box ─────────────────────────────────────────────────
ad_comp = adaptive_result.get("composite_score", 0.0)
ad_art  = adaptive_result.get("avg_response_time", 0.0)
ad_fstd = adaptive_result.get("fairness_std", 0.0)

# Find the best classic per dimension
best_art  = min(
    (v.get("avg_response_time", 9999), k)
    for k, v in all_results.items() if "error" not in v and k != "TrueAdaptive"
)
best_fstd = min(
    (v.get("fairness_std", v.get("avg_waiting_time", 9999)), k)
    for k, v in all_results.items() if "error" not in v and k != "TrueAdaptive"
)

ai_wins = []
if ad_art  <= best_art[0]  + 0.5: ai_wins.append(f"Response Time ({ad_art:.2f} ticks)")
if ad_fstd <= best_fstd[0] + 0.5: ai_wins.append(f"Fairness / WT Std ({ad_fstd:.2f})")
if adaptive_result.get("context_switches", 0) <= min(
    v.get("context_switches", 9999)
    for k, v in all_results.items() if "error" not in v and k != "TrueAdaptive"
) + 1:
    ai_wins.append("Context Switches")

wins_html = "".join(
    f'<span class="badge badge-srtf" style="margin:2px 4px;">{w}</span>'
    for w in ai_wins
) if ai_wins else "<em style='color:#64748b;'>No dimension win in this run — workload is fully CPU-bound.</em>"

st.markdown(
    f'<div class="justification-box" style="margin-top:1rem;">'
    f'<b>🏆 Adaptive AI Wins On:</b>&nbsp; {wins_html}<br>'
    f'<span style="font-size:0.85rem;color:#94a3b8;margin-top:6px;display:block;">'
    f'Composite Score: <b style="color:#f093fb;">{ad_comp:.3f}</b> '
    f'(weights: AWT 35% + ART 35% + CS-per-process 15% + Fairness 15%)'
    f'</span></div>',
    unsafe_allow_html=True,
)

st.markdown("---")


# ─────────────────────────────────────────────────────────────────────────────
# ── SECTION 6: FEATURE VECTOR ──
# ─────────────────────────────────────────────────────────────────────────────

st.markdown('<div class="section-heading">🔬 ML Feature Vector (Initial Batch)</div>',
            unsafe_allow_html=True)

try:
    from ml.predictor import extract_features, FEATURE_COLS
    feat_vec = extract_features(processes)
    feat_fig = go.Figure(go.Bar(
        x=FEATURE_COLS, y=feat_vec.tolist(),
        marker=dict(color=["#667eea","#f093fb","#4ade80","#fbbf24","#67e8f9"],
                    opacity=0.85, line=dict(width=0)),
        text=[f"{v:.3f}" for v in feat_vec],
        textposition="outside",
        textfont=dict(size=12, color="#e2e8f0"),
    ))
    feat_fig.update_layout(
        title=dict(text="<b>Feature Vector fed to ML Classifier (initial batch)</b>",
                   font=dict(size=14, color="#e2e8f0")),
        xaxis=dict(color="#94a3b8", tickfont=dict(size=12)),
        yaxis=dict(color="#94a3b8", showgrid=True,
                   gridcolor="rgba(255,255,255,0.07)"),
        paper_bgcolor="rgba(10,14,26,0)",
        plot_bgcolor="rgba(10,14,26,0.4)",
        font=dict(family="Inter, sans-serif", color="#e2e8f0"),
        height=320, margin=dict(l=40, r=30, t=60, b=60),
    )
    st.plotly_chart(feat_fig, use_container_width=True, key="feature_chart")
except Exception:
    pass


# ─────────────────────────────────────────────────────────────────────────────
# ── FOOTER ──
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("---")
st.markdown("""
<div style="text-align:center;color:#475569;font-size:0.85rem;padding:1rem 0;">
    🧠 Adaptive AI-Driven CPU Scheduler v2.0 &nbsp;·&nbsp;
    True Runtime Adaptation Edition &nbsp;·&nbsp;
    Streamlit + scikit-learn + Plotly
</div>""", unsafe_allow_html=True)
