"""
metrics/calculator.py
----------------------
Calculates and formats CPU scheduling performance metrics.

Metrics supported:
    - Average Waiting Time (AWT)
    - Average Turnaround Time (ATAT)
    - Average Response Time (ART)
    - Throughput (processes per unit time)
    - CPU Utilisation (%) – estimated from burst times vs total span
    - Context Switch Count

Each function accepts either a list of Process objects or a raw
simulation result dict from SimulationEngine / AdaptiveScheduler.
"""

from __future__ import annotations

import math
from typing import List, Dict, Any, Optional

import pandas as pd

from core.process import Process


# ---------------------------------------------------------------------------
# Individual metric calculators
# ---------------------------------------------------------------------------

def average_waiting_time(processes: List[Process]) -> float:
    """Mean time processes spend waiting in the ready queue."""
    if not processes:
        return 0.0
    return round(sum(p.waiting_time for p in processes) / len(processes), 4)


def average_turnaround_time(processes: List[Process]) -> float:
    """Mean time from arrival to completion."""
    if not processes:
        return 0.0
    return round(sum(p.turnaround_time for p in processes) / len(processes), 4)


def average_response_time(processes: List[Process]) -> float:
    """Mean time from arrival to first CPU assignment."""
    if not processes:
        return 0.0
    return round(sum(p.response_time for p in processes) / len(processes), 4)


def throughput(processes: List[Process]) -> float:
    """
    Processes completed per unit time over the full simulation span.
    span = max(completion_time) - min(arrival_time)
    """
    if not processes:
        return 0.0

    completions = [p.completion_time for p in processes if p.completion_time is not None]
    arrivals = [p.arrival_time for p in processes]
    if not completions:
        return 0.0

    span = max(completions) - min(arrivals)
    return round(len(processes) / span if span > 0 else float(len(processes)), 4)


def cpu_utilisation(processes: List[Process]) -> float:
    """
    Estimated CPU utilisation (%) over the full simulation span.

    Utilisation = total_cpu_burst_time / total_span × 100
    """
    if not processes:
        return 0.0

    completions = [p.completion_time for p in processes if p.completion_time is not None]
    arrivals = [p.arrival_time for p in processes]
    if not completions:
        return 0.0

    span = max(completions) - min(arrivals)
    total_burst = sum(p.burst_time for p in processes)

    return round(min(100.0, (total_burst / span * 100) if span > 0 else 100.0), 2)


def waiting_time_std(processes: List[Process]) -> float:
    """Standard deviation of waiting times (measures fairness)."""
    if len(processes) < 2:
        return 0.0
    wts = [p.waiting_time for p in processes]
    mean = sum(wts) / len(wts)
    variance = sum((w - mean) ** 2 for w in wts) / len(wts)
    return round(math.sqrt(variance), 4)


# ---------------------------------------------------------------------------
# Aggregate report
# ---------------------------------------------------------------------------

def compute_metrics(processes: List[Process], context_switches: int = 0) -> Dict[str, Any]:
    """
    Compute a full metrics report for a finished simulation.

    Args:
        processes       : Completed Process objects with metrics populated.
        context_switches: Number of context switches recorded by the scheduler.

    Returns:
        Dictionary with all key metrics.
    """
    return {
        "process_count": len(processes),
        "avg_waiting_time": average_waiting_time(processes),
        "avg_turnaround_time": average_turnaround_time(processes),
        "avg_response_time": average_response_time(processes),
        "throughput": throughput(processes),
        "cpu_utilisation_pct": cpu_utilisation(processes),
        "waiting_time_std": waiting_time_std(processes),
        "context_switches": context_switches,
    }


# ---------------------------------------------------------------------------
# DataFrame builders
# ---------------------------------------------------------------------------

def processes_to_dataframe(processes: List[Process]) -> pd.DataFrame:
    """
    Convert a list of completed Process objects into a styled DataFrame
    suitable for display in the Streamlit dashboard.

    Returns:
        pandas DataFrame with one row per process.
    """
    rows = [p.to_dict() for p in processes]
    df = pd.DataFrame(rows)

    # Reorder columns for readability.
    ordered_cols = [
        "PID", "Arrival Time", "Burst Time", "Priority", "I/O Frequency",
        "Start Time", "Completion Time",
        "Waiting Time", "Turnaround Time", "Response Time",
    ]
    existing = [c for c in ordered_cols if c in df.columns]
    return df[existing]


def comparison_dataframe(results: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
    """
    Build a side-by-side comparison DataFrame from multiple scheduler results.

    Args:
        results: Dict keyed by algorithm name, each containing metric fields.

    Returns:
        DataFrame with algorithm names as rows and key metrics as columns.
    """
    rows = []
    for algo_key, res in results.items():
        if "error" in res:
            continue

        procs = res.get("processes", [])
        wts   = [p.waiting_time for p in procs] if procs else []
        if len(wts) >= 2:
            mean_wt = sum(wts) / len(wts)
            fairness = round((sum((w - mean_wt) ** 2 for w in wts) / len(wts)) ** 0.5, 3)
        else:
            fairness = 0.0

        rows.append({
            "Algorithm":           res.get("algorithm", algo_key),
            "Avg Waiting Time":    res.get("avg_waiting_time", 0.0),
            "Avg Turnaround Time": res.get("avg_turnaround_time", 0.0),
            "Avg Response Time":   res.get("avg_response_time", 0.0),
            "Throughput":          res.get("throughput", 0.0),
            "Context Switches":    res.get("context_switches", 0),
            "Fairness (WT Std)":   fairness,
        })
    return pd.DataFrame(rows).set_index("Algorithm")


def metrics_summary_dict(result: Dict[str, Any]) -> Dict[str, str]:
    """
    Format key metrics as a human-readable string dictionary for card display.

    Args:
        result: Single scheduler result dict.

    Returns:
        Dict of {metric_name: formatted_string}.
    """
    return {
        "Avg Waiting Time": f"{result.get('avg_waiting_time', 0.0):.3f} ticks",
        "Avg Turnaround Time": f"{result.get('avg_turnaround_time', 0.0):.3f} ticks",
        "Avg Response Time": f"{result.get('avg_response_time', 0.0):.3f} ticks",
        "Throughput": f"{result.get('throughput', 0.0):.4f} proc/tick",
        "Context Switches": str(result.get("context_switches", 0)),
    }
