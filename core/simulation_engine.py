"""
core/simulation_engine.py
--------------------------
Simulation engine that orchestrates process generation, ready-queue management,
and context-switch tracking across multiple scheduling runs.

Key responsibilities:
    - Generate synthetic Process batches from user-defined parameters.
    - Run one or more schedulers and collect results.
    - Track and expose context-switch counts for comparison.
"""

import random
from typing import List, Dict, Any, Optional

from core.process import Process
from core.schedulers import (
    fcfs,
    sjf_non_preemptive,
    sjf_preemptive,
    round_robin,
    priority_scheduling,
)


# ---------------------------------------------------------------------------
# Process factory
# ---------------------------------------------------------------------------

def generate_processes(
    count: int,
    max_arrival_time: int = 20,
    burst_time_range: tuple = (2, 20),
    io_probability: float = 0.4,
    seed: Optional[int] = None,
) -> List[Process]:
    """
    Generate a batch of synthetic processes for simulation.

    Args:
        count           : Number of processes to create.
        max_arrival_time: Upper bound for random arrival times.
        burst_time_range: (min, max) for random burst times.
        io_probability  : Probability [0, 1] that a process is I/O-heavy.
        seed            : Optional random seed for reproducibility.

    Returns:
        Sorted list of Process objects (by arrival_time, then pid).
    """
    if seed is not None:
        random.seed(seed)

    processes: List[Process] = []
    for i in range(1, count + 1):
        arrival = random.randint(0, max_arrival_time)
        burst = random.randint(*burst_time_range)
        priority = random.randint(1, 10)
        io_freq = round(random.uniform(0.3, 0.9), 3) if random.random() < io_probability else round(random.uniform(0.0, 0.2), 3)

        processes.append(Process(
            pid=i,
            arrival_time=arrival,
            burst_time=burst,
            priority=priority,
            io_frequency=io_freq,
        ))

    processes.sort(key=lambda p: (p.arrival_time, p.pid))
    return processes


# ---------------------------------------------------------------------------
# Simulation runner
# ---------------------------------------------------------------------------

class SimulationEngine:
    """
    Orchestrates simulation runs across all scheduling algorithms.

    Usage:
        engine = SimulationEngine(processes)
        results = engine.run_all(time_quantum=4)
    """

    ALGORITHM_KEYS = ["FCFS", "SJF", "RR", "Priority"]

    def __init__(self, processes: List[Process]):
        if not processes:
            raise ValueError("Process list must not be empty.")
        self.processes = processes

    def run_all(self, time_quantum: int = 4) -> Dict[str, Dict[str, Any]]:
        """
        Run all four classic schedulers on the stored process batch.

        Args:
            time_quantum: Time slice for Round Robin.

        Returns:
            Dictionary keyed by algorithm name, each containing:
                - 'processes'       : List[Process] with filled metrics.
                - 'timeline'        : List of (pid, start, end) tuples.
                - 'context_switches': int.
                - 'algorithm'       : Human-readable name string.
                - 'avg_waiting_time'     : float.
                - 'avg_turnaround_time'  : float.
                - 'avg_response_time'    : float.
                - 'throughput'           : float (processes / total time unit).
        """
        results: Dict[str, Dict[str, Any]] = {}

        schedulers = {
            "FCFS":     lambda: fcfs(self.processes),
            "SJF":      lambda: sjf_non_preemptive(self.processes),
            "SRTF":     lambda: sjf_preemptive(self.processes),
            "RR":       lambda: round_robin(self.processes, time_quantum),
            "Priority": lambda: priority_scheduling(self.processes),
        }

        for key, run_fn in schedulers.items():
            try:
                result = run_fn()
                result.update(self._compute_aggregate_metrics(result["processes"]))
                results[key] = result
            except Exception as exc:  # pragma: no cover
                results[key] = {"error": str(exc), "algorithm": key}

        return results

    def run_single(self, algorithm: str, time_quantum: int = 4) -> Dict[str, Any]:
        """
        Run a single named algorithm.

        Args:
            algorithm   : One of 'FCFS', 'SJF', 'RR', 'Priority'.
            time_quantum: Used only when algorithm == 'RR'.

        Returns:
            Result dict (same schema as run_all entries).
        """
        dispatch = {
            "FCFS": lambda: fcfs(self.processes),
            "SJF": lambda: sjf_non_preemptive(self.processes),
            "SRTF": lambda: sjf_preemptive(self.processes),
            "RR": lambda: round_robin(self.processes, time_quantum),
            "Priority": lambda: priority_scheduling(self.processes),
        }

        key = algorithm.upper()
        if key not in dispatch:
            raise ValueError(
                f"Unknown algorithm '{algorithm}'. "
                f"Choose from: {list(dispatch.keys())}"
            )

        result = dispatch[key]()
        result.update(self._compute_aggregate_metrics(result["processes"]))
        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_aggregate_metrics(procs: List[Process]) -> Dict[str, float]:
        """Compute average metrics and throughput for a completed simulation."""
        n = len(procs)
        if n == 0:
            return {
                "avg_waiting_time": 0.0,
                "avg_turnaround_time": 0.0,
                "avg_response_time": 0.0,
                "throughput": 0.0,
            }

        total_wt = sum(p.waiting_time for p in procs)
        total_tt = sum(p.turnaround_time for p in procs)
        total_rt = sum(p.response_time for p in procs)

        # Throughput = processes completed per unit time (full span).
        all_completions = [p.completion_time for p in procs if p.completion_time is not None]
        all_arrivals = [p.arrival_time for p in procs]
        span = (max(all_completions) - min(all_arrivals)) if all_completions else 1
        throughput = n / span if span > 0 else float(n)

        wts = [p.waiting_time for p in procs]
        mean_wt = total_wt / n
        fairness_std = round(
            (sum((w - mean_wt) ** 2 for w in wts) / n) ** 0.5, 3
        )

        return {
            "avg_waiting_time":    round(total_wt / n, 3),
            "avg_turnaround_time": round(total_tt / n, 3),
            "avg_response_time":   round(total_rt / n, 3),
            "throughput":          round(throughput, 4),
            "fairness_std":        fairness_std,
        }

    def context_switch_summary(self, results: Dict[str, Dict[str, Any]]) -> Dict[str, int]:
        """Extract context-switch counts for all algorithms."""
        return {
            key: res.get("context_switches", 0)
            for key, res in results.items()
            if "error" not in res
        }
