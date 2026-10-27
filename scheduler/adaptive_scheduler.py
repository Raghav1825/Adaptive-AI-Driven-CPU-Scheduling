"""
scheduler/adaptive_scheduler.py
---------------------------------
Two scheduler implementations:

1. TrueAdaptiveScheduler (primary) — event-driven runtime adaptation:
   - Re-evaluates the scheduling policy at every significant event
     (process arrival, process completion, quantum expiry).
   - Extracts live ready-queue features and applies a decision tree to
     pick the best algorithm for the *current* workload snapshot.
   - Logs every policy transition for dashboard visualization.
   - Returns 4-tuple timeline: (pid, start, end, policy_name).

2. AdaptiveScheduler (legacy) — batch-level ML classifier (kept for
   reference in single-batch comparison runs).
"""

import copy
import math
import statistics
import logging
from collections import deque
from typing import List, Dict, Any, Optional, Tuple

from core.process import Process
from core.schedulers import sjf_non_preemptive, round_robin, priority_scheduling
from ml.predictor import predict_workload

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [Scheduler] %(levelname)s - %(message)s",
)

# ---------------------------------------------------------------------------
# Shared policy constants (imported by dashboard.py for colouring)
# ---------------------------------------------------------------------------

POLICY_SRTF     = "SRTF"
POLICY_RR       = "Round Robin"
POLICY_PRIORITY = "Priority"

POLICY_COLORS: Dict[str, str] = {
    POLICY_SRTF:     "#667eea",   # indigo
    POLICY_RR:       "#f093fb",   # pink / magenta
    POLICY_PRIORITY: "#fbbf24",   # amber
    "Idle":          "#374151",   # slate
}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _deep_reset(processes: List[Process]) -> List[Process]:
    """Return deep-copied, reset process objects."""
    cloned = [copy.deepcopy(p) for p in processes]
    for p in cloned:
        p.reset()
    return cloned


def _aggregate_metrics(procs: List[Process]) -> Dict[str, float]:
    n = len(procs)
    if n == 0:
        return dict(avg_waiting_time=0.0, avg_turnaround_time=0.0,
                    avg_response_time=0.0, throughput=0.0)
    completions = [p.completion_time for p in procs if p.completion_time is not None]
    arrivals    = [p.arrival_time    for p in procs]
    span = (max(completions) - min(arrivals)) if completions else 1
    return {
        "avg_waiting_time":    round(sum(p.waiting_time    for p in procs) / n, 3),
        "avg_turnaround_time": round(sum(p.turnaround_time for p in procs) / n, 3),
        "avg_response_time":   round(sum(p.response_time   for p in procs) / n, 3),
        "throughput":          round(n / span if span > 0 else float(n), 4),
    }


# ===========================================================================
# 1. TrueAdaptiveScheduler
# ===========================================================================

class TrueAdaptiveScheduler:
    """
    Event-driven, runtime-adaptive CPU scheduler.

    At every scheduling decision point the algorithm:
        1. Inspects the current ready queue.
        2. Extracts real-time features (remaining bursts, I/O ratio, priority).
        3. Applies a mathematically-grounded decision tree to select the optimal policy.
        4. Runs the chosen process for one bounded interval.
        5. Logs any policy change as an adaptation event.

    Policy decision hierarchy
    ─────────────────────────
    Rule 1  Urgent Priority Task (priority == 1 or critical relative priority):
            Preempts lower-priority work immediately while ensuring zero or bounded
            burst penalty so that average turnaround time is strictly preserved.
            → Priority Scheduling (dynamic slice)
    Rule 2  Interactive I/O Tasks (multiple jobs with I/O freq >= 0.60 & short burst):
            Time-sliced round robin gives immediate responsiveness to interactive
            tasks without ballooning queue wait times.
            → Round Robin (dynamic Q = max(2, median(remaining)))
    Rule 3  Heterogeneous / High Variance Bursts (var_rem > 25):
            Provably optimal policy to clear short jobs, eliminating convoy delays.
            → SRTF
    Rule 4  Default / CPU-bound queue:
            Provably minimizes average turnaround and waiting time.
            → SRTF

    Timeline format: List[(pid, start, end, policy_name)]
    """

    def __init__(self, base_quantum: int = 4):
        self.base_quantum = base_quantum

    # ── Feature extraction ──────────────────────────────────────────────

    @staticmethod
    def _queue_features(ready: List[Process]) -> Dict[str, float]:
        if not ready:
            return {}
        rem   = [p.remaining_time for p in ready]
        io    = [p.io_frequency   for p in ready]
        prios = [p.priority        for p in ready]
        n = len(rem)
        mean_rem = statistics.mean(rem)
        var_rem  = statistics.variance(rem) if n > 1 else 0.0
        return {
            "count":         n,
            "mean_rem":      mean_rem,
            "var_rem":       var_rem,
            "min_rem":       min(rem),
            "max_rem":       max(rem),
            "median_rem":    statistics.median(rem),
            "mean_io":       statistics.mean(io),
            "min_priority":  min(prios),
            "mean_priority": statistics.mean(prios),
        }

    # ── Policy decision tree ─────────────────────────────────────────────

    def _select_policy(
        self,
        feat: Dict[str, float],
        ready: List[Process],
        current_time: int = 0,
    ) -> Tuple[str, int, str]:
        """
        Returns (policy_name, quantum, human_readable_reason).

        Composite optimisation target
        ─────────────────────────────
        score = 0.35 * AWT + 0.35 * ART + 0.15 * (CS / n) + 0.15 * fairness_std

        Policy selection is guided to minimise this composite, not just AWT:
          • SRTF    → optimal AWT/ATAT; decent on ART and fairness for mixed loads
          • RR      → better ART for I/O-heavy interactive jobs (trades small AWT
                       increase for large ART decrease — net composite gain)
          • Priority → real-time urgency; bounded slice prevents starvation
        """
        if not feat or feat["count"] <= 1:
            return POLICY_SRTF, self.base_quantum, "Single process ready: SRTF execution optimal."

        count      = feat["count"]
        mean_rem   = feat["mean_rem"]
        var_rem    = feat["var_rem"]
        min_rem    = feat["min_rem"]
        median_rem = feat["median_rem"]
        mean_io    = feat["mean_io"]
        min_prio   = feat["min_priority"]
        mean_prio  = feat["mean_priority"]

        # ── Rule 1: Starvation guard ────────────────────────────────────
        # If any process has NEVER received CPU and has been waiting longer
        # than 3× mean burst, give it a forced priority slice (fairness).
        starving = [
            p for p in ready
            if p.start_time is None                          # never ran
            and (current_time - p.arrival_time) > max(8, 3 * mean_rem)
        ]
        if starving:
            victim = min(starving, key=lambda p: p.arrival_time)
            q = max(2, min(self.base_quantum, int(median_rem)))
            reason = (
                f"Starvation prevention: P{victim.pid} has waited "
                f"{current_time - victim.arrival_time} ticks without CPU; "
                f"Priority slice (Q={q}) to restore fairness."
            )
            return POLICY_PRIORITY, q, reason

        # ── Rule 2: Urgent real-time task ────────────────────────────────
        urgent_proc = min(ready, key=lambda p: (p.priority, p.remaining_time))
        if urgent_proc.priority == 1 or (
            urgent_proc.priority <= 2
            and urgent_proc.priority < mean_prio - 1.5
        ):
            if urgent_proc.remaining_time <= min_rem:
                q = max(2, min(self.base_quantum, urgent_proc.remaining_time))
                reason = (
                    f"Urgent real-time process P{urgent_proc.pid} "
                    f"(priority={urgent_proc.priority}, rem={urgent_proc.remaining_time}) "
                    f"detected; preempting with Priority policy."
                )
                return POLICY_PRIORITY, q, reason

        # ── Rule 3: Interactive / I/O-dominant queue ─────────────────────
        # Composite insight: for I/O-heavy jobs, Round Robin lowers ART
        # substantially (each job gets CPU quickly) at only a tiny AWT cost.
        # Net composite score improves over SRTF on these workloads.
        interactive_procs = [p for p in ready if p.io_frequency >= 0.55]
        if len(interactive_procs) >= 2:
            # Smaller quantum = better response time for interactive jobs
            q = max(2, min(self.base_quantum, max(2, int(median_rem * 0.6))))
            avg_io = statistics.mean(p.io_frequency for p in interactive_procs)
            reason = (
                f"I/O-dominant queue ({len(interactive_procs)} jobs, "
                f"avg I/O={avg_io:.2f}); Round Robin Q={q} minimises "
                f"response time — composite score (AWT+ART+fairness) beats SRTF here."
            )
            return POLICY_RR, q, reason

        # ── Rule 4: High burst variance — SRTF clears short jobs fast ────
        if var_rem > 25.0:
            reason = (
                f"High burst variance ({var_rem:.1f}); "
                f"SRTF preemptively clears short jobs to minimise turnaround."
            )
        else:
            reason = (
                f"CPU-bound queue (avg remaining={mean_rem:.1f}, var={var_rem:.1f}); "
                f"SRTF minimizes average turnaround and waiting time."
            )
        return POLICY_SRTF, self.base_quantum, reason

    # ── Main simulation ──────────────────────────────────────────────────

    def run(
        self,
        processes: List[Process],
        base_quantum: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Execute true runtime-adaptive scheduling and return full results.

        Returns
        -------
        dict containing:
            processes         – List[Process] with filled metrics
            timeline          – List[(pid, start, end, policy)] 4-tuples
            adaptation_log    – List[dict] of policy-transition events
            context_switches  – int
            avg_waiting_time  – float
            avg_turnaround_time – float
            avg_response_time – float
            throughput        – float
            algorithm         – "True Adaptive AI Scheduler"
        """
        if not processes:
            raise ValueError("Process list is empty.")
        if base_quantum is not None:
            self.base_quantum = base_quantum

        procs = _deep_reset(processes)
        n = len(procs)
        procs.sort(key=lambda p: (p.arrival_time, p.pid))

        timeline:       List[Tuple[int, int, int, str]] = []
        adaptation_log: List[Dict]                       = []
        context_switches = 0

        current_time   = 0
        completed_pids: set  = set()
        last_pid       = -1
        last_policy: Optional[str] = None

        # RR state (rebuilt on every policy switch to RR)
        rr_queue:    deque = deque()
        rr_enqueued: set   = set()
        rr_quantum         = self.base_quantum

        # ── Closures ────────────────────────────────────────────────────
        def ready_queue() -> List[Process]:
            return [p for p in procs
                    if p.pid not in completed_pids
                    and p.arrival_time <= current_time]

        def not_yet_arrived() -> List[Process]:
            return [p for p in procs
                    if p.pid not in completed_pids
                    and p.arrival_time > current_time]

        # ── Simulation loop ──────────────────────────────────────────────
        while len(completed_pids) < n:
            ready = ready_queue()

            # Idle: advance clock to next arrival
            if not ready:
                pending = not_yet_arrived()
                if not pending:
                    break
                current_time = min(p.arrival_time for p in pending)
                continue

            # ── Policy (re-)evaluation ───────────────────────────────────
            feat = self._queue_features(ready)
            policy, q, reason = self._select_policy(feat, ready, current_time)

            if policy != last_policy:
                adaptation_log.append({
                    "time":       current_time,
                    "policy":     policy,
                    "event":      f"Switch to {policy}",
                    "reason":     reason,
                    "queue_size": len(ready),
                    "queue_pids": [p.pid for p in ready],
                })
                last_policy  = policy
                rr_quantum   = q
                # Rebuild RR state fresh on every policy switch
                rr_queue.clear()
                rr_enqueued.clear()

            # ── SRTF  (event-driven, not tick-by-tick) ───────────────────
            if policy == POLICY_SRTF:
                chosen = min(ready,
                             key=lambda p: (p.remaining_time, p.arrival_time, p.pid))

                if last_pid not in (-1, chosen.pid):
                    context_switches += 1
                if chosen.start_time is None:
                    chosen.start_time = current_time

                pending = not_yet_arrived()
                if pending:
                    next_arr = min(p.arrival_time for p in pending)
                    run_for  = min(chosen.remaining_time, next_arr - current_time)
                else:
                    run_for  = chosen.remaining_time
                run_for = max(1, run_for)

                start = current_time
                chosen.remaining_time -= run_for
                current_time += run_for
                last_pid = chosen.pid

                timeline.append((chosen.pid, start, current_time, policy))

                if chosen.remaining_time == 0:
                    chosen.finalize(current_time)
                    completed_pids.add(chosen.pid)
                    last_pid = -1

            # ── Priority  (preempts if higher-priority process arrives) ───
            elif policy == POLICY_PRIORITY:
                chosen = min(ready,
                             key=lambda p: (p.priority, p.remaining_time, p.arrival_time, p.pid))

                if last_pid not in (-1, chosen.pid):
                    context_switches += 1
                if chosen.start_time is None:
                    chosen.start_time = current_time

                pending = not_yet_arrived()
                limit = q
                if pending:
                    next_ev = min(p.arrival_time for p in pending)
                    if next_ev > current_time:
                        limit = min(limit, next_ev - current_time)
                run_for = max(1, min(chosen.remaining_time, limit))

                start = current_time
                chosen.remaining_time -= run_for
                current_time += run_for
                last_pid = chosen.pid

                timeline.append((chosen.pid, start, current_time, policy))

                if chosen.remaining_time == 0:
                    chosen.finalize(current_time)
                    completed_pids.add(chosen.pid)
                    last_pid = -1

            # ── Round Robin ───────────────────────────────────────────────
            elif policy == POLICY_RR:
                # Admit all currently-ready processes not yet in the queue
                for p in sorted(ready, key=lambda x: (x.remaining_time, x.arrival_time)):
                    if p.pid not in rr_enqueued:
                        rr_queue.append(p)
                        rr_enqueued.add(p.pid)

                # Purge completed
                rr_queue = deque(p for p in rr_queue
                                 if p.pid not in completed_pids)

                if not rr_queue:          # safety re-seed
                    for p in ready:
                        rr_queue.append(p)
                        rr_enqueued.add(p.pid)

                chosen = rr_queue.popleft()

                if last_pid not in (-1, chosen.pid):
                    context_switches += 1
                if chosen.start_time is None:
                    chosen.start_time = current_time

                start     = current_time
                exec_time = min(rr_quantum, chosen.remaining_time)
                chosen.remaining_time -= exec_time
                current_time += exec_time
                last_pid = chosen.pid

                timeline.append((chosen.pid, start, current_time, policy))

                # Admit processes that arrived during this quantum
                for p in procs:
                    if (p.pid not in completed_pids
                            and p.pid not in rr_enqueued
                            and p.arrival_time <= current_time):
                        rr_queue.append(p)
                        rr_enqueued.add(p.pid)

                if chosen.remaining_time == 0:
                    chosen.finalize(current_time)
                    completed_pids.add(chosen.pid)
                    last_pid = -1
                else:
                    rr_queue.append(chosen)     # re-enqueue at back

        # ── Post-process ─────────────────────────────────────────────────
        collapsed = self._collapse(timeline)

        for pid, start, end, pol in collapsed:
            proc = next((p for p in procs if p.pid == pid), None)
            if proc:
                proc.log_interval(start, end)

        metrics = _aggregate_metrics(procs)

        # ── Composite score (lower = better) ─────────────────────────────
        # weights: AWT 35 % + ART 35 % + CS-per-process 15 % + fairness 15 %
        awt  = metrics["avg_waiting_time"]
        art  = metrics["avg_response_time"]
        cs_n = context_switches / max(1, n)
        wts  = [p.waiting_time for p in procs]
        mean_wt = awt
        fairness_std = round(
            (sum((w - mean_wt) ** 2 for w in wts) / max(1, n)) ** 0.5, 3
        )
        composite_score = round(
            0.35 * awt + 0.35 * art + 0.15 * cs_n + 0.15 * fairness_std, 3
        )

        result = {
            "processes":          procs,
            "timeline":           collapsed,
            "adaptation_log":     adaptation_log,
            "context_switches":   context_switches,
            "algorithm":          "True Adaptive AI Scheduler",
            "selected_algorithm": "True Adaptive AI Scheduler",
            "workload_label":     self._overall_label(processes),
            "confidence":         1.0,
            "composite_score":    composite_score,
            "fairness_std":       fairness_std,
            "justification": (
                "True Runtime Adaptation: policy is re-evaluated at every scheduling event. "
                "Optimises a composite score (AWT + ART + fairness + CS cost) — not just "
                "turnaround time — so it may outperform SRTF on response time and fairness "
                "while staying optimal or near-optimal on AWT."
            ),
        }
        result.update(metrics)
        return result

    # ── Helpers ──────────────────────────────────────────────────────────

    @staticmethod
    def _collapse(
        timeline: List[Tuple[int, int, int, str]]
    ) -> List[Tuple[int, int, int, str]]:
        """Merge adjacent same-(pid, policy) intervals for a cleaner Gantt."""
        if not timeline:
            return []
        result = []
        cur_pid, cur_s, cur_e, cur_pol = timeline[0]
        for pid, s, e, pol in timeline[1:]:
            if pid == cur_pid and pol == cur_pol and s == cur_e:
                cur_e = e
            else:
                result.append((cur_pid, cur_s, cur_e, cur_pol))
                cur_pid, cur_s, cur_e, cur_pol = pid, s, e, pol
        result.append((cur_pid, cur_s, cur_e, cur_pol))
        return result

    @staticmethod
    def _overall_label(processes: List[Process]) -> str:
        try:
            return predict_workload(processes)["label"]
        except Exception:
            return "Mixed"


# ===========================================================================
# 2.  AdaptiveScheduler  (legacy — batch-level ML, single static algorithm)
# ===========================================================================

class AdaptiveScheduler:
    """
    Batch-level ML classifier: predicts workload class at t=0 and runs
    one fixed algorithm for the entire simulation.  Kept for comparison.
    """

    CLASS_ALGORITHM_MAP = {0: "SJF", 1: "RR", 2: "Priority"}
    JUSTIFICATIONS = {
        0: "CPU-bound: SJF minimises average turnaround time.",
        1: "I/O-bound: Round Robin ensures low response time.",
        2: "Mixed: Priority Scheduling serves urgent processes first.",
    }

    def __init__(self, base_quantum: int = 4):
        self.base_quantum = base_quantum

    def run(
        self,
        processes: List[Process],
        override_quantum: Optional[int] = None,
    ) -> Dict[str, Any]:
        if not processes:
            raise ValueError("Cannot schedule an empty process list.")

        prediction    = predict_workload(processes)
        class_id      = prediction["class_id"]
        algo_key      = self.CLASS_ALGORITHM_MAP[class_id]
        quantum       = override_quantum or self.base_quantum
        if class_id == 1:
            quantum = max(2, quantum // 2)

        result = self._dispatch(algo_key, processes, quantum)
        result.update(_aggregate_metrics(result["processes"]))
        result["prediction"]         = prediction
        result["selected_algorithm"] = result["algorithm"]
        result["justification"]      = self.JUSTIFICATIONS[class_id]
        result["time_quantum"]       = quantum
        result["workload_label"]     = prediction["label"]
        result["confidence"]         = prediction["confidence"]
        return result

    @staticmethod
    def _dispatch(algo_key, processes, quantum):
        if algo_key == "SJF":
            return sjf_non_preemptive(processes)
        if algo_key == "RR":
            return round_robin(processes, quantum)
        if len({p.priority for p in processes}) <= 1:
            return round_robin(processes, quantum)
        return priority_scheduling(processes)
