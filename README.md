# Adaptive AI-Driven CPU Scheduling

A modern, high-performance Python simulation of **True Runtime-Adaptive AI CPU Scheduling** alongside classical operating system schedulers. 

Unlike static schedulers or simple one-time batch classifiers, this scheduler continuously inspects the active ready queue, dynamically switches scheduling algorithms during runtime execution, calculates dynamic quantum slices, and optimizes a multi-objective composite score balancing latency, fairness, and turnaround time.

---

## ✨ Key Features

| Feature | Details |
|---|---|
| **True Runtime Adaptation** | Mid-execution policy switching triggered at scheduling events and quantum boundaries |
| **Multi-Objective Composite Optimization** | Optimizes $0.35 \times \text{AWT} + 0.35 \times \text{ART} + 0.15 \times \text{CS/proc} + 0.15 \times \sigma_{\text{waiting}}$ |
| **Starvation Prevention** | Dynamic aging & priority escalation for processes waiting $> 3\times$ mean burst |
| **Dynamic Round Robin Quantum** | Automatically scaled quantum $Q = \max(2, \min(Q_{\text{base}}, \text{median}(\text{rem})))$ for interactive queues |
| **Classic Schedulers** | FCFS, SJF (Non-Preemptive), SRTF (Preemptive SJF), Round Robin, and Priority |
| **Workload Input Modes** | 🎲 **Random Generator**, 📋 **Classic OS Presets** (Convoy, Interactive, Interrupts), ✏️ **Manual Table Input** |
| **Visual Analytics** | Dark glassmorphism Streamlit UI, multi-algorithm Gantt charts, adaptation audit log, and benchmark metric tables |

---

## 🚀 Quickstart

### 1 — Navigate to the project

```powershell
cd adaptive-cpu-scheduler
```

### 2 — Activate the virtual environment

```powershell
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
```

### 3 — Install dependencies (if not already installed)

```powershell
pip install -r requirements.txt
```

### 4 — (Optional) Train the Batch ML Workload Classifier

```powershell
python -m ml.data_generator      # Creates data/workload_dataset.csv
python -m ml.train_model         # Trains Random Forest model -> ml/model.joblib
```

### 5 — Launch the Interactive Streamlit Dashboard

```powershell
streamlit run dashboard.py
```

Open **http://localhost:8501** in your browser.

---

## 🧠 True Runtime Adaptation vs. Static Schedulers

Classical schedulers commit to a single policy regardless of changing workload characteristics:
- **FCFS**: Prone to the convoy effect when long CPU tasks block short tasks.
- **SJF / SRTF**: Mathematically optimal for Average Waiting Time (AWT) and Turnaround Time (ATAT), but starves long jobs and degrades interactive response time.
- **Round Robin**: High response time performance for interactive jobs, but increases average turnaround time for heterogeneous workloads.
- **Priority**: Guarantees urgent tasks run first, but risks indefinite starvation for lower-priority tasks.

### Adaptive Policy Decision Hierarchy

At every scheduling decision point (process completion or quantum expiration), `TrueAdaptiveScheduler` inspects the active ready queue:

1. **Starvation Guard**: If any task has zero CPU time and waiting time exceeds $3\times \text{mean burst}$, a bounded priority slice restores fairness.
2. **Urgent Priority Interruption**: Critical real-time tasks (`Priority == 1` or high relative priority with competitive burst) preempt lower-priority work immediately.
3. **Interactive I/O Queue**: When multiple interactive jobs ($\text{I/O Frequency} \ge 0.55$) are ready, Round Robin with dynamic $Q$ delivers fast initial response times ($\text{ART}$).
4. **Heterogeneous / CPU-bound (SRTF Core)**: For high burst variance or general compute batches, SRTF clears short jobs rapidly to minimize overall turnaround time.

---

## 📊 Workload Modes in Dashboard

1. **🎲 Random Generator**:
   - Customize process count, arrival time spans, burst ranges, priority distributions, and I/O probability.
2. **📋 Classic OS Presets**:
   - **Convoy Effect Demo**: Long CPU burst followed by short jobs (demonstrating preemption benefits).
   - **Mixed Batch & Interactive**: Alternates between heavy compute jobs and bursty I/O tasks.
   - **High-Priority Interruption**: Urgent real-time interrupts arriving mid-execution.
3. **✏️ Manual Table Input**:
   - Directly edit and enter custom PIDs, Arrival Times, Burst Times, Priorities, and I/O Frequencies in an editable interactive table.

---

## 📁 Project Structure

```
adaptive-cpu-scheduler/
├── core/
│   ├── process.py              # Process data class, metrics tracking, deepcopy reset
│   ├── schedulers.py           # FCFS, SJF, SRTF, Round Robin, Priority implementations
│   └── simulation_engine.py    # Process factory, batch runner & aggregate metric evaluator
├── scheduler/
│   ├── __init__.py
│   └── adaptive_scheduler.py   # TrueAdaptiveScheduler (Runtime Adaptation) & Legacy ML Scheduler
├── ml/
│   ├── data_generator.py       # Synthetic workload dataset generator
│   ├── train_model.py          # Random Forest training pipeline
│   └── predictor.py            # Feature extraction and classification inference
├── metrics/
│   └── calculator.py           # Comparison DataFrames, fairness calculations, metric formatters
├── dashboard.py                # Full-featured Streamlit dashboard with Plotly Gantt & analytics
├── requirements.txt            # Project dependencies
└── README.md
```

---

## 📐 Benchmark Metrics Evaluated

- **Average Waiting Time (AWT)**: Time processes spend waiting in the ready queue.
- **Average Turnaround Time (ATAT)**: Total elapsed time from arrival to completion.
- **Average Response Time (ART)**: Time from arrival to first CPU allocation.
- **Throughput**: Processes completed per unit of simulation time.
- **Context Switches**: Number of preemption and dispatch transitions.
- **Fairness ($\sigma_{\text{WT}}$)**: Standard deviation of waiting times (lower indicates more uniform service).
- **Composite Score**: Weighted multi-objective metric evaluating holistic scheduling performance:
  $$\text{Composite Score} = 0.35 \times \text{AWT} + 0.35 \times \text{ART} + 0.15 \times \left(\frac{\text{Context Switches}}{n}\right) + 0.15 \times \sigma_{\text{WT}}$$

---