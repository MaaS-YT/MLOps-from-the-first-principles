# Lesson 3.3 — Orchestration and Scale: Prefect + Spark

> **Lesson 3.3** — The two distinct problems that end a notebook pipeline's usefulness: scale (data too large for one machine) and reliability (a pipeline that runs unattended must recover from failure). Two tools, two problems.

| | |
|---|---|
| **Problem this solves** | A pipeline that works in a notebook runs once, manually. A production pipeline runs on a schedule, handles failures gracefully, retries automatically, and processes data that may be larger than any single machine's memory. These require different infrastructure. |
| **Mental model** | Orchestration and distribution are orthogonal. Prefect answers: *"when does this run, and what happens if it fails?"* Spark answers: *"how do we process data that doesn't fit on one machine?"* A mature system needs both — independently. |
| **What the lecture demonstrates** | Spark: a PySpark ML pipeline that partitions data across workers — the same logic as pandas but distributed; Prefect: a flow with tasks, retries, scheduling, and a UI showing run history and failures |
| **Where this fits** | This module adds the **Infrastructure layer** — Prefect schedules the entire system shown in the map; Spark scales the Data Layer to production data volumes. |

---

# Data Pipeline Part 3 — Orchestration & Distribution

Sub-demos: `spark-demo/` — distributed computation with PySpark. `prefect-demo/` — workflow orchestration with Prefect.

---

## The Two Problems This Part Solves

Parts 1 and 2 of this series built a data pipeline that works. It ingests raw data,
validates it, engineers features, and produces a training-ready dataset. Run in a
notebook, it produces correct results.

But a pipeline that only runs when you manually trigger it in a notebook is not a
production system. Two distinct problems remain:

**Problem 1 — Scale**: Data that doesn't fit in memory, or computation that takes
too long on one core. This is the **distribution** problem.

**Problem 2 — Reliability**: A pipeline that needs to run on a schedule, recover
from failures, retry individual steps, and let you observe what happened.
This is the **orchestration** problem.

These are separate concerns. A pipeline can be distributed without being orchestrated,
and orchestrated without being distributed. In production, most mature pipelines need both.

---

## Mental Model 1 — Distribution

### What distribution actually means

Distribution is the act of splitting a problem across multiple units of compute
so they work in parallel. The goal is to escape the constraints of a single machine.

A single machine has a fixed amount of RAM and a fixed number of CPU cores.
Any computation that exceeds those limits either fails (out-of-memory) or takes
too long (single-threaded). Distribution solves both by adding more machines —
or more cores — to the problem.

The mental model is a **coordinator and workers**:

```
Coordinator (Driver)
│  Holds the full plan
│  Divides data into chunks (partitions)
│  Assigns chunks to workers
│  Collects and merges results
│
├── Worker A  →  processes partition 0, 1, 2
├── Worker B  →  processes partition 3, 4, 5
└── Worker C  →  processes partition 6, 7, 8
```

The coordinator never holds all the data at once. Workers only see their own
partitions. The result is assembled by the coordinator at the end.

---

### The partition is the unit of work

Everything in distributed computation flows from one idea: **the partition**.

A partition is a chunk of the dataset small enough to fit in one worker's memory.
The system creates partitions automatically when data is loaded. Every transformation
you write operates on partitions in parallel — the same code runs on each worker,
on its own slice of data, simultaneously.

```
Full dataset (too large for one machine)
  │
  ├── Partition 0   →  Worker A applies transform  →  result 0
  ├── Partition 1   →  Worker B applies transform  →  result 1
  ├── Partition 2   →  Worker C applies transform  →  result 2
  └── ...
                               │
                          Merge results
```

The partition count determines parallelism. More partitions means finer-grained
parallelism and better utilization across many workers — but also more coordination
overhead. The right partition count depends on data size and cluster shape.

---

### Lazy evaluation: the plan before the execution

Distributed systems cannot afford to execute every operation as it's written.
Network calls are expensive. Sending data between workers is expensive. Redundant
reads of the same data are expensive.

The solution is **lazy evaluation**: transformations are recorded as a logical plan.
Nothing executes until you request a result. When you do, the engine optimizes the
full plan — pushing filters down, merging passes, eliminating redundant shuffles —
and then executes the optimized plan in one sweep.

```
Code written (logical plan built):
  df.filter(...)
    .groupBy(...)
    .agg(...)

No execution yet.

Code triggered (action called):
  df.count()   ← action

Engine optimizes the full plan, then executes.
```

This is different from Pandas, which executes eagerly — every operation runs
immediately. Lazy evaluation is what allows distributed systems to be efficient
at scale.

---

### What distribution does NOT solve

Distribution is about **throughput and scale**. It does not help with:

- Running a pipeline on a schedule
- Retrying a step that fails midway through
- Alerting you when something goes wrong
- Giving you visibility into what ran, when, and with what result

Those problems belong to orchestration.

---

## Mental Model 2 — Orchestration

### What orchestration actually means

Orchestration is the management of a workflow: its scheduling, execution,
failure handling, and observability. Where distribution answers "how do we
compute this at scale?", orchestration answers "how do we make sure this runs
reliably, repeatedly, and visibly?"

The mental model is a **conductor, score, and musicians**:

```
Scheduler (Conductor)
│  Knows when the flow should run
│  Triggers execution at the right time
│  Monitors progress
│
Flow (Score)
│  The complete description of what should happen
│  Defined as code — not config, not a UI
│  Composed of tasks in dependency order
│
Tasks (Musicians)
│  Individual steps of the pipeline
│  Each has a defined input, output, and retry policy
│  Failures are isolated — one task failing doesn't corrupt others
│
Worker (Execution engine)
   Picks up scheduled runs
   Executes tasks on real infrastructure
   Reports status back to the scheduler
```

---

### The flow-task hierarchy

The central abstraction in orchestration is the **flow**: a function that defines
a complete pipeline by calling tasks in sequence or in parallel.

```python
@task(retries=3, retry_delay_seconds=10)
def fetch_data():
    return "data.csv"

@task
def process_data(path: str):
    return "processed.parquet"

@flow(name="training_pipeline")
def training_pipeline():
    raw       = fetch_data()
    processed = process_data(raw)
    model     = train_model(processed)
    deploy_model(model)
```

The key properties of this hierarchy:

**Tasks are the retry boundary.** If `process_data` fails, the orchestrator
retries that task from its input — not the entire flow. `fetch_data` is not
re-run. This is critical: in a long pipeline, re-running from the beginning
every time a downstream step fails wastes enormous compute.

**The flow defines the dependency graph.** When `process_data(raw)` receives
`raw` as an argument, the orchestrator knows `process_data` depends on `fetch_data`.
Dependencies are expressed through data flow in code, not through a separate
configuration file.

**The flow is the schedulable unit.** You deploy a flow, not a task. The scheduler
triggers the flow on a cron schedule; the flow decides which tasks to run and in
what order.

---

### The scheduler-worker separation

Orchestrators split the concern of "knowing when to run" from "actually running":

```
Orchestrator Server
│  Stores flow metadata, schedules, run history
│  Exposes a UI and API
│  Sends work to workers when schedule fires
│  Never executes user code directly
│
Worker (runs on your infrastructure)
│  Polls the server for scheduled work
│  Pulls the flow code and executes it
│  Reports task state (running, completed, failed) back to server
│  Executes on whatever infrastructure you configure — local, Docker, Kubernetes
```

This separation means the orchestrator server is lightweight and stateless with
respect to execution. Workers are disposable — if a worker crashes, the server
reschedules the run to another worker. The run history and state live in the server,
not in the worker.

In the Prefect demo, the server runs at `localhost:4200`, and a local process
worker polls it. In production, the server is hosted (Prefect Cloud or self-hosted)
and workers run inside your data infrastructure.

---

### Observability: the output of orchestration

The deepest value of an orchestrator is not scheduling — cron can schedule.
It is **observability**: a structured record of every run, every task, every
log line, and every failure, queryable from a single UI.

Without orchestration, you know a pipeline ran because the output file exists.
You do not know when it ran, how long each step took, which step failed last Tuesday,
or whether the data from three weeks ago came from a clean run or a partial retry.

With orchestration, every run is a first-class object:

```
Run ID:     abc-123
Flow:       training_pipeline
Started:    2025-03-10 02:00:01
Duration:   4m 37s
Status:     Completed

Tasks:
  fetch_data      ✓  0m 12s
  process_data    ✓  2m 18s   (retried once: network timeout at 0m 03s)
  train_model     ✓  1m 55s
  deploy_model    ✓  0m 12s
```

This record exists for every run, forever, without any extra instrumentation.
It is the difference between operating a data system and guessing at it.

---

## How Distribution and Orchestration Fit Together

They operate at different layers of the stack and solve different problems.
Neither replaces the other.

```
Orchestration layer (Prefect, Airflow, Dagster)
  What runs, when, in what order, with what retry policy

        │
        │  triggers
        ▼

Execution layer (your Python code, Spark jobs, SQL queries)
  The actual computation

        │
        │  may delegate scale to
        ▼

Distribution layer (Spark, Dask, Ray)
  How a single computation step handles data too large for one machine
```

A Prefect flow can call a Spark job as one of its tasks. Prefect handles the
scheduling, retry, and observability. Spark handles the distributed computation
inside that task. Each tool does what it is designed for.

---

## Where This Fits in the Pipeline Progression

```
Part 1  Manual pipeline       ingest, validate, feature engineering, one notebook
Part 2  Feature store (Feast) centralized feature serving, versioned, reusable
Part 3  Distribution          Spark for data that doesn't fit in memory        ← spark-demo
        Orchestration         Prefect for scheduling, retries, observability   ← prefect-demo
  ↓
Part 4  Monitoring            detect data drift before it reaches the model
Part 5  CI/CD for ML          automated testing and deployment of pipelines
```

Parts 1 and 2 are about correctness. Part 3 is about production-readiness.
A pipeline that is correct but not orchestrated is a script. A pipeline that
is orchestrated but not distributed is a scheduled script. Both together are
a production data system.

---

## Sub-Demo Reference

| Demo | Tool | What it shows |
|---|---|---|
| `spark-demo/` | PySpark | Distributed DataFrames, lazy evaluation, MLlib Pipeline, time-based split |
| `prefect-demo/` | Prefect | `@flow` / `@task` decorators, retries, scheduling, deployment, local UI |

Read `spark-demo/SPARK_GUIDE.md` and `prefect-demo/README.md` for the tool-specific details.
