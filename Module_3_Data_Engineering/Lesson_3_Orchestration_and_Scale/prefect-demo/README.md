# Prefect Demo — Workflow Orchestration

> **Lesson 3.3 companion** — This demo shows the two core ideas Prefect adds to any pipeline: tasks with automatic retry, and a deployment with a schedule. Read the Lesson 3.3 guide for the why; this file is the how.

| | |
|---|---|
| **Problem this solves** | A pipeline that works in a notebook runs once, manually. Prefect makes the same pipeline run on a schedule, survive transient failures, and expose a UI where you can see every run's status, logs, and duration. |
| **Mental model** | A `@flow` is the pipeline. `@task`s are the steps. Prefect sits outside both and answers three questions: *Did it run? Did it succeed? When does it run next?* |
| **What the demo shows** | A mocked ML pipeline (fetch → process → train → deploy) decorated with `@flow`/`@task`, with retries, scheduled every 30 seconds, visible in the Prefect UI. |

---

## The Core Concepts

### Tasks and flows

```python
@task(retries=3, retry_delay_seconds=10)
def fetch_data():
    ...         # Prefect retries this up to 3 times if it raises an exception

@flow
def training_pipeline():
    raw = fetch_data()       # tasks are called inside flows
    ...
```

The `@task` decorator tells Prefect: "this is a unit of work — track it separately,
retry it on failure, log its duration." The `@flow` wraps them into a pipeline that
has a single run record in the UI.

Without Prefect, a failure anywhere in the pipeline crashes the whole script silently.
With Prefect, each task failure is caught, retried, and reported. The UI shows exactly
which task failed and why.

### The execution model

```
prefect server start        starts the UI + API at localhost:4200
                            this is just an observer — it doesn't run your code

python example.py           this runs your code locally
                            Prefect's SDK sends run metadata to the server
                            the server stores it and shows it in the UI

prefect deploy              registers a deployment: flow + schedule + work pool
prefect worker start        this is the process that actually executes runs
                            the worker polls the server for scheduled runs
                            when a run is due, the worker executes it
```

The server and the worker are separate processes on purpose: the server is stateless
(just an API + database), the worker is where execution happens. In production the
worker runs in a container on a schedule; the server can be Prefect Cloud.

### Schedules

A deployment in `prefect.yaml` includes a schedule:

```yaml
schedules:
  - cron: "* * * * *"       # every minute
  - interval: 30            # every 30 seconds
```

The worker checks for due runs at the interval and executes them. This is how you
turn a one-off script into a recurring pipeline — no cron jobs, no custom scheduler.

---

## Running the Demo

### Option 1 — Quick local run (no server)

Just run the flow directly. Output goes to stdout.

```bash
uv sync
uv run python example.py
```

### Option 2 — Full UI demo (server + worker + scheduled runs)

**Terminal 1** — start the server:

```bash
uv run prefect server start
# UI is now at http://localhost:4200
```

**Terminal 2** — create a work pool, deploy, start a worker:

```bash
export PREFECT_API_URL="http://localhost:4200/api"

uv run prefect work-pool create "local-process" --type process
uv run prefect deploy          # reads prefect.yaml, prompts for which flow
uv run prefect worker start -p "local-process" --type process
```

Open `http://localhost:4200` → Deployments → watch runs execute on schedule.

To stop: `Ctrl+C` in Terminal 2, then Terminal 1.

---

## What to look for in the UI

| UI section | What it shows |
|---|---|
| Flow Runs | Every execution — status, duration, start time |
| Logs | Full stdout/stderr for each task within a run |
| Deployments | Scheduled flows and their next run time |
| Work Pools | Which workers are alive and polling |

The Gantt-style task timeline is the most useful view for debugging: it shows which
task in a pipeline is slow or failing, with the retry history visible inline.

---

## The `example.py` structure

```
fetch_data()      → retries=3 — simulates a flaky data source (network, S3)
process_data()    → no retries — if transform logic fails, it's a code bug
train_model()     → no retries — deterministic given fixed data
deploy_model()    → no retries — idempotent write to model registry
```

The retry policy is deliberate: only tasks that fail for *transient* reasons
(network blips, rate limits, temporary unavailability) should be retried.
Tasks that fail for *logic* reasons should surface the error immediately.

---

## Quick Reference

```bash
# Run locally
uv run python example.py

# Start server
uv run prefect server start

# Deploy and schedule
uv run prefect deploy

# Start worker
uv run prefect worker start -p "local-process" --type process

# List deployments
uv run prefect deployments ls

# Cron syntax: min hour day month weekday
# 0 6 * * *   = daily at 6am
# */15 * * * * = every 15 minutes
```
