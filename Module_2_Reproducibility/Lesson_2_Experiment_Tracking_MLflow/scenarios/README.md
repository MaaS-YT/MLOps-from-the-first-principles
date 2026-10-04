# Scenarios — How MLflow Scales

The main lesson uses a single pipeline (NYC taxi) with a single team shape: you, on your laptop, with a local SQLite backend.

But MLflow looks different at different scales. These three scenarios show the same concepts applied to three team sizes.

| # | Scenario | Backend | Artifacts | What it demonstrates |
|---|----------|---------|-----------|---------------------|
| 01 | [Single scientist](scenario-1.ipynb) | Local filesystem (`mlruns/`) | Local filesystem | The simplest possible setup. No server. |
| 02 | [Small team](scenario-2.ipynb) | SQLite via local server | Local filesystem | One data scientist, one cross-functional team. Shared server. |
| 03 | [Multiple scientists](scenario-3.ipynb) | Postgres via remote server | S3 | The production pattern. EC2 server, Postgres backend, S3 artifacts. |

These are the **official MLflow tutorial notebooks.** They're the reference for how MLflow is deployed at each scale.

---

## What changes between them

Only three things:

**1. The tracking URI.** Where MLflow writes runs.

```python
# Scenario 01 — no server, local files
# (uses the default: file:./mlruns)

# Scenario 02 — local server with SQLite backend
mlflow.set_tracking_uri("http://127.0.0.1:5000")

# Scenario 03 — remote server with Postgres backend
mlflow.set_tracking_uri(f"http://{TRACKING_SERVER_HOST}:5000")
```

**2. The backend store.** Where the run metadata lives.

- Scenario 01: files in `mlruns/`
- Scenario 02: SQLite database (`backend.db`)
- Scenario 03: Postgres (managed by the server)

**3. The artifact store.** Where model files and other artifacts live.

- Scenario 01: files in `mlruns/`
- Scenario 02: local filesystem
- Scenario 03: S3 bucket

**Everything else is identical.** The `mlflow.log_param(...)`, `mlflow.log_metric(...)`, `mlflow.sklearn.log_model(...)` calls don't change.

**That's the point.** MLflow scales by changing where things are stored, not by changing the training code.

---

## When to read them

**After the ladder.** The four rungs teach the concepts — runs, tracking server, registry. The scenarios show the same concepts deployed at three scales.

**As reference.** If you're setting up MLflow for your team, look at the scenario that matches your size.

**As a decision aid.** Before you commit to an infrastructure, see what each scenario requires. Scenario 01 has zero setup. Scenario 03 requires AWS, Postgres, and S3 configuration.

---

## Running them

### Scenario 01 — no setup

Just run the notebook. It uses the default file-based backend. Runs land in `mlruns/` in the current directory.

View them:

```bash
mlflow ui --backend-store-uri file:./mlruns
```

### Scenario 02 — local server

Start the MLflow server in a terminal first:

```bash
mlflow server --backend-store-uri sqlite:///backend.db
```

The server runs at http://127.0.0.1:5000. Then run the notebook.

### Scenario 03 — remote server

Requires an AWS account and a deployed MLflow server. This is **not runnable locally.** The notebook is here as a reference for what the production setup looks like.

The MLflow documentation has a step-by-step guide for deploying the server to EC2. It's out of scope for this lesson — Module 5 (Cloud & Infra) covers this pattern in depth.

---

## What each scenario proves

**Scenario 01** — MLflow works with zero infrastructure. Start here.

**Scenario 02** — A shared server is one command away. Still no cloud needed.

**Scenario 03** — The same API scales to a full production deployment. Same `log_param`, same `log_metric`, same `log_model`.

**The progression is the point.** You don't rewrite your code when your team grows. You move the backend.