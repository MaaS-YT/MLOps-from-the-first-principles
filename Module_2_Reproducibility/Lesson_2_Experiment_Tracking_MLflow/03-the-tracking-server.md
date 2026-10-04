# 03 — The Tracking Server

> [← Previous: The MLflow Run](02-the-mlflow-run.md) · [Next: The Model Registry →](04-the-model-registry.md)

---

## Where the database actually lives

In the last rung, we said MLflow is a database + an API. But we never said **where** the database is.

That's what the **tracking URI** answers.

```python
mlflow.set_tracking_uri("sqlite:///mlflow.db")
```

This line tells MLflow: *"store all runs in the SQLite file `mlflow.db`, next to this notebook."*

Change the URI, and you change where runs are stored — without changing a single line of training code.

That's the abstraction. **Runs don't care where they live.** They care that they can be written and read. The tracking URI is the switch.

---

## The three kinds of backend

MLflow supports three deployment shapes. They differ only in where the data lives and who can see it.

### 1. Local file — `mlruns/`

```
mlflow.set_tracking_uri("file:./mlruns")
```

Or just omit it — this is the default.

Runs are stored as a directory tree on disk:

```
mlruns/
├── 0/                          ← default experiment
├── 1/                          ← your experiment
│   ├── <run_id>/
│   │   ├── meta.yaml           ← params, tags, status
│   │   ├── metrics/
│   │   ├── params/
│   │   ├── tags/
│   │   └── artifacts/
│   └── meta.yaml
└── models/                     ← registered models
```

**Pros:** zero setup. Works out of the box. Portable.

**Cons:** file-based, no real schema, no concurrency. Two writers to the same directory can corrupt it. The UI works, but the registry is crippled — no aliases, limited versioning.

**When to use:** solo projects, personal experiments, a quick sanity check.

### 2. SQLite — `mlflow.db`

```
mlflow.set_tracking_uri("sqlite:///mlflow.db")
```

One file. Real SQL. Real schema.

**Pros:** one file to copy. Real queries. Full registry support. Faster than file-based.

**Cons:** single-writer. If two processes write at once, you can corrupt the database. Not suitable for teams — you can't have three people writing to the same SQLite file over a network.

**When to use:** solo projects that need a real registry. This is what the lesson uses.

### 3. Remote server — `http://...`

```
mlflow.set_tracking_uri("http://mlflow.mycompany.com:5000")
```

MLflow runs as a server. Clients talk to it over HTTP. The server writes to a real database (Postgres, MySQL) and a real artifact store (S3, GCS, Azure Blob).

**Pros:** multiple writers. Team-wide. Real backend. Real registry. Real artifact store.

**Cons:** requires infrastructure. You have to run the server, provision the DB, configure the artifact store.

**When to use:** teams, production, anything that involves more than one person.

---

## The pattern that matters

**The training code doesn't change.** Only the tracking URI changes.

```python
# Same code, three backends
mlflow.set_tracking_uri("file:./mlruns")                # local
mlflow.set_tracking_uri("sqlite:///mlflow.db")          # local with registry
mlflow.set_tracking_uri("http://mlflow.internal:5000")  # remote team server

with mlflow.start_run():
    model.fit(X_train, y_train)
    mlflow.log_params(model.get_params())
    mlflow.log_metrics({...})
```

**Same `log_param`, same `log_metric`, same `log_model`.** The code is portable across all three backends.

This is why MLflow is powerful. You prototype locally with `mlruns/`, move to SQLite when you need the registry, and switch to a remote server when your team grows. **No code changes.**

---

## The artifact store

The tracking backend stores *metadata* — run IDs, params, metrics, tags, status. It does **not** store the artifacts themselves.

Artifacts — model files, plots, checkpoints — go to an **artifact store**, configured separately.

```
tracking backend                artifact store
─────────────────                ──────────────
runs table                       model.pkl
params table                     plot.png
metrics table                    feature_importance.csv
tags table
  │
  │  each run has an artifact_uri
  └────────────────────────────────►  location in the artifact store
```

For the local file backend, the artifact store is `mlruns/<experiment_id>/<run_id>/artifacts/` — same directory tree.

For SQLite, the artifact store defaults to `./mlruns/<experiment_id>/<run_id>/artifacts/` — a directory that lives next to the database.

For the remote server, the artifact store is configured separately — S3, GCS, Azure Blob, or a shared filesystem.

**The two are decoupled.** You can point SQLite at an S3 bucket, or a remote server at a local directory. The tracking backend stores where the artifact lives; the artifact store holds the artifact itself.

---

## What the UI actually is

Open the MLflow UI:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Two things happen:

1. MLflow starts a **local web server** on port 5000.
2. That server reads the SQLite file and renders it as HTML.

The UI is **not a separate application.** It's a Flask app that runs `SELECT * FROM runs WHERE ...` and formats the results. Every page load is a database query.

You can see this in the URL. Click on a run in the UI:

```
http://localhost:5000/#/experiments/1/runs/<run_id>
```

That `<run_id>` is the primary key in the `runs` table. The UI is showing you the row.

The "Metrics" tab on a run? Reading the `metrics` table with `WHERE run_id = '<run_id>'`.

The "Models" tab? Reading the `registered_models` and `model_versions` tables.

**The UI is a window into the database.** There's no magic. Just queries rendered as HTML.

---

## What the lesson uses

For this lesson, we use **SQLite**.

```python
mlflow.set_tracking_uri("sqlite:///mlflow.db")
```

**Why SQLite and not the default file-based backend?**

Because the file-based backend doesn't fully support the registry. No aliases, no proper versioning, and stage transitions are limited. Since rung 04 is about the registry, we need the backend that supports it.

**Why SQLite and not a remote server?**

Because the lesson is local. There's no team yet. The reader runs the notebook on their laptop. A remote server would require setup (Postgres, artifact storage, hosted MLflow) that has nothing to do with teaching the concepts.

**The SQLite backend is the right middle ground:** real schema, real registry, zero setup.

---

## Starting the UI

From the lesson directory:

```bash
cd Module_2_Reproducibility/Lesson_2_Experiment_Tracking_MLflow
uv run mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
```

Leave it running. Open http://localhost:5000.

**Terminal output:**

```
[INFO] Starting gunicorn ...
[INFO] Listening at: http://127.0.0.1:5000
[INFO] Using worker: sync
[INFO] Booting worker with pid: ...
```

**The server is now reading `mlflow.db`.** Every request to the browser becomes a query. Every write to the database (from the notebook) is immediately visible after refreshing the page.

---

## What you see in the UI

**Experiments view** — lists all experiments. You'll see `nyc_taxi_duration` (yours) and `Default`.

Click the experiment name. You see the runs table.

**Runs table** — one row per run. Sortable, filterable, searchable.

**Click a run** — the run detail page. Params, metrics, tags, and artifacts.

**Artifacts tab** — the model artifact and everything else the run produced. Click the `model/` folder to see the artifact contract: `MLmodel`, `model.pkl`, `conda.yaml`, `requirements.txt`.

**Models tab** (left sidebar) — the registry. Empty until we register a model in rung 04.

**Compare button** — select multiple runs and click "Compare" to see them side by side. Metrics in a table, params in a diff, scatter plots for hyperparameter analysis.

Every one of those views is a database query.

---

## Why this matters

The tracking server abstracts away *where* runs live. The same code writes to local files, SQLite, or a remote Postgres.

**For the reader, this means three things:**

1. **You can start local and grow.** Begin with `mlruns/`, move to SQLite when you need a registry, move to a server when you have a team. No rewrites.

2. **The UI is not the system.** The system is the database. The UI is a convenience. If you understand the schema, you can build your own views — with `mlflow.search_runs()` in Python, or with any tool that reads SQLite.

3. **Reproducibility is not tied to a specific machine.** When you move to a remote server, runs are stored on the server, not on your laptop. Your teammate sees the same runs. Your CI system sees the same runs. Your production system pulls models from the same registry.

The tracking server is the piece that makes MLflow a **team tool** instead of a personal logging library.

---

## What comes next

You have six runs in the database. You have a UI that queries them. You have artifacts stored next to them.

But the runs are still individual training executions. To ship one to production, you need something more: a **named, versioned, staged artifact** — the *registered model*.

That's rung 04.

---

## Checkpoint

You should now be able to answer:

- **What is a tracking URI?** The connection string that tells MLflow where to store runs.
- **What are the three backend types?** Local files (`mlruns/`), SQLite (`mlflow.db`), remote server (HTTP).
- **How does the training code change between backends?** It doesn't. Only the tracking URI changes.
- **What's the artifact store?** Where model files and other artifacts live. Configured separately from the tracking backend.
- **What is the UI, structurally?** A Flask app that reads the tracking backend and renders it as HTML. Every page is a query.
- **Why does this lesson use SQLite?** Full registry support, no setup, single file.
- **When do you switch to a remote server?** When you have a team or CI system that needs shared access.

**Next:** [The Model Registry →](04-the-model-registry.md)