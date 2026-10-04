# 02 — The MLflow Run

> [← Previous: The Run Problem](01-the-run-problem.md) · [Next: The Tracking Server →](03-the-tracking-server.md)

---

## What a run is

A run is **one execution of your training code.** One model trained. One set of params. One set of metrics. One artifact.

In `02-pipeline-tracked.ipynb`, every iteration of the training loop is one run. Six models → six runs.

Under the hood, a run is a row in the `runs` table of the MLflow database. Everything else — params, metrics, tags, artifacts — are rows in separate tables with a foreign key pointing back to the run.

```
runs
 ├── run_id: 47c45b47...
 ├── experiment_id: 1
 ├── status: FINISHED
 ├── start_time, end_time
 └── artifact_uri

params          metrics          tags           artifacts
 ├── run_id      ├── run_id       ├── run_id     ├── run_id
 ├── key         ├── key          ├── key        ├── path
 └── value       ├── value        └── value      └── ...
                 └── timestamp
```

**One run, many rows across tables.** That's the shape.

When the MLflow UI shows you a run, it queries those tables and renders the result. When you sort runs by `val_r2`, the query sorts the metrics table. When you filter by `model_family`, it filters the tags table.

The whole system is a relational database with a specific schema. Nothing more exotic than that.

---

## The `with` block

The context manager is how you open and close a run.

```python
with mlflow.start_run(run_name="Random Forest") as run:
    model.fit(X_train, y_train)
    mlflow.log_params(...)
    mlflow.log_metrics(...)
    mlflow.sklearn.log_model(...)
```

What happens, step by step:

1. **`mlflow.start_run()`** — inserts a row into the `runs` table with status `RUNNING`. Returns a `Run` object with a unique `run_id`.
2. **Everything inside the block** — each `mlflow.log_*` call writes rows to the corresponding table, all pointing at this run's `run_id`.
3. **Exit the block** — MLflow updates the run's status to `FINISHED` and records `end_time`.

If an exception is raised inside the block, MLflow catches it, sets the run's status to `FAILED`, and re-raises. The run still exists in the database — with its params and metrics up to the crash. That's why crashed runs are still visible in the UI.

**The `with` block is a transaction.** Open, write, close. Consistent.

---

## What each log call writes

Five methods do all the work.

### `mlflow.log_param(key, value)`

Writes one row to the `params` table.

```python
mlflow.log_param('n_estimators', 100)
mlflow.log_param('max_depth', 20)
mlflow.log_param('data_version', '4a1073c0...')
```

**Purpose:** capture the *inputs*. The hyperparameters, the data hash, the config values.

**Constraint:** params are **immutable** once set. If you try to log the same param twice, MLflow raises an error. That's deliberate — a run's config shouldn't change mid-flight.

**Tip:** use `mlflow.log_params(dict)` to log a whole dictionary at once:

```python
params = model.get_params()
mlflow.log_params({k: str(v) for k, v in params.items()})
```

Why `str(v)`? MLflow params must be strings or numbers. Functions, class instances, and `None` need to be coerced.

### `mlflow.log_metric(key, value)`

Writes one row to the `metrics` table.

```python
mlflow.log_metric('val_r2', 0.82)
mlflow.log_metric('val_mae', 4.31)
mlflow.log_metric('training_time', 12.5)
```

**Purpose:** capture the *outputs*. Accuracy, loss, R², MAE, time, anything numeric.

**Unlike params, metrics can be updated.** Call `log_metric('val_r2', x)` twice in the same run, and MLflow keeps the latest value. This is deliberate — during training, a metric might be logged per epoch.

**For per-step metrics** (loss per epoch, for example):

```python
for epoch in range(10):
    loss = train_epoch()
    mlflow.log_metric('loss', loss, step=epoch)
```

The `step` argument creates a time series. The UI plots it as a line chart instead of a single value.

### `mlflow.set_tag(key, value)`

Writes one row to the `tags` table.

```python
mlflow.set_tag('model_family', 'Random Forest')
mlflow.set_tag('data_source', 'dvc')
mlflow.set_tag('final_model', 'true')
```

**Purpose:** capture *context*. Not hyperparameters (those are params), not results (those are metrics). Things like "which team ran this", "which pipeline stage", "is this a candidate for production".

**Tags are searchable.** In the UI, you can filter runs by tag: `tags.model_family = 'Random Forest'`. In the Python API, you can search by tag too. This is how you organize a large experiment into groups.

### `mlflow.log_artifact(local_path)`

Copies a file into the run's artifact store and records its location.

```python
mlflow.log_artifact('feature_importance.png')
mlflow.log_artifact('confusion_matrix.csv')
```

**Purpose:** save *any file* the run produced. Plots, tables, configs, saved models.

Every run has its own directory in the artifact store:

```
mlruns/
└── <experiment_id>/
    └── <run_id>/
        └── artifacts/
            ├── feature_importance.png
            └── confusion_matrix.csv
```

The `artifact_uri` column in the `runs` table points at this directory.

### `mlflow.sklearn.log_model(model, artifact_path)`

A specialized version of `log_artifact` for sklearn models. Writes the model *plus* metadata:

- `model.pkl` — the serialized model
- `MLmodel` — the artifact contract (signature, framework, versions)
- `conda.yaml` — the exact environment to reload it
- `requirements.txt` — same, in pip format
- `python_env.yaml` — the Python version and pip dependencies

**Purpose:** save the model in a self-describing, portable format. Any MLflow-aware system can load and serve it without additional code.

We'll cover the artifact contract in detail in rung 04.

---

## What the notebook logs

Look at cell 9 of `02-pipeline-tracked.ipynb`. For each model in the portfolio:

```python
with mlflow.start_run(run_name=name) as run:
    # Train
    model.fit(X_train_processed, y_train)
    
    # Evaluate
    y_train_pred = model.predict(X_train_processed)
    y_val_pred = model.predict(X_val_processed)
    
    metrics = { 'train_r2': ..., 'val_r2': ..., ... }
    
    # Log hyperparameters
    mlflow.log_params(model.get_params())
    
    # Log context
    mlflow.log_param('model_type', name)
    mlflow.log_param('data_version', dvc_hash)
    mlflow.log_param('train_samples', split_info['train_size'])
    ...
    
    # Log metrics
    mlflow.log_metrics(metrics)
    
    # Log tags
    mlflow.set_tag('model_family', name)
    mlflow.set_tag('data_source', 'dvc')
    
    # Log the model
    mlflow.sklearn.log_model(model, 'model', signature=signature)
```

**Six runs.** One per model. Each run has ~20 params, 8 metrics, 2 tags, and 1 model artifact.

By the time the loop finishes, the database has six complete records. Each one is comparable, filterable, and reproducible.

---

## Why this matters for the reader

Look at what the run captures versus what notebook 01 captured.

| Question | Notebook 01 | Notebook 02 |
|----------|-------------|-------------|
| Which model won? | Printed to terminal, gone | Queryable in the UI |
| What were its hyperparameters? | Not recorded | In the `params` table |
| What was its val R²? | Printed, scrolled away | In the `metrics` table |
| Which data version trained it? | Not recorded | In the `params` table |
| Where's the model file? | Somewhere on disk | In the artifact store, linked to the run |
| Can I compare this run to last week's? | No | Yes, in the UI |

**The run is the atom of reproducibility.** Once every training execution becomes a run, everything downstream becomes possible — comparison, promotion, lineage, serving.

---

## The search API

Once runs are in the database, you can query them.

The UI is one way. The Python API is another:

```python
experiment = mlflow.get_experiment_by_name("nyc_taxi_duration")

runs_df = mlflow.search_runs(
    experiment_ids=[experiment.experiment_id],
    order_by=['metrics.val_r2 DESC'],
)
```

**`runs_df` is a pandas DataFrame.** Columns are `params.*`, `metrics.*`, `tags.*`, `run_id`, `start_time`, and so on.

The `where` clause is a filter:

```python
runs_df = mlflow.search_runs(
    experiment_ids=[experiment.experiment_id],
    filter_string="params.model_type = 'Random Forest' AND metrics.val_r2 > 0.8",
    order_by=['metrics.val_r2 DESC'],
)
```

**This is the payoff of the database.** The comparison table from notebook 01 was hand-built from a Python dict. The comparison table in notebook 02 is a query — and it can filter, sort, and join across runs.

Same idea, but the data is permanent and queryable.

---

## What comes next

You have six runs in the database. Each one is a complete record.

But there's a question the runs don't answer: **where does the database live?**

Right now, it lives in a file — `mlflow.db` — on your laptop. If your teammate wants to see your runs, they need your file. If you want to share runs across a team, you need a server.

That's the **tracking server** — rung 03.

---

## Checkpoint

You should now be able to answer:

- **What's a run, structurally?** A row in the `runs` table, with params, metrics, tags, and artifacts attached as foreign-key rows.
- **What does `log_param` write?** A row in the `params` table.
- **What does `log_metric` write?** A row in the `metrics` table. Can be updated within a run; supports `step` for time series.
- **What does `set_tag` write?** A row in the `tags` table. Searchable, filterable.
- **What does `log_model` write?** The model file, plus `MLmodel`, `conda.yaml`, and `requirements.txt` — a self-describing artifact package.
- **What's the `with` block?** A transaction. Open, write, close. Crashed runs are still visible with their status set to `FAILED`.
- **What's `search_runs`?** A query against the database, returned as a pandas DataFrame.

**Next:** [The Tracking Server →](03-the-tracking-server.md)