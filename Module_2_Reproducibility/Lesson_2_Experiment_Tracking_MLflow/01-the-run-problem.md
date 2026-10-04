# 01 — The Run Problem

> [← Back to the ladder](README.md) · [Next: The MLflow Run →](02-the-mlflow-run.md)

---

## What you just did

You ran `01-pipeline-untracked.ipynb`. It worked. It cleaned the NYC taxi data, engineered 30+ features, trained 6 models, tuned the best one, and saved it to disk.

You looked at the comparison table. Random Forest won. Its val R² was something like 0.82. Its val MAE was around 4.3 minutes. You ran the test set — final R² and MAE printed. You saved the model to `models_nyc_taxi/<timestamp>/model.pkl`.

Everything worked.

**Now close the notebook.**

---

## What happens next

Open the notebook tomorrow. Run it again. What's different?

Maybe you changed `RF_N_ESTIMATORS` from 100 to 200. Maybe you added a feature. Maybe the taxi data got updated with new rows.

The notebook runs. New numbers print. A new model saves to a new timestamped folder.

Now answer these questions:

- Which run was better — this one or yesterday's?
- What were the hyperparameters of yesterday's Random Forest?
- How much did yesterday's model differ from today's?
- Which one should you ship to production?
- Yesterday's `model.pkl` — which version of the data trained it?

**You can't answer any of them.** The numbers printed to the terminal and scrolled away. The comparison table lived in memory and died with the kernel. The `model.pkl` file has no metadata attached — no hyperparameters, no metrics, no data version.

You have a folder of `.pkl` files and no way to tell them apart.

**That's the problem MLflow solves.**

---

## Why this is worse than it sounds

It's not just "you can't compare two runs." It's that every downstream task depends on being able to answer those questions.

**Debugging.** The model in production is suddenly giving bad predictions. Is it the same model you shipped last week? Or did someone overwrite the `.pkl`? You don't know — you have no record.

**Collaboration.** A teammate wants to build on your work. You send them the notebook. They get different numbers. Why? You don't remember what data version you used. They can't reproduce your result.

**Compliance.** A regulator asks what data trained the credit-risk model. You point at a `.pkl` file and shrug.

**Iteration.** You want to try a new feature. Was it better than the last feature you tried? You don't remember. You didn't keep a log.

**Production.** You want to roll back to last week's model. You have five `.pkl` files in the folder. Which one is last week's?

Every one of these breaks the moment you close the notebook. The pipeline is a black box that produced an output and forgot everything.

---

## What you actually need

You need a **record** for every run. Not a log file you write by hand. Not a spreadsheet. A **structured, queryable, comparable record** that captures:

| What | Why |
|------|-----|
| The **parameters** | So you know what configuration produced the result |
| The **metrics** | So you can compare across runs |
| The **artifacts** | So you can retrieve the model later |
| The **context** | So you know which code, data, and git commit produced it |
| The **lineage** | So you can trace the model back to its inputs |

And you need this record to live **outside the notebook** — so it survives the kernel restart. So it's visible to your whole team. So it can be queried, sorted, filtered, and compared.

That's what an experiment tracker is.

---

## What MLflow actually is

MLflow sounds abstract until you see what it actually is under the hood.

**MLflow is two things:**

1. **A database** — where runs, params, metrics, and artifacts are stored
2. **An API** — a way to write to that database (from Python) and read from it (from Python or the UI)

That's it. Underneath the branding, MLflow is:

```
┌─────────────────────────────────────────┐
│                                         │
│            MLflow UI (browser)          │
│                                         │
└───────────────┬─────────────────────────┘
                │  reads
                ▼
┌─────────────────────────────────────────┐
│                                         │
│      SQLite / Postgres / MySQL          │
│                                         │
│   ┌──────────────┬──────────────────┐   │
│   │ runs table   │ params table     │   │
│   │ metrics table│ artifacts table  │   │
│   │ tags table   │ registered models│   │
│   └──────────────┴──────────────────┘   │
│                                         │
└───────────────┬─────────────────────────┘
                │  writes
                ▲
┌───────────────┴─────────────────────────┐
│                                         │
│       Python API (mlflow.log_*)         │
│                                         │
└─────────────────────────────────────────┘
                │
                ▲
        your training loop
```

**Your training loop calls `mlflow.log_param(...)` and `mlflow.log_metric(...)`.**
MLflow serializes those calls into SQL rows.
The rows go into a database file — `mlflow.db`.
The UI is just a web app reading those rows.

When you open http://localhost:5000 and see the runs table, you're looking at rows from the database.

When you sort by `val_r2`, you're sending a `SELECT ... ORDER BY val_r2 DESC` to that database.

When you compare two runs side-by-side, you're doing a join between two rows.

**It's a database with a friendly UI.** That's the whole trick.

---

## Why a database (not a file)

Three reasons:

**1. Queries.** "Show me all runs where `val_r2 > 0.8` and `n_estimators >= 200`." In a file, you'd grep. In a database, one query.

**2. Concurrency.** Two people can log runs at the same time. Files would conflict.

**3. Structure.** Runs have a schema — params, metrics, tags, artifacts. A flat file can't express that cleanly.

This is why MLflow's default backend for a shared project is a **SQL database** (SQLite locally, Postgres or MySQL in production). The `mlruns/` directory is the file-based fallback — fine for a solo project, painful for a team.

**In this lesson, we use SQLite.** One file — `mlflow.db` — that acts as the backend store.

---

## The shape of a run

Every time you call `mlflow.start_run()`, MLflow creates a new row in the `runs` table. That row has columns for:

- `run_id` — a unique identifier
- `experiment_id` — which experiment this belongs to
- `status` — RUNNING, FINISHED, FAILED
- `start_time`, `end_time`
- `artifact_uri` — where the run's files are stored
- `user_id` — who ran it

Then, when you call:

- `mlflow.log_param('n_estimators', 100)` → adds a row to `params`
- `mlflow.log_metric('val_r2', 0.82)` → adds a row to `metrics`
- `mlflow.log_artifact('model.pkl')` → writes the file to the artifact store, adds a row to `artifacts`
- `mlflow.set_tag('model_family', 'Random Forest')` → adds a row to `tags`

Every call is a database write. Every query is a database read.

The `with mlflow.start_run(...)` block is a transaction. When you exit the block, MLflow closes the transaction and the run's status becomes FINISHED.

**That's the whole model.** A run is a database row. Everything else is metadata attached to it.

---

## The shape of a registered model

A registered model is also a database row — just in a different table.

When you call:

```python
mlflow.register_model("runs:/<run_id>/model", "nyc_taxi_predictor")
```

MLflow adds a row to the `registered_models` table for `nyc_taxi_predictor`, and a row to the `model_versions` table for version 1 of that model. The version references the run that produced it. The run references the params, metrics, and artifacts.

**So the chain is:**

```
model version  →  run  →  params + metrics + artifacts
       ↑
   "production"
   (a stage or an alias)
```

When you later load `models:/nyc_taxi_predictor/Production`, MLflow:

1. Queries the `model_versions` table for the version with stage `Production`
2. Follows the version's `run_id` to the `runs` table
3. Follows the run's `artifact_uri` to the artifact store
4. Loads the model file

Three database lookups and one filesystem read. That's how "load by stage" works.

---

## What the notebook does next

`02-pipeline-tracked.ipynb` is the same pipeline as before — same data, same models, same metrics — but every step is now logged to this database.

You'll see:

- `mlflow.set_tracking_uri("sqlite:///mlflow.db")` — point MLflow at a database
- `mlflow.set_experiment("nyc_taxi_duration")` — create a container for the runs
- `with mlflow.start_run(run_name=...):` — open a transaction, log everything, close it
- `mlflow.log_params(...)` — writes to the `params` table
- `mlflow.log_metrics(...)` — writes to the `metrics` table
- `mlflow.sklearn.log_model(...)` — writes the model to the artifact store
- `mlflow.search_runs(...)` — reads from the database instead of a local DataFrame
- `mlflow.register_model(...)` — creates a row in `registered_models`

By the end of the notebook, the database has ~7 runs and 1 registered model. The UI shows all of it.

**The pipeline is now a database — queryable, comparable, permanent.**

---

## Checkpoint

You should now be able to answer:

- **What breaks when you close the untracked notebook?** The comparison, the hyperparameters, the metrics, the ability to compare runs — all gone.
- **What is MLflow, literally?** A database + an API. Nothing more.
- **Why a database and not a file?** Structured queries, concurrency, schema.
- **What is a run?** A row in the `runs` table, with params, metrics, tags, and artifacts attached as foreign-key rows.
- **What is a registered model?** A row in the `registered_models` table, pointing at a version, pointing at a run, pointing at artifacts.

**Next:** [The MLflow Run →](02-the-mlflow-run.md)