# Lesson 2.2 — Experiment Tracking with MLflow

> **Module 2 · Reproducibility** · [← Module 1: The ML System](../../Module_1_ML_Systems_Intro/) · [Lesson 2.3: W&B →](../Lesson_3_Experiment_Tracking_WandB/)

---

## The Problem

You trained six models on the NYC taxi dataset. Random Forest won. You tuned it, tested it, and saved it.

Now close the notebook.

Reopen it tomorrow. Run it again. What changed?

- Which run was better — this one or yesterday's?
- What were the hyperparameters of yesterday's Random Forest?
- Which data version trained it?
- Which `.pkl` file on your disk is the good one?

**You can't answer any of them.** The numbers printed to the terminal and scrolled away. The comparison table lived in memory and died with the kernel. The `.pkl` file has no metadata attached.

**The run link in the chain is broken.**

```
data → code → environment → run → artifact → model
                                  ✗
```

You have the code (git), the data (DVC), and the environment (Docker). But every training execution leaves no trace. You cannot compare runs, cannot reproduce them, cannot promote them.

That's the problem MLflow solves.

---

## The Mental Model

**MLflow is a database + an API.**

Underneath the branding:

- A **database** — where runs, params, metrics, tags, and artifacts are stored
- An **API** — Python methods to write to it, a web UI to read from it

That's the whole system. Every `mlflow.log_param(...)` writes a row. Every `mlflow.search_runs(...)` reads rows. The UI is a Flask app that queries the database and renders the results as HTML.

The same three-part shape you saw in Module 1 and Lesson 2.1:

```
.mlflow run   →  metadata + artifacts  →  loadable model
definition    →  artifact              →  instance
.dvc file     →  hash + remote         →  checked-out data
Dockerfile    →  image                 →  container
```

Same triune. Different link in the chain.

---

## The Ladder

Read the four rungs in order. Each assumes the previous.

| Rung | Guide | What you learn |
|------|-------|----------------|
| 01 | [The Run Problem](01-the-run-problem.md) | Why losing your trained models is worse than it sounds |
| 02 | [The MLflow Run](02-the-mlflow-run.md) | What a run is — params, metrics, tags, artifacts |
| 03 | [The Tracking Server](03-the-tracking-server.md) | Where the database lives. Local, SQLite, remote. |
| 04 | [The Model Registry](04-the-model-registry.md) | Register, stage, alias, load by name |

**Start with rung 01.** Don't skip to the commands. The commands only make sense once the problem is clear.

**Two notebooks** accompany the rungs:

- `01-pipeline-untracked.ipynb` — the same pipeline with **no tracking**. Run this first.
- `02-pipeline-tracked.ipynb` — the same pipeline, every step logged to MLflow.

The contrast between them is the lesson.

**Scenarios:** [`scenarios/`](scenarios/) — three small notebooks showing how MLflow is used at different team sizes.

---

## The Four-Step Progression

Where MLflow sits in the longer arc of experiment tracking:

```
1. Print statements       nothing persists, no comparison
2. Ad-hoc logs           spreadsheets, comments, memory — fragile
3. MLflow local          runs in mlruns/, solo scientist       ← rung 01–02
4. MLflow with registry  stages, aliases, load by name         ← rung 03–04
5. Team tracking server  shared backend, Postgres, S3
6. Fully automated CI/CD pipelines log runs on every commit
```

MLflow covers rungs 3 and 4 of this ladder. Modules 5 and 6 cover 5 and 6.

---

## Quick Start

Run this after walking the ladder — it's the destination, not the starting point.

```bash
cd Module_2_Reproducibility/Lesson_2_Experiment_Tracking_MLflow

# Pull the dataset (DVC)
uv run dvc pull data/sample_nyc_taxi.csv.dvc

# Start the MLflow UI in a separate terminal
uv run mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000

# Open http://localhost:5000

# Run the tracked notebook
jupyter lab 02-pipeline-tracked.ipynb

# After it finishes, refresh the UI — 6 runs are now visible
```

**Verify the registry:**

```python
import mlflow
from mlflow import MlflowClient

mlflow.set_tracking_uri("sqlite:///mlflow.db")
client = MlflowClient()

# Show the registered model versions
for v in client.search_model_versions("name='nyc_taxi_predictor'"):
    print(f"v{v.version}  stage={v.current_stage}  run={v.run_id[:8]}")
```

Expected:

```
v1  stage=Production  run=abc12345
```

**Load by stage:**

```python
model = mlflow.sklearn.load_model("models:/nyc_taxi_predictor/Production")
```

Load by stage or alias — not by file path. That's the whole point.

---

## What You Should Be Able to Answer at the End

- **What is MLflow, underneath the branding?** A database + an API. Runs are rows.
- **What is a run?** One training execution, stored as a row with params, metrics, tags, and artifacts attached.
- **What does `log_param` write vs `log_metric` vs `set_tag`?** Rows in the `params`, `metrics`, and `tags` tables respectively.
- **Where does the database live?** Whatever the tracking URI says — a local file, SQLite, or a remote server.
- **What's the artifact store?** Where model files and other artifacts go. Configured separately from the tracking backend.
- **What's a registered model?** A named, versioned pointer to a run's model artifact.
- **What are stages and aliases?** Two ways to label which version is the "current" one. Stages are classic; aliases are modern.
- **What does the registry decouple?** The model *reference* (a name) from the model *artifact* (a file). Serving code loads by name.
- **What's the triune here?** Run → artifact package → loadable model. Definition → artifact → instance.

---

## Where This Fits

This lesson is the **run link** in the reproducibility chain.

```
data → code → environment → run → artifact → model
 ↑                    ↑       ↑        ↑       ↑
2.1                 Module 1  2.2     2.2     2.3
DVC                 Docker   MLflow  MLflow  W&B
```

- **Backward:** Lesson 2.1 gave you versioned data. This lesson anchors runs to a specific data version.
- **Backward:** Module 1 gave you a container that runs identically anywhere. MLflow runs on top of that container.
- **Forward (Lesson 2.3):** W&B extends MLflow with cloud-native collaboration and artifact lineage across teams.
- **Forward (Module 4):** Optimized models are also artifacts. They log to MLflow the same way.
- **Forward (Module 5):** Kubernetes pulls versioned artifacts from a registry. The registry is what this lesson teaches.

> 📍 **Full chain diagram:** [`../README.md`](../README.md)

---

## Files

| File | Purpose |
|------|---------|
| `01-the-run-problem.md` … `04-the-model-registry.md` | The four rungs — read in order |
| [`01-pipeline-untracked.ipynb`](01-pipeline-untracked.ipynb) | The pipeline with no tracking |
| [`02-pipeline-tracked.ipynb`](02-pipeline-tracked.ipynb) | The same pipeline, fully tracked |
| [`scenarios/`](scenarios/) | The three official MLflow tutorial notebooks (single scientist, small team, multi-scientist) |
| [`data/`](data/) | DVC-tracked NYC taxi sample (100K rows) |
| [`mock-remote/`](mock-remote/) | Local DVC remote |
| [`scripts/download_data.py`](scripts/download_data.py) | One-shot dataset downloader |
| `mlflow.db` | The SQLite tracking backend (regenerated by running the notebook) |
| `models_nyc_taxi/` | Local model output (regenerated by running the notebook) |

---

## The Scenarios

The lesson uses NYC taxi as its single demo. But MLflow looks different at different scales. The `scenarios/` folder has the three official MLflow tutorial notebooks — one per team size:

| Scenario | Team size | Backend | Artifacts |
|----------|-----------|---------|-----------|
| [01 — Single scientist](scenarios/scenario-1.ipynb) | One person | Local files (`mlruns/`) | Local files |
| [02 — Small team](scenarios/scenario-2.ipynb) | Cross-functional team | SQLite via local server | Local files |
| [03 — Multiple scientists](scenarios/scenario-3.ipynb) | Distributed team | Postgres via remote server | S3 |

Each notebook is self-contained. Only the tracking URI changes between them — the training code is identical.

**Read them after the ladder.** Scenario 01 runs with zero setup. Scenario 02 requires a local MLflow server. Scenario 03 is a reference for production — not runnable without AWS.

**Start with:** [`scenarios/README.md`](scenarios/README.md)
---

## Key Terms

| Term | Definition |
|------|------------|
| **Tracking URI** | Connection string telling MLflow where runs are stored |
| **Experiment** | A named collection of runs |
| **Run** | One training execution. Params, metrics, tags, artifacts. |
| **Params** | Inputs: hyperparameters, config, data version |
| **Metrics** | Outputs: R², MAE, time, loss |
| **Tags** | Context: team, model family, stage of experiment |
| **Artifact** | Any file the run produced |
| **Artifact store** | Where artifacts live — separate from the tracking backend |
| **MLmodel file** | The artifact contract. Framework, environment, signature. |
| **Signature** | Declared input/output schema of a model |
| **Registered model** | A named, versioned pointer to a run's model artifact |
| **Stage** | Exclusive label: None, Staging, Production, Archived |
| **Alias** | Named, non-exclusive pointer. Modern replacement for stages. |
| **`MLflowClient`** | Python class for registry operations and query API |

---

## Checkpoint

You should now be able to answer:

- **What does MLflow store?** Runs — as rows in a database. Params, metrics, tags, artifacts attached.
- **Where does it store them?** Whatever the tracking URI points at — file, SQLite, or remote server.
- **What's the difference between a run and a registered model?** A run is one execution. A registered model is a named, versioned pointer to a run's artifact.
- **How do you load by stage?** `mlflow.sklearn.load_model("models:/name/Production")`.
- **How do you load by alias?** `mlflow.sklearn.load_model("models:/name@champion")`.
- **Why does this matter for production?** Serving code loads by name, not file path. Promoting a new version doesn't change any serving code.

**Start here:** [Rung 01 — The Run Problem →](01-the-run-problem.md)

---

## How to Use This Lesson

**For a first read:** open `01-pipeline-untracked.ipynb`, run it end to end, then read rung 01. Then open `02-pipeline-tracked.ipynb`, run it, and read rungs 02–04.

**For the registry:** if you already understand MLflow tracking and just want the registry, jump to rung 04 and cells 13–19 of notebook 02.

**For team patterns:** read the scenarios after the ladder. Each is self-contained.

---

> *A notebook is where you experiment.*
> *MLflow is where the experiment lives.*