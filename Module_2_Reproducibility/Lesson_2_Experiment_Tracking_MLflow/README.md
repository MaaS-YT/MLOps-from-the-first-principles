# Lesson 2.2 — Experiment Tracking with MLflow

> **Lesson 2.2** — Why unstructured experiment logs fail at scale, how MLflow turns every training run into a reproducible record, and how the model registry creates a clean handoff from experimentation to deployment.

| | |
|---|---|
| **Problem this solves** | After 20 experiments, you cannot answer: "What were the exact parameters for that 97% run?" Without structured logging, experiment results live in comments, memory, and spreadsheets — none of which survive team growth or time. |
| **Mental model** | MLflow is structured logging for ML. Each run is a record with three parts: what went **in** (params), what came **out** (metrics), and what was **produced** (artifacts). The registry is the promotion system: experiments → staging → production. |
| **What the lecture demonstrates** | Training three classifiers in a loop → logging params/metrics/models to MLflow → comparing runs in the UI → registering the best model → promoting it to Production stage |
| **Where this fits** | MLflow sits in the **Training Layer**. It connects experimentation (notebooks) to deployment (model registry). The serving system loads by stage (`Production`), not by run ID — making model updates independent of serving code. |

---

# MLflow Experiment Tracking — Concepts & Guide

Demo: `exp-tracking.ipynb` — Iris classification, three-model comparison and registry.

---

## Part 1 — The Experiment Tracking Problem

### The notebook chaos problem

ML development is experimental by nature. You try an algorithm. You tune parameters.
You try a different algorithm. You preprocess differently. You repeat.

Each attempt generates numbers — accuracy, loss, F1. After twenty experiments,
the critical questions become:

- Which configuration produced the best result?
- What exactly were the parameters for that run?
- Can I reproduce it?
- Can I compare two runs side by side?

Without a tracking system, the answers live in your memory, in comments in the notebook,
in a spreadsheet you may or may not have updated. This doesn't scale beyond a handful
of experiments, and it makes reproducibility nearly impossible.

**Experiment tracking is structured logging for ML experiments.**
Every run is a record: the parameters that went in, the metrics that came out,
and the artifacts that were produced. All queryable, all comparable.

---

### The model handoff problem

You found the best model. Now someone needs to use it — another service, a colleague,
a deployment system. They need to know:

- Which file is the model?
- Which version? (You trained four times this week.)
- What dependencies does it need to load correctly?
- How is it expected to be called?

This is the **model handoff problem**. Without a registry, the answer is email threads
and Slack messages. With a registry, there is a single catalog where models are
versioned, documented, and promoted through stages.

---

### The two problems MLflow solves

```
Experiment Tracking    "Which of my experiments was best, and how do I reproduce it?"
                       → Structured log: params + metrics + artifacts per run

Model Registry         "Which model is in production, and how do I update it safely?"
                       → Versioned catalog: register → staging → production → archived
```

This demo covers both.

---

### Where MLflow fits in the MLOps progression

```
1. Ad-hoc notebooks    experiments live in your head and notebook outputs
2. Spreadsheet logs    manual, error-prone, not linked to artifacts
3. MLflow tracking     every run logged automatically, comparable in UI    ← you are here
4. MLflow registry     promoted models, lifecycle management               ← and here
5. Automated training  pipelines trigger new runs on a schedule
6. Full CD4ML          continuous training + automated deployment
```

---

## Part 2 — Core MLflow Concepts

### The experiment

An experiment is a named container for a group of related runs.
It answers: "What problem am I trying to solve?" — not "How am I solving it?"

```
Experiment: "Iris_Classification_Comparison"
  All runs that attempt to classify Iris flowers belong here.
  Different algorithms, different parameters — same experiment.
```

In production, one experiment maps to one model version objective.
Experiments accumulate over the lifetime of a project.

---

### The run

A run is a single execution with a defined start and end.
It is the atomic unit of tracking.

```
Run: "RandomForest"
  Parameters (inputs):     model_type, n_estimators, random_state
  Metrics (outputs):       accuracy, f1_score, log_loss
  Artifacts (files):       the serialized model, plots, data samples
  Metadata (context):      start time, duration, git commit, run ID
```

Every run gets a unique `run_id`. That ID is the permanent reference.
Given a run ID, you can always find the exact model it produced.

---

### Parameters vs metrics vs artifacts

| Type | What it is | Examples |
|---|---|---|
| Parameter | Inputs you chose | `n_estimators=100`, `learning_rate=0.001` |
| Metric | Outputs you measured | `accuracy=0.97`, `log_loss=0.017` |
| Artifact | Files the run produced | trained model, confusion matrix, feature importances |

The distinction matters for the UI: parameters are used for filtering and grouping runs;
metrics are plotted over steps; artifacts are downloadable and deployable.

---

### Why log_loss, not just accuracy?

On the Iris dataset, all three models in the demo achieve 100% accuracy.
Accuracy alone cannot distinguish them.

**Log loss** measures not just whether predictions are correct, but how confident
they are. A model that correctly predicts class A with 51% probability and one that
does it with 99% probability both score the same on accuracy — log loss tells them apart.

```
lower log_loss = more confident correct predictions

RandomForest: 0.017  ← most confident
SVC:          0.077
LogReg:       0.111  ← least confident
```

In production, confidence matters: a model that is barely more likely than chance
to predict the right class is dangerous in high-stakes decisions, even if its
accuracy looks fine. Log loss is the better selection criterion when accuracies are tied.

---

### The model registry

The registry is a catalog, not a storage system. It organizes pointers to model artifacts
that were already logged during runs.

A registered model has versions and stages:

```
Model: "IrisBestModel"
  Version 1    (trained 2025-01-15)  stage: None → Staging → Production
  Version 2    (trained 2025-02-01)  stage: None → Staging (testing in progress)
  Version 3    (trained 2025-03-10)  stage: None (just registered)
```

Promoting a model from Staging to Production is a deliberate human (or automated) decision.
The serving system loads by stage (`Production`), not by version number.
When you promote Version 2, the serving system picks it up without a code change.

This is the **promotion workflow**: the interface between experimentation and deployment.

---

## Part 3 — The Demo Walkthrough

### What the demo does

Three classifiers are trained on the same Iris dataset in a loop.
Each run is logged to MLflow with identical structure — same params tracked,
same metrics tracked, same model format. The best run is then registered.

The demo shows:
1. How the experiment/run structure maps to real experiment management
2. How metrics are compared across runs
3. How the registry decouples the model from the code that serves it

---

### The context manager pattern

```python
with mlflow.start_run(run_name=name) as run:
    # train, evaluate, log
```

The `with` block defines the run's lifetime. MLflow records start time on entry
and end time + status on exit — even if an exception occurs. The run is always
cleanly closed. `run.info.run_id` is available inside the block.

---

### The log_model call

```python
mlflow.sklearn.log_model(model, name="model", input_example=X_test[:5])
```

This does more than `joblib.dump`. It saves:
- The serialized model (via joblib internally)
- A `conda.yaml` — the exact Python environment needed to reload it
- A `MLmodel` file — the model's signature (input/output schema)
- An `input_example` — a sample of real data for documentation and schema inference

The result is a **self-describing model package**.
MLflow can serve it as a REST endpoint with zero additional code:
`mlflow models serve -m runs:/{run_id}/model`.

---

### The registry connection

```python
mlflow.register_model(
    model_uri=f"runs:/{best_run_id}/model",
    name="IrisBestModel"
)
```

`model_uri` is a pointer, not a copy. The artifact stays where it was logged.
The registry just adds a named, versioned reference to it.

Loading from the registry decouples consuming code from specific run IDs:

```python
# Fragile: tied to a specific run
model = mlflow.sklearn.load_model("runs:/94d7b467.../model")

# Robust: tied to a stage — survives model updates
model = mlflow.sklearn.load_model("models:/IrisBestModel/Production")
```

The serving system never changes its code. You update the model by registering
a new version and promoting it.

---

## Part 4 — The Bigger Picture

### Where MLflow sits in the system

```
Experiments (notebooks, pipelines)
         │
         │  mlflow.log_param / log_metric / log_model
         ▼
MLflow Tracking Server
  ├── Run metadata (params, metrics, tags)
  └── Artifact store (model files, plots)
         │
         │  mlflow.register_model
         ▼
MLflow Model Registry
  └── Named models with versions and stages
         │
         │  mlflow.sklearn.load_model("models:/Name/Production")
         ▼
Serving system (API, batch job, notebook)
```

MLflow is the connection between experimentation and deployment.
Everything upstream produces artifacts. Everything downstream consumes them.
The registry is the handoff point.

---

### MLflow vs W&B

| | MLflow | W&B |
|---|---|---|
| Hosting | Self-hosted (local or server) | Cloud service |
| Model registry | First-class, built-in | Via Artifacts |
| Data lineage | Not built-in | Native (artifact graphs) |
| Team sharing | Needs shared tracking server | Immediate (cloud) |
| Best for | Enterprise, self-hosted | Team dashboards, quick start |

Both track experiments. The choice depends on your infrastructure constraints.
W&B is covered in the next lesson.

---

## Quick Reference

### Start the UI

```bash
mlflow ui
# → http://localhost:5000
```

### Core logging pattern

```python
mlflow.set_experiment("my_experiment")

with mlflow.start_run(run_name="my_run") as run:
    mlflow.log_param("n_estimators", 100)
    mlflow.log_metric("accuracy", 0.97)
    mlflow.sklearn.log_model(model, "model")
    print(run.info.run_id)
```

### Register and load

```python
# Register the best run's model
mlflow.register_model(f"runs:/{run_id}/model", "MyModel")

# Load by stage (used in serving)
model = mlflow.sklearn.load_model("models:/MyModel/Production")

# Load latest (used in notebooks)
model = mlflow.sklearn.load_model("models:/MyModel/latest")
```

### Serve a model as REST API

```bash
mlflow models serve -m "models:/MyModel/Production" -p 5001
# POST http://localhost:5001/invocations {"inputs": [[5.1, 3.5, 1.4, 0.2]]}
```

---

## Official Documentation

- MLflow tracking: https://mlflow.org/docs/latest/tracking.html
- Model registry: https://mlflow.org/docs/latest/model-registry.html
- MLflow concepts: https://mlflow.org/docs/latest/concepts.html
- Sklearn flavor: https://mlflow.org/docs/latest/python_api/mlflow.sklearn.html
