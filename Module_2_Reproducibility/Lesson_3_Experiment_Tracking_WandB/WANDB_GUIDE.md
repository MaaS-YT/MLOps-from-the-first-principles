# Lesson 2.3 — Experiment Tracking with Weights & Biases

> **Lesson 2.3** — Why self-hosted experiment tracking breaks down for teams, how W&B's artifact lineage solves the data provenance problem, and the LSTM demo as a complete multi-stage pipeline with full traceability.

| | |
|---|---|
| **Problem this solves** | Two problems MLflow doesn't solve out of the box: (1) shared visibility — your teammate can't see your local MLflow server; (2) data provenance — you know which model won, but not which data version trained it. |
| **Mental model** | W&B is a cloud-native MLflow with one extra layer: **artifacts**. An artifact is any versioned file (dataset, model, evaluation). `use_artifact()` creates a lineage edge. The full chain — raw data → processed data → model — becomes a queryable graph. |
| **What the lecture demonstrates** | Two demos: (1) a two-stage sklearn pipeline with explicit artifact hand-off between data upload and training; (2) a three-stage LSTM pipeline showing real-time training curves, model checkpointing with aliases, and the full lineage graph in the W&B UI |
| **Where this fits** | W&B completes the **Training Layer** alongside MLflow. The key addition is artifact lineage — you can trace any production model back through its training run to the exact dataset version that produced it. |

---

# Weights & Biases — Concepts & Guide

Demo: `sklearn_demo.ipynb` (house price prediction) + `ts_lstm_demo.ipynb` (sales forecasting LSTM).

---

## Part 1 — What W&B Adds Over Basic Tracking

### The shared visibility problem

MLflow (previous lesson) solves the personal experiment chaos problem.
But teams have a second problem: shared visibility.

A data scientist logs experiments to their local MLflow server.
Their manager can't see them. Their colleague is working on the same model and doesn't
know what's been tried. The ML engineer deploying the model has no access to the run history.

**W&B is a cloud-native experiment tracker.** Every run syncs to wandb.ai in real time.
The whole team sees the same dashboards, the same runs, the same artifacts — without
setting up a shared server.

---

### The data provenance problem

MLflow tracks models. But models are downstream of data.
If the data changes — different preprocessing, different filter, different version —
the model changes. And unless you know *which* data version trained *which* model,
you can't reproduce or debug anything.

W&B solves this with **Artifacts**: versioned objects for anything that matters —
raw datasets, processed datasets, models, evaluation reports.

```
raw-sales-data:v0
    ↓  used by: preprocess-data run
processed-sales-data:v0 (train.csv + validation.csv)
    ↓  used by: training run
sales-forecasting-lstm:v0 (best_model_bundle.pth)
```

This graph — the artifact lineage — answers the question:
"What exactly was this model trained on, and how was that data produced?"

---

### Where W&B fits in the MLOps progression

```
1. Print statements        no history, no comparison
2. MLflow (self-hosted)    structured local logs, model registry
3. W&B (cloud)             team-visible, real-time, artifact lineage    ← you are here
4. W&B Sweeps              automated hyperparameter search
5. W&B Reports             shareable experiment summaries for stakeholders
6. Full observability      W&B + feature store + model monitoring
```

---

## Part 2 — Core W&B Concepts

### The Run

A Run is a single execution of your code — one training job, one data upload,
one preprocessing step. It is the same concept as an MLflow run, but with richer
real-time visualization.

Every run belongs to a **project** (the task) and carries a **job_type** (the stage):

```
project = "sales-forecasting"   (the task — all runs related to this belong here)
job_type = "ingest-data"        (the stage — what this particular run does)
job_type = "preprocess-data"
job_type = "training"
```

`job_type` is not just documentation. In W&B's lineage graph, it helps group and
visualize the pipeline stages as a connected flow.

---

### The Artifact

An Artifact is a versioned, named collection of files.
Unlike metrics (which are numbers), Artifacts are anything file-based:
datasets, models, plots, evaluation results.

```
Artifact concept:
  name        = "raw-sales-data"          logical identifier
  type        = "raw_dataset"             category for organization
  version     = v0, v1, v2...             auto-incremented by W&B
  files       = raw_sales_data.csv        the actual content
  metadata    = {"source": "UCI"}         any JSON-serializable info
```

W&B auto-versions: first upload creates `v0`, next creates `v1`.
`latest` is a mutable alias always pointing to the newest version.

**Using** an artifact (not just logging it) creates the lineage link:

```python
artifact = run.use_artifact("raw-sales-data:latest")  # creates link in lineage graph
```

After this call, W&B knows: "This run consumed raw-sales-data:v0."
The lineage graph is built automatically from `use_artifact` and `log_artifact` calls.

---

### The Config

```python
with wandb.init(config={"learning_rate": 0.001, "epochs": 50}) as run:
    config = wandb.config  # use this, not the local dict
    lr = config.learning_rate
```

Passing config to `wandb.init` and reading from `wandb.config` (not the local dict)
is the pattern that enables **W&B Sweeps** — automated hyperparameter search.
A Sweep can inject different config values into the same training script.
If you read from a local dict, Sweeps can't override it.

Writing code in this pattern from the start costs nothing and unlocks automation later.

---

### wandb.log — scalars and live charts

```python
wandb.log({"epoch": epoch, "train_loss": 0.34, "val_loss": 0.41})
```

Unlike MLflow's per-run metrics, `wandb.log` supports **per-step logging**.
Call it once per epoch (or per batch), and W&B builds a live time-series chart
you can watch as training runs — useful for catching divergence early.

---

## Part 3 — The sklearn Demo (House Price Prediction)

### What the demo shows

Two pipeline stages, two separate W&B runs, one artifact connecting them:

```
load_data()     Upload California Housing dataset as an artifact
    ↓
train()         Download the artifact → train Random Forest → log metrics + model
```

The separation into two runs with explicit artifact hand-off is the key pattern.
It means:
- The data upload step can run once, even if training runs ten times
- You always know exactly which data version each training run used
- Swapping the dataset (use a different version) doesn't require changing the training code

---

### The diagnostic plots

```python
wandb.sklearn.plot_regressor(model, x_train, x_test, y_train, y_test)
```

W&B generates residuals plots, learning curves, and outlier candidates automatically
and logs them as interactive charts. This requires no matplotlib code on your part.

These plots are important not just for evaluation but for **debugging**:
systematic patterns in residuals indicate the model is missing a feature or the
relationship is non-linear; a learning curve that hasn't plateaued suggests more
data would help.

---

## Part 4 — The LSTM Demo (Sales Forecasting)

### What the demo shows

A three-stage pipeline entirely managed through W&B artifacts:

```
ingest_raw_data()      Download UCI Online Retail → raw-sales-data artifact
        ↓
preprocess_data()      541k transactions → daily sales → train/val split
                       Log data quality metrics → processed-sales-data artifact
        ↓
train_forecaster()     LSTM on daily sales → per-epoch loss tracking
                       Checkpoint best model → sales-forecasting-lstm artifact
```

Each stage is a W&B run. Each stage's output is an artifact used by the next stage.
The full lineage is visible in W&B's lineage graph.

---

### The time series preprocessing concepts

**Why aggregate to daily?**
Transaction-level data (one row per item sold) is too granular for forecasting.
The model needs to learn patterns over time — day-level granularity captures
weekly seasonality, which is the dominant signal in retail sales.

**Why `asfreq("D", fill_value=0)`?**
If there are days with no sales (weekends, holidays), they won't appear in the raw data.
Forecasting models require a continuous, gap-free time series. Missing days must be
explicitly represented as zero-sales days, not omitted.

**Why split chronologically, not randomly?**
Random splitting on time series data leaks future patterns into training.
If a sample from December ends up in training and its neighbor (a day in the same week)
ends up in validation, the model has already "seen" the context of the validation period.
Chronological split mirrors the real condition: the model always predicts the future
from the past.

---

### The LSTM concepts

**What problem LSTM solves:**
Standard feedforward networks see one input at a time, with no memory of previous inputs.
Forecasting requires memory — yesterday's sales affect today's prediction.

LSTM (Long Short-Term Memory) maintains a hidden state that carries information
across time steps. It decides what to remember, what to forget, and what to output
at each step. This is why it works well for sequences.

**The sliding window:**
The LSTM is trained on windows: "given these 30 days of sales, predict day 31."
Each window shifts by one day. This converts a time series into a standard supervised
learning problem.

```
Window 1:  [day1 ... day30] → predict day31
Window 2:  [day2 ... day31] → predict day32
Window 3:  [day3 ... day32] → predict day33
```

**Why reset hidden state per batch?**
The hidden state carries information from the previous time step.
If you don't reset between unrelated sequences (different windows), the model
carries contaminated state from one window into the next — a form of data leakage
in the time dimension.

---

### Model checkpointing with artifact aliases

```python
artifact = wandb.Artifact("sales-forecasting-lstm", type="model",
                            metadata={"epoch": epoch, "val_loss": best_loss})
run.log_artifact(artifact, aliases=["best", f"epoch_{epoch}"])
```

**Aliases** are mutable pointers to artifact versions.
As training progresses and better checkpoints are found, the `"best"` alias
moves to the new version. Code that loads `"sales-forecasting-lstm:best"` always
gets the best checkpoint without knowing its version number.

This is the pattern behind safe model updates in production:
the serving system loads `"best"` (or `"production"`). You update the model
by moving the alias — no serving code changes.

---

## Part 5 — The Bigger Picture

### What artifact lineage gives you

Without lineage, debugging a model failure requires answering:
"What data was it trained on? Which version? Was it the preprocessed or raw data?
What were the preprocessing steps?" — and hoping someone recorded this somewhere.

With lineage, one click on any model artifact in W&B shows:
the exact data version it was trained on, the run that produced that data,
and the raw data that run processed. The entire chain is auditable.

This becomes critical when:
- A model degrades in production (is it a data problem or a model problem?)
- A regulatory audit requires you to document model training data
- You want to retrain with last month's data, not this month's

---

### W&B vs MLflow

| | W&B | MLflow |
|---|---|---|
| Hosting | Cloud (wandb.ai) | Self-hosted |
| Real-time charts | Yes, live during training | After run completes |
| Artifact lineage | Native, visual graph | Not built-in |
| Team collaboration | Immediate | Requires shared server |
| Model registry | Via Artifacts + aliases | Dedicated registry |
| Best for | Teams, quick setup | Self-hosted, enterprise |

---

## Quick Reference

### Login

```bash
wandb login
```

### Minimal tracking

```python
import wandb

with wandb.init(project="my_project", config={"lr": 0.001}) as run:
    for epoch in range(10):
        loss = train_one_epoch()
        wandb.log({"loss": loss, "epoch": epoch})
```

### Log and consume an artifact

```python
# Log (upload)
artifact = wandb.Artifact("my_data", type="dataset")
artifact.add_file("data.csv")
run.log_artifact(artifact)

# Consume (download + create lineage link)
artifact = run.use_artifact("my_data:latest")
path = artifact.download()
df = pd.read_csv(f"{path}/data.csv")
```

### Watch model gradients (PyTorch)

```python
wandb.watch(model, log_freq=100)  # logs gradients and weights
```

---

## Official Documentation

- W&B quickstart: https://docs.wandb.ai/quickstart
- Artifacts: https://docs.wandb.ai/guides/artifacts
- Artifact lineage: https://docs.wandb.ai/guides/artifacts/explore-and-traverse-an-artifact-graph
- PyTorch integration: https://docs.wandb.ai/guides/integrations/pytorch
- W&B Sweeps: https://docs.wandb.ai/guides/sweeps
