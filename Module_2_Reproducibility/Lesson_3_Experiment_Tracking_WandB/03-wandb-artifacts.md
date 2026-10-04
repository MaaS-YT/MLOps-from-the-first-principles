# 03 — W&B Artifacts

> [← Previous: The W&B Run](02-the-wandb-run.md) · [Next: The Lineage Graph →](04-the-lineage-graph.md)

---

## What an artifact is

An artifact is a **versioned, named collection of files.**

Not a run's output. Not a log entry. A first-class object with:

- A **name** — `raw-sales-data`
- A **type** — `raw_dataset`, `processed_dataset`, `model`
- **Versions** — `v0`, `v1`, `v2`, ...
- **Aliases** — `latest`, `best`, `staging`, `production`
- **Metadata** — any JSON-serializable info
- **Lineage** — which runs consumed it, which runs produced it

**If a run is one execution, an artifact is one version of a thing.**

A dataset. A model. A report. Any file that matters enough to be referenced by name.

---

## The mechanism

Two operations. That's it.

**Log** — write an artifact from a run:

```python
artifact = wandb.Artifact(
    name="raw-sales-data",
    type="raw_dataset",
    description="Raw online retail sales data from UCI.",
)
artifact.add_file("raw_sales_data.csv")
run.log_artifact(artifact)
```

**Use** — read an artifact into a run:

```python
artifact = run.use_artifact("raw-sales-data:latest")
artifact_dir = artifact.download()
df = pd.read_csv(f"{artifact_dir}/raw_sales_data.csv")
```

**`log_artifact` produces. `use_artifact` consumes.**

And here's the thing: **each of these calls draws an edge in the lineage graph.**

- `log_artifact` → edge from run to artifact ("this run produced it")
- `use_artifact` → edge from artifact to run ("this run consumed it")

The graph is not something you build. It's something that **falls out of your code.** If you use and log artifacts correctly, the graph appears.

---

## Versioning

W&B auto-versions artifacts. Same name, different calls → different versions.

```python
# First call
run1.log_artifact(artifact)  # → raw-sales-data:v0

# Second call, after data changes
run2.log_artifact(artifact)  # → raw-sales-data:v1

# Third call
run3.log_artifact(artifact)  # → raw-sales-data:v2
```

**Version numbers are automatic. You never assign them.**

**`latest` is a special alias.** It always points at the most recent version. When you write `run.use_artifact("raw-sales-data:latest")`, you get v2 (or whatever's newest).

**Custom aliases** are how you mark important versions:

```python
run.log_artifact(artifact, aliases=["best", "production"])
```

Now `raw-sales-data:best` resolves to that specific version. When you promote a new version to `best`, you move the alias.

**This is the pattern that makes the registry work.** We'll see it again in rung 05.

---

## The two-stage pipeline

The notebook's first two stages illustrate the artifact hand-off pattern.

### Stage 1 — `ingest_raw_data()`

```python
def ingest_raw_data():
    with wandb.init(project=PROJECT, job_type="ingest-data") as run:
        # Download
        df = pd.read_excel(RAW_DATA_URL)
        df.to_csv("raw_sales_data.csv", index=False)

        # Log as artifact
        artifact = wandb.Artifact(
            name="raw-sales-data",
            type="raw_dataset",
        )
        artifact.add_file("raw_sales_data.csv")
        run.log_artifact(artifact)
```

**What this does:**
- Downloads the UCI Online Retail dataset (~540K transactions)
- Saves it as a local CSV
- Logs it to W&B as an artifact named `raw-sales-data` of type `raw_dataset`

**What it produces:** the first version of `raw-sales-data`. The data is now a versioned, immutable snapshot in W&B.

### Stage 2 — `preprocess_data()`

```python
def preprocess_data():
    with wandb.init(project=PROJECT, job_type="preprocess-data") as run:
        # Consume the raw artifact
        raw_artifact = run.use_artifact("raw-sales-data:latest")
        raw_dir = raw_artifact.download()
        df = pd.read_csv(f"{raw_dir}/raw_sales_data.csv")

        # Clean, aggregate, split
        # ...

        # Log the processed artifact
        artifact = wandb.Artifact(
            name="processed-sales-data",
            type="processed_dataset",
        )
        artifact.add_file("train.csv")
        artifact.add_file("validation.csv")
        run.log_artifact(artifact)
```

**What this does:**
- Loads the raw artifact via `use_artifact`
- Cleans, aggregates to daily sales, splits into train/val
- Logs the processed data as a new artifact

**What it produces:** `processed-sales-data` — a two-file artifact (`train.csv` + `validation.csv`).

### The connection

**Look at what happened between the two stages:**

```
raw-sales-data (v0)
    │
    │  consumed by
    ▼
ingest run  ──►  preprocess run
                      │
                      │  produces
                      ▼
                processed-sales-data (v0)
```

W&B recorded the edge automatically. When the training stage later consumes `processed-sales-data:latest`, another edge is drawn. The chain grows.

**No explicit graph definition. No YAML file. No DAG declaration.**

Just calls to `use_artifact` and `log_artifact`. The graph emerges.

---

## Why this is different from MLflow

MLflow has artifacts too. But they're attached to runs, not first-class objects.

| MLflow | W&B |
|--------|-----|
| Artifact is a file at a path | Artifact is a versioned object |
| Referenced by `runs:/<run_id>/path` | Referenced by `name:version` or `name:alias` |
| No version history across runs | Versions accumulate automatically |
| No lineage edges | Lineage edges drawn by use/log calls |
| Manual promotion to registry | Aliases as first-class promotion |

**The difference is scope.** MLflow's artifacts are "things a run produced." W&B's artifacts are "things with their own lifecycle."

That's why W&B can build a lineage graph and MLflow can't. The graph requires first-class objects with edges between them, not files with paths.

---

## The pattern to internalize

**Every stage of a pipeline should:**

1. **Consume** the previous stage's output via `use_artifact`
2. **Produce** its own output via `log_artifact`

**Nothing should be passed between stages any other way.**

Not via local files. Not via environment variables. Not via arguments.

**Only via artifacts.**

Why? Because artifacts carry their own lineage. When stage 3 uses `processed-sales-data:latest`, the graph shows exactly which version was used. If stage 3 runs again tomorrow and `latest` has moved, the graph shows a different version was used. **The history is preserved.**

If stages passed data via local files, none of this would be visible. The graph would be a flat list of runs with no connections.

**The discipline of "always use artifacts" is what makes the graph meaningful.**

---

## What the notebook produces

By the time the two data stages finish, W&B has:

**Two artifacts:**
- `raw-sales-data` (v0) — the downloaded CSV
- `processed-sales-data` (v0) — the cleaned train/val split

**Two runs:**
- `ingest-data` — produced `raw-sales-data`
- `preprocess-data` — consumed `raw-sales-data`, produced `processed-sales-data`

**One lineage edge:**
- `raw-sales-data` → `preprocess-data` → `processed-sales-data`

The training stage (rung 02) adds to this. By the end of the notebook, the graph has four stages and three artifacts. We'll see it in rung 04.

---

## Why this matters for reproducibility

**Data is the most common source of irreproducibility.** Change the data, and every model trained on it becomes unreproducible.

Without artifact versioning:
- "Which version of the CSV did we train on?" — you don't know
- "The model was better last month" — with what data?
- "Retrain with the same data" — same as what?

With W&B artifacts:
- Every training run's `use_artifact` call recorded the exact version
- Every dataset version is immutable — v0 is always v0
- The lineage graph shows the chain from raw data to trained model

**Artifacts turn data into a first-class citizen of the pipeline.** Not a file on disk that might change. A versioned object that doesn't.

Same discipline as DVC. Different mechanism.

---

## Checkpoint

You should now be able to answer:

- **What's an artifact in W&B?** A versioned, named collection of files. First-class object, not a run output.
- **What are the two operations?** `log_artifact` (produce) and `use_artifact` (consume).
- **How is versioning automatic?** Each `log_artifact` call on the same name creates a new version.
- **What's `latest`?** A special alias that always points at the most recent version.
- **What's a custom alias?** A named pointer to a specific version — `best`, `staging`, `production`.
- **How does the lineage graph get built?** Automatically, from `use_artifact` and `log_artifact` calls.
- **Why use artifacts instead of passing files between stages?** Artifacts carry lineage. Files don't.

**Next:** [The Lineage Graph →](04-the-lineage-graph.md)