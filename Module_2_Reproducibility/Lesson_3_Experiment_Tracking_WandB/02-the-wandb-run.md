# 02 — The W&B Run

> [← Previous: The Team Problem](01-the-team-problem.md) · [Next: W&B Artifacts →](03-wandb-artifacts.md)

---

## What a run is

A run is one execution of your code. One stage of the pipeline. One training job. One data upload.

Same abstraction as MLflow. Same shape:

```
wandb.init()  →  log_* calls  →  sync to cloud
```

The difference is where it goes. MLflow writes to a database file or server you configured. W&B syncs to the cloud.

**That's the whole difference at the run level.** Same API shape, different destination.

---

## The `with` block

Same context manager pattern as MLflow:

```python
with wandb.init(project="sales-forecasting", job_type="training", config=config) as run:
    model.fit(...)
    wandb.log({"train_loss": ..., "val_loss": ...})
```

What happens, step by step:

1. **`wandb.init()`** — creates a run. Assigns a unique run ID. Starts syncing to the cloud.
2. **Everything inside the block** — every `wandb.log()`, `wandb.watch()`, `log_artifact()` call writes to this run.
3. **Exit the block** — the run is closed. Its status is set. The sync completes.

**One important difference from MLflow:** W&B streams as it runs. You can watch charts update live in the browser while the training loop is still executing.

---

## The four things a run holds

### 1. Config — the inputs

```python
with wandb.init(config={"learning_rate": 0.0005, "epochs": 15}) as run:
    config = wandb.config
    lr = config.learning_rate
```

Same idea as `mlflow.log_param`, but with a crucial difference:

**In W&B, you declare the config at `init()` time, then read it from `wandb.config`.**
**In MLflow, you log params incrementally during the run.**

Why it matters: because the config is declared upfront, W&B **Sweeps** can inject alternative configs into the same script without any code changes. Your training script says "read `learning_rate` from `wandb.config`" — the sweep provides different values for different runs.

**This is why you read from `wandb.config`, not from a local dict.** A local dict can't be overridden by a sweep.

In the notebook:

```python
TRAIN_CONFIG = {
    "learning_rate": 0.0005,
    "epochs": 15,
    "sequence_length": 30,
    "hidden_layer_size": 50,
    "seed": 1,
}

with wandb.init(project=PROJECT, job_type="training", config=TRAIN_CONFIG) as run:
    config = wandb.config
    model = LSTMModel(hidden_layer_size=config.hidden_layer_size)
```

The config is the run's DNA. Two runs are comparable only if you know what config each had.

### 2. Metrics — the outputs

```python
wandb.log({"epoch": epoch, "train_loss": avg_train_loss, "val_loss": avg_val_loss})
```

Same idea as `mlflow.log_metric`, but two differences:

**Per-step logging.** `wandb.log` supports a `step` argument, so you can log per epoch, per batch, per anything. W&B builds a time-series chart instead of a single point.

**Live streaming.** Each `wandb.log` call pushes to the cloud. If you're watching the dashboard while training runs, you see the chart update in real time.

In the notebook:

```python
for epoch in range(config.epochs):
    # ... train loop ...
    # ... validation loop ...
    
    wandb.log({
        "epoch": epoch,
        "train_loss": avg_train_loss,
        "val_loss": avg_val_loss,
    })
```

Each epoch produces one row in the metrics table. Fifteen epochs → fifteen rows. The W&B dashboard plots them as a curve.

**Why this matters for deep learning:** you want to catch divergence *while* training, not after. If the val loss starts climbing at epoch 6, you'd rather know at epoch 6 than after the job finishes.

### 3. Tags — context

Same as MLflow:

```python
wandb.init(
    project="sales-forecasting",
    job_type="training",
    tags=["lstm", "time-series", "sales"],
)
```

Tags make runs searchable. In the UI, filter by tag. In the Python API, query by tag.

**`job_type` is a special tag.** It tells W&B what stage of the pipeline this run is. `ingest-data`, `preprocess-data`, `training`, `register` — each stage gets its own `job_type`.

Why this matters: **the lineage graph uses `job_type` to color and group the nodes.** Without it, every run looks the same. With it, you can see the pipeline structure at a glance.

### 4. Artifacts — the outputs that persist

```python
artifact = wandb.Artifact(name="sales-forecasting-lstm", type="model")
artifact.add_file("best_model_bundle.pth")
run.log_artifact(artifact, aliases=["best", f"epoch_{epoch}"])
```

This is where W&B diverges from MLflow.

**In MLflow:** an artifact is a file attached to a run.
**In W&B:** an artifact is a **first-class object** with its own version history, aliases, and lineage.

We'll cover artifacts in detail in rung 03. For now: **every `log_artifact()` call creates a new version of an artifact.** If you call it every epoch, you get one version per epoch.

That's how you accumulate checkpoints. Every improvement → a new version. Aliases point at the interesting ones.

---

## `wandb.watch()` — the deep learning tool

This is the feature MLflow has no equivalent for.

```python
wandb.watch(model, log_freq=100)
```

Hooks into a PyTorch model and logs:

- **Gradients** — histograms per layer, updated every `log_freq` steps
- **Weights** — histograms showing how the model evolves
- **Model topology** — a graph of the architecture

In the notebook's training loop, it's called once after the model is initialized:

```python
model = LSTMModel(hidden_layer_size=config.hidden_layer_size)
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

# Hook: W&B now tracks gradients for every layer, every 100 steps
wandb.watch(model, log_freq=100)
```

**What the reader sees in the W&B UI:**

Six gradient histograms in the training run's dashboard. Two LSTM weight matrices. Two LSTM biases. The linear layer's weight. The linear layer's bias. Each one shows the distribution of gradients at each logged step.

**Why this matters:**

- **Vanishing gradients** — the histograms collapse toward zero. The model stops learning.
- **Exploding gradients** — the histograms grow toward infinity. The loss explodes.
- **Unstable training** — the histograms oscillate wildly. Learning rate is too high.
- **Dead layers** — a histogram stays flat. That layer isn't contributing.

**These are invisible without instrumentation.** You can't tell a dead LSTM layer from a working one by looking at the loss curve alone. The gradient histograms show you exactly what's happening inside.

**MLflow has no built-in equivalent.** You'd write custom callbacks to log gradients. W&B does it in one line.

---

## What the notebook logs

Look at cells 12–13 of `pipeline-pytorch.ipynb`. During the training loop, for each epoch:

```python
# Log metrics
wandb.log({
    "epoch": epoch,
    "train_loss": avg_train_loss,
    "val_loss": avg_val_loss,
})
```

And once, before the loop:

```python
# Hook gradient tracking
wandb.watch(model, log_freq=100)
```

And once per improvement:

```python
# Checkpoint
artifact = wandb.Artifact(name="sales-forecasting-lstm", type="model")
artifact.add_file("best_model_bundle.pth")
run.log_artifact(artifact, aliases=["best", f"epoch_{epoch}"])
```

**By the time training finishes, the run holds:**

- The full config (hyperparameters, sequence length, seed)
- 15 rows of train/val metrics
- 6 gradient histograms streaming in real time
- Multiple checkpoint versions of the model artifact
- Two aliases (`best`, `epoch_N`) pointing at the last one

**That's the atomic unit of reproducibility for deep learning.** One run. Everything captured. Queryable, comparable, shareable.

---

## Why this matters for the chain

The run is the atom of the pipeline. Everything else — artifacts, lineage, the registry — builds on the run.

**If the run is well-instrumented, everything downstream is possible:**

- Compare two runs → see which config won
- Trace a model back to a run → see its hyperparameters
- Reproduce a training run → use the same config + data
- Debug a divergence → inspect the gradient histograms

**If the run is not well-instrumented, none of it is possible.** This is why we set up the run first, artifacts second.

Same discipline as MLflow. Same chain. Different altitude.

---

## Checkpoint

You should now be able to answer:

- **What's a run in W&B?** One execution of your code — a row in the W&B cloud, synced in real time.
- **Where does `wandb.config` differ from MLflow params?** Config is declared at `init()` time, enabling Sweeps. MLflow params are logged incrementally.
- **What does `wandb.log` do differently?** Per-step logging (time series) and live streaming.
- **What's `job_type`?** A special tag that tells W&B what stage of the pipeline this run is. Used by the lineage graph for grouping.
- **What does `wandb.watch()` log?** Gradients, weights, model topology. Nothing MLflow has an equivalent for.
- **Why does this matter for deep learning?** Vanishing gradients, dead layers, unstable training — all invisible without instrumentation.

**Next:** [W&B Artifacts →](03-wandb-artifacts.md)