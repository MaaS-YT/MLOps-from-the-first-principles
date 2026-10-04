# Lesson 2.3 — Experiment Tracking with Weights & Biases

> **Module 2 · Reproducibility** · [← Module 1: The ML System](../../Module_1_ML_Systems_Intro/) · [← Lesson 2.2: MLflow](../Lesson_2_Experiment_Tracking_MLflow/)

---

## The Problem

You finished the MLflow lesson. You have working tracking. You can log runs, compare experiments, register models, load by alias.

Now add a teammate.

They open their MLflow UI. Your runs aren't there.

**The team link in the chain is broken.**

```
data → code → environment → run → artifact → model
                                                ↑
                              team visibility ✗
```

MLflow stores runs on your laptop. Their MLflow stores runs on theirs. Two separate universes.

You could:
- Copy the SQLite file
- Set up a shared tracking server
- Move everything to Postgres

All three are real solutions. All three are **your problem now** — infrastructure you have to build, maintain, and secure.

That's what W&B solves.

---

## The Mental Model

**W&B is cloud-native experiment tracking with artifact lineage.**

Two ideas:

**1. Runs are the same as MLflow.** One execution — config, metrics, artifacts. Same shape.

**2. Lineage is new.** Every artifact declares its inputs and outputs. The graph emerges automatically from those declarations.

Put them together:

```
run + artifacts + lineage  =  a pipeline you can see
```

The signature feature is the **lineage graph**. Not a feature you turn on — a natural result of structuring your pipeline correctly.

---

## The Triune, Again

Same shape as Module 1 and the previous lessons:

```
.wandb run        →  artifacts + metrics  →  loadable model
definition        →  artifact             →  instance
.dvc file         →  hash + remote        →  checked-out data
Dockerfile        →  image                →  container
MLflow run        →  model artifact       →  registered model
```

In W&B, the triune applies at two levels:

**Within a run:**
- **Definition:** `wandb.init(config=...)`
- **Artifact:** the run's metrics, logs, checkpoints
- **Instance:** the live dashboard in the W&B UI

**Across the pipeline:**
- **Definition:** `use_artifact()` + `log_artifact()` calls
- **Artifact:** the lineage graph itself
- **Instance:** any node you click to inspect

Same three-part shape, different scale.

---

## The Five Rungs

Read in order. Each assumes the previous.

| Rung | Guide | What you learn |
|------|-------|----------------|
| 01 | [The Team Problem](01-the-team-problem.md) | Why local tracking breaks for teams |
| 02 | [The W&B Run](02-the-wandb-run.md) | `wandb.init`, config, log, `wandb.watch()` |
| 03 | [W&B Artifacts](03-wandb-artifacts.md) | Versioned datasets and models. The hand-off pattern. |
| 04 | [The Lineage Graph](04-the-lineage-graph.md) | W&B's signature feature — the pipeline as a graph |
| 05 | [The W&B Registry](05-the-wandb-registry.md) | Aliases (`best`, `staging`, `production`), load by name |

**Start with rung 01.** Don't skip to the notebook. The commands only make sense once the problem is clear.

**The notebook:** [`pipeline-pytorch.ipynb`](pipeline-pytorch.ipynb) — a four-stage LSTM pipeline that maps to rungs 02–05.

**Scenarios:** [`scenarios/`](scenarios/) — an offline-mode demo.

---

## The Five-Step Progression

Where W&B sits in the longer arc of team tracking:

```
1. Print statements       no history, no comparison
2. Local MLflow           structured logs, solo scientist
3. Shared MLflow          requires ops — server, DB, storage
4. W&B                    cloud-native, shared by default      ← you are here
5. W&B Sweeps + Reports   automated search + shareable summaries
6. Full observability     W&B + feature store + monitoring
```

W&B is rung four. It gets you the shared visibility that solo MLflow can't provide — without the ops cost of a shared MLflow server.

---

## Quick Start

Run this after walking the ladder — it's the destination, not the starting point.

```bash
# Prerequisite — log in to W&B (free account at wandb.ai)
wandb login

# Open the notebook
cd Module_2_Reproducibility/Lesson_3_Experiment_Tracking_WandB
jupyter lab pipeline-pytorch.ipynb

# Run all cells. Four stages execute in order:
#   1. Ingest raw UCI data → logs raw-sales-data artifact
#   2. Preprocess → logs processed-sales-data artifact
#   3. Train LSTM → logs sales-forecasting-lstm artifacts (per epoch)
#   4. Register → adds staging + production aliases

# Then open the project URL printed by the notebook
```

**Verify the lineage graph in the W&B UI:**

1. Open your project on wandb.ai
2. Click **Artifacts** in the left sidebar
3. Click on `sales-forecasting-lstm`
4. Click the **Lineage** tab

The full pipeline is rendered as a graph — raw data → processed → model.

**Load the production model:**

```python
import wandb
import torch

run = wandb.init()
artifact = run.use_artifact("sales-forecasting-lstm:production")
artifact_dir = artifact.download()

bundle = torch.load(f"{artifact_dir}/best_model_bundle.pth")
print(f"Loaded model from epoch {bundle['config']}")
print(f"Config: {bundle['config']}")
```

Three lines to go from artifact name to loadable model.

---

## What You Should Be Able to Answer at the End

- **What does W&B add over MLflow?** Cloud-native runs (shared by default) and automatic artifact lineage.
- **What is a W&B run?** One execution — config, metrics, artifacts. Same abstraction as MLflow, different destination.
- **What's the difference between `log_artifact` and `use_artifact`?** Produce and consume. Each draws a lineage edge.
- **How is versioning handled?** Automatic. Each `log_artifact` on the same name creates a new version.
- **What is the lineage graph?** A visual representation of the pipeline, drawn automatically from use/log calls.
- **What's the discipline that makes it work?** Every stage consumes the previous stage's artifact. Nothing else.
- **What are the three aliases that matter?** `best` (automatic), `staging`, `production` (manual).
- **How does a serving system load a production model?** By name and alias — `sales-forecasting-lstm:production`.
- **What's the triune?** run → artifact → instance. At two scales (within a run, across the pipeline).

---

## Where This Fits

This lesson is the **team link** in the reproducibility chain.

```
data → code → environment → run → artifact → model
 ↑                    ↑       ↑       ↑        ↑
2.1                 Module 1  2.2     2.2     2.3
DVC                 Docker   MLflow  MLflow  W&B
```

- **Backward (Lesson 2.1):** DVC gives you versioned data. W&B's artifacts give you a second layer — the same data, tracked in the pipeline context.
- **Backward (Lesson 2.2):** MLflow gives you runs. W&B gives you runs *plus* lineage, shared by default.
- **Forward (Module 3):** Data pipelines produce artifacts. W&B's lineage graph is the visual proof of those pipelines.
- **Forward (Module 4):** Optimized models are also artifacts. Same pattern — logged, versioned, aliased.
- **Forward (Module 5):** Kubernetes pulls from a registry. W&B's aliases are the interface.
- **Forward (Module 6):** Observability instruments the deployed model. The lineage graph shows where it came from.

> 📍 **Full chain diagram:** [`../README.md`](../README.md)

---

## Files

| File | Purpose |
|------|---------|
| `01-the-team-problem.md` … `05-the-wandb-registry.md` | The five rungs — read in order |
| [`pipeline-pytorch.ipynb`](pipeline-pytorch.ipynb) | The four-stage LSTM pipeline — the demo |
| [`scenarios/`](scenarios/) | Offline-mode demo |
| `_archive/` | Old material (pre-refactor) |

---

## Key Terms

| Term | Definition |
|------|------------|
| **W&B** | Weights & Biases — cloud-native experiment tracking + artifact lineage |
| **Project** | A named collection of runs (same concept as an MLflow experiment) |
| **Run** | One execution — config, metrics, artifacts |
| **Job type** | A tag describing the pipeline stage (`ingest-data`, `training`, etc.) |
| **Config** | Declared at `wandb.init()`. Enables W&B Sweeps. |
| **Artifact** | A versioned, named collection of files. First-class object. |
| **Version** | Auto-assigned — `v0`, `v1`, `v2` — for each log call |
| **Alias** | A mutable pointer — `latest`, `best`, `staging`, `production` |
| **Lineage graph** | A visual representation of the pipeline, drawn automatically |
| **`wandb.watch()`** | Hook that logs gradients, weights, and model topology |
| **`use_artifact`** | Consume an artifact. Draws an edge in the lineage graph. |
| **`log_artifact`** | Produce an artifact. Draws an edge in the lineage graph. |
| **Registry** | A curated space for production models. W&B offers aliases + a UI feature. |

---

## Checkpoint

You should now be able to answer:

- **What problem does W&B solve that MLflow doesn't?** Shared visibility + data provenance + lineage — all by default.
- **What's the trade?** MLflow gives you control. W&B gives you velocity. Different optimization targets.
- **What is a registered model?** A named artifact with lifecycle aliases (`staging`, `production`).
- **What makes the lineage graph possible?** First-class artifacts + every stage declaring its inputs/outputs.
- **What's the chain?** `data → code → environment → run → artifact → model`.
- **What did this lesson fix vs Lesson 2.2?** Lesson 2.2 fixed the run link for an individual. Lesson 2.3 fixed the team link.

**Start here:** [Rung 01 — The Team Problem →](01-the-team-problem.md)

---

## How to Use This Lesson

**For a first read:** read rungs 01–05 in order. Then run the notebook. Then open the W&B UI and walk through the lineage graph.

**For the deep-learning angle:** if you already know tracking from MLflow, jump to rung 02 (the `wandb.watch()` section) and rung 04 (the lineage graph).

**For reference:** the Key Terms table above and the W&B docs at [docs.wandb.ai](https://docs.wandb.ai).

---

> *MLflow taught you what tracking is.*
> *W&B teaches you what tracking is capable of when the infrastructure disappears.*