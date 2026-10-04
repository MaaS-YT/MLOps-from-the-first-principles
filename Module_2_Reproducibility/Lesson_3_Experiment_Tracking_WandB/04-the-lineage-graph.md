# 04 — The Lineage Graph

> [← Previous: W&B Artifacts](03-wandb-artifacts.md) · [Next: The W&B Registry →](05-the-wandb-registry.md)

---

## What the lineage graph is

It's a picture of your pipeline.

Not a diagram you drew. Not a YAML file you wrote. Not a Mermaid block in a README.

**A live graph, drawn by W&B, from your actual code.**

Every `use_artifact` call adds an edge from an artifact to a run.
Every `log_artifact` call adds an edge from a run to an artifact.

Do this consistently, and the graph emerges.

---

## What it looks like

For our notebook, the lineage graph renders as:

```
   ┌──────────────┐       ┌──────────────────┐       ┌────────────────────┐
   │ raw-sales-   │       │  preprocess-     │       │ processed-sales-   │
   │ data:v0      │──────▶│  data run        │──────▶│ data:v0            │
   └──────────────┘       └──────────────────┘       └────────┬───────────┘
                                                              │
                                                              │ consumed by
                                                              ▼
                                                     ┌──────────────────┐
                                                     │  training run    │
                                                     └────────┬─────────┘
                                                              │
                                                              │ produces
                                                              ▼
                                            ┌─────────────────────────────────┐
                                            │ sales-forecasting-lstm:v0..v10  │
                                            │  aliases: best, epoch_N         │
                                            └──────────────┬──────────────────┘
                                                           │
                                                           │ consumed by
                                                           ▼
                                                  ┌──────────────────┐
                                                  │  register run    │
                                                  └──────────────────┘
```

**Every node is clickable.** Click `raw-sales-data:v0` → see the CSV, its metadata, the run that produced it. Click `sales-forecasting-lstm:best` → see the model checkpoint, its training metrics, the data it was trained on.

**Every edge is meaningful.** An arrow from artifact to run means "this run consumed it." An arrow from run to artifact means "this run produced it."

**The graph is the audit trail.**

---

## Where to find it

Open your project on wandb.ai. In the left sidebar:

1. Click **Artifacts** (the icon that looks like a stack of pancakes)
2. Click any artifact — say, `sales-forecasting-lstm`
3. At the top of the artifact page, click the **Lineage** tab

You'll see the graph above. You can:
- **Pan** — click and drag
- **Zoom** — scroll wheel
- **Click a node** — inspect it
- **Expand** — see more of the graph as you click outward

Try it now:

```
https://wandb.ai/<your-username>/sales-forecasting/artifacts/sales-forecasting-lstm
```

Replace `<your-username>` with your W&B username (the one from `wandb login`).

---

## What the graph answers

The lineage graph answers the questions that manual tracking can't.

**"How was this model created?"**

Click the model → click the training run → click the processed data → click the raw data. The full chain in four clicks.

**"What data trained the model in production?"**

Click the `production` alias on the model → click the training run → click the `use_artifact` line → the exact version of `processed-sales-data` is shown.

**"If I need to retrain, what data do I use?"**

Click the training run → the exact data version is captured in the graph. Use the same version — same result.

**"A teammate joined. How do I show them the pipeline?"**

Send the URL. They click through the graph. The entire pipeline structure is visible in one screen.

**"A regulator asked how we trained this."**

Export the lineage view. Every node, every edge, every metadata field. Auditable.

**These questions have been the source of ML team pain for a decade.** The lineage graph makes them answerable in seconds.

---

## Why this is different

MLflow doesn't have a lineage graph. You can trace a model to a run, and a run to its params and metrics — but not from raw data to model through every intermediate step.

DVC tracks data versions with git commits. But the graph is git's commit graph, not a pipeline graph. It shows time, not data flow.

**W&B is the only one of the three that renders the pipeline as a graph.** That's its signature feature.

---

## What makes the graph possible

Two things have to be true for the graph to exist:

**1. Artifacts must be first-class objects.**

Not "files attached to a run." Versioned, named, aliased objects. This is what makes edges possible — you can't draw an edge to a file path, you can only draw an edge to an object.

We covered this in rung 03.

**2. Every stage must declare its inputs and outputs.**

`use_artifact` declares an input. `log_artifact` declares an output. If your code reads a CSV from disk without calling `use_artifact`, the graph is blind to it.

**This is the discipline.** Any data your code reads has to come through `use_artifact`. Any data it writes has to go through `log_artifact`. If both are true, the graph is complete.

---

## The rule that makes it work

**Every stage consumes the previous stage's artifact. Nothing else.**

Not a file path.
Not a shared volume.
Not an environment variable pointing at a CSV.

**An artifact.**

```python
# Bad — the graph doesn't see this
df = pd.read_csv("raw_sales_data.csv")

# Good — the graph draws an edge
artifact = run.use_artifact("raw-sales-data:latest")
artifact_dir = artifact.download()
df = pd.read_csv(f"{artifact_dir}/raw_sales_data.csv")
```

Both do the same thing. Only the second one is visible in the graph.

**The graph is a mirror of your discipline.** If you always use artifacts, the graph is complete. If you sometimes bypass them, the graph has holes.

---

## What the notebook's graph shows

If you ran `pipeline-pytorch.ipynb` end to end, your graph has:

**Six nodes:**

- 3 artifact nodes — `raw-sales-data`, `processed-sales-data`, `sales-forecasting-lstm`
- 4 run nodes — `ingest-data`, `preprocess-data`, `training`, `register`

Wait, that's seven. Let me count:

- 3 artifact nodes
- 4 run nodes

**Seven nodes. Six edges:**

```
raw-sales-data → [ingest-data run]      (edge from ingest-data producing raw-sales-data)
raw-sales-data → [preprocess-data run]  (edge from preprocess-data consuming raw-sales-data)
[preprocess-data run] → processed-sales-data  (edge from preprocess-data producing processed-sales-data)
processed-sales-data → [training run]   (edge from training consuming processed-sales-data)
[training run] → sales-forecasting-lstm (edge from training producing the model)
sales-forecasting-lstm → [register run] (edge from register consuming the model)
```

Six edges. One clear pipeline.

**Walk the graph from raw-sales-data to sales-forecasting-lstm. That's the model's full provenance.**

No notes. No documentation. No "here's how we built it." The graph is the answer.

---

## What you can do with it

**Debug data issues.**

A model's performance degraded. Instead of wondering "was it the data or the code?" — click into the graph. Compare the `processed-sales-data` version used by the working model vs the current model. If they differ, you found it.

**Audit compliance.**

A regulator asks what data trained the production model. Click the model → click the training run → the exact data version is documented. No manual investigation.

**Onboard teammates.**

Send the URL. Your new teammate sees the pipeline structure in one screen. No verbal explanation needed.

**Retrain with the same inputs.**

When you want to reproduce a run, the graph tells you exactly what data to use. Check out that data version, run the same code, same config. Reproducible.

**Track lineage across projects.**

Artifacts can be shared across projects. A dataset versioned in one project can be consumed by a run in another. The graph spans both.

---

## Why this is the lesson's payoff

We've been building toward this since rung 01.

- **Rung 01** — the team needs to see each other's work
- **Rung 02** — every stage becomes a run
- **Rung 03** — every stage declares its inputs and outputs as artifacts
- **Rung 04** — the graph falls out of the first three

The graph is not a feature you turn on. It's the natural result of the discipline we've been building.

**That's the point.** W&B's value isn't "cloud MLflow." It's that if you structure your pipeline correctly — runs and artifacts — you get a traceability graph for free.

---

## A note on MLflow

MLflow doesn't have this. It has:

- Runs in a database
- Artifacts attached to runs
- A model registry
- A UI

What it doesn't have is a rendering of the pipeline as a graph. You can find all the info in the DB, but there's no view that shows the chain.

**This is the gap that W&B fills.** It's why the module needs both lessons.

If you're building a solo project, MLflow is enough. If you're building a team project with a pipeline, the lineage graph is worth the trade.

---

## Checkpoint

You should now be able to answer:

- **What is the lineage graph?** A visual representation of the pipeline, drawn automatically from `use_artifact` and `log_artifact` calls.
- **Where does it live?** In the W&B UI, on the Lineage tab of any artifact.
- **What are the nodes?** Runs and artifacts.
- **What are the edges?** "Consumed by" and "produced by" relationships.
- **What makes it possible?** First-class artifacts (rung 03) + every stage declaring its inputs/outputs.
- **What is the discipline?** Every stage consumes the previous stage's artifact. Nothing else.
- **What does it answer?** "How was this model created?" — in four clicks instead of an investigation.
- **Why doesn't MLflow have this?** Its artifacts are files attached to runs, not first-class objects with edges. No graph is possible.

**Next:** [The W&B Registry →](05-the-wandb-registry.md)