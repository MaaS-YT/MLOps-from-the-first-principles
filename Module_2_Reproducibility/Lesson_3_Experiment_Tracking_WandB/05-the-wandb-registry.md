# 05 — The W&B Registry

> [← Previous: The Lineage Graph](04-the-lineage-graph.md) · [Back to the ladder](README.md)

---

## The problem the registry solves

You've trained ten versions of the model. W&B has all ten — each one a version of the `sales-forecasting-lstm` artifact, each with its own metadata (epoch, val_loss).

The training code knows which one is best (`best` alias). But "best" is a moving target. Every time a new checkpoint beats the previous best, `best` moves.

Now a serving system needs to load the current production model.

**How does it reference it?**

**Option A — by run ID.**

```python
artifact = run.use_artifact("sales-forecasting-lstm:v7")
```

Brittle. Version numbers change. You have to update the serving code every time.

**Option B — by alias.**

```python
artifact = run.use_artifact("sales-forecasting-lstm:production")
```

**This is what the registry is for.**

The name is stable. The alias `production` is stable. When you promote a new version, the alias moves and the serving code picks it up on the next call.

---

## What a registered model is in W&B

It's a **named, curated artifact** with lifecycle aliases.

Think of it as an artifact that's been marked as "this is a thing we care about." It's not a new file — the underlying artifact already exists. The registry just adds:

- A **curated name** — `sales-forecasting-lstm`
- **Lifecycle aliases** — `staging`, `production`
- **A registry entry** — a central place where production models live

**No new artifact is stored.** The registry adds names and aliases. That's all.

---

## Two paths to the registry

W&B offers two paths. Both work. They serve different purposes.

### Path A — Aliases on artifacts (portable)

This is what the notebook uses.

You take an artifact version and add aliases:

```python
artifact = run.use_artifact("sales-forecasting-lstm:best")

# Add promotion aliases
for alias in ("staging", "production"):
    if alias not in artifact.aliases:
        artifact.aliases.append(alias)
artifact.save()
```

Now the artifact has three aliases: `best`, `staging`, `production`. Any code that loads `sales-forecasting-lstm:production` gets this version.

**Pros:**
- Works on all W&B plans
- No UI required
- Fully scriptable
- Same pattern as MLflow aliases

**Cons:**
- No central "registry" view in the UI
- Aliases are managed per-artifact, not in a catalog

### Path B — Linked registry (UI + plan-dependent)

W&B's newer feature. You can link an artifact to a **registry** — a curated collection of production models. The UI provides:

- A centralized "Registry" section in the sidebar
- Per-model history across projects
- Team-wide access control

**How to use it:**

1. Go to your artifact page in the W&B UI
2. Click **"Link to Registry"** (top-right of the artifact view)
3. Pick or create a registry — say, `production-models`
4. Pick or create a collection — say, `sales-forecasters`
5. Assign a `staging` alias
6. Click **Link**

Now the model lives in the registry. From any project, you can pull from `models:/<registry-name>/<collection-name>:<alias>`.

**Pros:**
- Centralized registry view
- Team-wide curation
- Access control

**Cons:**
- Plan-dependent (may require a paid tier)
- Requires the UI (or the Registry API, which is newer)
- More ceremony

**The notebook uses Path A** — aliases directly on the artifact. It's portable, scriptable, and works everywhere. If your plan supports Path B, the same artifact can be linked via the UI.

---

## The aliases pattern

Three aliases matter for production. Same pattern as MLflow.

| Alias | Meaning | Who moves it |
|-------|---------|--------------|
| **`best`** | Best checkpoint from training | The training loop (automatic) |
| **`staging`** | Under evaluation for production | The ML engineer promoting |
| **`production`** | Live model served to users | The ML engineer promoting |

**`best`** is automatic. Every time the training loop finds a better checkpoint, it re-logs the artifact with the `best` alias. The alias moves.

**`staging`** and **`production`** are manual. A human (or a CI job) moves them explicitly. That's the point — the promotion is deliberate.

### Why this works

The serving system loads `sales-forecasting-lstm:production`. It doesn't know about versions. It doesn't know about `best`. It just knows: "give me the production model."

When you promote a new version:
```python
artifact = run.use_artifact("sales-forecasting-lstm:best")
artifact.aliases.append("production")
artifact.save()
```

The `production` alias now points at the new version. The serving system picks it up on the next load.

**No serving code change. No restart. No config update.** Just an alias move.

---

## What the notebook's `register_and_promote()` does

Look at cell 15 of `pipeline-pytorch.ipynb`:

```python
def register_and_promote():
    with wandb.init(project=PROJECT, job_type="register") as run:
        # Load the best checkpoint from training
        artifact = run.use_artifact("sales-forecasting-lstm:best")

        print(f"Best checkpoint: {artifact.qualified_name}")
        print(f"Version:         {artifact.version}")
        print(f"Aliases:         {artifact.aliases}")

        # Add promotion aliases
        for alias in ("staging", "production"):
            if alias not in artifact.aliases:
                artifact.aliases.append(alias)
        artifact.save()

        print("\nAdded aliases: 'staging', 'production'")
        print(f"View run: {run.get_url()}")
```

**What it does:**

1. Opens a new W&B run (`job_type="register"`)
2. Loads the best checkpoint from the previous stage (creates a lineage edge from `sales-forecasting-lstm` to this run)
3. Prints what it found — the artifact's full name, version, current aliases
4. Adds `staging` and `production` aliases
5. Saves the artifact — the aliases are now attached to the version

**What this produces:**

- A fourth run (`register`) in the W&B project
- A lineage edge: `sales-forecasting-lstm:best` → `register` run
- The `sales-forecasting-lstm` artifact now has aliases: `best`, `epoch_N`, `staging`, `production`

**This is the promotion.** The model is now marked as "we consider this a candidate for production." If your team uses the W&B Registry feature, the same artifact can be linked there.

---

## Loading by alias

The final pattern. Any code that needs the model — a notebook, a serving script, a batch job — loads by alias:

```python
import wandb

run = wandb.init()
artifact = run.use_artifact("sales-forecasting-lstm:production")
artifact_dir = artifact.download()

# Load the bundle
import torch
bundle = torch.load(f"{artifact_dir}/best_model_bundle.pth")
model_state_dict = bundle["model_state_dict"]
scaler = bundle["scaler"]
config = bundle["config"]

# Now you have everything needed to run inference
```

**Three things to notice:**

1. **No version number.** The code loads `:production`, not `:v7`. If production moves to v8 tomorrow, this code still works.

2. **No file path.** The artifact is downloaded from W&B. Not from a shared drive, not from S3, not from a container. From W&B, by name and alias.

3. **Everything is bundled.** The `best_model_bundle.pth` contains the model weights, the scaler, and the config. The model isn't just the weights — it's the full pipeline needed to reproduce the inference.

**That's the handoff.** From training to serving. From experimentation to production. Through a name and an alias.

---

## Registry vs stages (revisiting the MLflow comparison)

In the MLflow lesson, we covered both stages and aliases. W&B uses aliases only — no concept of exclusive stages.

| MLflow | W&B |
|--------|-----|
| Stages (`Staging`, `Production`, `Archived`) | Aliases (`staging`, `production`) |
| Exclusive — one stage per version | Non-exclusive — many aliases per version |
| Auto-archives the previous Production | No auto-archive |
| Classic pattern | Modern pattern |

**W&B made the modern choice.** Aliases are more flexible:

- A version can be `champion` and `validated` and `eu-region` at once
- Moving an alias doesn't archive anything
- You can have custom lifecycle names beyond the fixed set

**The tradeoff:** stages were more opinionated. `Production` meant something specific. Aliases are just names — your team decides what `production` means.

**In practice, this is fine.** Conventions emerge. Teams agree that `production` means "live." W&B just doesn't enforce it.

---

## What comes next

The model is registered. The aliases are set. The serving system can load by name.

**That's the end of the chain.**

```
data → code → environment → run → artifact → model
                        ✓        ✓       ✓
```

Module 1 fixed the environment.
Lesson 2.1 fixed the data.
Lesson 2.2 fixed the run (for an individual).
Lesson 2.3 fixed the team.

**The chain is complete.** Every link has a tool. Every tool is in place.

What's next in the course is what happens *after* the model is registered:

- **Module 3** — Data pipelines. Automate the training data flow.
- **Module 4** — Optimization. Compress the model for deployment.
- **Module 5** — Serving. Ship the registered model to Kubernetes.
- **Module 6** — Observability. Watch it in production.

The registry is the bridge between Module 2 and everything after. Every deployment loads `:production`. Every retrain logs a new version. The alias moves.

---

## Checkpoint

You should now be able to answer:

- **What is a registered model in W&B?** A named, curated artifact with lifecycle aliases.
- **What are the three aliases that matter?** `best` (automatic), `staging`, `production` (manual).
- **How does a serving system reference a production model?** By name and alias — `sales-forecasting-lstm:production`.
- **What happens when a new version is promoted?** The alias moves. The serving code doesn't change.
- **What's the difference from MLflow stages?** W&B uses aliases only — non-exclusive, no auto-archive.
- **What does the notebook's `register_and_promote()` do?** Loads the `best` checkpoint, adds `staging` and `production` aliases, saves.

**You've finished the ladder.** Return to the [README](README.md) for the [Quick Start](README.md#quick-start) and [Where This Fits](README.md#where-this-fits).