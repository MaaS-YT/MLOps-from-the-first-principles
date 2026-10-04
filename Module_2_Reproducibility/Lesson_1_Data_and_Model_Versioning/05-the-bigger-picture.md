# 05 — The Bigger Picture

> [← Previous: Key Concepts](04-key-concepts.md) · [Back to the ladder](README.md)

---

## Why this matters for the whole system

Data versioning isn't a tool you learn in isolation. It's one link in a chain that makes the entire ML system reproducible.

### Experiment reproducibility

> *"Reproduce run X"* requires the same **code**, the same **data**, and the same **environment**.

- Git gives you the code.
- DVC gives you the data.
- Docker gives you the environment.

Without all three, you cannot reconstruct the run.

### Debugging model degradation

> *"The model was better last month."*

Checkout last month's git tag. `dvc checkout`. Run the training script. You have the same data, same code, same model. Now you can diff against today and find what changed.

Without versioning, you're comparing two models you can't reconstruct.

### Auditing and compliance

> *"What data was this model trained on?"*

`git log` the `.dvc` file. Find the exact hash. Find the exact data. **Auditable answer in three commands.**

In regulated industries — healthcare, finance, autonomous vehicles — this isn't optional. It's the difference between shipping a model and not shipping it.

### Team collaboration

Everyone pulls the same data version for the same git commit. No more "works on my machine" for data.

A teammate clones the repo, runs `git checkout` and `dvc pull`, and has exactly what you had. No emails. No Slack threads. No ambiguity.

---

## The complete reproducibility stack

Every layer solves one problem:

```
Code version     →  git
Data version     →  DVC
Environment      →  Docker / conda
Experiments      →  MLflow / W&B
Features         →  Feast (Module 3)
```

Module 1 filled in the environment layer. This lesson fills in the data layer. Module 3 fills in the features layer. Each one is a prerequisite for the next.

Without data versioning:
- MLflow runs reference datasets you can't reproduce.
- W&B artifacts point to files that may have changed.
- Feast features are built from data that isn't traceable.

**DVC is the foundation for everything downstream.**

---

## What comes next

- **Lesson 2.2 — MLflow:** you've versioned the data. Now version the *run* that used it.
- **Lesson 2.3 — W&B:** you've versioned your work. Now make it visible to your team.
- **Module 3 — Data Engineering:** you've learned to version data manually. Now build pipelines that produce versioned data automatically.

You've just fixed the **first link** in the chain. There are more.

---

## Checkpoint

You should now be able to answer:

- **What's the reproducibility stack?** code → data → environment → experiments → features.
- **What does DVC enable for MLflow and W&B?** Without data versioning, their run references are meaningless.
- **Why is this link foundational?** Every downstream tool assumes the data is traceable. DVC makes it so.
- **What comes next in the chain?** The run link — that's Lesson 2.2.

**You've finished the ladder.** Return to the [README](README.md) for the [Quick Start](README.md#quick-start) and [Where This Fits](README.md#where-this-fits).