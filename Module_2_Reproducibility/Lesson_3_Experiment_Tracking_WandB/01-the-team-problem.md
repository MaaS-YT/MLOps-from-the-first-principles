# 01 — The Team Problem

> [← Back to the ladder](README.md) · [Next: The W&B Run →](02-the-wandb-run.md)

---

## Where you are

You finished the MLflow lesson. You have a working tracking setup. You can log runs, compare experiments, register models, and load by stage or alias.

It works.

For you.

---

## Now add a teammate

Your colleague wants to see your latest experiment. They open their own MLflow UI.

**Your runs aren't there.**

MLflow stores runs in a SQLite file on *your* laptop. Their SQLite file is on *their* laptop. Two separate universes.

You could:
- Copy the file over
- Set up a shared server
- Move everything to Postgres

All three are real solutions. All three are **your problem now** — infrastructure you have to build, maintain, and secure.

---

## The three failures that appear when there's more than one person

**1. Shared visibility — "I can't see what you ran."**

Your teammate can't see your runs. Your manager can't see your latest. The ML engineer who needs to serve your model has no access to its hyperparameters.

Everyone relies on screenshots, Slack threads, and "hey, what did you get on that one?"

**2. Data provenance — "I know which model won, but not which data trained it."**

MLflow tracks runs. Runs reference artifacts. But there's no native way to answer: *"This production model — which version of the training data produced it?"*

You'd have to manually trace: model → run → artifact → dataset → ... and hope every link was recorded.

**3. Lineage — "How was this model actually created?"**

The question a new teammate asks on their first day. Or an auditor. Or you, six months from now.

Without lineage, the answer is a manual investigation across three tools.

---

## Why MLflow doesn't solve these

MLflow is **self-hosted by design.** It gives you the tools to build a shared setup, but doesn't assume one. That's a feature — you can run it offline, in a classified environment, or as a solo scientist.

But self-hosted means:
- One person runs the server
- Everyone has to configure their client to point at it
- The server has to be running when anyone wants to log
- Artifact storage has to be somewhere everyone can reach

For a solo project, this is fine. For a team, it's infrastructure.

**W&B was designed for the team case first.** Everything is cloud-native. Everyone sees the same thing by default.

---

## What W&B changes

**Shared by default.**
Every `wandb.init()` syncs to the cloud. Your teammate opens a URL. They see your runs.

**Artifacts have lineage by default.**
When a run calls `use_artifact()`, W&B draws an edge from the artifact to the run. When a run calls `log_artifact()`, W&B draws an edge from the run to the artifact. The lineage graph is built automatically — you don't configure anything.

**No infrastructure to build.**
No server. No database. No artifact storage. You sign up, you `wandb login`, you run code. Everything works.

That's the trade:
- MLflow gives you **control** — you own every piece.
- W&B gives you **velocity** — none of it is your problem.

---

## The mental model

W&B is **cloud-native experiment tracking with artifact lineage.**

Two things to hold in your head:

**1. The run abstraction is the same as MLflow.**
A run is one execution. It has config, metrics, and artifacts. Same shape.

**2. The lineage graph is new.**
Every artifact has inputs and outputs. Every edge is a relationship: "this run consumed that artifact" or "this run produced that artifact." The graph is the traceability.

Put them together: runs are the atoms, artifacts are the molecules, lineage is the chemistry between them.

---

## The triune, again

Same shape as Module 1 and the previous two lessons:

```
.wandb run            →  run + metrics + artifacts  →  loadable model
definition            →  artifact                    →  instance
.dvc file             →  hash + remote               →  checked-out data
Dockerfile            →  image                       →  container
```

In this lesson, the triune applies at two levels:

**Within a run:**
- **Definition:** `wandb.init(config=...)`
- **Artifact:** the run's metrics, logs, and checkpoint files
- **Instance:** the live dashboard in the W&B UI

**Across the pipeline:**
- **Definition:** `use_artifact()` and `log_artifact()` calls
- **Artifact:** the lineage graph itself
- **Instance:** any node you click to inspect

Same three-part shape, different scale. **Learn it once, use it everywhere.**

---

## Where W&B fits in the chain

```
data → code → environment → run → artifact → model
 ↑                    ↑       ↑       ↑        ↑
2.1                 Module 1  2.2     2.2     2.3
DVC                 Docker   MLflow  MLflow  W&B
```

- **Module 1** fixed the **environment** link — Docker.
- **Lesson 2.1** fixed the **data** link — DVC.
- **Lesson 2.2** fixed the **run** link — MLflow, for a single person.
- **Lesson 2.3** fixes the **team** link — W&B, for the whole team.

They compound. Fixing data alone leaves runs untraceable. Fixing runs alone leaves the team blind. You need all three.

**W&B doesn't replace MLflow.** It's not "MLflow but cloud." It's the piece that closes the team gap MLflow leaves open.

---

## What this lesson will show

By the end of the ladder, you'll have:

- A multi-stage ML pipeline (ingest → preprocess → train → register)
- Every stage logged as a W&B run
- Every stage's output logged as an artifact
- The full lineage graph — clickable, traceable, auditable
- A registered model with `staging` and `production` aliases
- A shareable URL — send it to a teammate, they see everything

**Not a toy.** A real deep learning pipeline, fully tracked, fully visible, fully reproducible.

---

## Checkpoint

You should now be able to answer:

- **What does MLflow not solve?** Team visibility, data provenance, and lineage — all three by default.
- **What does W&B change?** Everything is cloud-native. Runs sync automatically. Lineage is drawn automatically.
- **What's the trade?** MLflow gives you control. W&B gives you velocity. Different optimization targets.
- **What's the triune here?** run → artifact → instance, at two scales (within a run, across the pipeline).
- **Where does W&B fit in the chain?** It's the **team link** — the piece that makes reproducibility a team property, not just an individual one.

**Next:** [The W&B Run →](02-the-wandb-run.md)