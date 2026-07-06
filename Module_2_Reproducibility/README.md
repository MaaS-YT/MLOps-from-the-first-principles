# Module 2 — Reproducibility

*You can't improve what you can't reproduce.*

The hardest thing about a model that works is proving it works — and being able to
make it work again tomorrow, with a different teammate, on a different machine.
This module solves that problem at three levels: the data, the experiment, and the team.

---

## Lessons

| Lesson | Topic | Problem It Solves | Guide |
|--------|-------|-------------------|-------|
| 2.1 | [Data & Model Versioning — DVC](Lesson_1_Data_and_Model_Versioning/) | Which data version trained the model in production? | [DVC_GUIDE.md](Lesson_1_Data_and_Model_Versioning/DVC_GUIDE.md) |
| 2.2 | [Experiment Tracking — MLflow](Lesson_2_Experiment_Tracking_MLflow/) | 20 experiments, no record of which was best or how to reproduce it | [README.md](Lesson_2_Experiment_Tracking_MLflow/README.md) |
| 2.3 | [Experiment Tracking — W&B](Lesson_3_Experiment_Tracking_WandB/) | Team visibility and artifact lineage across experiments | [WANDB_GUIDE.md](Lesson_3_Experiment_Tracking_WandB/WANDB_GUIDE.md) |

---

## What This Module Builds

```
VERSIONING LAYER (Lesson 2.1)
  git commit  →  code + .dvc pointer
  dvc push    →  actual data to remote
  Any result is reproducible: checkout the commit, pull the data, run the code

TRAINING LAYER (Lessons 2.2–2.3)
  MLflow  →  structured run logs + model registry (self-hosted)
  W&B     →  cloud-native tracking + artifact lineage (team-wide)
```

Each lesson solves one reproducibility failure mode:
- 2.1: the data changes and you don't know which version trained which model
- 2.2: you have 20 experiments with no structured record of what ran
- 2.3: your teammates can't see your experiments, and you can't trace a model back to its data

---

## Where This Fits

This module sits between the Serving Layer (Module 1) and the Data Engineering layer
(Module 3). Reproducibility is the prerequisite for everything that follows:
you can't debug a model, retrain it safely, or monitor it in production
unless every artifact is traceable to the exact code, data, and parameters that produced it.

Open `SYSTEM_MAP.md` at the repo root for the full system view.
