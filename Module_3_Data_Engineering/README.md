# Module 3 — Data Engineering for ML

*The model is only as good as the data that reaches it.*

Two models trained on the same algorithm with the same hyperparameters can perform
completely differently in production — because their data differed in ways nobody noticed.
This module builds the infrastructure that makes data trustworthy: validated, versioned,
consistently computed, and reliably delivered at scale.

---

## Lessons

| Lesson | Topic | Problem It Solves | Guide |
|--------|-------|-------------------|-------|
| 3.1 | [Data Pipelines](Lesson_1_Data_Pipelines/) | Leakage, validation failures, and temporal errors in training data | [DATA_PIPELINE_GUIDE.md](Lesson_1_Data_Pipelines/DATA_PIPELINE_GUIDE.md) |
| 3.2 | [Feature Store — Feast](Lesson_2_Feature_Store/) | Training and serving compute features differently — silently | [FEAST_GUIDE.md](Lesson_2_Feature_Store/FEAST_GUIDE.md) |
| 3.3 | [Orchestration + Scale](Lesson_3_Orchestration_and_Scale/) | The pipeline works in a notebook — it won't run reliably at scale | [DATA_PIPELINE_PART3_GUIDE.md](Lesson_3_Orchestration_and_Scale/DATA_PIPELINE_PART3_GUIDE.md) |

---

## What This Module Builds

```
DATA LAYER (Lesson 3.1)
  Raw Sources → Ingest → Validate → Engineer → Label
  Temporal cutoff enforced — no future data leaks into training

FEATURE STORE (Lesson 3.2)
  One feature definition for both training and serving
  Offline Store (history) → Training dataset
  Online Store (latest)   → Live serving (<10ms lookup)

INFRASTRUCTURE (Lesson 3.3)
  Prefect  →  @flow / @task, retries, scheduling, monitoring
  Spark    →  partitioned computation across many cores
```

Each lesson solves one data failure mode:
- 3.1: training data contains future information the model won't have at serve time
- 3.2: the same feature is computed differently in training and production — the model degrades silently
- 3.3: the pipeline runs manually in a notebook — it breaks in production and nobody knows

---

## Where This Fits

This module builds the **Data Layer** in the system map — the foundation all training depends on.
Bad data here propagates through every downstream component: the experiment tracker records
a run based on corrupt features; the model registry promotes a model trained on leaked data;
the serving layer delivers predictions from a model that never saw real production inputs.

Open `SYSTEM_MAP.md` at the repo root for the full system view.
