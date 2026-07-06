# Lesson 3.2 — Feature Store with Feast

| | |
|---|---|
| **Problem this solves** | Training and serving compute the same feature differently. The model trains on one distribution and serves on another — accuracy drops silently and nobody knows why. This is training-serving skew, and it is the most common production ML failure mode. |
| **Mental model** | A feature store is a shared library for feature definitions. Write the feature once. Both the training pipeline and the serving API read from the same store. The offline store holds full history for training; the online store holds the latest value per entity for low-latency serving. |
| **What the lecture demonstrates** | Defining features in Feast → materializing to the online store → fetching historical features for training → fetching live features for serving — both using the exact same feature definition |
| **Where this fits** | This is the **Feature Store** in the system map. It sits between the data pipeline (Lesson 3.1) and both the training run (Module 2) and the serving API (Lesson 1.1). |

---

## Files

| File | Purpose |
|------|---------|
| `FEAST_GUIDE.md` | Full guide: training-serving skew, offline vs online store, point-in-time correctness |
| `pipeline_with_feast.ipynb` | Demo: defining features, materializing, fetching for training and serving |

**Start with:** `FEAST_GUIDE.md`

---

## Run the Demo

```bash
uv run --no-sync jupyter notebook pipeline_with_feast.ipynb
```
