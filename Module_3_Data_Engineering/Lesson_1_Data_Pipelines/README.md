# Lesson 3.1 — Data Pipelines

| | |
|---|---|
| **Problem this solves** | Manually cleaned notebooks are not pipelines. They leak future data into training, skip validation, and can't run unattended. The result: a model that looks good in evaluation and fails silently in production. |
| **Mental model** | A data pipeline is a function: raw data in, clean labeled features out. Every step is explicit and ordered. The temporal cutoff is the most critical step — it enforces that features only use information available before the label's timestamp. |
| **What the lecture demonstrates** | Building a churn prediction pipeline: ingest → validate schema → engineer features → apply temporal cutoff → label → output a training-ready dataset |
| **Where this fits** | This is the **Data Layer** entry point. It produces the training dataset that the experiment tracker (Module 2) and feature store (Lesson 3.2) depend on. |

---

## Files

| File | Purpose |
|------|---------|
| `DATA_PIPELINE_GUIDE.md` | Full guide: ETL vs ELT, validation, temporal cutoff, leakage patterns |
| `data_pipeline.ipynb` | Demo: end-to-end churn prediction pipeline |

**Start with:** `DATA_PIPELINE_GUIDE.md`

---

## Run the Demo

```bash
uv run --no-sync jupyter notebook data_pipeline.ipynb
```
