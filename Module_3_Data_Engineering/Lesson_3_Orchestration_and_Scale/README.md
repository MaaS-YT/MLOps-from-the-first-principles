# Lesson 3.3 — Orchestration and Scale: Prefect + Spark

| | |
|---|---|
| **Problem this solves** | The pipeline works in a notebook. It fails in production because (1) the data doesn't fit on one machine, and (2) nothing retries it when it fails at 3am. Two separate problems — two separate tools. |
| **Mental model** | Prefect solves reliability: every step is a tracked task with retries, logging, and scheduling. Spark solves scale: every step is distributed across many cores. A production pipeline needs both. |
| **What the lecture demonstrates** | Prefect: decorating a pipeline with `@flow`/`@task`, deploying with a cron schedule, watching retries in the UI. Spark: processing 100M rows across partitions, distributed ML with MLlib. |
| **Where this fits** | Prefect and Spark are **Infrastructure** in the system map. They are not data tools — they are the runtime that makes the data pipeline (Lesson 3.1) production-grade. |

---

## Structure

```
Lesson_3_Orchestration_and_Scale/
  DATA_PIPELINE_PART3_GUIDE.md   ← main guide: both tools, the production problem
  prefect-demo/
    README.md                    ← Prefect companion
    example.py                   ← @flow/@task pipeline with schedule
  spark-demo/
    SPARK_GUIDE.md               ← Spark reference guide
    pyspark-vs-pandas.ipynb      ← Spark vs Pandas at 100M rows
    spark-ml-pipeline.ipynb      ← distributed ML pipeline with MLlib
    spark-ml.py                  ← same pipeline as a script
```

**Start with:** `DATA_PIPELINE_PART3_GUIDE.md`, then follow into the sub-demos.

---

## Run the Demos

```bash
# Prefect
uv run --no-sync python prefect-demo/example.py
prefect server start              # then open http://localhost:4200

# Spark (requires Java)
uv run --no-sync jupyter notebook spark-demo/pyspark-vs-pandas.ipynb
uv run --no-sync python spark-demo/spark-ml.py
```
