# Lesson 3.1 — Data Pipeline: Building Clean Training Data

> **Lesson 3.1** — Why manually cleaned notebooks are not pipelines, how the temporal cutoff eliminates the most common form of data leakage, and how to build a repeatable feature engineering process with an explicit validation layer.

| | |
|---|---|
| **Problem this solves** | A model trained on leaked data passes evaluation and fails in production. The root cause is almost always a temporal error: features were computed using information that wouldn't have existed at prediction time. A structured pipeline enforces the boundary. |
| **Mental model** | The cutoff is a wall in time. Everything to the left of the wall is what your model can see. Everything to the right is what it is predicting. Any feature that crosses the wall is leakage. Build the pipeline so the wall is enforced structurally, not by convention. |
| **What the lecture demonstrates** | Ingesting from three source formats (CSV, JSON, SQLite) → validating schema and distributions → engineering RFM features with a strict cutoff → generating churn labels from the label window → splitting temporally (not randomly) |
| **Where this fits** | This module builds the **Data Layer** — the foundation of the system. All upstream modules (serving, tracking, versioning) assume clean, correctly-labeled training data. This module is where that data is built. |

---

# Data Pipeline Part 1 — Concepts & Guide

Demo: `data_pipeline.ipynb` — Multi-source ingestion, validation, RFM feature engineering, churn labeling.

---

## Part 1 — What Is a Data Pipeline?

### The problem it solves

A data scientist downloads a CSV, cleans it manually in a notebook, builds features.
It works. Then:

- Next month there's new data — they redo every step manually, possibly differently
- A second model needs the same features — they copy the code, which then diverges
- A cleaning step silently filters the wrong rows — nobody knows until the model degrades
- The ML team needs the data in a different format — they rebuild from scratch

These failures have nothing to do with the model. They are failures of process.

A data pipeline is a structured, repeatable, auditable process:
raw data comes in one end, a clean ready-to-use dataset comes out the other.
Same inputs always produce the same outputs. Every step is recorded.

---

### ETL vs ELT

Two patterns dominate data pipeline design:

```
ETL  (Extract → Transform → Load)
  Pull raw data from sources
  Clean and transform it into the final form
  Store the result
  
  Classic. Transformation happens before storage.
  Good when: storage is limited, only the final form is needed.

ELT  (Extract → Load → Transform)
  Pull raw data from sources
  Store it immediately (raw, unchanged)
  Transform when needed, by whoever needs it
  
  Modern. Raw data is preserved.
  Good when: storage is cheap, multiple teams need different transforms,
             you want to be able to re-derive features from scratch.
```

Real pipelines are often hybrids. The demo is a deliberate hybrid:
the data team runs ETL to produce intermediate features; the ML team then
runs ELT to produce preprocessed training data. Each stage has its own
Extract, Load, and Transform phases with a different consumer in mind.

---

### The three data sources

Real data lives in multiple systems. This demo uses three deliberately different formats:

```
sales.csv          Flat file        Transactional records, realistic noise
events.json        JSON             User behavior events, semi-structured
customers.db       SQLite           Master data, relational
```

The diversity is intentional. Every organization has data scattered across
different systems with different formats, schemas, and quality levels.
A pipeline's first responsibility is to bring these into a coherent, validated whole.

---

### Where this fits in the MLOps progression

```
1. Manual notebook         clean once, not repeatable
2. Reusable functions      extract/transform as functions, still manual execution
3. Data pipeline           structured, repeatable, auditable         ← you are here
4. Feature store           centralized, versioned, serving-ready     (Part 2)
5. Pipeline orchestration  Prefect/Airflow schedules and monitors
6. Data quality monitoring detect upstream data drift before it reaches the model
```

---

## Part 2 — The Temporal ML Problem

### The cutoff concept

This is the most important concept in this pipeline — and the one most commonly
misunderstood.

When building a churn prediction model, you're asking: "Will this customer
stop buying from us?" But you can only ask this question at a specific point in time.

That point is the **cutoff**: the moment you imagine yourself making predictions.
Features are computed from data *before* the cutoff. Labels are determined from
data *after* the cutoff.

```
obs_start              cutoff                     label_end
    │                    │                             │
    ├────────────────────┤─────────────────────────────┤
    │  observation window│       label window           │
    │  (what we can see) │  (what we're predicting)    │
    │                    │                             │
  Features built         Did the customer              churn = 1 if no purchase
  from purchases here    buy anything here?            churn = 0 if at least one
```

**Why this matters:**
If you accidentally include purchases from after the cutoff when computing features,
the model sees information it wouldn't have had at prediction time.
This is **data leakage** — it inflates training metrics but the model fails in production.

The cutoff is the dividing line between "what you know" and "what you're predicting."
Every temporal ML pipeline must enforce it rigorously.

---

### The observation window

Not all history is relevant. A customer's purchase behavior from three years ago
may not predict their behavior next month.

The **observation window** defines how far back to look:

```python
obs_start = cutoff - timedelta(days=200)
```

Features are computed from purchases in `[obs_start, cutoff)`.
The choice of window length is a domain decision — 200 days for this churn problem —
but it always has the same structure: a lookback from the cutoff.

---

### The label window

After the cutoff, there's a defined window to observe the outcome:

```python
label_end = cutoff + timedelta(days=60)
```

A customer is labeled `churn=1` if they made **no** purchase in `[cutoff, label_end)`.

```python
has_future_purchase = sales_label.groupby("customer_id")["sale_id"].count()
feats["label_churn"] = (has_future_purchase == 0).astype(int)
```

The 60-day window is a business decision: "we consider a customer churned if
they haven't bought in 60 days." Different businesses define churn differently.
The pipeline structure stays the same; only the window changes.

---

## Part 3 — Data Validation

### Why validation is a first-class concern

Garbage in, garbage out. But worse: **silent** garbage in.

A model trained on corrupted data doesn't fail with an error — it trains fine,
evaluates on corrupted validation data, and produces optimistic metrics.
You discover the problem only when it fails in production.

Validation catches corruption early, at the point of ingestion, with clear error messages.
It is not optional — it is the most important quality gate in the pipeline.

---

### What validation covers

Three categories of problems are handled in the demo:

**Structural validation** — Does the data have the right shape?
```python
for c in ["sale_id", "customer_id", "amount", "ts"]:
    if c not in df.columns:
        raise ValueError(f"Missing column {c}")
```
A missing column is always a pipeline failure. Fail fast, fail loudly.

**Type coercion** — Is the data the right type?
```python
df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
df["ts"] = pd.to_datetime(df["ts"], errors="coerce")
```
`errors="coerce"` converts unparseable values to `NaN` instead of raising an exception.
This preserves the pipeline but marks bad values for downstream handling.

**Business rule validation** — Is the data plausible?
```python
# Negative amounts are invalid (returns or data errors)
df.loc[df["amount"] <= 0, "amount"] = np.nan

# Extreme outliers capped at 95th percentile × 5
cap = df["amount"].quantile(0.95) * 5
df.loc[df["amount"] > cap, "amount"] = cap
```

Not every anomaly is an error — outliers happen. The cap prevents extreme values
from dominating statistical aggregations without discarding the data entirely.

---

### The imputation strategy

After validation, some `amount` values are `NaN`. These need to be filled.

The demo uses a two-level fallback:

```python
# Level 1: fill with that customer's own median purchase amount
df["amount"] = df.groupby("customer_id")["amount"].transform(
    lambda s: s.fillna(s.median())
)

# Level 2: fill remaining NaN with the global median
df["amount"] = df["amount"].fillna(df["amount"].median())
```

Level 1 is more informative: a customer who typically spends $200 should have their
missing amount estimated at $200, not the global median of $150. Customer-level
imputation uses that customer's own behavior pattern.

Level 2 handles cases where a customer has only one transaction and it's NaN —
no customer-level median exists, so fall back to global.

---

## Part 4 — Feature Engineering

### RFM — the standard framework

Customer behavior features are typically summarized using **RFM**:
Recency, Frequency, Monetary. This framework has been used in marketing analytics
for decades and maps directly onto ML features.

| Feature | What it measures | Signal |
|---|---|---|
| Recency | Days since last purchase | Recent buyers are less likely to churn |
| Frequency | Purchase count in obs window | Frequent buyers are more engaged |
| Monetary | Total spend in obs window | High spenders have higher switching cost |

The demo extends RFM with three additional features:

| Feature | What it measures |
|---|---|
| `avg_amount` | Typical spend per purchase |
| `max_amount` | Highest single purchase (identifies big-ticket buyers) |
| `avg_gap_days` | Average days between purchases (buying rhythm) |
| `tenure_days` | Days since signup (loyalty signal) |

Together, these features capture a customer's engagement pattern in a compact,
interpretable vector — the kind that generalizes well across model types.

---

### The NaN problem in RFM features

`avg_gap_days` requires at least two purchases to compute — one interval needs two endpoints.
Customers with fewer than 2 purchases in the observation window get `NaN`.

```
avg_gap_days: 92 NaN    → customers with < 2 purchases
monetary:     37 NaN    → customers with 0 purchases in observation window
```

These NaNs are **informative**. A customer with `avg_gap_days = NaN` either:
- Just signed up and has only bought once
- Has been inactive and made no repeated purchases

In the ML preprocessing stage, these are imputed with the mean. In a production system,
you might also create a boolean `has_repeat_purchase` feature to preserve the signal.

---

## Part 5 — The Hybrid ETL/ELT Structure

### Phase 1: Data team ETL

```
Extract:   Read from CSV + JSON + SQLite
Validate:  Type coercion, outlier capping, deduplication, imputation
Transform: Compute RFM features, apply cutoff window, generate churn label
Load:      Write to out/features.parquet + out/features.csv + out/features.json
           Write out/data_dictionary.json + run_metadata.json + qa_report.json
```

The Load phase saves three file formats (parquet, csv, json) because downstream
consumers have different needs: the feature store (Feast in Part 2) reads parquet;
humans read csv; APIs read json.

The **QA report** and **data dictionary** are the handoff documents.
The ML team picks up the data not just as a file but with documentation:
what each column means, what the cutoff was, what the churn rate is.

---

### Phase 2: ML team ELT

```
Extract:   ML team reads out/features.json (simulates fetching from intermediate store)
Load:      Save a local copy to data.csv before any transformation
Transform: sklearn Pipeline — imputation, scaling, encoding, train/val split
```

The Load-before-Transform step is important: save the raw intermediate before
any ML-specific processing. If the preprocessing turns out to be wrong (wrong
imputation strategy, wrong scaler), you can rerun preprocessing without
rerunning the expensive upstream pipeline.

---

### The sklearn Pipeline for preprocessing

The preprocessing is done with sklearn's `Pipeline` and `ColumnTransformer`.

The critical principle is **fit on train only, transform both**:

```python
# Split FIRST
X_train, X_valid, y_train, y_valid = train_test_split(X, y, stratify=y)

# Fit the preprocessor on TRAINING DATA ONLY
X_train_processed = preprocessor.fit_transform(X_train)

# Apply the same learned transforms to validation
X_valid_processed = preprocessor.transform(X_valid)
```

`fit_transform` learns statistics from the data (mean for imputation, mean/std for scaling).
If you fit on the full dataset before splitting, the validation data's statistics
contaminate the training statistics — a subtle leakage that inflates validation metrics.

The Pipeline also ensures the same sequence of transforms is always applied in the same order.
Save the fitted `preprocessor` as an artifact; load it in production to apply the exact
same transforms to live data that were applied during training.

---

## Part 6 — The Leakage Checklist

Leakage is the most common reason a model "works in the notebook" but fails in production.

| Stage | Leakage risk | Guard |
|---|---|---|
| Feature computation | Using data after the cutoff | `obs_mask = ts < cutoff` |
| Churn label | Using purchases in the feature window | `label_mask = ts >= cutoff` |
| Train/val split | Fitting scaler before splitting | Always split first |
| Imputation | Imputing with global stats before split | `fit_transform` on train only |
| Feature selection | Selecting features on validation performance | Use train data only |

Each row is an independent failure mode. A model can pass all other checks and still
leak through one of these channels.

---

## Quick Reference

### Key parameters

```python
cutoff    = 75th percentile of timestamps  # prediction point in time
obs_days  = 200                            # lookback window for features
label_days = 60                            # lookahead window for churn label
```

### Key output files

```
out/features.parquet          Feature matrix (used by Feast in Part 2)
out/features.csv              Same, human-readable
out/data_dictionary.json      Column definitions
out/run_metadata.json         Run parameters and timestamps
out/qa_report.json            Data quality summary (churn rate, row counts)
data.csv                      Intermediate saved before ML preprocessing
```

### Run the notebook

```bash
jupyter notebook data_pipeline.ipynb
# or open in VS Code and run all cells
```

---

## Official Documentation

- pandas ETL patterns: https://pandas.pydata.org/docs/user_guide/
- sklearn Pipeline: https://scikit-learn.org/stable/modules/pipeline.html
- ColumnTransformer: https://scikit-learn.org/stable/modules/generated/sklearn.compose.ColumnTransformer.html
- RFM analysis: https://en.wikipedia.org/wiki/RFM_(market_research)
