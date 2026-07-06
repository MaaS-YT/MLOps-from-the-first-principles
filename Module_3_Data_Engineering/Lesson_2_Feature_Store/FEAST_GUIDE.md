# Lesson 3.2 — The Feature Store: Eliminating Training-Serving Skew

> **Lesson 3.2** — The most common cause of production ML failure is not model quality. It is that training and serving compute the same feature differently. A feature store is the infrastructure that makes this impossible.

| | |
|---|---|
| **Problem this solves** | Your model achieves 92% on the validation set. In production, it consistently makes wrong predictions. The data has not changed. The model has not changed. But features are computed differently in training (pandas, batch) vs. serving (Python, real-time). This silent divergence is **training-serving skew**. |
| **Mental model** | A feature store is a shared warehouse for ML features. Features are manufactured goods: computed once, stored centrally, consumed identically by training and serving. No more duplicate definitions drifting apart. |
| **What the lecture demonstrates** | Building a Feast feature repo → writing features to the offline store → retrieving point-in-time correct training data → materializing to the online store → serving features via a live API lookup |
| **Where this fits** | Feast sits in the **Data Layer**, between the data pipeline (Module 6) and the training run (Modules 4–5). It is the contract that guarantees training data and serving data are computed identically. |

---

## Part 1 — The Problem in Detail

### Why training-serving skew happens

Module 6 built a data pipeline that produces a clean training dataset.
That pipeline runs in pandas, in a batch job, on historical data.

Now your model is deployed as an API. A request arrives. The API needs
the same features the model was trained on — but it must compute them in
real time, from a live database, using whatever code the serving team wrote.

The two computations are independent. Differences accumulate:
- Training normalizes `tenure` by dividing by the training set maximum (say 72)
- Serving normalizes `tenure` by dividing by today's maximum (which is 74)
- Training computes `monthly_charges` as a float
- Serving queries a database that returns it as a string

Every difference is a vector shift between the distribution the model learned
and the distribution it receives at inference. Predictions degrade.
Nobody notices until the metrics are already bad.

### Why this is hard to catch

Training-serving skew does not cause errors. The serving API returns predictions.
They look reasonable. The model has not "broken." The performance metric slowly drifts
over weeks until someone runs an evaluation and realizes accuracy dropped from 92% to 81%.
By then the root cause is buried in the history of two independent codebases.

### What a feature store does

A feature store solves the problem structurally, not procedurally:

- Features are **defined once** in a shared schema
- The offline store holds **historical feature values** for training
- The online store holds **the latest feature values** for serving
- Both training and serving call the same store API with the same feature names
- There is no second implementation to drift

```
Without feature store:
  Training code (pandas)   defines MonthlyCharges → 65.0
  Serving code (SQL)       defines MonthlyCharges → "65"  ← silent skew

With feature store:
  Feature definition once  MonthlyCharges: float, computed as X
  Training:   store.get_historical_features() → 65.0
  Serving:    store.get_online_features()     → 65.0
```

---

## Part 2 — The Two Stores

A feature store contains two separate databases with different performance
profiles, used at different points in the ML lifecycle.

### The Offline Store

The offline store holds the full history of every feature value for every entity.
When you ask "what were this customer's features on November 15th?", the offline store
answers by looking up the nearest row before that timestamp.

It is optimized for large historical scans, not for low-latency single lookups.
Typical backends: Parquet files (local/S3), BigQuery, Snowflake, Hive.

Used by: **training** — `store.get_historical_features()` queries the offline store.

### The Online Store

The online store holds only the **latest** feature value for each entity.
It is a key-value store: "given customer ID 42, what are their current features?"

It is optimized for single-entity lookups under 10ms — the latency budget
of a real-time serving API. It does not keep history.

Used by: **serving** — `store.get_online_features()` queries the online store.

### The Boundary Between Them

```
Offline Store (full history)
  customer_id=42, event_timestamp=Jan 1,  MonthlyCharges=60.0
  customer_id=42, event_timestamp=Feb 1,  MonthlyCharges=65.0
  customer_id=42, event_timestamp=Mar 1,  MonthlyCharges=70.0
                │
                │  materialize()  ← copies latest value per entity
                ▼
Online Store (latest only)
  customer_id=42,  MonthlyCharges=70.0
```

`materialize()` is the sync operation. It reads the offline store and updates
the online store with the latest value for each entity. In production this runs
on a schedule (hourly, daily) to keep the online store fresh.

---

## Part 3 — Point-in-Time Correctness

This is the most important concept in this module.

When training a churn model, each row in the training dataset represents
a customer at a specific moment in time: the moment the churn event
was observed (or the moment a non-churn was recorded). The features for
that row must be the features that existed **at that moment** — not features
from a later update.

```
label_timestamp (churn observed):    March 15
                                           │
feature value at training time:      ◄────┘
  MonthlyCharges = $65.00   (Feb 28 — last known before churn)

feature value at a later date:
  MonthlyCharges = $70.00   (April 2 — a later billing update)
```

If you accidentally use the April 2 value in a row labeled March 15, the model
is trained on information that did not exist at prediction time. This is leakage
through time, and it causes the model to perform worse in production than
it appeared to in training.

Feast handles point-in-time correctness automatically in `get_historical_features()`.
For each row in the entity DataFrame (which carries a timestamp), Feast finds the
latest feature row in the offline store **before** that timestamp. You do not need
to write this join manually — but you need to understand why it matters.

---

## Part 4 — Feast Architecture

### The Four Core Objects

**Entity**
The primary key of your feature store — the thing features describe.
In this demo: `customer_id`. In other systems: `user_id`, `item_id`, `driver_id`.

**DataSource**
Where Feast reads raw feature data from. This module uses local Parquet files.
In production this could be BigQuery, Redshift, or a Kafka stream.

**FeatureView**
A group of related features, their schema, and their TTL (how long a value
is considered valid). One FeatureView maps to one data source.

**FeatureService**
A named bundle of features for a specific model. Rather than listing individual
feature names in your training code, you reference a FeatureService — a contract
between the feature store and the model. When features change, the FeatureService
is updated in one place.

### The Feature Repo Structure

```
feat_telco_repo/
├── feature_store.yaml   ← project config, registry path, online/offline backends
├── registry.db          ← metadata: feature definitions, entity mappings
├── online_store.db      ← latest values per entity (SQLite in dev, Redis in prod)
└── data/
    ├── telco_features.parquet   ← feature values over time (offline store source)
    └── entity_labels.parquet    ← churn outcomes + timestamps (entity_df)
```

`feature_store.yaml` is the configuration file that declares backends.
Swap `offline_store.type` from `file` to `bigquery` and the same feature code
runs at petabyte scale — with no changes to training or serving code.

---

## Part 5 — The Pipeline in This Module

The demo builds a complete training and serving pipeline for Telco churn prediction.

### Step 1 — Compute and store features

The data pipeline (Module 6 output) computes features for each customer.
These feature rows, indexed by `customer_id` and `event_timestamp`, are written
to a Parquet file and registered as a Feast DataSource.

### Step 2 — Define the feature store schema

```python
customer = Entity(name="customer_id")

source = FileSource(
    path="data/telco_features.parquet",
    event_timestamp_column="event_timestamp",
    created_timestamp_column="created_at",
)

customer_stats = FeatureView(
    name="customer_stats",
    entities=[customer],
    source=source,
    schema=[Field(name="tenure", dtype=Float32), ...],
    ttl=timedelta(days=200),
)

store.apply([customer, customer_stats])
```

`store.apply()` writes the schema to the registry. It does not move data.
It registers the definitions so Feast knows what exists and where to find it.

### Step 3 — Generate training data with point-in-time correctness

```python
training_df = store.get_historical_features(
    entity_df=entity_df,   # rows with customer_id and timestamp (churn date)
    features=["customer_stats:tenure", "customer_stats:MonthlyCharges"],
).to_df()
```

For each row in `entity_df`, Feast finds the latest feature values
from `customer_stats` that occurred **before** that row's timestamp.
The result is a leak-free training dataset.

### Step 4 — Materialize to the online store

```python
store.materialize_incremental(end_date=datetime.now(timezone.utc))
```

This copies the latest feature value per customer from the offline Parquet
into the online SQLite store. In production, this step runs on a schedule
(e.g., hourly) to keep the online store fresh.

### Step 5 — Serve features in real time

```python
result = store.get_online_features(
    features=["customer_stats:tenure", "customer_stats:MonthlyCharges"],
    entity_rows=[{"customer_id": "7590-VHVEG"}],
).to_dict()
```

The serving API calls this during prediction. The same feature names, the same
data types, the same Feast registry that was used in training. Training-serving
skew is eliminated structurally.

---

## Part 6 — Where Feast Fits in the System

```
Module 6 output:                   Module 7 input:
features.parquet                →  Feast DataSource

Training flow:
  entity_df (customer + label ts)
       │  store.get_historical_features()
       ▼
  Training dataset (point-in-time correct)
       │
  Model training (Modules 4–5)

Serving flow:
  POST /predict { customer_id: "123" }
       │  store.get_online_features()
       ▼
  Latest features for customer 123
       │
  Model inference (Module 1)
       ▼
  { "churn_probability": 0.87 }
```

This is the **consistency guarantee** that makes the system reliable.
Both arrows — training and serving — go through the same store with the same schema.
The model was trained on exactly the features it will receive at inference time.

---

## Part 7 — When to Use a Feature Store

Feature stores add complexity. They are not always the right choice.

```
One model, one team, one notebook     → Feast is unnecessary overhead
Multiple models sharing features      → Feast starts paying off
Real-time serving with <50ms budget   → Feast online store is essential
Regulatory audit requires traceability → Feast registry provides it
Multiple teams own different features  → Feast resolves ownership and versioning
```

The rule of thumb: if training and serving code are maintained by different teams
or run in different environments, use a feature store. The risk of silent divergence
grows with team size and system age.

---

## Quick Reference

### Initialize and apply

```python
from feast import FeatureStore
store = FeatureStore(repo_path="feat_telco_repo")
store.apply([customer, customer_stats, svc])
```

### Training data (offline, point-in-time correct)

```python
training_df = store.get_historical_features(
    entity_df=entity_df,
    features=["customer_stats:tenure", "customer_stats:MonthlyCharges"],
).to_df()
```

### Materialize offline → online

```python
store.materialize_incremental(end_date=datetime.now(timezone.utc))
```

### Serving (online, real-time)

```python
result = store.get_online_features(
    features=["customer_stats:tenure", "customer_stats:MonthlyCharges"],
    entity_rows=[{"customer_id": "7590-VHVEG"}],
).to_dict()
```

### Key concepts to remember

| Concept | What it means |
|---------|--------------|
| Offline store | Full feature history — for training |
| Online store | Latest values per entity — for serving |
| `materialize()` | Syncs offline → online for a time range |
| `materialize_incremental()` | Syncs from last materialization to now |
| TTL | How long a feature value is considered valid before it's treated as stale |
| Point-in-time | For each label timestamp, use only features that existed before that moment |
| Training-serving skew | Same feature defined differently in training and serving — causes silent model degradation |

---

## Official Documentation

- Feast concepts: https://docs.feast.dev/concepts/overview
- Point-in-time joins: https://docs.feast.dev/getting-started/concepts/point-in-time-joins
- Feature views: https://docs.feast.dev/concepts/feature-view
- Offline stores: https://docs.feast.dev/reference/offline-stores/
- Online stores: https://docs.feast.dev/reference/online-stores/
