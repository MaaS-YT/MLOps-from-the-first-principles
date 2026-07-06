# Data Pipeline Part 3 — Apache Spark & MLlib

Demos: `pyspark-vs-pandas.ipynb` — Spark vs Pandas at 100M rows. `spark-ml-pipeline.ipynb` / `spark-ml.py` — distributed ML pipeline with time-based split and LinearRegression.

---

## Part 1 — Why Spark Exists

### The pandas wall

Pandas is the right tool for data that fits in memory. It's fast, expressive, and
widely understood. Most ML projects start here and should.

The problem appears when data grows past a single machine's RAM. Pandas loads the
entire dataset into memory. When you hit 50M rows with multiple wide columns, two
things happen:

- Memory usage spikes — operations that require copies double or triple it
- Processing is single-threaded — one CPU core does all the work

The `pyspark-vs-pandas.ipynb` demo makes this concrete: 100 million rows.
Pandas crashes the kernel. Spark handles it without issue, distributes the work
across all cores, and never loads the full dataset into the driver's memory at once.

This is the **pandas wall** — the point where a single-machine, in-memory tool
stops being the right answer.

---

### What Spark actually does

Spark is a distributed computation engine. Instead of loading data into one process,
it splits the data into **partitions** and distributes those partitions across workers.
Each worker processes its own partitions in parallel. The driver coordinates, collects
results, and handles the final aggregation.

```
Driver (your code)
  │
  ├── Worker 1: partitions 0–3   → local compute
  ├── Worker 2: partitions 4–7   → local compute
  └── Worker 3: partitions 8–11  → local compute
                                           │
                                    results merged on driver
```

In local mode (`master("local[*]")`), the driver and all workers run in the same JVM
on your machine, using all available CPU cores. This is how the demos run — no cluster
needed, but the same code runs unchanged on a real cluster.

---

### Lazy evaluation

Spark does not execute operations immediately. When you call `.withColumn()` or
`.filter()`, Spark records the transformation in a logical plan but does nothing.
Execution is triggered by an **action** — `.count()`, `.show()`, `.collect()`,
`.write`.

```python
df = spark.range(N)                      # no execution
    .withColumn("sales", F.rand() * 1000) # no execution
    .withColumn("city", ...)              # no execution

_ = df.count()   # ← action: execution happens here
```

This allows Spark to optimize the full plan before running it — push filters down,
merge transformations, reorder operations. The optimizer (Catalyst) often produces
a plan significantly faster than the naive execution order.

---

### Where Spark fits in the MLOps progression

```
1. Pandas notebook        fits in RAM, single core, fast for small data
2. Pandas + chunking      manual batching, hard to maintain, still one core
3. Dask / Polars          drop-in Pandas alternatives, single-machine parallelism
4. PySpark local          full Spark on one machine, same API as cluster   ← you are here
5. PySpark on cluster     same code, distributed across many machines
6. Spark + Delta Lake     ACID transactions, time travel, streaming + batch unified
```

The key insight: code written for local Spark runs on a cluster without changes.
Learning the API locally is learning the production API.

---

## Part 2 — The SparkSession

Every Spark program starts with a `SparkSession`. It is the entry point to all
Spark functionality — SQL, DataFrames, MLlib, streaming.

```python
spark = (
    SparkSession.builder
    .appName("SimpleSparkMLPipeline")
    .master("local[*]")
    .config("spark.driver.memory", "4g")
    .getOrCreate()
)
```

| Config | What it does |
|---|---|
| `appName` | Label shown in Spark UI and logs |
| `master("local[*]")` | Run locally, use all CPU cores |
| `spark.driver.memory` | RAM available to the driver process |
| `getOrCreate()` | Reuse an existing session if one exists (safe for notebooks) |

`local[*]` is the most important setting for local development. The `*` means
"use all available cores." Replace with `local[4]` to limit to 4 cores.

Always call `spark.stop()` at the end. It releases resources and cleanly shuts
down the session. In a notebook, forgetting this across kernel restarts can leave
orphaned JVM processes.

---

## Part 3 — DataFrames vs Pandas

Spark DataFrames look like Pandas but behave differently in one critical way:
**they are immutable**. Every transformation returns a new DataFrame; no operation
modifies in place.

```python
# Pandas — in-place possible:
df["hour"] = df["ds"].dt.hour

# Spark — always returns a new DataFrame:
df = df.withColumn("hour", F.hour("ds").cast("double"))
```

The Spark column operations come from `pyspark.sql.functions` (imported as `F`).
These are distributed-safe functions that operate on the entire column across
all partitions.

### Common operations

```python
# Add a column
df = df.withColumn("new_col", F.col("a") * 2)

# Filter rows
df = df.filter(F.col("rank") <= cutoff)

# Aggregate
df.groupBy("city").agg(F.sum("sales").alias("total_sales"))

# Window function
w = Window.orderBy("ds")
df = df.withColumn("rank", F.row_number().over(w))
```

Window functions in Spark work like SQL window functions — they compute a value
for each row relative to a group of rows, without collapsing the DataFrame.
`F.row_number().over(w)` assigns a sequential rank based on the ordering column.

---

## Part 4 — Time-Based Split

The ML pipeline uses a time-based train/test split, not a random one. This is
the correct approach for any time-series or temporal prediction problem.

```python
w = Window.orderBy("ds")
df_ranked = df.withColumn("rank", F.row_number().over(w))

n      = df_ranked.count()
cutoff = int(n * 0.8)

train = df_ranked.filter(F.col("rank") <= cutoff).drop("rank")
test  = df_ranked.filter(F.col("rank") >  cutoff).drop("rank")
```

**Why not random split?**

A random split on temporal data creates leakage. If a test row from timestamp T+10
was used to estimate any statistics, and its "future" data ended up in the training
set, the model has information it would never have had at prediction time.

Time-based split enforces the invariant: training data is always earlier than test
data. The model is evaluated exactly as it would be deployed — trained on the past,
tested on the future.

The 80/20 split means the first 8,000 rows (earliest timestamps) are training data
and the last 2,000 rows (most recent timestamps) are the test set.

---

## Part 5 — The MLlib Pipeline

MLlib is Spark's built-in machine learning library. It follows the same
`Pipeline` abstraction as sklearn: a sequence of stages, each transforming
the DataFrame in turn. The pipeline is fitted on training data and applied to test data.

### The three stages

```python
imputer = Imputer(
    inputCols=["hour", "feature_a", "feature_b"],
    outputCols=["hour_imp", "feature_a_imp", "feature_b_imp"]
)

assembler = VectorAssembler(
    inputCols=["hour_imp", "feature_a_imp", "feature_b_imp"],
    outputCol="features"
)

lr = LinearRegression(labelCol="y", featuresCol="features")

pipeline = Pipeline(stages=[imputer, assembler, lr])
```

**Stage 1 — Imputer**: Fills missing values. Learns the mean of each column from
the training data. Outputs new columns with `_imp` suffix, leaving originals intact.

**Stage 2 — VectorAssembler**: Combines individual feature columns into a single
`DenseVector` column called `features`. This is a hard requirement for all MLlib
algorithms — they expect a single vector column, not individual columns.

```
Before: hour_imp | feature_a_imp | feature_b_imp
After:  features → DenseVector([14.0, 0.312, 7.841])
```

**Stage 3 — LinearRegression**: Trains on the `features` vector. `labelCol` is the
target; `featuresCol` is the assembled vector.

### Fit and transform

```python
model = pipeline.fit(train)       # stages learn from training data
preds = model.transform(test)     # apply to test, adds "prediction" column
```

`pipeline.fit(train)` returns a `PipelineModel` — a fitted version of each stage.
The Imputer now knows the training means. The LinearRegression now has learned
coefficients. `model.transform(test)` applies these fitted stages to the test set
without re-learning anything.

This is the same fit-on-train-only discipline as sklearn, enforced by the API.

---

## Part 6 — Evaluation

```python
evaluator = RegressionEvaluator(
    labelCol="y",
    predictionCol="prediction",
    metricName="r2"
)

r2_test = evaluator.evaluate(preds_test)
print(f"R²(test): {r2_test:.4f}")
```

R² (coefficient of determination) measures how much of the variance in `y` the
model explains. A score of 1.0 is a perfect fit; 0.0 means the model does no
better than predicting the mean.

On this synthetic dataset the true relationship is:
```
y = 2.0 * feature_a + 0.3 * feature_b + noise
```

The model recovers this exactly — R² is close to 1.0 — which confirms the pipeline
is correct. In a real dataset, R² depends on signal quality, feature richness, and
model complexity.

Other available metrics: `rmse`, `mae`, `mse`.

---

## Part 7 — Pandas vs Spark: When to Use Which

| | Pandas | PySpark |
|---|---|---|
| Data size | Fits in RAM (< ~10M rows) | Larger than RAM, or 10M+ rows |
| Execution | Eager (immediate) | Lazy (deferred until action) |
| Parallelism | Single core | Multi-core / multi-node |
| Mutability | In-place operations allowed | Immutable — always returns new DF |
| Ecosystem | sklearn, matplotlib, scipy | MLlib, Spark SQL, Delta Lake |
| Startup cost | Instant | JVM startup (~5–10 seconds) |
| Debugging | Easy — print anywhere | Harder — distributed stack traces |

**Default to Pandas.** Use Spark when:
- Data does not fit in memory
- Computation is too slow on a single core at your data size
- You need to run the same code on a cluster in production

The cost of Spark — JVM overhead, lazy evaluation complexity, verbosity — is
only worth paying when the scale genuinely demands it.

---

## Quick Reference

### Start a session

```python
from pyspark.sql import SparkSession, functions as F, Window

spark = (
    SparkSession.builder
    .appName("MyApp")
    .master("local[*]")
    .config("spark.driver.memory", "4g")
    .getOrCreate()
)
```

### Core DataFrame operations

```python
df.withColumn("col", F.expr)     # add/replace column
df.filter(F.col("x") > 0)        # filter rows
df.drop("col")                    # remove column
df.count()                        # action: triggers execution
df.show(5)                        # action: print first 5 rows
```

### MLlib pipeline pattern

```python
from pyspark.ml import Pipeline
from pyspark.ml.feature import Imputer, VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator

pipeline = Pipeline(stages=[imputer, assembler, model])
fitted   = pipeline.fit(train)
preds    = fitted.transform(test)
score    = RegressionEvaluator(labelCol="y", metricName="r2").evaluate(preds)
```

### Run the script

```bash
python spark-ml.py
# or open spark-ml-pipeline.ipynb in Jupyter / VS Code
```

---

## Official Documentation

- PySpark DataFrame API: https://spark.apache.org/docs/latest/api/python/reference/pyspark.sql/dataframe.html
- MLlib pipelines: https://spark.apache.org/docs/latest/ml-pipeline.html
- MLlib regression: https://spark.apache.org/docs/latest/ml-classification-regression.html
- Spark configuration: https://spark.apache.org/docs/latest/configuration.html
