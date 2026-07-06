# Import core PySpark components:
# - SparkSession: entry point to Spark
# - functions (F): built-in column operations
# - Window: for window-based computations (like ranking)
from pyspark.sql import SparkSession, functions as F, Window
from pyspark.sql.functions import col

# Import ML pipeline components from Spark MLlib
from pyspark.ml import Pipeline
from pyspark.ml.feature import Imputer, VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator


# -----------------------
# Spark session
# -----------------------
# Initialize a local Spark session:
# - appName: just a label for the job
# - master("local[*]"): run locally using all CPU cores
# - driver.memory: max memory Spark can use on the driver
spark = (
    SparkSession.builder
    .appName("SimpleSparkMLPipeline")
    .master("local[*]")
    .config("spark.driver.memory", "4g")
    .getOrCreate()
)

# Reduce log verbosity (avoid too many INFO logs)
spark.sparkContext.setLogLevel("WARN")


# -----------------------
# Data generation
# -----------------------
# Number of synthetic rows to generate
n_rows = 10_000

# Convert a fixed timestamp string into UNIX time (seconds)
start_ts = F.unix_timestamp(F.lit("2024-01-01 00:00:00"))

# Create synthetic dataset
df = (
    # spark.range generates a DataFrame with a single column "id" from 0 to n_rows-1
    spark.range(n_rows)

    # Create a timestamp column increasing by 60 seconds per row (1-minute intervals)
    .withColumn("ts", start_ts + (col("id") * 60))

    # Convert UNIX timestamp to proper timestamp type
    .withColumn("ds", F.from_unixtime(col("ts")).cast("timestamp"))

    # Generate random feature_a ~ Normal(0,1)
    .withColumn("feature_a", F.randn(seed=42))

    # Generate random feature_b ~ Uniform(0,10)
    .withColumn("feature_b", F.rand(seed=1337) * 10.0)

    # Create target variable y as a linear combination of features + noise
    # y = 2*feature_a + 0.3*feature_b + Gaussian noise
    .withColumn(
        "y",
        2.0 * col("feature_a") +
        0.3 * col("feature_b") +
        F.randn(seed=7) * 0.5
    )

    # Drop intermediate timestamp column
    .drop("ts")
)


# -----------------------
# Time-based split
# -----------------------
# Define a window ordered by timestamp (ds)
w = Window.orderBy("ds")

# Rank rows by time (earliest = 1, latest = n)
df_ranked = df.orderBy("ds").withColumn("rank", F.row_number().over(w))

# Count total rows (triggers Spark execution)
n = df_ranked.count()

# Define 80% cutoff index
cutoff = int(n * 0.8)

# Split into train (first 80%) and test (last 20%) based on time
train = df_ranked.filter(F.col("rank") <= cutoff).drop("rank")
test  = df_ranked.filter(F.col("rank") >  cutoff).drop("rank")


# -----------------------
# Pipeline
# -----------------------
# IMPORTANT: Feature engineering is done AFTER split to avoid data leakage

# Extract hour of day from timestamp as a numeric feature
train = train.withColumn("hour", F.hour("ds").cast("double"))
test  = test.withColumn("hour", F.hour("ds").cast("double"))

# Imputer fills missing values (if any) using mean strategy by default
# Outputs new columns with "_imp" suffix
imputer = Imputer(
    inputCols=["hour", "feature_a", "feature_b"],
    outputCols=["hour_imp", "feature_a_imp", "feature_b_imp"]
)

# Combine multiple feature columns into a single vector column "features"
# Required input format for Spark ML models
assembler = VectorAssembler(
    inputCols=["hour_imp", "feature_a_imp", "feature_b_imp"],
    outputCol="features"
)

# Define linear regression model
# - labelCol: target variable
# - featuresCol: input vector
lr = LinearRegression(
    labelCol="y",
    featuresCol="features"
)

# Create ML pipeline chaining preprocessing + model
pipeline = Pipeline(stages=[imputer, assembler, lr])

# Train the model on training data
model = pipeline.fit(train)


# -----------------------
# Evaluation
# -----------------------
# Generate predictions on test set
preds_test = model.transform(test)

# Define evaluation metric (R² = coefficient of determination)
evaluator = RegressionEvaluator(
    labelCol="y",
    predictionCol="prediction",
    metricName="r2"
)

# Compute R² score on test data
r2_test = evaluator.evaluate(preds_test)

# Print formatted result
print(f"R²(test): {r2_test:.4f}")


# Stop Spark session (clean shutdown)
spark.stop()