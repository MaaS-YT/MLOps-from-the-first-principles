# Foundations of Monitoring and Observability in MLOps

> *"Deployment marks not the end of the journey, but the beginning of a new and complex phase — the 'day two' problem."*

A model that achieves 99% accuracy in validation can silently degrade to 75% in production. Unlike traditional software failures that are loud and obvious (crashes, 404 errors), ML failures are frequently silent — the API returns predictions with 200 OK status codes, yet the predictions become progressively less accurate and less useful.

This guide covers the **core fundamentals** you need to understand before implementing monitoring tools. It provides the mental model, taxonomy of failures, and statistical techniques that underpin every monitoring system.

---

## The Problem: Silent Degradation

A traditional software bug is often loud and obvious:
- A server crashes
- A webpage returns a 404 error
- An application throws an exception

**ML failures are different.** The system continues to operate from a technical standpoint, even as its outputs become nonsensical. This silent degradation can go unnoticed for weeks or months, slowly eroding business value and user trust.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    SILENT DEGRADATION CURVE                             │
│                                                                         │
│  1.00 ─┐                                                               │
│        │  ─── Training Accuracy                                        │
│  0.95 ─┤    ╲                                                          │
│        │     ╲                                                         │
│  0.90 ─┤      ╲─── Production Accuracy                                 │
│        │       ╲                                                       │
│  0.85 ─┤        ╲                                                     │
│        │         ╲                                                    │
│  0.80 ─┤          ╲                                                   │
│        │           ╲                                                  │
│  0.75 ─┤            ╲                                                 │
│        │             ╲                                                │
│  0.70 ─┤              ╲                                               │
│        └─────────────────────────────────────────────────────────────  │
│        0    1    2    3    4    5    6    7    8    9   10            │
│                              Time (Months)                             │
│                                                                         │
│  The gap widens silently. The API still returns predictions.           │
│  But the predictions are increasingly wrong.                           │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Taxonomy of Failures

Failures in a production ML system fall into two broad categories:

### 1. Software System Failures

These are issues that can affect any complex software system:

| Failure Type | Examples |
|--------------|----------|
| **Dependency failures** | Upstream service breaks, third-party package changes API |
| **Deployment errors** | Wrong model binary deployed, missing permissions |
| **Hardware failures** | CPUs overheat, GPUs fail, network infrastructure goes down |
| **Distributed system bugs** | Workflow scheduler errors, data pipeline joins fail |

> 💡 **Key insight:** MLOps is, to a large extent, an engineering discipline. Strong software engineering and DevOps skills are essential.

### 2. ML-Specific Failures

These are the **subtle failures** unique to systems that learn from data. They don't cause crashes but result in degradation of predictive performance. They occur silently because the system continues to operate.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    ML-SPECIFIC FAILURE TYPES                            │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  1. DATA DRIFT                                                   │   │
│  │     Input distribution changes (P(X) changes)                    │   │
│  │     Example: User demographics shift, new customer segments      │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  2. CONCEPT DRIFT                                                │   │
│  │     Relationship between inputs and outputs changes (P(Y|X))     │   │
│  │     Example: Post-pandemic user behavior changed permanently     │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  3. TRAINING-SERVING SKEW                                        │   │
│  │     Training data processing differs from serving processing     │   │
│  │     Example: Feature normalization applied differently           │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  4. OUTLIERS                                                     │   │
│  │     Individual data points far outside training distribution     │   │
│  │     Example: $15M transaction when model trained on <$1M         │   │
│  └─────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Understanding Each Failure Type

### 1. Data Drift

**Definition:** Changes in the input data distribution (covariate shift).

**The Core Problem:** The assumption that training and production data come from the same distribution is almost always violated in the real world.

**Examples:**
- A model trained on last year's transaction data; this year, a new demographic uses the platform
- A marketing change attracts a younger audience → age feature distribution shifts
- Seasonal patterns: holiday shopping behavior vs. regular season

**What to Monitor:**
```
Feature Statistics to Track:
├── Mean
├── Standard Deviation
├── Minimum/Maximum
├── Percentiles (5th, 25th, 50th, 75th, 95th)
├── Distribution shape
├── Correlations between features
```

**Important Nuance:** Not all data drift is catastrophic. Sometimes models generalize fine to the new distribution (e.g., a slight shift in age might not matter if the model is robust). Significant drift, however, can break the model's assumptions.

---

### 2. Concept Drift

**Definition:** Changes in the relationship/mapping between inputs and outputs (the underlying concept the model is trying to predict).

**The Core Problem:** Even if the input distribution is the same, the output mapping has changed.

```
BEFORE DRIFT:
┌─────────────────────────────────────────────────────────────────┐
│  Feature 1 ──┐                                                 │
│              ├──► Target = 1                                   │
│  Feature 2 ──┘                                                 │
│                                                                 │
│  Same input ranges: Feature 1 [0,10], Feature 2 [2,10]        │
└─────────────────────────────────────────────────────────────────┘

AFTER DRIFT (Concept Drift):
┌─────────────────────────────────────────────────────────────────┐
│  Feature 1 ──┐                                                 │
│              ├──► Target = 0                                   │
│  Feature 2 ──┘                                                 │
│                                                                 │
│  Input ranges haven't changed!                                  │
│  But the relationship has.                                     │
└─────────────────────────────────────────────────────────────────┘
```

**Real-World Examples:**

| Domain | Example |
|--------|---------|
| **Customer Churn** | Before pandemic: users with <2 logins/week churned. After pandemic: even highly active users began canceling due to budget cuts. |
| **Fraud Detection** | Fraudsters adapt: they start mimicking genuine behavior after learning the model's patterns. |
| **Recommendation Systems** | User preferences evolve over time; what was popular last year may not be popular now. |
| **Credit Scoring** | Economic conditions change; the same financial behavior may indicate different risk levels. |

**Key Difference from Data Drift:**

```
Data Drift                    Concept Drift
─────────────────────────────────────────────────────────────────
P(X) changes                  P(Y|X) changes
"Input changed"               "The rules changed"
Sometimes model handles it     Almost always requires retraining
"Things look different"        "Things work differently"
```

---

### 3. Training-Serving Skew

**Definition:** The feature data used during model training differs from the feature data used during online inference.

**The Core Problem:** This is a **self-inflicted wound** caused by process or implementation errors — not external changes in the world.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    TRAINING-SERVING SKEW                                 │
│                                                                         │
│  TRAINING PIPELINE                      SERVING PIPELINE                │
│  ┌─────────────────────┐               ┌─────────────────────┐         │
│  │  Raw Data           │               │  Raw Data           │         │
│  │  ↓                  │               │  ↓                  │         │
│  │  Feature Engineer   │               │  Feature Engineer   │         │
│  │  ↓                  │               │  ↓                  │         │
│  │  Normalize: (x-μ)/σ │               │  Normalize: x/σ     │         │
│  │  ↓                  │               │  ↓                  │         │
│  │  Train Model        │               │  Model Prediction   │         │
│  └─────────────────────┘               └─────────────────────┘         │
│                                                                         │
│  ⚠️ BUG: Normalization is different → Inputs are skewed!              │
│  The model sees different data than it was trained on.                 │
└─────────────────────────────────────────────────────────────────────────┘
```

**Common Causes:**

| Cause | Example |
|-------|---------|
| **Separate codebases** | Training in Spark/Pandas, serving in C++ — subtle differences in feature implementation |
| **Data pipeline bugs** | A data source becomes unavailable → NaNs or default values in production |
| **Time window discrepancies** | Training uses 30-day window, serving uses 15-day window by mistake |
| **Missing transformation** | Log transformation applied in training but forgotten in serving |

**Mitigation Strategy:** Use a **Feature Store** (covered in Module 3) that provides a centralized repository for feature definitions and logic, ensuring the same transformations in both training and serving.

---

### 4. Outliers

**Definition:** Individual data points that are very different from the training data.

**The Core Problem:** The model was never trained on such data → predictions will be unreliable.

**Example:**
- Fraud detection model trained on transactions up to $1M
- One day, a $15M transaction arrives
- The model's prediction on it is unreliable

**Detection Approaches:**
- Simple range checks: "Does this feature value fall outside expected bounds?"
- Statistical methods: Z-score, Isolation Forest
- Domain-specific rules

**Handling Strategies:**
1. **Log** the outlier for analysis
2. **Route** the case for manual review
3. **Use a fallback model** designed for edge cases
4. **Flag** the prediction as unreliable

---

## Techniques to Detect Drift

### Data Drift Detection

#### 1. Kullback-Leibler (KL) Divergence

**What it measures:** How one distribution \(Q(x)\) (current data) diverges from a reference \(P(x)\) (baseline).

\[
D_{KL}(P \parallel Q) = \sum_{x} P(x) \log \frac{P(x)}{Q(x)}
\]

**Interpretation:**
- \(D_{KL} = 0\) → Identical distributions (no drift)
- Higher value → Greater divergence (possible drift)

**Where else it's used:**
- Loss function in t-SNE algorithm
- Model compression using Knowledge Distillation (covered in Module 4)

---

#### 2. Population Stability Index (PSI)

**What it measures:** How feature proportions shift across binned ranges.

\[
PSI = \sum_{i}(Q_{i} - P_{i})\ln \left(\frac{Q_{i}}{P_{i}}\right)
\]

Where:
- \(P_{i}\) = proportion of baseline data in bin \(i\)
- \(Q_{i}\) = proportion of current data in bin \(i\)

**Rule of Thumb:**
```
PSI < 0.1          → No significant drift
0.1 ≤ PSI < 0.25   → Moderate drift
PSI ≥ 0.25         → Significant drift (take action)
```

---

#### 3. Kolmogorov-Smirnov (KS) Test

**What it measures:** The maximum difference between two cumulative distribution functions.

\[
D = \sup_{x}|F_{1}(x) - F_{2}(x)|
\]

Where:
- \(F_{1}\) = training data CDF
- \(F_{2}\) = current data CDF
- \(\sup_{x}\) = maximum value over all \(x\)

```
┌─────────────────────────────────────────────────────────────────┐
│                    KS TEST VISUALIZED                            │
│                                                                 │
│  1.0 ─┐                                                         │
│        │    Training CDF ──┬───────────────                    │
│  0.8 ─┤      Current CDF ─┼───────────────                    │
│        │                   │                                    │
│  0.6 ─┤                   │  ← Maximum gap (D)                │
│        │                   │                                    │
│  0.4 ─┤                   │                                    │
│        │                   │                                    │
│  0.2 ─┤                   │                                    │
│        │                   │                                    │
│  0.0 ─┴─────────────────────────────────────────────────────    │
│        0    1    2    3    4    5    6    7    8    9   10    │
│                                                                 │
│  Null Hypothesis: Both datasets come from same distribution    │
│  p-value < 0.01 → Reject H₀ → Drift detected                   │
└─────────────────────────────────────────────────────────────────┘
```

**Interpretation:**
- **Large D** → Large gap between distributions
- **Small p-value** (e.g., p < 0.01) → Likely different distributions → Drift detected

---

### Concept Drift Detection

#### With Labels (Supervised)

If labels are available (even with delay), track performance metrics:

```
Metric     = Accuracy, F1, AUC, etc.
Baseline   = Metric on training or stable period
Current    = Metric on recent production data (with labels)

Alert if:  |Current - Baseline| > τ (tolerance threshold)
```

**Example (Credit Card Fraud):**
- Banks communicate ground truth feedback to card networks with a 30-45 day delay
- When feedback arrives, compare predictions against actual outcomes
- If accuracy drops significantly → concept drift has occurred

---

#### Without Labels (Unsupervised)

**Method 1: Drift Classifier**

```
┌─────────────────────────────────────────────────────────────────┐
│                    DRIFT CLASSIFIER APPROACH                     │
│                                                                 │
│  1. Combine old + new samples                                   │
│  2. Label old = 0, new = 1                                     │
│  3. Train a simple binary classifier                           │
│  4. If accuracy or AUC > 0.7 → distributions are different    │
│                                                                 │
│  Old Data (0)  ──┐                                             │
│                  ├──► Binary Classifier ──► Accuracy > 0.7?   │
│  New Data (1)  ──┘                                             │
│                                                                 │
│  High accuracy means the classifier can distinguish old vs new │
│  → Data has drifted                                            │
└─────────────────────────────────────────────────────────────────┘
```

**Method 2: Confidence Entropy**

For probabilistic models, track the entropy of predictions:

\[
H(p) = -\sum_{y} p(y | x) \log p(y | x)
\]

Where \(p(y | x)\) is the predicted probability distribution.

**Interpretation:**
- **Low entropy** → High confidence (model knows what it's doing)
- **High entropy** → Low confidence (model is uncertain, seeing unfamiliar data)

If average entropy rises, the model is seeing data it doesn't understand.

---

### ADWIN (Adaptive Windowing)

**What it is:** An online algorithm for detecting concept drift in streaming data.

**The Core Idea:** ADWIN keeps a sliding window of recent observations and constantly checks if the average behavior in the earlier part of the window differs from the later part.

```
┌─────────────────────────────────────────────────────────────────┐
│                    ADWIN ALGORITHM                               │
│                                                                 │
│  Step 1: Maintain a window W of recent data points             │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │  [x₁, x₂, x₃, x₄, x₅, x₆, x₇, x₈, x₉, x₁₀]               ││
│  └─────────────────────────────────────────────────────────────┘│
│                                                                 │
│  Step 2: Split W into two sub-windows                          │
│  ┌─────────────────────┬─────────────────────────────────────┐  │
│  │  W₁ (older samples) │     W₂ (newer samples)             │  │
│  │  [x₁, x₂, x₃, x₄]   │     [x₅, x₆, x₇, x₈, x₉, x₁₀]    │  │
│  └─────────────────────┴─────────────────────────────────────┘  │
│                                                                 │
│  Step 3: Compare means using Hoeffding's inequality            │
│                                                                 │
│  |μ₁ - μ₂| > √(1/2m × ln(4/δ))                                │
│                                                                 │
│  Step 4: If difference exceeds bound → drift detected          │
│  → Remove W₁ (older samples), keep W₂ (new concept)           │
└─────────────────────────────────────────────────────────────────┘
```

**Why ADWIN is Powerful:**
- **Fully online** → Works as data streams in (no batches)
- **Adaptive** → Automatically determines window size based on how quickly data changes
- **No manual tuning** → No need to pre-define window size

---

## Setting Up Alerts

Alerts should be configured for both functional and operational issues:

### Functional Alerts (ML-Specific)

| Alert Condition | Threshold | Action |
|-----------------|-----------|--------|
| Feature mean shifted | > 3 standard deviations | Investigate data pipeline |
| KS test p-value | < 0.01 | Check feature distribution |
| PSI | > 0.25 | Data drift detected |
| Model output distribution shift | > 20% | Check for concept drift |
| Average confidence entropy | > threshold | Model seeing unfamiliar data |
| Accuracy drop | > 5% | Immediate retraining needed |

### Operational Alerts (System Health)

| Alert Condition | Threshold | Action |
|-----------------|-----------|--------|
| Error rate | > 1% for 5 minutes | Check logs, rollback if needed |
| Latency (p95) | > 1 second | Scale horizontally, optimize |
| No predictions | > 10 minutes | Check upstream pipeline |
| CPU usage | > 90% | Scale out |
| Memory usage | > 90% | Investigate memory leak |

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    MULTI-LAYER ALERTING                                 │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  FEATURE LEVEL (Data Drift)                                     │   │
│  │  ───────────────────────────────────────────────────────────── │   │
│  │  Feature X mean shifted by > 3σ                                │   │
│  │  Alert when blue line crosses red threshold                    │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  DISTRIBUTION LEVEL (KS Test)                                   │   │
│  │  ───────────────────────────────────────────────────────────── │   │
│  │  KS test p-value drops below 0.01                              │   │
│  │  Alert when distribution has significantly changed             │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  OUTPUT LEVEL (Model Predictions)                               │   │
│  │  ───────────────────────────────────────────────────────────── │   │
│  │  Average prediction shifted beyond allowed range               │   │
│  │  Alert when model behavior changes unexpectedly                 │   │
│  └─────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Logging and Observability

### What is Observability?

**Observability** means having the tools and data to understand what's happening inside your system.

For an ML service, this includes:
- Traditional logging and monitoring (CPU, memory, request rates)
- ML-specific logging (input features, predictions, performance)

**The Goal:** Be able to answer questions when something goes wrong:
- "Did something change in the input data?"
- "What was the model prediction on that problematic case?"
- "Are we seeing more errors or slow responses than usual?"

---

### What to Log

#### 1. Predictions and Inputs

**Best Practice:** Log model inputs and outputs for at least a sample of requests (if not all).

**Privacy Consideration:** Data might be sensitive. Options:
- Log transformed features instead of raw data
- Log aggregate statistics
- Implement sampling (log 1% of requests in detail)

**Why It Matters:**
When ground truth feedback arrives (e.g., fraud label 30-45 days later), you can:
- Go back to the logged predictions
- See what features those transactions had
- Understand where the model went wrong
- Collect fresh labeled data for retraining

#### 2. System Metrics

| Metric | Why It Matters |
|--------|----------------|
| **Latency** | p95 response time — catch slowdowns early |
| **Throughput** | Requests per second — understand capacity |
| **Error Rate** | HTTP 500s, custom errors — detect bugs |

#### 3. Resource Metrics

| Metric | Why It Matters |
|--------|----------------|
| **CPU** | Maxed out? Need to scale out |
| **Memory** | Climbing? Possible memory leak |
| **GPU** | 5% utilization? Over-provisioned |

#### 4. Application-Specific Logs

| Information | Why It Matters |
|-------------|----------------|
| **Feature retrieval timing** | Pinpoint bottlenecks |
| **Preprocessing steps** | Detect skew |
| **Model version** | Track which model made each prediction |

---

### Logging for Feedback Loops

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    LOGGING FEEDBACK LOOP                                 │
│                                                                         │
│  ┌─────────────────┐                                                   │
│  │  Production     │                                                   │
│  │  Data           │                                                   │
│  └────────┬────────┘                                                   │
│           │                                                             │
│           ▼                                                             │
│  ┌─────────────────┐           ┌─────────────────────────────────────┐ │
│  │  Log Features   │──────────▶│  Store with Prediction & Timestamp │ │
│  │  & Predictions  │           └─────────────────────────────────────┘ │
│  └─────────────────┘                                                   │
│                                                                         │
│           ┌─────────────────────────────────────────────────────────────│
│           │                                                             │
│           ▼                                                             │
│  ┌─────────────────┐           ┌─────────────────────────────────────┐ │
│  │  Ground Truth   │──────────▶│  Join with Logged Predictions      │ │
│  │  (delayed)      │           │  → New Training Dataset             │ │
│  └─────────────────┘           └─────────────────────────────────────┘ │
│                                                                         │
│           ┌─────────────────────────────────────────────────────────────│
│           │                                                             │
│           ▼                                                             │
│  ┌─────────────────┐                                                   │
│  │  Retrain Model  │                                                   │
│  │  on New Data    │                                                   │
│  └─────────────────┘                                                   │
└─────────────────────────────────────────────────────────────────────────┘
```

**Key Insight:** Logging is also data collection. Think of it as building the dataset for your next model version.

---

## Quick Reference

### Drift Detection Techniques

| Technique | Use Case | What It Measures |
|-----------|----------|------------------|
| **KL Divergence** | Data drift | Distribution divergence \(D_{KL}(P \parallel Q)\) |
| **PSI** | Data drift (binned) | Shift in proportions across bins |
| **KS Test** | Data drift | Maximum gap between CDFs |
| **ADWIN** | Concept drift (streaming) | Mean difference across windows |
| **Drift Classifier** | Concept drift (no labels) | Accuracy of distinguishing old vs new |
| **Confidence Entropy** | Unsupervised drift | Model uncertainty level |

### PSI Thresholds

```
PSI < 0.1     → No significant drift
0.1 - 0.25    → Moderate drift
> 0.25        → Significant drift
```

### Alert Threshold Examples

```
Feature Mean Shift   → 3σ from training mean
KS Test p-value      → < 0.01
PSI                  → > 0.25
Error Rate           → > 1% for 5 minutes
Latency (p95)        → > 1 second
CPU Usage            → > 90%
Memory Usage         → > 90%
```

---

## Key Takeaways

1. **ML failures are silent.** Unlike crashes or 404 errors, ML models degrade quietly. The API still returns predictions — they're just wrong.

2. **Understand the taxonomy.** Data drift (inputs change), concept drift (rules change), training-serving skew (process error), and outliers (extreme points) are the four main failure types.

3. **Data drift isn't always bad.** Sometimes models generalize fine. Monitor and set thresholds based on business impact, not just p-values.

4. **Concept drift almost always requires action.** When the relationship between inputs and outputs changes, retraining on recent data is usually necessary.

5. **Training-serving skew is preventable.** Use a feature store. Keep training and serving transformations in sync.

6. **Log everything you can.** You need historical data to:
   - Debug failures
   - Detect drift
   - Collect training data for the next model version

7. **Monitor both functional and operational metrics.**
   - Functional: model quality, drift, data quality
   - Operational: latency, errors, resource usage

---

## Official Documentation & Resources

- **Evidently AI:** https://docs.evidentlyai.com/
- **Prometheus:** https://prometheus.io/docs/
- **Grafana:** https://grafana.com/docs/
- **Kolmogorov-Smirnov Test:** https://en.wikipedia.org/wiki/Kolmogorov%E2%80%93Smirnov_test
- **Population Stability Index:** https://en.wikipedia.org/wiki/Population_stability_index
- **ADWIN:** https://arxiv.org/abs/1404.6591
- **OpenTelemetry:** https://opentelemetry.io/docs/

---

## What's Next

- **Lesson 6.1:** Hands-on data drift detection with Evidently AI
- **Lesson 6.2:** Production monitoring with Prometheus and Grafana
- **Integration:** Combining drift detection and metrics for complete observability
```