# Lesson 6.1 — Evidently AI for Data Drift Monitoring

| | |
|---|---|
| **Problem this solves** | Models degrade in production because the data distribution changes. Without proactive monitoring, you discover this when user complaints arrive. Evidently automatically detects statistical drift in your input features before it impacts model performance. |
| **Mental model** | Training data is a snapshot of the past. Production data is the present. When the distribution shifts — incomes change, user behavior evolves, seasons change — the model's assumptions break. Evidently compares reference (training) and current (production) data using statistical tests and flags when distributions diverge. |
| **What the demo shows** | Generate reference data (training distribution) and current data (production with drift). Run Evidently reports for data drift and data quality. Visualize drift detection with interactive HTML reports. |
| **Where this fits** | First layer of the observability stack. Monitor data before it enters the model to catch drift early. Use alongside performance monitoring (accuracy, precision) to distinguish data issues from model issues. |

---

## Files

| File | Purpose |
|------|---------|
| `evidently-demo.ipynb` | Full Jupyter notebook: data generation, drift detection, data quality reports, and interactive HTML export |
| `drift_report.html` | Sample drift report with statistical test results and visualizations |
| `data_summary.html` | Sample data quality report with distribution summaries and missing values |

**Start with:** `evidently-demo.ipynb`

```bash
# Run the notebook
jupyter notebook evidently-demo.ipynb

# Or run as a script
jupyter nbconvert --to python evidently-demo.ipynb --output evidently-demo.py
python evidently-demo.py
```

---

## What the Demo Shows

### Data Generation

Reference data represents the training distribution:
```python
ref = pd.DataFrame({
    "age": np.random.normal(30, 4, 1000),           # mean 30, std 4
    "income": np.random.normal(60000, 10000, 1000), # mean 60k, std 10k
    "transactions": np.random.normal(12, 3, 1000)   # mean 12, std 3
})
```

Current data simulates production drift:
```python
curr = pd.DataFrame({
    "age": np.random.normal(30, 4, 1000),           # no drift
    "income": np.random.normal(70000, 15000, 1000), # mean + variance drift
    "transactions": np.random.normal(15, 5, 1000)   # moderate drift
})
```

### Drift Detection

Evidently runs statistical tests for each feature:

| Feature | Reference Distribution | Current Distribution | Drift Detected |
|---------|----------------------|---------------------|----------------|
| `age` | Normal(30, 4) | Normal(30, 4) | ❌ No |
| `income` | Normal(60k, 10k) | Normal(70k, 15k) | ✅ Yes |
| `transactions` | Normal(12, 3) | Normal(15, 5) | ✅ Yes |

### Statistical Tests Used

Evidently uses multiple statistical tests depending on the feature type:

- **Kolmogorov-Smirnov** — continuous features (age, income)
- **Chi-square** — categorical features
- **Wasserstein distance** — distribution comparison

### Reports Generated

1. **Data Drift Report** (`drift_report.html`)
   - Drift detection for each feature with p-values
   - Dataset-level drift summary
   - Interactive visualizations (histograms, Q-Q plots)

2. **Data Quality Report** (`data_summary.html`)
   - Summary statistics (mean, std, min, max, quantiles)
   - Missing value analysis
   - Data distribution visualizations

---

## Key Concepts

### 1. Data Drift vs Concept Drift

```
Data Drift     →  Input distribution changes (P(X) changes)
Concept Drift  →  Relationship between input and output changes (P(Y|X) changes)

Example:
  Data Drift:  Users become younger (age distribution shifts)
  Concept Drift: User behavior changes, younger users now buy different products
```

### 2. Statistical Significance vs Business Impact

A statistically significant drift (p-value < 0.05) doesn't always mean business impact. Monitor:
- **Magnitude of drift** — how far has the distribution shifted?
- **Business metrics** — does the drift correlate with performance degradation?
- **Alert thresholds** — set thresholds based on business impact, not just p-values

### 3. When to Check for Drift

```
Frequency       Strategy
─────────────── ──────────────────────────────────
Per batch       For high-frequency predictions (recommended)
Per day         For daily batch predictions
Per week        For low-frequency models
Before retraining  Check if drift warrants retraining
After deployment  Monitor for immediate distribution shift
```

---

## Production Integration Patterns

### 1. API-Based Monitoring (as shown in Lesson 6.2)

```python
@app.get("/drift-report")
def drift_report():
    df_curr = load_request_log()  # stored predictions
    df_ref = pd.read_csv("reference.csv")
    
    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=df_ref, current_data=df_curr)
    
    return {
        "dataset_drift": drift_detected,
        "drifted_columns": drifted_columns,
        "report_path": "drift_report.html"
    }
```

### 2. Scheduled Monitoring

```python
# Cron job or Airflow task
def scheduled_drift_check():
    df_curr = load_week_of_predictions()
    df_ref = load_training_data()
    report = run_drift_report(df_ref, df_curr)
    if report.dataset_drift:
        send_alert("Data drift detected in production")
        trigger_retraining()
```

### 3. Streaming Monitoring

```python
# Real-time drift detection with sliding windows
def monitor_stream(prediction_batch):
    window = get_sliding_window(prediction_batch, size=1000)
    if detect_drift(window):
        alert_drift()
```

---

## Quick Reference

### Evidently Report API

```python
from evidently import Report
from evidently.presets import DataDriftPreset, DataSummaryPreset

# Data drift report
drift_report = Report(metrics=[DataDriftPreset()])
drift_report.run(reference_data=ref, current_data=curr)
drift_report.save_html("drift_report.html")

# Data quality report
quality_report = Report(metrics=[DataSummaryPreset()])
quality_report.run(reference_data=ref, current_data=curr)
quality_report.save_html("data_summary.html")

# Access results programmatically
results = drift_report.dict()
drift_detected = results["metrics"][0]["value"]["count"] > 0
```

### Statistical Tests

| Test | Use Case | Threshold |
|------|----------|-----------|
| Kolmogorov-Smirnov | Continuous distributions | p-value < 0.05 |
| Chi-square | Categorical distributions | p-value < 0.05 |
| Wasserstein | Distribution distance | Context-dependent |

### Common Drift Thresholds

```
Conservative:  p < 0.01 (fewer false positives)
Default:       p < 0.05 (standard statistical significance)
Lenient:       p < 0.10 (more sensitive, more false positives)
```

---

## Official Documentation

- Evidently AI Docs: https://docs.evidentlyai.com/
- Evidently GitHub: https://github.com/evidentlyai/evidently
- Data Drift Detection: https://docs.evidentlyai.com/presets/data-drift
- Statistical Tests: https://docs.evidentlyai.com/reference/metrics
```
