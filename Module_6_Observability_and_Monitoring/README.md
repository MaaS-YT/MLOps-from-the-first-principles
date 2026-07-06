# Module 6 — Observability and Monitoring

*A model in production is a living system — it needs vital signs.*

A model that achieves 99% accuracy in validation may silently degrade to 75% in production due to data drift, concept drift, or infrastructure issues. This module covers the observability stack needed to detect these issues before they impact users — from statistical drift detection to production-grade metrics with Prometheus and Grafana.

---

## 📚 Foundations First

Before diving into the tools, understand **why** monitoring matters and **how** drift detection works:

👉 **[Read the Foundations Guide →](FOUNDATIONS.md)**

This guide covers:
- Taxonomy of ML failures (data drift, concept drift, training-serving skew, outliers)
- Statistical drift detection techniques (KL Divergence, PSI, KS Test, ADWIN)
- Logging and observability best practices
- Alerting strategies and thresholds

---

## Lessons

| Lesson | Topic | Problem It Solves | Guide | Status |
|--------|-------|-------------------|-------|--------|
| 6.1 | [Evidently](Evidently/) | Data drift goes undetected without proactive statistical monitoring | [README](Evidently/README.md) · [evidently-demo.ipynb](Evidently/evidently-demo.ipynb) | ✓ |
| 6.2 | [Prometheus & Grafana](prometheus_and_Grafana/) | Production metrics need a scalable, battle-tested monitoring stack with latency, throughput, and system metrics | [README](prometheus_and_Grafana/README.md) · [app.py](prometheus_and_Grafana/app.py) · [dashboard.json](prometheus_and_Grafana/dashboard.json) | ✓ |

---

## What This Module Builds

```
OBSERVABILITY STACK

FOUNDATIONS (FOUNDATIONS.md)
  Taxonomy of Failures →  Data Drift, Concept Drift, Training-Serving Skew, Outliers
  Detection Techniques →  KL Divergence, PSI, KS Test, ADWIN
  Logging Principles  →  What to log, how to log, feedback loops

EVIDENTLY (Lesson 6.1)
  Data Drift       →  detect distribution shifts in input features
  Data Quality     →  summary statistics and missing value analysis
  Statistical Tests →  Kolmogorov-Smirnov, Chi-square, Wasserstein distance

PROMETHEUS + GRAFANA (Lesson 6.2)
  Metrics Export   →  expose latency, error rates, prediction distribution
  Time Series DB   →  Prometheus stores metrics with labels for querying
  Dashboards       →  Grafana visualizes trends, alerts, and system metrics
  Load Testing     →  simulate production traffic with batch and concurrent requests

INTEGRATION (Production Pattern)
  Alert Rules      →  trigger notifications on drift detection threshold
  Automated Retraining →  pipeline triggers when drift exceeds acceptable levels
```

---

## Integration: Evidently + Prometheus/Grafana

The two lessons work together to provide complete observability:

```
┌──────────────────────────────────────────────────────────┐
│                   PREDICTION API                        │
│  ┌────────────────────────────────────────────────┐     │
│  │  app.py (FastAPI)                             │     │
│  │  • /predict                                   │     │
│  │  • /predict/batch                             │     │
│  │  • /drift-report                              │     │
│  └───────────────┬───────────────────────────────┘     │
│                  │                                       │
│         ┌────────┴────────┐                             │
│         │ request_log.csv  │      ┌──────────────┐      │
│         │ (raw predictions)│      │ /metrics     │      │
│         └────────┬────────┘      └──────┬────────┘      │
│                  │                       │              │
│         ┌────────▼────────┐     ┌────────▼────────┐     │
│         │   Evidently     │     │   Prometheus    │     │
│         │ • Drift Report  │     │ • Time-series   │     │
│         │ • Data Quality  │     │ • Alerts        │     │
│         └─────────────────┘     └────────┬────────┘     │
│                                           │              │
│                                  ┌────────▼────────┐     │
│                                  │    Grafana      │     │
│                                  │ • Dashboards    │     │
│                                  │ • Visualization │     │
│                                  └─────────────────┘     │
└──────────────────────────────────────────────────────────┘
```

### Complete Monitoring Flow

1. **Client sends prediction request** → `/predict` or `/predict/batch`
2. **App logs the request** → `request_log.csv`
3. **Prometheus scrapes metrics** → Every 15s from `/metrics`
4. **Grafana visualizes** → Dashboards from Prometheus data
5. **Evidently analyzes** → On-demand drift report from `request_log.csv`
6. **Alert triggers** → When drift exceeds threshold or metrics degrade

### Alerting Strategy

| Condition | Threshold | Action |
|-----------|-----------|--------|
| Data drift share | > 30% | Send Slack alert, schedule retraining |
| Prediction latency | p95 > 1s | Scale up replicas, optimize model |
| Error rate | > 1% | Rollback deployment, investigate |
| CPU usage | > 80% | Scale horizontally |
| Memory usage | > 85% | Investigate memory leak, increase limits |

### Monitoring Dashboard Layout

A production-ready Grafana dashboard typically includes:

| Panel | Metric | Purpose |
|-------|--------|---------|
| Request Rate | `rate(prediction_requests[1m])` | Monitor traffic patterns |
| Latency Heatmap | `prediction_latency_seconds_bucket` | Identify slow predictions |
| Error Rate | `rate(prediction_errors[5m])` | Detect failures |
| CPU/Memory | `system_cpu_usage`, `system_memory_usage` | Resource utilization |
| Data Drift | `data_drift_share` | Distribution changes |
| Prediction Distribution | Histogram of predictions | Output drift detection |

---

## Where This Fits

This module sits at the production phase of the MLOps lifecycle. Module 5 (Cloud and Infrastructure) gets the model deployed; this module keeps it healthy once it's running. Together with Module 4 (Model Optimization), they form the production engineering layer that bridges model development and real-world deployment.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    THE MLOPS LIFECYCLE                                   │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  Module 1: The ML System (FastAPI, Docker)                     │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                              │                                          │
│                              ▼                                          │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  Module 2: Reproducibility (DVC, MLflow, W&B)                  │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                              │                                          │
│                              ▼                                          │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  Module 3: Data Engineering (Feast, Prefect, Spark)            │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                              │                                          │
│                              ▼                                          │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  Module 4: Model Optimization & Serving (Compression, gRPC)    │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                              │                                          │
│                              ▼                                          │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  Module 5: Cloud & Infrastructure (Kubernetes, AWS)            │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                              │                                          │
│                              ▼                                          │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  Module 6: Observability & Monitoring (Evidently, Prometheus)  │   │
│  └─────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────┘
```

Open `SYSTEM_MAP.md` at the repo root for the full system view.

---

## Quick Start

```bash
# Start the FastAPI server with Prometheus metrics
cd prometheus_and_Grafana
uvicorn app:app --reload --port 8000

# In another terminal, test the monitoring system
python test_1.py

# Generate drift report
curl http://localhost:8000/drift-report

# View metrics
curl http://localhost:8000/metrics
```

---

## What You'll Learn

By the end of this module, you'll understand:

- **Statistical drift detection** with Evidently AI
- **Production monitoring** with Prometheus metrics and Grafana dashboards
- **System metrics** (CPU, memory) alongside application metrics
- **Load testing patterns** for monitoring system capacity
- **Alerting strategies** for data drift and performance degradation

---

## Next Steps After This Module

Once you have monitoring in place, consider:

1. **Automated Retraining**: Trigger model retraining when drift exceeds threshold
2. **Alert Integration**: Connect to Slack, PagerDuty, or OpsGenie
3. **SLO/SLI Definition**: Set Service Level Objectives for model performance
4. **A/B Testing Monitoring**: Compare model versions with drift detection
5. **Cost Optimization**: Monitor inference costs and scaling decisions

---

## Module Structure

```
Module_6_Observability_and_Monitoring/
├── README.md                    # Module overview (this file)
├── FOUNDATIONS.md               # Core concepts: drift, detection, logging
├── Evidently/
│   ├── README.md                # Lesson 6.1 guide
│   ├── evidently-demo.ipynb     # Jupyter notebook with drift detection
│   ├── drift_report.html        # Sample drift report
│   └── data_summary.html        # Sample data quality report
└── prometheus_and_Grafana/
    ├── README.md                # Lesson 6.2 guide
    ├── app.py                   # FastAPI app with Prometheus metrics
    ├── test_1.py               # Sequential load test
    ├── test_2.py               # Concurrent load test
    ├── generate_reference_data.py
    ├── generate_dashboard.py
    ├── prometheus-config.yaml
    ├── prometheus-deployment.yaml
    ├── grafana-deployment.yaml
    ├── dashboard.json
    ├── request_log.csv
    ├── reference.csv
    ├── app.log
    └── drift_report.html
```

---

## Official Documentation

- **Evidently AI:** https://docs.evidentlyai.com/
- **Prometheus:** https://prometheus.io/docs/
- **Grafana:** https://grafana.com/docs/
- **FastAPI:** https://fastapi.tiangolo.com/
- **prometheus_client:** https://github.com/prometheus/client_python
- **starlette_exporter:** https://github.com/stephenhillier/starlette_exporter
- **Kubernetes:** https://kubernetes.io/docs/
```

