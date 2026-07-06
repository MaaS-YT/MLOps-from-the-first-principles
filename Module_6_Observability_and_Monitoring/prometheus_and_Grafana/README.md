# Lesson 6.2 — Prometheus & Grafana for Production Monitoring

| | |
|---|---|
| **Problem this solves** | You can't rely on logs alone in production. You need real-time metrics, dashboards, and alerting to catch issues before they impact users. Prometheus collects and stores time-series metrics; Grafana visualizes them. |
| **Mental model** | Every prediction request generates metrics: latency, success/failure, prediction value. Prometheus scrapes these metrics from your application every few seconds, storing them as time-series data. Grafana queries Prometheus to build dashboards showing trends, anomalies, and system health. |
| **What the demo shows** | FastAPI app with Prometheus metrics (request count, latency, system CPU/memory). Two load-testing scripts simulate traffic (100 sequential requests, 5000 concurrent requests). Complete Kubernetes deployment for Prometheus and Grafana with pre-configured dashboards. |
| **Where this fits** | Production monitoring layer. After models are deployed (Module 5), Prometheus/Grafana provide the observability needed to keep them running reliably. Integrates with Evidently for comprehensive monitoring. |

---

## Files

| File | Purpose |
|------|---------|
| `app.py` | FastAPI application with Prometheus metrics, request logging, and drift report endpoint |
| `test_1.py` | Sequential load test: 100 requests, batch processing, drift report generation |
| `test_2.py` | Concurrent load test: 5000 requests with 20 parallel workers |
| `generate_reference_data.py` | Generate reference (training) data for drift detection |
| `generate_dashboard.py` | Script to generate/update Grafana dashboard configuration |
| `prometheus-config.yaml` | Prometheus configuration with scrape targets and retention |
| `prometheus-deployment.yaml` | Kubernetes deployment for Prometheus |
| `grafana-deployment.yaml` | Kubernetes deployment for Grafana |
| `dashboard.json` | Grafana dashboard JSON (latency, requests, system metrics, drift) |
| `request_log.csv` | Persistent request log for drift detection |
| `reference.csv` | Reference data for drift detection |
| `app.log` | Application logs |
| `drift_report.html` | Generated drift report |

**Start with:** `app.py` and run the load tests

```bash
# Start the server
uvicorn app:app --reload --port 8000

# Generate reference data (first time only)
python generate_reference_data.py

# Run tests
python test_1.py    # 100 requests + drift report
python test_2.py    # 5000 concurrent requests

# Check metrics
curl http://localhost:8000/metrics
```

---

## What This Demo Shows

### 1. Metrics Export with Prometheus

The app exposes three types of metrics:

```python
from prometheus_client import Counter, Histogram, Gauge

REQUEST_COUNT = Counter("prediction_requests", "Total prediction requests")
LATENCY = Histogram("prediction_latency_seconds", "Model latency")
CPU_USAGE = Gauge("system_cpu_usage", "System CPU usage percentage")
MEMORY_USAGE = Gauge("system_memory_usage", "System memory usage percentage")
```

**Counter**: Only increases (total requests)
**Histogram**: Buckets latency values (track distribution)
**Gauge**: Current value (CPU/memory usage)

### 2. Batch Prediction for Efficiency

```python
@app.post("/predict/batch")
def predict_batch(data_list: List[InputData]):
    REQUEST_COUNT.inc(len(data_list))  # Count each prediction
    
    for data in data_list:
        with LATENCY.time():  # Measure each prediction
            pred = model.predict(data.dict())
    
    append_batch_request_log(data_dicts)  # Batch log append
```

Batch processing reduces:
- I/O overhead (single log write vs N writes)
- Network overhead (single HTTP request)
- Latency per prediction

### 3. Production-Grade Request Logging

```python
LOG_FILE = "request_log.csv"

def append_batch_request_log(data_list):
    """Append multiple requests to CSV file"""
    file_exists = os.path.exists(LOG_FILE)
    with open(LOG_FILE, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=["age", "income", "transactions"])
        if not file_exists:
            writer.writeheader()
        writer.writerows(data_list)  # Single write for batch
```

This allows drift detection on historical data without a database:
- `request_log.csv` stores all predictions
- Evidently loads and analyzes this data
- Simple, fast, and portable

### 4. System Metrics Monitoring

```python
@app.on_event("startup")
def start_monitoring():
    def monitor_system_metrics():
        while True:
            CPU_USAGE.set(cpu_percent(interval=1))
            MEMORY_USAGE.set(virtual_memory().percent)
            time.sleep(1)
    
    thread = threading.Thread(target=monitor_system_metrics, daemon=True)
    thread.start()
```

System metrics are scraped alongside application metrics in Grafana:
- CPU usage (%) — detect resource contention
- Memory usage (%) — detect memory leaks or scaling needs

### 5. Load Testing Patterns

**Sequential (test_1.py):** Simulates realistic user traffic
```python
# 100 sequential predictions with batch endpoint
data_batch = [...]  # 100 requests in one batch
response = session.post(f"{API_BASE_URL}/predict/batch", json=data_batch)
```

**Concurrent (test_2.py):** Simulates peak load
```python
# 5000 concurrent requests with 20 parallel workers
with ThreadPoolExecutor(max_workers=20) as executor:
    futures = [executor.submit(make_prediction_request) for _ in range(5000)]
```

Use test_1 for:
- Generating realistic traffic patterns
- Testing the drift report endpoint
- Validating monitoring setup

Use test_2 for:
- Stress testing the application
- Observing metric collection under load
- Measuring throughput limits

---

## Architecture Overview

```
                    ┌─────────────────┐
                    │   Prometheus    │
                    │  (Scrape every  │
                    │   15 seconds)   │
                    └────────┬────────┘
                             │
┌─────────────────────────┐  │  ┌─────────────────────────┐
│   FastAPI Application   │──┼──│   Grafana Dashboard     │
│  • /predict            │  │  │  • Latency graphs       │
│  • /predict/batch      │  │  │  • Request rates        │
│  • /drift-report       │  │  │  • System metrics       │
│  • /metrics (export)   │  │  │  • Drift alerts         │
└─────────────────────────┘  │  └─────────────────────────┘
                             │
                    ┌────────┴────────┐
                    │   Request Log   │
                    │  (CSV storage)  │
                    └─────────────────┘
```

---

## Kubernetes Deployment

### Prometheus Configuration

```yaml
# prometheus-config.yaml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'fastapi-app'
    static_configs:
      - targets: ['app:8000']
```

### Prometheus Deployment (K8s)

```yaml
# prometheus-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: prometheus
spec:
  replicas: 1
  selector:
    matchLabels:
      app: prometheus
  template:
    metadata:
      labels:
        app: prometheus
    spec:
      containers:
      - name: prometheus
        image: prom/prometheus:latest
        args:
          - "--config.file=/etc/prometheus/prometheus.yaml"
        ports:
        - containerPort: 9090
        volumeMounts:
        - name: config
          mountPath: /etc/prometheus
      volumes:
      - name: config
        configMap:
          name: prometheus-config
```

### Grafana Deployment (K8s)

```yaml
# grafana-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: grafana
spec:
  replicas: 1
  selector:
    matchLabels:
      app: grafana
  template:
    metadata:
      labels:
        app: grafana
    spec:
      containers:
      - name: grafana
        image: grafana/grafana:latest
        ports:
        - containerPort: 3000
        env:
        - name: GF_SECURITY_ADMIN_PASSWORD
          value: "admin"
```

---

## Key Concepts

### 1. Metrics Types

| Type | Use Case | Example |
|------|----------|---------|
| Counter | Ever-increasing values | Request count, errors |
| Histogram | Distribution of values | Latency, response sizes |
| Gauge | Current value | CPU usage, memory usage |
| Summary | Like Histogram but client-side | Percentiles, quantiles |

### 2. Dashboard Design Principles

**RED Method** (for services):
- **Rate** — requests per second
- **Errors** — error rate
- **Duration** — latency distribution

**USE Method** (for systems):
- **Utilization** — CPU, memory usage
- **Saturation** — queue length, throttling
- **Errors** — error counters

### 3. Alert Rules

```yaml
# Prometheus alert rules
groups:
- name: app_alerts
  rules:
  - alert: HighErrorRate
    expr: rate(prediction_errors[5m]) > 0.01
    annotations:
      summary: "Error rate above 1%"
  
  - alert: HighLatency
    expr: histogram_quantile(0.95, rate(prediction_latency_seconds_bucket[5m])) > 1
    annotations:
      summary: "95th percentile latency > 1s"
  
  - alert: DataDriftDetected
    expr: data_drift_share > 0.3
    annotations:
      summary: "Data drift detected in production"
```

---

## Quick Reference

### Server Commands

```bash
# Start server
uvicorn app:app --reload --port 8000

# Check metrics
curl http://localhost:8000/metrics

# Generate drift report
curl http://localhost:8000/drift-report

# View API docs
open http://localhost:8000/docs
```

### Test Commands

```bash
# Sequential test (100 requests)
python test_1.py

# Concurrent test (5000 requests, 20 workers)
python test_2.py

# Generate reference data
python generate_reference_data.py

# Generate dashboard
python generate_dashboard.py
```

### Kubernetes Commands

```bash
# Deploy Prometheus
kubectl apply -f prometheus-deployment.yaml
kubectl apply -f prometheus-config.yaml

# Deploy Grafana
kubectl apply -f grafana-deployment.yaml

# Access services
kubectl port-forward svc/prometheus 9090:9090
kubectl port-forward svc/grafana 3000:3000

# Check logs
kubectl logs -f deployment/prometheus
kubectl logs -f deployment/grafana
```

### Docker Commands

```bash
# Build image
docker build -t prediction-app .

# Run container
docker run -p 8000:8000 prediction-app

# Run with Prometheus sidecar
docker-compose up -d
```

---

## Official Documentation

- Prometheus: https://prometheus.io/docs/
- Grafana: https://grafana.com/docs/
- FastAPI: https://fastapi.tiangolo.com/
- prometheus_client: https://github.com/prometheus/client_python
- starlette_exporter: https://github.com/stephenhillier/starlette_exporter
- Kubernetes: https://kubernetes.io/docs/
```
