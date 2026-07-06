# System Map — The Production ML System

This document shows the complete system this course builds.
Keep it open across all lectures. Every lesson adds one layer.

---

## The Full System

```
╔══════════════════════════════════════════════════════════════════════════════╗
║                        THE PRODUCTION ML SYSTEM                              ║
╚══════════════════════════════════════════════════════════════════════════════╝

  ┌──────────────────────────────────────────────────────────────┐
  │                        DATA LAYER                            │
  │                                                              │
  │  Raw Sources                                                 │
  │  ┌────────────┐  ┌────────────┐  ┌────────────────────┐    │
  │  │  CSV files  │  │  JSON logs │  │  SQL / databases   │    │
  │  └─────┬──────┘  └─────┬──────┘  └─────────┬──────────┘    │
  │        └───────────────┼─────────────────────┘              │
  │                        │  [Lesson 3.1]                       │
  │                        ▼                                     │
  │              Data Pipeline                                   │
  │              Ingest → Validate → Engineer → Label            │
  │                        │                                     │
  │                        ▼  [Lesson 3.2]                       │
  │              ┌──────────────────────┐                        │
  │              │     Feature Store    │                        │
  │              │  ┌────────────────┐  │                        │
  │              │  │ Offline Store  │  │  Historical features   │
  │              │  │ (full history) │  │  for training          │
  │              │  └────────┬───────┘  │                        │
  │              │           │          │                        │
  │              │  materialize()       │                        │
  │              │           │          │                        │
  │              │  ┌────────▼───────┐  │                        │
  │              │  │ Online Store   │  │  Latest features       │
  │              │  │ (per entity)   │  │  for serving (<10ms)   │
  │              │  └────────────────┘  │                        │
  │              └──────────────────────┘                        │
  └──────────────────────────────────────────────────────────────┘
            │ get_historical_features()          │ get_online_features()
            ▼                                    ▼

  ┌──────────────────────────┐      ┌──────────────────────────────────┐
  │     TRAINING LAYER       │      │         SERVING LAYER            │
  │                          │      │                                  │
  │  Training Run            │      │  Prediction API                  │
  │  [Lessons 2.2–2.3]       │      │  [Lesson 1.1]                    │
  │                          │      │                                  │
  │  ┌────────────────────┐  │      │  ┌──────────────────────────┐   │
  │  │ Experiment Tracker │  │      │  │ FastAPI                  │   │
  │  │ (MLflow / W&B)     │  │      │  │                          │   │
  │  │                    │  │      │  │ POST /predict            │   │
  │  │ • params logged    │  │      │  │  → fetch features        │   │
  │  │ • metrics logged   │  │      │  │  → load model            │   │
  │  │ • model artifact   │  │      │  │  → return prediction     │   │
  │  └─────────┬──────────┘  │      │  └──────────────────────────┘   │
  │            │              │      │           │                      │
  │  Model Registry           │      │  Docker Container [Lesson 1.2] │
  │  (staging → production)   │      │  Same env everywhere           │
  └────────────┬──────────────┘      └──────────────────────────────────┘
               │
               │  [Lesson 2.1]
               ▼
  ┌──────────────────────────┐
  │     VERSIONING LAYER     │
  │                          │
  │  DVC + Git               │
  │                          │
  │  git commit → code       │
  │  dvc commit → data       │
  │                          │
  │  Any model artifact      │
  │  traceable to exact      │
  │  data + code + params    │
  └────────────┬─────────────┘
               │
               │  [Lesson 4.1]
               ▼
  ┌──────────────────────────────────────────────────┐
  │     OPTIMIZATION LAYER                            │
  │                                                  │
  │  Pruning → Quantization → Distillation → ONNX    │
  │                                                  │
  │  Model goes from "best accuracy in training"     │
  │  to "deployable in production":                  │
  │  smaller, faster, hardware-agnostic              │
  └────────────────────┬─────────────────────────────┘
                       │  [Lesson 4.2]
                       ▼
  ┌──────────────────────────────────────────────────┐
  │     SERVING TRANSPORT LAYER                       │
  │                                                  │
  │  TorchScript   Compile model → portable IR       │
  │                No Python needed at runtime       │
  │                                                  │
  │  LibTorch      Load IR in C++ via                │
  │                torch::jit::load()                │
  │                                                  │
  │  gRPC          Serve over binary RPC             │
  │                Protobuf contract, HTTP/2          │
  │                Lower latency than REST           │
  └──────────────────────────────────────────────────┘

  ── INFRASTRUCTURE ──────────────────────────────────────────────────────────

  [Lesson 1.2]  Docker       Everything containerized. No "works on my machine."

  [Lesson 3.3]  Prefect      Schedules and monitors the entire pipeline.
                             Retries failures. Alerts on anomalies.

  [Lesson 3.3]  Spark        Scales the data pipeline beyond one machine.
                             Partition → distribute → aggregate.

  ── DEPLOYMENT LAYER (Module 5) ────────────────────────────────────────────

  ┌──────────────────────────────────────────────────────────────────────┐
  │                                                                      │
  │  Lesson 5.1 — Kubernetes (Kind → EKS/GKE/AKS)                       │
  │                                                                      │
  │  ┌────────────────────────────────────────────────────────────────┐  │
  │  │                     Kubernetes Cluster                          │  │
  │  │  ┌──────────────────────────────────────────────────────────┐  │  │
  │  │  │                    Control Plane                         │  │  │
  │  │  │  API Server  │  etcd  │  Scheduler  │  Controller Mgr   │  │  │
  │  │  └──────────────────────────────────────────────────────────┘  │  │
  │  │                              │                                   │  │
  │  │  ┌──────────────────────────────────────────────────────────┐  │  │
  │  │  │                    Worker Nodes                          │  │  │
  │  │  │  ┌───────────────┐  ┌───────────────┐  ┌────────────┐  │  │  │
  │  │  │  │ Node 1        │  │ Node 2        │  │ Node 3     │  │  │  │
  │  │  │  │ ┌───────────┐ │  │ ┌───────────┐ │  │ ┌────────┐ │  │  │  │
  │  │  │  │ │ Pod:Model │ │  │ │ Pod:Model │ │  │ │ Pod:   │ │  │  │  │
  │  │  │  │ │ replica 1 │ │  │ │ replica 2 │ │  │ │ Model  │ │  │  │  │
  │  │  │  │ └───────────┘ │  │ └───────────┘ │  │ │replica3│ │  │  │  │
  │  │  │  │ ┌───────────┐ │  │ ┌───────────┐ │  │ └────────┘ │  │  │  │
  │  │  │  │ │ Service   │ │  │ │ kube-proxy│ │  │            │  │  │  │
  │  │  │  │ │ (LB)      │ │  │ └───────────┘ │  │            │  │  │  │
  │  │  │  │ └───────────┘ │  └───────────────┘  └────────────┘  │  │  │
  │  │  │  └───────────────┘                                      │  │  │
  │  │  └──────────────────────────────────────────────────────────┘  │  │
  │  └────────────────────────────────────────────────────────────────┘  │
  │                                                                      │
  │  Core Abstractions (Lesson 5.1)                                      │
  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌────────────┐ │
  │  │    Pod      │  │ Deployment  │  │  Service    │  │ Namespace  │ │
  │  │ 1+ container│  │  replica    │  │ stable IP   │  │ logical    │ │
  │  │ shared net  │  │  management │  │ + DNS + LB  │  │ isolation  │ │
  │  └─────────────┘  └─────────────┘  └─────────────┘  └────────────┘ │
  │                                                                      │
  │  ── CLOUD INFRASTRUCTURE (Lesson 5.2) ──────────────────────────────│
  │                                                                      │
  │  ┌────────────────────────────────────────────────────────────────┐  │
  │  │                      AWS Cloud                                 │  │
  │  │  ┌──────────────────────────────────────────────────────────┐  │  │
  │  │  │  IAM          │  VPC          │  EC2        │  S3        │  │  │
  │  │  │  (Auth)       │  (Network)    │  (Compute)  │  (Storage)  │  │  │
  │  │  └──────────────────────────────────────────────────────────┘  │  │
  │  │  ┌──────────────────────────────────────────────────────────┐  │  │
  │  │  │  EKS (Elastic Kubernetes Service)                         │  │  │
  │  │  │  • Managed control plane                                  │  │  │
  │  │  │  • Worker node groups (EC2)                               │  │  │
  │  │  │  • ECR for container images                               │  │  │
  │  │  │  • Load Balancer for service exposure                     │  │  │
  │  │  └──────────────────────────────────────────────────────────┘  │  │
  │  └────────────────────────────────────────────────────────────────┘  │
  └──────────────────────────────────────────────────────────────────────┘

  ── OBSERVABILITY LAYER (Module 6) ────────────────────────────────────────

  ┌─────────────────────────────────────────────────────────────────────────┐
  │                      OBSERVABILITY STACK                                 │
  │                                                                         │
  │  ┌──────────────────────────────────────────────────────────────────┐   │
  │  │  EVIDENTLY (Lesson 6.1)                                          │   │
  │  │  • Data drift detection (statistical tests)                      │   │
  │  │  • Data quality reports (summary statistics, missing values)     │   │
  │  │  • Interactive HTML reports with visualizations                  │   │
  │  │  • Trigger: on-demand or scheduled                               │   │
  │  └──────────────────────────────────────────────────────────────────┘   │
  │                                                                         │
  │  ┌──────────────────────────────────────────────────────────────────┐   │
  │  │  PROMETHEUS + GRAFANA (Lesson 6.2)                              │   │
  │  │  • Metrics export: latency, request count, error rates           │   │
  │  │  • System metrics: CPU usage, memory usage                      │   │
  │  │  • Time-series DB: Prometheus scrapes every 15s                 │   │
  │  │  • Dashboards: Grafana visualizes trends and anomalies          │   │
  │  │  • Alerts: configured on threshold breaches                     │   │
  │  └──────────────────────────────────────────────────────────────────┘   │
  │                                                                         │
  │  ┌──────────────────────────────────────────────────────────────────┐   │
  │  │  INTEGRATION PATTERN                                             │   │
  │  │  • request_log.csv stores all predictions                        │   │
  │  │  • Evidently analyzes historical logs for drift                  │   │
  │  │  • Prometheus scrapes /metrics endpoint                         │   │
  │  │  • Grafana shows combined view: metrics + drift                 │   │
  │  └──────────────────────────────────────────────────────────────────┘   │
  └─────────────────────────────────────────────────────────────────────────┘

  Drift Detection Flow:
  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
  │   /predict   │───▶│ request_log  │───▶│   Evidently  │───▶│   Alert if   │
  │   requests   │    │   .csv       │    │   Report     │    │   drift > 30%│
  └──────────────┘    └──────────────┘    └──────────────┘    └──────────────┘

  Metrics Flow:
  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
  │   /metrics   │───▶│  Prometheus  │───▶│   Grafana    │───▶│  Dashboard   │
  │   endpoint   │    │  (scrape)    │    │  (query)     │    │  (visualize) │
  └──────────────┘    └──────────────┘    └──────────────┘    └──────────────┘
```

---

## What Each Lesson Adds to the System

| Lesson | What Gets Added | Where in the Diagram | Status |
|--------|----------------|----------------------|--------|
| 1.1 | Serving API (FastAPI) + first container | Serving Layer | ✓ |
| 1.2 | Docker depth: images, networking, Compose | Infrastructure | ✓ |
| 2.1 | DVC versioning for data + models | Versioning Layer | ✓ |
| 2.2 | MLflow: structured run logs + model registry | Training Layer | ✓ |
| 2.3 | W&B: cloud tracking + artifact lineage | Training Layer | ✓ |
| 3.1 | Data pipeline: ingest + validate + features | Data Layer | ✓ |
| 3.2 | Feature store: offline + online, training-serving consistency | Data Layer | ✓ |
| 3.3 | Orchestration (Prefect) + distribution (Spark) | Infrastructure | ✓ |
| 4.1 | Compression: pruning + quantization + distillation + ONNX | Optimization Layer | ✓ |
| 4.2 | Serving transport: TorchScript + LibTorch + gRPC | Serving Transport Layer | ✓ |
| 5.1 | Kubernetes: pods, deployments, services, rolling updates, self-healing, auto-scaling | Deployment Layer | ✓ |
| 5.2 | Cloud infrastructure: AWS (EC2, S3, ECR, VPC, IAM, EKS) | Deployment Layer | ✓ |
| 6.1 | Data drift detection with Evidently (statistical tests, quality reports) | Observability Layer | ✓ |
| 6.2 | Production metrics: Prometheus + Grafana (latency, throughput, system metrics, dashboards) | Observability Layer | ✓ |

---

## The Five Core Problems

Every lesson in this course is a solution to one of five fundamental problems
in production ML:

### Problem 1 — Reproducibility
*"I can't recreate this result."*

A model that works in a notebook is not reproducible if you can't reconstruct
the exact data, code, parameters, and environment that produced it.

| Solution | Lesson |
|---------|--------|
| Version data alongside code | DVC — Lesson 2.1 |
| Log every experiment parameter and artifact | MLflow — Lesson 2.2, W&B — Lesson 2.3 |
| Containerize the environment | Docker — Lesson 1.2 |

### Problem 2 — Consistency
*"The model behaves differently in training than in production."*

Training-serving skew: the features, preprocessing, and data distributions
seen during training differ from what the model receives at inference time.
This is the most common cause of models that look good in evaluation
and fail silently in production.

| Solution | Lesson |
|---------|--------|
| One feature definition for training and serving | Feast — Lesson 3.2 |
| Enforce temporal cutoffs during feature engineering | Data Pipeline — Lesson 3.1 |
| Version models that training and serving both reference | DVC + Registry — Lessons 2.1–2.3 |

### Problem 3 — Scalability
*"It works on my laptop but fails on production data."*

Single-machine compute, unscheduled pipelines, and models too large to deploy
are all scalability failures at different layers of the system.

| Solution | Lesson |
|---------|--------|
| Distribute computation across many cores | Spark — Lesson 3.3 |
| Schedule and monitor pipelines reliably | Prefect — Lesson 3.3 |
| Compress models for deployment targets | Compression — Lesson 4.1 |
| Containerize for hardware-agnostic deployment | Docker — Lesson 1.2 |
| Scale the serving layer across many nodes with orchestration | Kubernetes — Lesson 5.1 |
| Deploy to cloud infrastructure with managed services | AWS — Lesson 5.2 |

### Problem 4 — Observability
*"The model was working. Now it isn't. We have no idea when it broke or why."*

A deployed model is a black box unless you instrument it. Data drifts.
User behavior changes. A feature pipeline silently starts producing nulls.
Without observability, you find out when a customer complains — not before.

| Solution | Lesson |
|---------|--------|
| Detect input distribution shift before accuracy drops | Evidently — Lesson 6.1 |
| Track system health: latency, errors, resource use | Prometheus + Grafana — Lesson 6.2 |
| Track model performance over time with live labels | Model monitoring — Lesson 5.2 (coming) |
| Alert on drift or performance degradation | Alerting rules — Lessons 6.2, 5.2 |

### Problem 5 — Automation
*"Every deployment is manual. One human error ships a broken model."*

A system that requires manual steps at every stage is fragile at scale.
CI/CD removes human error from the critical path by making the pipeline
the gatekeeper, not the engineer.

| Solution | Lesson |
|---------|--------|
| Automatically test code and validate data on every commit | CI pipeline — Coming |
| Block deployments where the new model underperforms the old | Model validation gate — Coming |
| Deploy automatically when all gates pass | CD pipeline / GitOps — Coming |
| Trigger retraining when drift is detected | Auto-retrain — Lessons 6.1 + Coming |

---

## What Kubernetes (Lesson 5.1) Adds Specifically

**Before Kubernetes (Lesson 4.2):**
- One container running on one machine
- Manual restart if it crashes
- Manual scaling (change Docker Compose replica count)
- Downtime during updates

**After Kubernetes (Lesson 5.1):**
- Multiple containers across multiple nodes in a cluster
- Self-healing: Kubernetes restarts failed containers automatically
- Auto-scaling: scale from 2 to 50 replicas based on CPU/requests
- Rolling updates: zero-downtime model deployments
- Service discovery: containers find each other via stable DNS names
- Load balancing: traffic distributed across all replicas

**The Demo in Lesson 5.1:**
- Train a linear regression model (y=2x)
- Wrap it in FastAPI + Docker
- Deploy to local Kind cluster
- Configure 2 replicas with rolling updates
- Test self-healing (delete a pod → it comes back)
- Test scaling (2 → 5 replicas)
- Test rolling update (v1 → v2 with zero downtime)

---

## What AWS (Lesson 5.2) Adds Specifically

**Before AWS (Lesson 5.1):**
- Local Kubernetes cluster (Kind/minikube)
- Limited to development machine resources
- No managed services
- Manual infrastructure management

**After AWS (Lesson 5.2):**
- Managed EKS control plane (no etcd/API server management)
- Worker node groups (auto-scaling EC2 instances)
- ECR for container image registry
- VPC for network isolation and security
- IAM for fine-grained access control
- S3 for model and data storage
- Load Balancer integration for service exposure

**The Demo in Lesson 5.2 (Coming):**
- Set up AWS CLI and IAM roles
- Create EKS cluster using `eksctl`
- Deploy the prediction app to EKS
- Configure Auto Scaling Groups
- Set up Load Balancer for public access
- Monitor with CloudWatch

---

## What Observability (Module 6) Adds Specifically

**Before Observability (Module 5):**
- Model deployed but no visibility into its behavior
- No detection of data drift (distribution changes)
- No system metrics (latency, throughput, resource usage)
- React to issues only after users complain

**After Observability (Module 6):**
- Evidently detects data drift before accuracy drops
- Prometheus collects real-time metrics from the application
- Grafana dashboards show trends and anomalies
- Alerts trigger on threshold breaches

**The Demos in Module 6:**
- **Evidently (Lesson 6.1):** Generate reference data, simulate drift, run reports
- **Prometheus + Grafana (Lesson 6.2):** FastAPI app with metrics, load testing, Kubernetes deployment

---

## Data Flow Through the System

The four arrows that connect the entire system:

```
1. Training data flow
   Raw sources → Data Pipeline → Feature Store (offline) → Training run
   Result: model artifact + experiment record

2. Serving data flow
   Live request → Feature Store (online) → Model → Prediction
   Result: prediction returned to caller in <100ms

3. Observability flow
   Predictions → request_log.csv → Evidently (drift detection)
   Metrics → /metrics endpoint → Prometheus → Grafana dashboard
   Result: health status + alerts

4. Update flow
   New raw data → re-run pipeline → new features → retrain
   → new experiment → promote to production → serving picks it up
   → monitoring confirms health
```

These four flows define the "data flywheel" of a production ML system.
The system improves continuously because each flow feeds the next.

---

## How This Course Covers the Stack

```
                        ┌────────────────┐
                        │  Module 6      │ Observability & Monitoring
                        │  (Lessons 6.1–2)│ Evidently + Prometheus/Grafana
                        └──────┬─────────┘
                               │
                        ┌──────┴─────────┐
                        │  Module 5      │ Cloud and Infrastructure
                        │  (Lessons 5.1–2)│ Kubernetes + AWS (EKS)
                        └──────┬─────────┘
                               │
                        ┌──────┴─────────┐
                        │  Lessons 4.1–2 │ Model Optimization & Serving
                        │ (Compression,  │
                        │ TorchScript,   │
                        │ LibTorch, gRPC)│
                        └──────┬─────────┘
                               │
                        ┌──────┴─────────┐
                        │  Lessons 3.1–3 │ Data Engineering
                        │ (Pipelines,    │
                        │ Feature Store, │
                        │ Prefect, Spark)│
                        └──────┬─────────┘
                               │
                        ┌──────┴─────────┐
                        │  Lessons 2.1–3 │ Reproducibility
                        │ (DVC, MLflow,  │
                        │  W&B)          │
                        └──────┬─────────┘
                               │
                        ┌──────┴─────────┐
                        │  Lessons 1.1–2 │ The ML System (foundation)
                        │ (FastAPI,      │
                        │  Docker)       │
                        └────────────────┘
```

The bottom (serving + containers) is introduced first because it is tangible:
you can call an API and see it work. Each layer above answers the next
unanswered question: "how did that model get there?", "what happens when
the data changes?", "how do we know it's still working?", "how do we stop
a human mistake from shipping a broken model?"

---

## One-Line Definitions (Reference)

| Term | One Line |
|------|---------|
| **Container** | A packaged app + its dependencies that runs identically anywhere |
| **Image** | The blueprint a container is built from |
| **DVC** | Git for data — tracks what data was used for each code commit |
| **Experiment run** | One training execution with logged params, metrics, and artifacts |
| **Model registry** | A versioned catalog that promotes models from staging to production |
| **Artifact** | Any file produced by a pipeline step (dataset, model, plot) |
| **Artifact lineage** | The chain: raw data → processed data → model → prediction |
| **ETL** | Extract → Transform → Load (transform before storage) |
| **ELT** | Extract → Load → Transform (transform at query time) |
| **Feature** | A computed input column used by a model (e.g., `purchases_last_30d`) |
| **Feature store** | Infrastructure that stores and serves features consistently |
| **Offline store** | Full feature history — used for training |
| **Online store** | Latest feature values per entity — used for serving |
| **Training-serving skew** | Training and serving compute features differently; model degrades |
| **Point-in-time correctness** | Using only data that was available before the label timestamp |
| **Temporal cutoff** | The boundary between "what you know" (features) and "what you predict" (label) |
| **Partition** | A chunk of data processed by one worker in a distributed system |
| **Orchestration** | Scheduling, monitoring, and recovering pipelines in production |
| **Pruning** | Removing weights with near-zero contribution |
| **Quantization** | Reducing weight precision (FP32 → INT8) |
| **Distillation** | Training a small model to mimic a large one |
| **ONNX** | Hardware-agnostic model format for deployment |
| **TorchScript** | Compiled, statically-typed subset of Python — makes PyTorch models portable and Python-free |
| **LibTorch** | PyTorch's C++ runtime — loads a TorchScript `.pt` file and runs inference without Python |
| **gRPC** | Binary RPC protocol — lower latency and higher throughput than REST |
| **Kubernetes (K8s)** | Container orchestration system that schedules and manages containers across a cluster of machines |
| **Pod** | Smallest deployable unit in Kubernetes; 1+ containers sharing network namespace |
| **Deployment** | Kubernetes controller that manages replicas, rolling updates, and rollbacks |
| **Service** | Kubernetes abstraction that provides a stable IP, DNS name, and load balancing for pods |
| **ReplicaSet** | Ensures a specified number of pod replicas are running at all times |
| **Ingress** | Exposes HTTP/HTTPS routes from outside the cluster to services |
| **Control plane** | Kubernetes brain: API server, etcd, scheduler, controller manager |
| **Worker node** | Kubernetes muscle: kubelet, kube-proxy, container runtime |
| **kubectl** | Command-line interface for interacting with Kubernetes clusters |
| **Kind** | Kubernetes in Docker — runs a local cluster for development |
| **Minikube** | Local Kubernetes cluster for development and testing |
| **EKS** | Elastic Kubernetes Service — AWS managed Kubernetes |
| **Rolling deployment** | Replace old pods one at a time — zero downtime, but mixed versions coexist briefly |
| **Canary deployment** | Route a small % of traffic to the new version; watch metrics before full rollout |
| **Blue-green deployment** | Run two identical environments; flip all traffic at once; easy rollback |
| **Shadow deployment** | New model receives real traffic but its predictions are not returned to users |
| **Data drift** | The input distribution at serving time diverges from the training distribution |
| **Concept drift** | The relationship between inputs and the target label changes over time |
| **Model monitoring** | Tracking prediction quality (accuracy, distribution) in production over time |
| **Infrastructure observability** | Tracking latency, throughput, error rate, CPU/memory of deployed services |
| **Evidently** | Open-source library for data drift detection and model monitoring |
| **Prometheus** | Time-series database for metrics collection with a pull-based model |
| **Grafana** | Visualization and dashboarding tool that queries Prometheus data |
| **Counter** | Prometheus metric that only increases (request count, errors) |
| **Histogram** | Prometheus metric that buckets values (latency distribution) |
| **Gauge** | Prometheus metric that represents a current value (CPU usage) |
| **EC2** | Elastic Compute Cloud — AWS virtual machines |
| **S3** | Simple Storage Service — AWS object storage |
| **ECR** | Elastic Container Registry — AWS container image registry |
| **VPC** | Virtual Private Cloud — AWS network isolation |
| **IAM** | Identity and Access Management — AWS authentication and authorization |
| **EKS** | Elastic Kubernetes Service — AWS managed Kubernetes |
| **eksctl** | CLI tool for creating and managing EKS clusters |
| **CloudWatch** | AWS monitoring and observability service |
| **Load Balancer** | Distributes traffic across multiple targets (EC2, pods) |
| **Auto Scaling** | Automatically adjusts capacity based on demand |
| **Node group** | A set of EC2 instances in EKS that share configuration |

---

## Lesson 5.1-2 Files Reference

### Lesson 5.1 — Kubernetes

| File | Purpose |
|------|---------|
| `train.py` | Trains linear regression model (y=2x) → `model.pkl` |
| `app.py` | FastAPI server with `/predict` and `/health` endpoints |
| `test.py` | Client that sends test requests to the API |
| `Dockerfile` | Containerizes the FastAPI app + model |
| `deployment.yaml` | Kubernetes Deployment: 2 replicas, rolling update strategy |
| `service.yaml` | Kubernetes Service: ClusterIP, load balancing across pods |
| `kind-config.yaml` | Optional Kind cluster configuration |
| `K8s.md` | Deep-dive learning guide for Kubernetes concepts |
| `README.md` | Step-by-step commands, experiments, and troubleshooting |

### Lesson 5.2 — Cloud and AWS

| File | Purpose |
|------|---------|
| `1-Cloud_Fundemetals.html` | Cloud computing fundamentals (IaaS, PaaS, SaaS, etc.) |
| `2-aws-intro.html` | AWS introduction: core services and architecture |
| `3-EKS.html` | EKS deep dive: cluster setup, node groups, deployments |

### Module 6 — Observability and Monitoring

#### Lesson 6.1 — Evidently

| File | Purpose |
|------|---------|
| `README.md` | Lesson guide: concepts, patterns, and quick reference |
| `evidently-demo.ipynb` | Full Jupyter notebook: data generation, drift detection, data quality reports |
| `drift_report.html` | Sample drift report with statistical test results and visualizations |
| `data_summary.html` | Sample data quality report with distribution summaries |

#### Lesson 6.2 — Prometheus & Grafana

| File | Purpose |
|------|---------|
| `README.md` | Lesson guide: concepts, architecture, and quick reference |
| `app.py` | FastAPI application with Prometheus metrics, request logging, drift endpoint |
| `test_1.py` | Sequential load test: 100 requests, batch processing, drift report |
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