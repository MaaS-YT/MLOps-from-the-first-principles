# Container Orchestration — Kubernetes & Model Deployment

**Demo Project:** `Kubernetes-example/` — FastAPI model serving with 2 replicas, Deployment + Service, Kind local cluster.

---

## Part 1 — Why Kubernetes Exists

### The container wall

Docker solved a critical problem: "it works on my machine" stopped being an excuse. You can package your FastAPI model server, all its Python dependencies, and the `model.pkl` file into a single image that runs identically anywhere Docker runs.

```
One container → fine.
Two containers → still fine.
Fifty containers across ten machines → chaos.
```

The problem appears when you move from **a container** to **many containers in production**. Your trained model is now a microservice. You have 2, 10, or 50 replicas running across multiple nodes. Questions appear immediately:

- Which node runs which container?
- What happens when a container crashes?
- How do you scale from 2 replicas to 10 under load?
- How do you update the model without dropping requests?
- How do containers find each other when IPs change constantly?

The `Kubernetes-example` project makes this concrete. With a single `deployment.yaml` you declare "2 replicas of my model API." Kubernetes handles:

| Capability | What it means for your model |
|------------|------------------------------|
| **Scheduling** | Picks which node runs each replica |
| **Self-healing** | If a pod dies, immediately recreates it |
| **Scaling** | `kubectl scale --replicas=10` changes live system |
| **Rolling updates** | `kubectl set image` updates without downtime |
| **Load balancing** | Distributes requests across all replicas |

This is the **container wall** — the point where manually managing containers stops being viable.

---

### What Kubernetes actually does

Kubernetes is a **container orchestration engine**. Instead of running `docker run` on individual machines, you declare a **desired state** (YAML files), and Kubernetes continuously works to make the actual state match your declaration.

```
Desired state (YAML)                Actual state (cluster)
│                                   │
├── 2 replicas                      ├── 2 pods running
├── image: k-demo:latest            ├── containers healthy
├── port: 80                        ├── service stable IP
│                                   │
└── kubectl apply ──────────────────┴── Reconciliation loop
```

The **reconciliation loop** is the core idea:

1. You declare: "I want 2 replicas of my model API"
2. Kubernetes checks: "I see 0 replicas running"
3. Kubernetes acts: creates 2 pods, schedules them on nodes
4. Loop repeats: constantly watches, fixes any drift

```
┌─────────────────────────────────────────────────────────────┐
│                    RECONCILIATION LOOP                      │
│                                                             │
│   Desired State ──┬──► Compare ──┬──► Gap? ──┬──► Act      │
│   (2 replicas)    │              │           │             │
│                   │              │           │             │
│   Actual State ───┘     (1 pod died)         └──► Create   │
│   (1 replica)                                      new pod  │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

In your `Kubernetes-example` project, you can see this in action:

```bash
# Kill a pod
kubectl delete pod <pod-name>

# Watch Kubernetes recreate it instantly
kubectl get pods -w
```

---

### Declarative vs Imperative

This is the most important shift in thinking.

| Approach | What you do | Kubernetes role |
|----------|-------------|-----------------|
| **Imperative** (Docker) | `docker run -p 80:80 k-demo` | Execute command, then forget |
| **Declarative** (K8s) | `kubectl apply -f deployment.yaml` | Continuously enforce state |

- **Imperative:** "Do this specific action now."
- **Declarative:** "Here's what I want the world to look like. Make it so and keep it so."

Your `deployment.yaml` is a **declaration**:

```yaml
spec:
  replicas: 2           # "I want 2 replicas"
  template:             # "Every replica should look like this"
    containers:
    - name: k-demo
      image: k-demo     # "Run this container"
```

You don't tell Kubernetes *how* to maintain 2 replicas. You just declare the desired state. The controllers figure out the rest.

---

### Where Kubernetes fits in the MLOps progression

```
1. Local script / notebook          model.predict() in process
2. Flask/FastAPI local              model as API, single process
3. Docker container                 model as portable unit
4. Docker Compose                   multi-container on one host      ← many stop here
5. Kubernetes local (Kind)          full orchestration API, local    ← you are here
6. Kubernetes cloud (EKS/GKE/AKS)   same YAML, production cluster
7. Kubernetes + Service Mesh        traffic splitting, canary, observability
```

**Key insight:** YAML written for Kind works unchanged on EKS, GKE, or AKS. Learning locally is learning production.

---

## Part 2 — The kubectl CLI

`kubectl` is the command-line tool for Kubernetes. It sends HTTP requests to the API server (the control plane's front door). Every command is an API call.

```bash
kubectl apply -f deployment.yaml    # create/update resources
kubectl get pods                    # list pods
kubectl logs -l app=k-demo          # logs from all pods with label
kubectl describe pod <name>         # detailed event history
kubectl delete pod <name>           # manual termination (controller recreates)
```

### Common commands for your project

| Command | What it does | When to use |
|---------|--------------|-------------|
| `kubectl get pods -o wide` | Show pods + which node they run on | Check deployment status |
| `kubectl get deployment k-demo-dep` | Show desired vs current replicas | Verify scaling worked |
| `kubectl get service k-demo` | Show service IP and port | Check internal endpoint |
| `kubectl logs -l app=k-demo --prefix` | Logs with pod names | Debug prediction errors |
| `kubectl port-forward service/k-demo 8000:80` | Local tunnel to service | Test without Ingress |
| `kubectl scale deployment k-demo-dep --replicas=5` | Change replica count | Handle load increase |
| `kubectl rollout status deployment/k-demo-dep` | Watch update progress | Verify rolling update |

**Pro tip:** Always use `-l` (label selector) to filter resources. Your pods have `app=k-demo`, so `-l app=k-demo` selects all related resources.

---

## Part 3 — Core Abstractions Mapped to Your Project

Kubernetes has many abstractions. Most production work uses these five regularly. Here's what each means in your `Kubernetes-example` project.

### Pod — smallest deployable unit

A pod is one or more containers that share the same network namespace (same IP). In your project, each pod runs one container with FastAPI + model.

```yaml
# From deployment.yaml template section
template:
  spec:
    containers:
    - name: k-demo
      image: k-demo
      ports:
      - containerPort: 80
```

**In your project:** `kubectl get pods` shows two pods. Each runs your model API.

### Deployment — manages replicas and rolling updates

A Deployment manages identical pods. It ensures the right number run, handles updates (replacing old pods with new ones gradually), and rolls back on failure.

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: k-demo-dep
spec:
  replicas: 2          # ← The most important line
  selector:
    matchLabels:
      app: k-demo      # ← "These pods belong to me"
```

**In your project:** The Deployment creates exactly 2 pods. Change `replicas: 5`, re-apply, and watch new pods appear.

### Service — stable network endpoint

Pods come and go. Every time a pod is recreated, it gets a new IP. A Service gives you a **stable IP and DNS name** that never changes, plus load balancing across all matching pods.

```yaml
apiVersion: v1
kind: Service
metadata:
  name: k-demo
spec:
  selector:
    app: k-demo        # ← Which pods to balance across
  ports:
  - port: 80           # ← Service port
    targetPort: 80     # ← Pod container port
```

**In your project:** Inside the cluster, any pod can reach your model API at `http://k-demo:80`. kube-proxy load balances across both replicas automatically.

### ConfigMap and Secret — configuration injection

Hardcoding database URLs, API keys, or model paths into the image forces a rebuild for every config change. ConfigMaps (plain text) and Secrets (base64-encoded) let you inject configuration at runtime.

```yaml
# Example ConfigMap (not in your current project)
apiVersion: v1
kind: ConfigMap
metadata:
  name: model-config
data:
  MODEL_VERSION: "2.1.0"
  LOG_LEVEL: "INFO"
```

**Try this:** Add a ConfigMap to pass model path or threshold values without rebuilding the image.

### Namespace — logical cluster partitioning

Namespaces isolate resources within the same cluster. Different teams or environments (dev, staging, prod) can share the same cluster without interfering.

```bash
kubectl get pods -n default          # your current namespace
kubectl create namespace production
kubectl apply -f deployment.yaml -n production
```

**In your project:** You're using the `default` namespace. For real deployment, you'd use separate namespaces for dev/staging/prod.

---

## Part 4 — Architecture: Control Plane + Worker Nodes

Your Kind cluster has the same architecture as a production Kubernetes cluster. Here's what runs inside.

### Control plane (brain — makes decisions)

| Component | What it does | In your project |
|-----------|--------------|-----------------|
| **kube-apiserver** | Front door, validates requests, stores state | `kubectl` commands go here |
| **etcd** | Distributed key-value store, source of truth | Stores your deployment YAML |
| **kube-scheduler** | Assigns pods to nodes | Decides which node gets each replica |
| **controller-manager** | Runs reconciliation loops | Keeps 2 replicas running |

In Kind, the control plane runs in a container named `kind-control-plane`:

```bash
docker exec kind-control-plane ps aux | grep -E "kube-|etcd"
```

### Worker nodes (muscle — run workloads)

| Component | What it does | In your project |
|-----------|--------------|-----------------|
| **kubelet** | Agent on each node, runs containers | Starts/stops your FastAPI container |
| **kube-proxy** | Manages network rules | Load balances to both replicas |
| **container-runtime** | Actually runs containers | Docker (or containerd) |

### Communication flow for your deployment

```
1. kubectl apply -f deployment.yaml
   │
   ▼
2. kube-apiserver (validates, stores in etcd)
   │
   ▼
3. controller-manager (sees 0 pods, wants 2)
   │
   ▼
4. creates 2 pod definitions (unscheduled)
   │
   ▼
5. kube-scheduler (assigns each pod to a node)
   │
   ▼
6. kubelet on each node (pulls image, starts container)
   │
   ▼
7. Pods running → Service gets endpoints
   │
   ▼
8. kubectl get pods shows 2/2 Running
```

---

## Part 5 — The Deployment YAML Explained

Your `deployment.yaml` line by line:

```yaml
apiVersion: apps/v1          # Stable API group for Deployments
kind: Deployment             # Resource type
metadata:
  name: k-demo-dep           # How kubectl refers to this Deployment
spec:
  replicas: 2                # Desired count — the most important field
  selector:                  # Deployment needs to find its pods
    matchLabels:             # Uses label matching
      app: k-demo            # Only manages pods with this label
  template:                  # Pod template — blueprint for each replica
    metadata:
      labels:
        app: k-demo          # Pod gets this label (matches selector above)
    spec:
      containers:
      - name: k-demo         # Container name (for logs, describe)
        image: k-demo        # Docker image to run
        imagePullPolicy: IfNotPresent  # Don't pull from registry — use local
        ports:
        - containerPort: 80  # Container listens here (FastAPI port)
```

### The selector + template relationship

```
Deployment
   │
   ├── selector.matchLabels: "app: k-demo"
   │        │
   │        │   "I manage any pod with app=k-demo"
   │        │
   │        ▼
   └── template.metadata.labels: "app: k-demo"
            │
            │   "Every pod I create gets this label"
            │
            ▼
        (Pods inherit the label, matching the selector)
```

This pattern — selector points to labels that template applies — appears everywhere in Kubernetes. It decouples the controller from the pods it manages.

### What happens when you apply this

```bash
kubectl apply -f deployment.yaml

# Sequence:
1. API server validates YAML
2. etcd stores the Deployment object
3. Deployment controller sees replicas: 2, current: 0
4. Controller creates 2 ReplicaSet objects
5. ReplicaSet controller creates 2 Pod objects (unscheduled)
6. Scheduler assigns nodes
7. kubelet starts containers
8. Status updates flow back up
```

---

## Part 6 — The Service YAML Explained

Your `service.yaml` line by line:

```yaml
apiVersion: v1               # Core API group
kind: Service                # Resource type
metadata:
  name: k-demo               # DNS name inside cluster (http://k-demo)
spec:
  selector:
    app: k-demo              # Which pods receive traffic
  ports:
  - name: http
    port: 80                 # Service listens on port 80
    targetPort: 80           # Forward to pod's port 80
  type: ClusterIP            # Internal-only (default)
```

### Service types

| Type | Accessibility | When to use |
|------|---------------|-------------|
| **ClusterIP** (default) | Inside cluster only | Internal services, microservice-to-microservice |
| **NodePort** | Node's IP: high port (30000-32767) | Simple external access, dev testing |
| **LoadBalancer** | Cloud provider's LB | Production external access |
| **Ingress** (separate resource) | HTTP/HTTPS routing | Domain-based routing, TLS termination |

Your project uses `ClusterIP` with `port-forward` for external access. In production, you'd use `LoadBalancer` (cloud) or `Ingress` (HTTP routing).

### How kube-proxy implements the Service

Every node runs `kube-proxy`. It watches for Services and Endpoint changes, then writes `iptables` rules:

```bash
# On any worker node (conceptual — rules look like this)
iptables -t nat -L KUBE-SERVICES

# Traffic to 10.96.0.1:80 (Service cluster IP)
# Randomly forward to either pod-ip-1:80 or pod-ip-2:80
```

This is why a Service gives you load balancing — kube-proxy distributes traffic across all pods matching the selector.

---

## Part 7 — The Core Operations

### Rolling update (zero downtime model deployment)

When you change the image (new model version), Kubernetes doesn't kill all pods at once.

```
Before: 2 pods running v1
   │
   ▼
Step 1: Start 1 v2 pod, keep 2 v1 pods running
   │
   ▼
Step 2: v2 pod ready → remove 1 v1 pod
   │
   ▼
Step 3: Start another v2 pod
   │
   ▼
Step 4: Remove last v1 pod
   │
   ▼
After: 2 pods running v2
```

**Your project demonstrates this:**

```bash
# Build new version
docker build -t k-demo:v2 .
kind load docker-image k-demo:v2 --name kind

# Update deployment (zero downtime)
kubectl set image deployment/k-demo-dep k-demo=k-demo:v2

# Watch the rolling update
kubectl rollout status deployment/k-demo-dep
kubectl get pods -w
```

**Why this matters for ML:** You can deploy a new model version without dropping a single inference request. Kubernetes waits for the new pod to be ready (health check passes) before removing an old pod.

### Self-healing (what happens when things break)

**Your project's self-healing in action:**

```bash
# Scenario: Pod crashes
kubectl delete pod <pod-name>

# Kubernetes immediately:
# 1. Detects pod missing (replica count mismatch)
# 2. Creates new pod definition
# 3. Scheduler assigns node
# 4. kubelet starts new container

# Result: Still 2 running pods within seconds
kubectl get pods
```

**Scenario: Node dies** (in a cloud cluster):
- Node controller marks node NotReady after timeout
- Pods on dead node are considered lost
- Replacement pods scheduled on healthy nodes

### Scaling (handling load)

```bash
# Manual scale (your project)
kubectl scale deployment k-demo-dep --replicas=5

# Automatic scale (Horizontal Pod Autoscaler)
kubectl autoscale deployment k-demo-dep --cpu-percent=70 --min=2 --max=10

# What HPA does:
# - Checks CPU every 15 seconds (default)
# - If CPU > 70%, adds replicas
# - If CPU < 70% for sustained period, removes replicas
```

For model serving, you'd scale based on:
- Requests per second
- GPU utilization
- Queue length
- Custom metrics from your inference server

---

## Part 8 — Local Development with Kind

Kind (Kubernetes in Docker) runs a full Kubernetes cluster inside Docker containers. It's how this demo works on your laptop.

### What Kind does

```
Your machine
   │
   ├── Docker daemon
   │     │
   │     ├── kind-control-plane (runs control plane components)
   │     ├── kind-worker      (runs your pods)
   │     └── kind-worker2     (runs your pods)
```

### Kind vs alternatives

| Tool | Use case | Production-like |
|------|----------|-----------------|
| **Kind** | CI, local dev, quick tests | Yes — real K8s inside containers |
| **Minikube** | Full VM, addons, more features | Yes — but heavier |
| **K3s** | Lightweight, edge, IoT | Yes — production capable |
| **Docker Desktop** | Mac/Windows built-in | Yes — but Mac/Windows only |

### Your complete Kind workflow

```bash
# 1. Create cluster
kind create cluster --name kind

# 2. Build image (Docker must be running)
docker build -t k-demo .

# 3. Load image into Kind (critical step!)
kind load docker-image k-demo --name kind

# 4. Deploy
kubectl apply -f deployment.yaml
kubectl apply -f service.yaml

# 5. Test
kubectl port-forward service/k-demo 8000:80
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d '{"x": 7.5}'

# 6. Clean up
kind delete cluster --name kind
```

**Why `kind load` is necessary:** Kind nodes are separate Docker containers — they can't see your local image cache without explicit loading.

---

## Part 9 — Labels and Selectors

Labels are the Kubernetes linking mechanism. Everything connects via labels.

### In your project

```yaml
# Deployment sets labels on pods
template:
  metadata:
    labels:
      app: k-demo        # ← Label key: value
      version: "v1"      # ← Can have multiple labels

# Deployment selector (finds its pods)
selector:
  matchLabels:
    app: k-demo          # ← Only cares about app label

# Service selector (finds pods to route to)
selector:
  app: k-demo            # ← Same label value
```

### Using labels in kubectl

```bash
# Get all pods with app=k-demo
kubectl get pods -l app=k-demo

# Get all resources with this label
kubectl get all -l app=k-demo

# Watch logs from all pods with the label
kubectl logs -l app=k-demo --tail=20 --prefix
```

Labels are your primary tool for grouping resources. Add a `version: v2` label during a canary deployment to target only new pods with specific traffic rules.

---

## Part 10 — When to Use Kubernetes

### Comparison matrix

| | Docker Compose | Kubernetes | Serverless |
|---|----------------|------------|------------|
| **Scale** | Single host | Multi-node cluster | Infinite (managed) |
| **Complexity** | Low | High | Low |
| **Startup time** | Instant | 10-30 sec | Cold start possible |
| **Stateful workloads** | Volumes | PVC, StatefulSet | Limited |
| **Cost** | Free | Control plane + nodes | Pay per request |
| **ML use case** | Local dev | Production serving | Sporadic inference |

### Decision guide

**Use Docker Compose when:**
- You have 1-3 containers on one host
- You don't need auto-scaling
- You're in local development
- Your ML model is a batch job, not a live API

**Use Kubernetes when:**
- You need multiple replicas across multiple machines
- You require zero-downtime rolling updates
- You need auto-scaling based on load
- Your model serves real-time traffic
- The same YAML runs in dev/staging/prod

**Use Serverless when:**
- Inference is intermittent (once per hour)
- Cold starts (1-5 seconds) are acceptable
- You want zero infrastructure management
- Your model fits in Lambda's 250MB limit

The `Kubernetes-example` project sits at the entry point of "Kubernetes when" — your first step from a single container to production orchestration.

---

## Quick Reference

### Essential kubectl commands for your project

```bash
# Cluster info
kubectl cluster-info
kubectl get nodes

# Deploy and manage
kubectl apply -f deployment.yaml
kubectl apply -f service.yaml
kubectl get deployments,services,pods

# Debugging
kubectl logs -l app=k-demo --prefix
kubectl describe pod <pod-name>
kubectl describe service k-demo

# Scaling and updates
kubectl scale deployment k-demo-dep --replicas=5
kubectl set image deployment/k-demo-dep k-demo=k-demo:v2
kubectl rollout status deployment/k-demo-dep

# Port forwarding (test external access)
kubectl port-forward service/k-demo 8000:80

# Clean up
kubectl delete -f deployment.yaml
kubectl delete -f service.yaml
```

### Your project files reference

| File | What it does |
|------|--------------|
| `train.py` | Trains y=2x linear regression model → `model.pkl` |
| `app.py` | FastAPI server with `/predict` and `/health` endpoints |
| `test.py` | Sends test request to `/predict`, prints response |
| `Dockerfile` | Packages app + model into container |
| `deployment.yaml` | Declares 2 replicas of your model API |
| `service.yaml` | Creates internal load balancer (ClusterIP) |

### YAML template for a new model service

```yaml
# deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: my-model-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: my-model
  template:
    metadata:
      labels:
        app: my-model
    spec:
      containers:
      - name: model-container
        image: my-model:latest
        ports:
        - containerPort: 8080
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "1Gi"
            cpu: "500m"
---
# service.yaml
apiVersion: v1
kind: Service
metadata:
  name: my-model-api
spec:
  selector:
    app: my-model
  ports:
  - port: 80
    targetPort: 8080
  type: ClusterIP
```

### Local Kind workflow (copy-paste ready)

```bash
# One-time setup
kind create cluster
docker build -t my-model .
kind load docker-image my-model

# Every deploy
kubectl apply -f deployment.yaml
kubectl apply -f service.yaml

# Test
kubectl port-forward service/my-model-api 8000:80
curl http://localhost:8000/predict
```

---

## Official Documentation

- kubectl cheatsheet: https://kubernetes.io/docs/reference/kubectl/quick-reference/
- Deployments: https://kubernetes.io/docs/concepts/workloads/controllers/deployment/
- Services: https://kubernetes.io/docs/concepts/services-networking/service/
- Kind (local cluster): https://kind.sigs.k8s.io/
- Kubectl install: https://kubernetes.io/docs/tasks/tools/