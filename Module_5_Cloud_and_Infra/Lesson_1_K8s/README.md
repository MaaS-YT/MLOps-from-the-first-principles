# Machine Learning Prediction Service

A simple machine learning prediction service using Python and FastAPI. This project demonstrates how to build and deploy a machine learning model as a REST API service within a containerized environment orchestrated by Kubernetes.

## Project Overview

This project contains:

- **Machine Learning Model**: A simple linear regression model (y = 2x) for predictions
- **API Service**: FastAPI-based REST API with prediction and health check endpoints
- **Docker Support**: Complete containerization setup
- **Kubernetes Deployment**: YAML configurations for container orchestration
- A simple test client for API validation

## Prerequisites

Before starting, ensure you have the following installed:

| Tool | Purpose | Verification command |
|------|---------|---------------------|
| **Docker** | Container runtime | `docker --version` |
| **kind** | Local Kubernetes cluster | `kind --version` |
| **kubectl** | Kubernetes command line | `kubectl version --client` |
| **Python 3.12+** | Running the code | `python --version` |

## Quick Start (TL;DR)

Run these commands in order. Each step is explained below.

```bash
# 1. Install dependencies
uv sync

# 2. Train the model
python train.py

# 3. Create Kubernetes cluster
kind create cluster --name kind

# 4. Build Docker image
docker build -t k-demo:latest .

# 5. Load image into Kind
kind load docker-image k-demo:latest --name kind

# 6. Deploy to Kubernetes
kubectl apply -f deployment.yaml
kubectl apply -f service.yaml

# 7. Wait for pods to be ready
kubectl wait --for=condition=ready pod -l app=k-demo --timeout=60s

# 8. Test the API
kubectl port-forward service/k-demo 8000:80 &
python test.py
```

**Expected output:** `{'prediction': 15.0}`

---

## Step-by-Step Setup and Usage

### 1. Install Dependencies

Using uv (recommended):

```bash
uv sync
```

Activate the virtual environment:

```bash
# Linux / macOS
source .venv/bin/activate

# Windows
.venv\Scripts\activate
```

Alternative using pip:

```bash
pip install -r requirements.txt
```

### 2. Train the Machine Learning Model

```bash
python train.py
```

**Expected output:**
```
Training model...
Model saved to model.pkl
Model coefficient: 2.00
Model intercept: 0.00
```

This trains a simple linear regression model on the pattern `y = 2x` and saves it to `model.pkl`.

### 3. Test the API Locally (Optional)

Test the FastAPI server on your local machine before containerizing.

Start the server:

```bash
uvicorn app:app --port 8000
```

**Expected output:**
```
INFO:     Uvicorn running on http://127.0.0.1:8000
INFO:     Application startup complete.
```

In another terminal, run the test script:

```bash
python test.py
```

**Expected output:**
```
Response: {'prediction': 15.0}
Time taken: 0.0428 seconds
```

You can also open `http://localhost:8000/docs` in your browser to see the interactive API documentation.

Press `Ctrl+C` to stop the server.

### 4. Build Docker Image

Make sure Docker is running on your machine:

```bash
docker ps
# Should show a table of running containers (may be empty)
```

Build the Docker image:

```bash
docker build -t k-demo:latest .
```

**Expected output (abbreviated):**
```
[+] Building 1.2s (10/10) FINISHED
 => => naming to docker.io/library/k-demo:latest
```

Verify the image was created:

```bash
docker images | grep k-demo
# Expected: k-demo    latest    [image-id]    [size]
```

### 5. Create Kubernetes Cluster (kind)

Create a local Kubernetes cluster using Kind:

```bash
kind create cluster --name kind
```

**Expected output:**
```
Creating cluster "kind" ...
 ✓ Ensuring node image (kindest/node:v1.34.0)
 ✓ Preparing nodes
 ✓ Starting control-plane
 ✓ Installing CNI
 ✓ Installing StorageClass
Set kubectl context to "kind-kind"
```

Verify the cluster is running:

```bash
kubectl cluster-info --context kind-kind
kubectl get nodes
# Expected: 1 node (or more) with STATUS "Ready"
```

### 6. Load Docker Image into Kind

Kind runs in separate Docker containers. Your local Docker image is not automatically visible to the Kind cluster. You must load it explicitly:

```bash
kind load docker-image k-demo:latest --name kind
```

**Expected output:**
```
Image: "k-demo:latest" with ID "sha256:..." loaded into kind
```

**Alternative:** Push the image to Docker Hub and use `imagePullPolicy: Always` in `deployment.yaml`.

### 7. Deploy to Kubernetes

Apply the Kubernetes configuration files:

```bash
kubectl apply -f deployment.yaml
kubectl apply -f service.yaml
```

**Expected output:**
```
deployment.apps/k-demo-dep created
service/k-demo created
```

Verify the deployment:

```bash
kubectl get deployments
kubectl get pods
kubectl get services
```

**Expected output for `kubectl get pods`:**
```
NAME                            READY   STATUS    RESTARTS   AGE
k-demo-dep-xxxxx-yyyyy          1/1     Running   0          10s
k-demo-dep-xxxxx-zzzzz          1/1     Running   0          10s
```

**Important:** Both pods should show `STATUS: Running`. If you see `Pending`, `ContainerCreating`, or `ErrImagePull`, wait a few seconds. If it persists, see the Troubleshooting section below.

Check pod logs (optional):

```bash
kubectl logs -l app=k-demo --prefix
```

### 8. Access the Service

#### Port Forwarding (for testing from your laptop)

Create a tunnel from your local machine to the Kubernetes service:

```bash
kubectl port-forward service/k-demo 8000:80 &
```

**Expected output:**
```
Forwarding from 127.0.0.1:8000 -> 80
Forwarding from [::1]:8000 -> 80
```

Now test the API:

```bash
python test.py
```

**Expected output:**
```
Response: {'prediction': 15.0}
Time taken: 0.xxxx seconds
```

Or use `curl` directly:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"x": 7.5}'
```

**Expected output:**
```json
{"prediction":15.0}
```

---

## Experiments (Learn How Kubernetes Works)

Once the service is running, try these experiments to see Kubernetes in action.

### Experiment 1: Self-Healing

Kubernetes automatically replaces failed pods.

```bash
# Get the current pods
kubectl get pods

# Delete one pod (copy the full name from above)
kubectl delete pod k-demo-dep-xxxxx-yyyyy

# Watch Kubernetes recreate it immediately
kubectl get pods -w
# Press Ctrl+C to stop watching
```

You'll see the old pod terminating and a new pod being created. Kubernetes maintains the `replicas: 2` state automatically.

### Experiment 2: Scaling

Scale your model service from 2 to 5 replicas:

```bash
kubectl scale deployment k-demo-dep --replicas=5

# Watch new pods appear
kubectl get pods -w
# Press Ctrl+C to stop watching

# Scale back down
kubectl scale deployment k-demo-dep --replicas=2
```

### Experiment 3: Rolling Update

Deploy a new version of your model without downtime:

```bash
# Build a new version (same code for demo)
docker build -t k-demo:v2 .
kind load docker-image k-demo:v2 --name kind

# Update the deployment
kubectl set image deployment/k-demo-dep k-demo=k-demo:v2

# Watch the rolling update
kubectl rollout status deployment/k-demo-dep

# Verify pods are running the new version
kubectl get pods
```

Kubernetes replaces pods one at a time, keeping the service available throughout.

---

## Troubleshooting

### Common Problems and Solutions

| Problem | What you see | Solution |
|---------|-------------|----------|
| **Kind cluster not created** | `unknown cluster "kind"` | Run `kind create cluster --name kind` first |
| **Pods stuck in Pending** | `STATUS: Pending` for >30 seconds | Run `kubectl describe pod <name>` to see why |
| **ErrImagePull / ImagePullBackOff** | Pod logs show `Failed to pull image` | Image not loaded: `kind load docker-image k-demo:latest --name kind` |
| **Port 8000 already in use** | `address already in use` | Kill existing port-forward: `pkill -f "port-forward"` or use different port: `kubectl port-forward service/k-demo 8080:80` |
| **Connection refused** | `curl: (7) Failed to connect` | Pods not ready yet. Run `kubectl get pods -w` and wait for `Running` |
| **No resources found** | `kubectl get pods` shows nothing | Wrong namespace? Try `kubectl get pods -n default` or check if deployment was created |
| **Model returns wrong prediction** | Response is not 2x | Model might not have loaded. Check logs: `kubectl logs -l app=k-demo` |

### Useful Debug Commands

```bash
# See detailed information about a failing pod
kubectl describe pod -l app=k-demo

# View logs from all pods
kubectl logs -l app=k-demo --prefix

# Check if service has endpoints (should show pod IPs)
kubectl get endpoints k-demo

# Watch pod status in real-time
kubectl get pods -w

# Check events in the cluster
kubectl get events --sort-by='.lastTimestamp'

# See resource usage (requires metrics server)
kubectl top pods
```

### Clean Up

To delete everything and free resources:

```bash
# Stop port forwarding
pkill -f "port-forward"

# Delete Kubernetes resources
kubectl delete -f deployment.yaml
kubectl delete -f service.yaml

# Delete the Kind cluster
kind delete cluster --name kind

# Remove Docker image (optional)
docker rmi k-demo:latest k-demo:v2
```

---

## Project File Structure

```
Kubernetes-example/
├── train.py              # Trains model and saves model.pkl
├── app.py                # FastAPI server with /predict and /health
├── test.py               # Simple client to test the API
├── model.pkl             # Trained model file
├── requirements.txt      # Python dependencies
├── Dockerfile            # Container definition
├── deployment.yaml       # Kubernetes Deployment (2 replicas)
├── service.yaml          # Kubernetes Service (ClusterIP)
└── README.md             # This file
```

---

## Next Steps

After this project, explore:

- **Ingress**: Expose your service with HTTP routing (`https://api.example.com/predict`)
- **ConfigMaps**: Pass configuration without rebuilding the image
- **Secrets**: Store API keys and credentials securely
- **Horizontal Pod Autoscaler (HPA)**: Auto-scale based on CPU or request rate
- **Helm**: Package and manage Kubernetes applications
- **Production clusters**: EKS (AWS), GKE (Google), AKS (Azure)

---

## Official Documentation

| Tool | Documentation |
|------|---------------|
| Kind | https://kind.sigs.k8s.io/ |
| kubectl | https://kubernetes.io/docs/reference/kubectl/ |
| Docker | https://docs.docker.com/ |
| FastAPI | https://fastapi.tiangolo.com/ |