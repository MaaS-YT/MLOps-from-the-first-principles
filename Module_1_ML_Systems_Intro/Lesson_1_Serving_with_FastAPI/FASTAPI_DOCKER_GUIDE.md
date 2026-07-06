# Lesson 1.1 — Serving a Model: FastAPI + Docker

> **Lesson 1.1** — The mechanics of wrapping a model in a REST API, containerizing it, and calling it from anywhere. Read `article.md` first for the "why"; this guide covers the "how."

| | |
|---|---|
| **Problem this solves** | A trained model is a Python object in memory. Nothing outside your machine can use it. A REST API gives it a stable, language-agnostic interface. A Docker container makes that API run identically everywhere. |
| **Mental model** | `train.py` runs once (offline). `app.py` runs forever (online). `test.py` verifies the handoff worked. These are three different jobs with different cadences — separating them is the pattern that scales to production. |
| **What the lecture demonstrates** | Training an Iris classifier → saving the model → wrapping it in FastAPI → containerizing with Docker → calling the API end-to-end |
| **Where this fits** | This is the **Serving Layer** — the last mile. Everything else in the course (versioning, tracking, pipelines, compression) exists to produce a better model for this layer to serve. |

---

# Serving ML Models with FastAPI & Docker — Concepts & Guide

Demo: `train.py` → `app.py` → `Dockerfile` — Iris species prediction API.

---

## Part 1 — From Model to Service (The Mental Model)

### The gap nobody talks about

You finish training a model. It has great accuracy. You close the notebook.

Nothing has changed in the world. No user has benefited. No system has been improved.
The model exists only as a Python object in memory — or at best, a `.pkl` file on your disk.

To make a model useful, something in the world must be able to *call* it — a web app,
a mobile app, another service, a batch job. That something needs a stable, well-defined
interface. That interface is a **REST API**.

---

### What "serving" means

Model serving is the practice of exposing a trained model as a network service.
Instead of importing the model directly, any consumer sends it data over HTTP
and receives predictions in return.

```
Without serving:
  Only Python code that imports your .pkl can use the model
  Only your machine can run it
  Every new user needs to install your dependencies

With serving:
  Any language, any device, any machine can call your model
  One running service handles all requests
  Consumers need nothing except the ability to make HTTP calls
```

This is the same reason databases have connection protocols, and why APIs power
every app on your phone. Serving turns a model into a product.

---

### The stateless service principle

A model serving API should be **stateless**: each request contains everything
the server needs to produce a response. The server does not remember previous requests.

This matters because:
- Multiple server instances can handle requests interchangeably (scaling)
- A crashed server can be restarted without losing anything
- Requests from different users don't interfere with each other

The model itself is loaded once at startup and stays in memory — but it is read-only.
It never changes between requests. The server is stateless even though it holds state.

---

### The health endpoint

Every production service exposes a `/health` endpoint.
It returns a simple "I am alive" response and nothing else.

Load balancers poll it every few seconds. Kubernetes uses it to know when to restart
a crashed pod. Monitoring systems alert when it stops responding.

The pattern is universal — you'll find it in every well-built API, ML or otherwise.

---

### Where this fits in the MLOps progression

```
1. Notebook model      no interface, no reuse, no deployment
2. Saved artifact      .pkl file — reusable in Python, still not a service
3. REST API            any client, any language → predictions    ← you are here
4. Containerized API   same API, runs identically everywhere     ← and here
5. Load-balanced API   multiple replicas, auto-scaling
6. Monitored API       log predictions, detect drift, alert on anomalies
```

---

## Part 2 — REST APIs and FastAPI

### How REST APIs work

REST (Representational State Transfer) is a convention for building networked services.
The client sends an **HTTP request** (a verb + a path + an optional body).
The server processes it and returns an **HTTP response** (a status code + a body).

```
Client                           Server
  │                                │
  │   POST /predict                │
  │   {"sepal_length": 5.1, ...}   │
  │ ──────────────────────────────►│
  │                                │  validate input
  │                                │  run model
  │                                │  format response
  │   200 OK                       │
  │   {"species": "setosa", ...}   │
  │ ◄──────────────────────────────│
```

`GET` is used for reading (no body, no side effects — like `/health`).
`POST` is used for sending data and getting a response (like `/predict`).

---

### What FastAPI adds

Building raw HTTP parsing from scratch is tedious and error-prone.
FastAPI handles the plumbing and gives you three things that matter for ML:

**Input validation** — You declare the shape of your input as a Python class.
FastAPI rejects malformed requests before your code ever runs. No defensive code needed.

**Automatic documentation** — Your API definition becomes interactive docs at `/docs`.
Anyone can explore and test the API in a browser without writing client code.

**Performance** — FastAPI is async-first and one of the fastest Python web frameworks.
For IO-bound workloads (waiting for DB, waiting for network), it handles concurrent
requests efficiently.

---

### The input schema

```python
class IrisFeatures(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float
```

This class is the **contract** between the client and the server. It says:
"Send me exactly these four fields, all floats. Anything else is rejected."

Pydantic (the library behind `BaseModel`) validates types, coerces compatible types
(a JSON integer becomes a float), and raises clear error messages for invalid inputs.
The demo uses it for a simple case — in production, it handles nested objects,
optional fields, ranges, custom validators.

---

## Part 3 — Containerization

### The dependency problem

Software depends on its environment: the operating system, installed libraries,
runtime versions, environment variables, file paths.

When you run a service on your laptop, these are all the right versions.
When someone else runs it, or when you deploy to a server, they might not be.

This problem — "it works on my machine" — caused so many production incidents that
the industry built an entire solution around it: **containerization**.

---

### What a container actually is

A container is a running process that is isolated from the rest of the system.
It has its own filesystem, its own network interface, its own process tree.
From inside the container, it looks like a standalone computer.
From outside, it looks like a regular process.

The isolation is provided by Linux kernel features (namespaces, cgroups) — not a full
virtual machine. This is why containers start in seconds and use minimal memory
compared to VMs.

---

### Images and containers

```
Image     A read-only snapshot. Filesystem + dependencies + configuration.
          Built once, stored, shared. Like a class definition.

Container A running instance of an image. Has a writable layer on top.
          Ephemeral: stop it, and the writable layer is gone.
          Like an object instantiated from a class.
```

You build an image once. You run it as many containers as you need, anywhere Docker runs.
The same image on your laptop and on a cloud server produces identical behavior.

---

### The Dockerfile

A Dockerfile is the recipe for building an image. Each instruction creates a layer.
Layers are cached — if a layer hasn't changed, Docker reuses it without rebuilding.

```dockerfile
FROM python:3.11-slim        # start from an existing base image
WORKDIR /app                 # all commands happen in /app inside the image
COPY requirements.txt /app/  # copy dependency list first (cache optimization)
RUN pip install -r requirements.txt  # install — cached unless requirements change
COPY app.py iris_model.pkl /app/     # copy app code (changes more often)
EXPOSE 8000                  # documentation: this container listens on 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
```

The copy-requirements-before-code ordering is a deliberate cache optimization:
source code changes frequently; dependencies change rarely. If you copied everything
at once, any code change would invalidate the `pip install` cache.

**`--host 0.0.0.0`** is required in containers. The default `127.0.0.1` binds only
to the container's loopback interface — unreachable from outside. `0.0.0.0` accepts
connections from any network interface, including the one Docker bridges to the host.

---

### Port mapping

A container is isolated — its ports are not the host's ports by default.

```
docker run -p 5000:8000 iris-api
            │     │
            │     └── container port (what uvicorn listens on)
            └──────── host port (what your browser connects to)
```

`localhost:5000` on your machine → forwarded to → port `8000` inside the container.
They don't have to be the same number.

---

## Part 4 — The Three-File Pattern

This demo separates concerns into three files that mirror three lifecycle stages:

```
train.py    Offline         Trains the model, saves iris_model.pkl
                            Runs once per training cycle. Never touches a network.

app.py      Online          Loads the model, accepts HTTP requests, returns predictions
                            Runs continuously. Never retrains.

test.py     Verification    Sends a real HTTP request, measures latency
                            Runs after deployment to confirm the system works end-to-end
```

The separation matters because these three things have different cadences, different
failure modes, and different responsibilities. Mixing them creates fragile systems.

In production, `train.py` becomes a scheduled pipeline (Prefect/Airflow).
`app.py` becomes a Kubernetes deployment. `test.py` becomes part of CI/CD.

---

## Part 5 — The Bigger Picture

### What the demo builds

```
Request  ──► FastAPI ──► model.predict() ──► Response
                │
           iris_model.pkl
           (loaded at startup)
```

### What production looks like

```
Client
  │
Load Balancer
  │         │         │
API pod    API pod   API pod     (multiple replicas for availability)
  │
Model artifact (fetched from registry at startup — not bundled in image)
  │
Prediction logger (every input + output stored for monitoring)
  │
Drift monitor (alerts when input distribution shifts)
```

The demo bundles the model into the Docker image. This is fine for learning.
In production, the image contains only serving code. The model is fetched from
a registry (MLflow, W&B) at startup — so model and code versions are independent.

---

## Quick Reference

### Run locally

```bash
uv sync && source .venv/bin/activate
uvicorn app:app --reload --port 5000
```

### Run in Docker

```bash
docker build -t iris-api .
docker run -p 5000:8000 iris-api
```

### Test

```bash
python test.py

# Or with curl:
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2}'

# Interactive docs:
# http://localhost:5000/docs
```

### Retrain

```bash
python train.py          # produces new iris_model.pkl
docker build -t iris-api .   # rebuild to pick up new model
```

---

## Official Documentation

- FastAPI: https://fastapi.tiangolo.com
- Pydantic: https://docs.pydantic.dev
- Docker: https://docs.docker.com
- Uvicorn: https://www.uvicorn.org
