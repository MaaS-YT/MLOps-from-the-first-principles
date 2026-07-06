# Lesson 1.2 — Docker in Depth

| | |
|---|---|
| **Problem this solves** | Software that works on your laptop fails on the server because the environments differ. This happens in every ML deployment if you don't address it explicitly. |
| **Mental model** | A container is not a virtual machine. It is a process with an isolated view of the filesystem and network — running on the same OS kernel as the host, with all dependencies baked into its image. |
| **What the lecture demonstrates** | Building an image → running a container → wiring a Node.js prediction API + Redis cache with Docker Compose → port mapping, inter-container DNS, environment variable injection |
| **Where this fits** | Docker is **infrastructure** in the system map. Every other component — the serving API, the training job, the pipeline — runs inside a container. This lesson explains why and how. |

---

## Files

| File | Purpose |
|------|---------|
| `docker_commands.md` | Full guide: containers vs VMs, Dockerfile internals, networking, Compose |
| `Dockerfile` | Container recipe for the Node.js API |
| `server.js` | Node.js Iris prediction API with Redis prediction caching |
| `package.json` | Node.js dependencies |
| `docker-compose.yml` | Wires API + Redis cache as a multi-service system |

**Start with:** `docker_commands.md`

---

## Quick Start

```bash
docker compose up --build

# First call — runs the classifier, caches the result
curl -X POST http://localhost:3000/predict \
  -H "Content-Type: application/json" \
  -d '{"sepal_length":5.1,"sepal_width":3.5,"petal_length":1.4,"petal_width":0.2}'

# Second call — same input, returns from cache
curl -X POST http://localhost:3000/predict \
  -H "Content-Type: application/json" \
  -d '{"sepal_length":5.1,"sepal_width":3.5,"petal_length":1.4,"petal_width":0.2}'

# Check hit rate
curl http://localhost:3000/stats
```

---

## The Language-Agnostic Point

The demo uses Node.js — deliberately. Docker wraps any language the same way.
The Dockerfile pattern, Compose structure, and networking concepts here apply
identically to Python, Go, or Java services. **Docker doesn't care what runs inside.**
