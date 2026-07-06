# Lesson 1.2 — Docker: Making Environments Reproducible

> **Lesson 1.2** — How containers work under the hood, why they solve the "works on my machine" problem, and how Docker Compose wires multi-service systems together.

| | |
|---|---|
| **Problem this solves** | Software that works on your laptop fails on the server because the environments differ. This happens in every ML deployment if you don't address it explicitly. |
| **Mental model** | A container is not a virtual machine. It is a process with an isolated view of the filesystem and network — running on the same OS kernel as the host, with all dependencies baked into its image. |
| **What the lecture demonstrates** | Building an image → running a container → wiring a Node.js prediction API + Redis cache with Docker Compose → understanding port mapping, inter-container DNS, and environment variable injection |
| **Where this fits** | Docker is **infrastructure** in the system map. Every other component — the serving API, the training job, the pipeline — runs inside a container. This module explains why and how. |

---

# Docker Crash Course — Concepts & Guide

Demo: `Dockerfile`, `docker-compose.yml`, `server.js` — Node.js Iris prediction API + Redis prediction cache.

---

## Part 1 — Why Containers Exist

### The environment problem

Software runs inside an environment. That environment includes:
the operating system, installed libraries, runtime versions, network config,
environment variables, available ports, file paths.

A piece of software that works perfectly in one environment can fail completely in another —
same code, different result. This causes:

- "Works on my machine" bugs that waste days to diagnose
- Deployment failures when the server has different library versions
- Onboarding problems when a new developer can't reproduce the dev environment
- Subtle production bugs when staging and prod differ slightly

This is not a niche problem. It is the single most common source of friction
between development and production in software engineering.

---

### What containerization solves

A container packages an application **together with everything it needs to run**:
the runtime, libraries, config, filesystem structure. The package is self-contained.
It runs identically on your laptop, a colleague's machine, and a cloud server.

The host machine provides only one thing: the ability to run containers.
It does not need the right Python version, the right libraries, or any specific config.
Everything the app needs is inside the container.

```
Without containers:
  Host must have: Node 20, correct npm packages, Redis installed, right env vars...
  Different for every developer, every server, every CI environment.

With containers:
  Host must have: Docker
  Everything else is inside the image. Language doesn't matter to Docker.
```

---

### Containers vs Virtual Machines

Both create isolated environments. The difference is what they virtualize:

```
Virtual Machine                     Container
────────────────────────────        ────────────────────────────
Application                         Application
Libraries                           Libraries
Full Guest OS (~GB)                 (shares host OS kernel)
Hypervisor (hardware emulation)     Container runtime (Docker)
Host OS                             Host OS
Hardware                            Hardware

Startup:  minutes                   Startup:  seconds
Size:     gigabytes                 Size:     megabytes
Isolation: hardware-level           Isolation: process-level
```

Containers are lighter because they don't emulate hardware or carry a full OS.
They share the host kernel but isolate the filesystem and processes.
For most dev and ML use cases, containers are the right trade-off.

---

### Where Docker fits in the MLOps progression

```
1. Local scripts           runs on your machine, breaks on others
2. Virtual environments    isolates Python packages — not the full environment
3. Docker containers       full environment isolation, reproducible     ← you are here
4. Docker Compose          multi-service orchestration                  ← and here
5. Container registry      share images across teams (Docker Hub, ECR)
6. Kubernetes              orchestrate containers at scale in production
```

---

## Part 2 — Images and Containers

### The image

An image is a **read-only snapshot** of a filesystem and its configuration.
It is built from a `Dockerfile` — a recipe that describes what to install and configure.

Images are layered. Each instruction in the Dockerfile adds a layer.
Layers are cached: if a layer hasn't changed, Docker reuses the cached version
without re-executing the instruction. This makes rebuilds fast.

Think of an image as a class definition: it describes a blueprint.
It does not run anything on its own.

---

### The container

A container is a **running instance of an image**.
Docker adds a thin writable layer on top of the image for the container's runtime state.
Stop the container, and that writable layer is discarded.

The image stays unchanged. You can start another container from the same image
and it starts clean, without any state from the previous run.

Think of a container as an object instantiated from a class:
- One image → many containers
- Containers are ephemeral; images are persistent
- Modifying a container doesn't change the image

---

### The Dockerfile

```dockerfile
FROM node:20-alpine               # base image: Node.js on minimal Alpine Linux
WORKDIR /app                      # working directory inside the image
COPY package*.json ./             # copy dependency manifest first
RUN npm install --production      # install deps (cached unless package.json changes)
COPY server.js ./                 # copy application code (changes more often)
EXPOSE 3000                       # document the port the app listens on
CMD ["node", "server.js"]         # command to run when the container starts
```

The copy-dependencies-before-source pattern is about **layer caching**.
Docker executes Dockerfile instructions top to bottom and caches each result.
If `package.json` is unchanged, Docker skips the `npm install` layer entirely on rebuild.
If you copied everything at once, any code change would invalidate the npm install cache.

This pattern is identical regardless of language — Python, Node, Go, Java all follow it:
copy the dependency manifest first, install, then copy source. **This is a Docker concept,
not a language concept.** The same mental model applies to `requirements.txt`, `go.mod`,
or `pom.xml`.

`CMD` uses array form ("exec form") — runs `node` directly as PID 1.
The alternative string form runs through `/bin/sh -c`, which adds a shell in between
and can cause problems with signal handling (SIGTERM not reaching the app on shutdown).

---

## Part 3 — Networking in Docker

### The isolation problem

By default, each container is isolated from every other container, and from the host.
A container running Redis on port 6379 is not reachable from another container —
not even one on the same machine.

This is intentional. Isolation is the point of containers.
But for multi-service apps, services need to talk to each other.

---

### Docker networks

A Docker network is a virtual private network that containers can join.
Containers on the same network can reach each other by **container name**.
Docker provides built-in DNS resolution: `redis` inside the network resolves to the
IP address of the container named `redis`.

```
ml-net (virtual network)
  ├── iris-api     reachable as "app" on this network
  └── iris-cache   reachable as "redis" on this network
```

The Node.js service connects to Redis with:
```
redis://redis:6379
```
`redis` is the service name in `docker-compose.yml` — not an IP, not localhost.
Docker resolves it via the network's built-in DNS.

No hardcoded IPs. If Docker reassigns IPs (it does), the connection string still works.

---

### Port mapping

A network inside Docker is private. To make a service reachable from your host machine
(your browser, a curl command), you map container ports to host ports:

```
-p 3000:3000     host port 3000 → container port 3000 (Node.js API)
-p 6379:6379     host port 6379 → container port 6379 (Redis — for debugging)
```

Container ports are for inter-container communication.
Host ports are for external access (your browser, tools on your machine).

They don't have to match — `-p 8080:3000` forwards host 8080 to container 3000.

---

## Part 4 — Docker Compose

### The problem with running containers manually

For a three-service app, manual `docker run` means:
three separate commands, manual network creation, manual dependency ordering,
easy to get flags wrong, hard to share with the team.

Docker Compose solves this with a single YAML file that declares the entire
multi-service setup. One `docker compose up` starts everything.
One `docker compose down` tears it all down cleanly.

---

### The two services in this demo

```yaml
services:
  app:    Node.js Iris prediction API (custom image — we build it)
  redis:  Prediction cache (official Redis image — no build needed)
```

Each service is a separate container. `app` uses a custom image built from our
`Dockerfile`. `redis` uses a pre-built official image pulled from Docker Hub.
This is the common pattern: your code gets built, dependencies get pulled.

The language in `app` is Node.js — deliberately. Docker wraps any language the
same way. The `Dockerfile` pattern, Compose structure, and networking concepts
you learn here apply identically to a Python, Go, or Java service.

---

### Dependency ordering and readiness

```yaml
depends_on:
  - redis
```

`depends_on` tells Compose: start `redis` before `app`.
**Important:** this only waits for the container to *start*, not for Redis to be *ready*.
Redis initializes quickly, but there's still a small window at startup.

The demo handles this gracefully: `server.js` checks `cache.isReady` before every
Redis operation and falls through to the classifier if the cache isn't ready yet.
`/health` reports the cache status so you can see it. In production you'd use a
Kubernetes readiness probe or a retry loop with exponential backoff.

---

### Restart policy

```yaml
restart: unless-stopped
```

| Policy | Behavior |
|---|---|
| `no` | Never restart automatically |
| `always` | Restart even after explicit stop |
| `on-failure` | Restart only on non-zero exit code |
| `unless-stopped` | Restart on crash, respect explicit stops |

`unless-stopped` is the right default for database containers in development:
they recover from crashes automatically but stop when you intend to stop the project.

---

### Environment variables

```yaml
environment:
  REDIS_URL: redis://redis:6379
```

Config that changes between environments (dev, staging, production) should never
be hardcoded in source code or images.

The app reads it at runtime:
```javascript
const REDIS_URL = process.env.REDIS_URL || 'redis://localhost:6379';
```

The `||` fallback is the local development default — when running outside Docker,
the variable is absent and the app tries localhost.
In Docker, the variable is injected by Compose and points to the Redis container.

In production, secrets (passwords, API keys) come from a secrets manager
(HashiCorp Vault, Kubernetes Secrets) — never from Compose files committed to git.

---

## Part 5 — The Bigger Picture

### What this demo represents

```
Client (curl / browser)
  │ HTTP :3000
Node.js API  (iris-api)     ← any language could be here
  │ TCP :6379 (via ml-net DNS — "redis" resolves automatically)
Redis cache  (iris-cache)
```

Two containers, one shared network, one `docker compose up` to run the whole thing.
The same pattern extends to any multi-service ML system: swap Node.js for Python,
add a model server, a database for logging predictions, a monitoring sidecar — each
as its own container on the same network. **Docker doesn't care what language runs inside.**

---

### From dev to production

| Concern | Development (this demo) | Production |
|---|---|---|
| Secrets | Env vars in compose file | Secrets manager (Vault, K8s Secrets) |
| Language | Node.js (could be anything) | Whatever your team uses |
| Scaling | Single app container | Kubernetes with replicas + HPA |
| Image storage | Local | Container registry (ECR, GCR, Docker Hub) |
| Redis | Single container | Redis Cluster or managed (ElastiCache) |

The concepts are identical. The infrastructure around them scales up.

---

## Quick Reference

### Build and run with Compose

```bash
docker compose up --build        # build the app image + start app + redis
docker compose up -d --build     # same, detached (background)
docker compose down              # stop and remove containers + networks
docker compose logs -f           # follow logs from all services
```

### Useful Docker commands

```bash
docker ps                       # running containers
docker ps -a                    # all containers (including stopped)
docker logs -f <container>      # follow container logs
docker exec -it <container> sh  # shell into a running container
docker images                   # list local images
docker rm -f $(docker ps -aq)  # remove all containers (clean slate)
```

### Access points

| Service | URL |
|---|---|
| Predict | http://localhost:3000/predict  (POST) |
| Health | http://localhost:3000/health   (GET) |
| Cache stats | http://localhost:3000/stats    (GET) |

```bash
# First call — runs the classifier, writes to cache, returns source:"model"
curl -X POST http://localhost:3000/predict \
  -H "Content-Type: application/json" \
  -d '{"sepal_length":5.1,"sepal_width":3.5,"petal_length":1.4,"petal_width":0.2}'

# Second call — same input, returns source:"cache"
curl -X POST http://localhost:3000/predict \
  -H "Content-Type: application/json" \
  -d '{"sepal_length":5.1,"sepal_width":3.5,"petal_length":1.4,"petal_width":0.2}'

# Check cache hit rate
curl http://localhost:3000/stats
```

---

## Official Documentation

- Docker overview: https://docs.docker.com/get-started/overview/
- Dockerfile reference: https://docs.docker.com/engine/reference/builder/
- Docker Compose: https://docs.docker.com/compose/
- Docker networking: https://docs.docker.com/network/
