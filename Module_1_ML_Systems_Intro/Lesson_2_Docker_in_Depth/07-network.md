# 07 — Wiring Containers by Hand

> [← Previous: Serving — A Second Image](06-serving.md) · [Next: Docker Compose →](08-compose.md)

---

## What We Have

Two images:

- `iris-trainer` — Python, produces `model.joblib`
- `iris-api` — Node.js, wants a Redis cache

One problem: **they can't see each other.**

By default, each container runs in its own network namespace. No route to anything. This is the isolation from step 04 working as designed.

Step 07 breaks the isolation *just enough* to let containers talk — and only the ones we want.

---

## The Three Things We Need

1. **A shared network** — so containers can route to each other
2. **DNS between them** — so we use names, not IPs
3. **Volumes** — so data escapes the container filesystem when needed

Compose (step 08) will hide all three. Do them manually first so you know what Compose is doing.

---

## Part 1 — Networks

### What Docker Networking Is

Docker gives every container a network interface. By default, it's attached to a **bridge network** called `bridge`, shared with every other container on the host.

That's a problem: everything is on the same network by default, and DNS is not enabled on the default bridge.

**Best practice: create a user-defined bridge network.** Containers on it can resolve each other by name.

```bash
docker network create ml-net
```

That's it. One command.

### What Just Happened

```bash
docker network inspect ml-net
```

You'll see:

- A subnet (e.g., `172.18.0.0/16`)
- A gateway
- An empty container list
- **A DNS resolver at `127.0.0.11`** — Docker's embedded DNS server

That DNS resolver is the magic. Every container on `ml-net` gets it. When one container asks for `redis`, Docker answers with the redis container's IP.

### Attach Containers

```bash
# Start Redis on the network, named "redis"
docker run -d \
  --name redis \
  --network ml-net \
  -v redis-data:/data \
  redis:7-alpine

# Start the API on the same network
docker run -d \
  --name iris-api \
  --network ml-net \
  -p 3000:3000 \
  -e REDIS_URL=redis://redis:6379 \
  iris-api
```

Look at the `-e REDIS_URL=redis://redis:6379`. The hostname `redis` — not an IP — is what the API uses. Docker's DNS resolves it.

**This is inter-container DNS.** No `/etc/hosts`, no hardcoded IPs, no service discovery layer. Just names on a shared network.

### Verify

```bash
# Is the API connected to Redis?
curl http://localhost:3000/health
# → {"status":"ok","cache":"connected"}
```

First prediction:

```bash
curl -X POST http://localhost:3000/predict \
  -H "Content-Type: application/json" \
  -d '{"sepal_length":5.1,"sepal_width":3.5,"petal_length":1.4,"petal_width":0.2}'
# → {"species":"setosa","confidence":1,"source":"model"}
```

Second prediction — same input:

```bash
curl -X POST http://localhost:3000/predict \
  -H "Content-Type: application/json" \
  -d '{"sepal_length":5.1,"sepal_width":3.5,"petal_length":1.4,"petal_width":0.2}'
# → {"species":"setosa","confidence":1,"source":"cache"}
```

`source: "cache"` — the API just used Redis. **The network did that.**

### Ports: What's Published vs What's Private

| Container | Port published to host? | Reachable from host? | Reachable from `ml-net`? |
|-----------|------------------------|----------------------|--------------------------|
| `iris-api` | Yes (`-p 3000:3000`) | Yes | Yes |
| `redis` | No | No | Yes, as `redis:6379` |

**This is the correct default.** Redis should not be exposed to your laptop. Only the API should be reachable from outside. Anything on `ml-net` can still reach Redis.

---

## Part 2 — Volumes

We brushed against volumes in step 05. Now let's be precise.

### The Problem

A container's filesystem is **ephemeral**. When the container exits, its filesystem is gone.

```bash
docker run --rm iris-trainer        # writes model.joblib inside the container
ls model.joblib                     # → not found. It's gone.
```

Three ways to move data across the container boundary:

| Mechanism | Direction | Use case |
|-----------|-----------|----------|
| **Bind mount** | Host directory ↔ Container directory | Development, artifact extraction |
| **Named volume** | Docker-managed storage ↔ Container | Persistent data (databases) |
| **tmpfs** | Memory only, never touches disk | Secrets, scratch space |

### Bind Mounts — The One We Need

```bash
docker run --rm \
  -v "$PWD/output:/out" \
  -e OUTPUT_DIR=/out \
  --user "$(id -u):$(id -g)" \
  -e HOME=/tmp \
  iris-trainer
```

This mounts the host's `output/` directory at `/out` inside the container. `train.py` reads `OUTPUT_DIR=/out`, writes the model there, and it appears at `output/model.joblib` on the host.

**Two-way, live.** Edit a file on the host → container sees it. Container writes a file → host sees it.

Two things step 05 hammered home, restated here:

- **The mount alone isn't enough.** The script must write to the mounted path. `OUTPUT_DIR` is how the host tells the container where to write.
- **The mount brings ownership with it.** Without `--user`, files come out owned by root. With `--user`, they come out owned by you.

For our training job: bind mount is exactly right. The output is a file we want on the host.

### Named Volumes — For State That Should Persist

```bash
# Create a named volume
docker volume create redis-data

# Mount it in Redis — Redis writes its AOF/RDB files there
docker run -d \
  --name redis \
  --network ml-net \
  -v redis-data:/data \
  redis:7-alpine
```

The volume is managed by Docker. It outlives the container. If you remove and re-create Redis, the data survives.

**Named volumes are not the same as bind mounts.** They're Docker-managed storage in a private area (`/var/lib/docker/volumes/`), not a directory you choose.

**Ownership is not an issue with named volumes.** Docker manages permissions internally — Redis writes as root to `/data`, and that's fine because `/data` lives inside Docker's storage, not on your host.

### What We Use Here

| Service | Mount | Type | Why |
|---------|-------|------|-----|
| `iris-api` | none | — | Stateless |
| `redis` | `redis-data:/data` | named | Cache should survive restarts |
| `iris-trainer` | `./output:/out` | bind | Artifact extraction |

The distinction matters: **bind mounts cross the container-host boundary; named volumes don't.** Ownership complications only arise with bind mounts.

---

## The Complete Manual Setup

```bash
# 1. Network
docker network create ml-net

# 2. Volume for Redis
docker volume create redis-data

# 3. Redis
docker run -d \
  --name redis \
  --network ml-net \
  -v redis-data:/data \
  redis:7-alpine

# 4. API
docker run -d \
  --name iris-api \
  --network ml-net \
  -p 3000:3000 \
  -e REDIS_URL=redis://redis:6379 \
  iris-api

# 5. Test
curl http://localhost:3000/health
```

**Count the moving parts:** network, volume, two `docker run` invocations, three flags each, hand-managed names. This is manageable for two containers. It is not manageable for ten.

That's what Compose fixes.

---

## Clean Up

```bash
docker rm -f iris-api redis
docker network rm ml-net
docker volume rm redis-data
```

Do this before step 08 — you'll rebuild the whole thing declaratively.

---

## Checkpoint

You should now be able to answer:

- **Why do containers need a shared network to talk?** Default isolation gives each container its own network namespace — no route.
- **What is `127.0.0.11`?** Docker's embedded DNS resolver, injected into every container on a user-defined network.
- **When do you use a bind mount vs a named volume?** Bind mount for host-visible files (artifacts); named volume for Docker-managed persistent state.
- **Why does ownership only matter for bind mounts?** Bind mounts cross into your host filesystem. Named volumes stay inside Docker's storage.
- **Why `OUTPUT_DIR` when a bind mount already exposes the path?** The mount makes the path visible; the script must be told to write there.

**Next:** [Docker Compose →](08-compose.md) — the whole system, declared in one file.
````
