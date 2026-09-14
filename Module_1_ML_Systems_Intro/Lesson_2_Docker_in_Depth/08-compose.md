# 08 — Docker Compose

> [← Previous: Wiring Containers by Hand](07-network.md) · [Back to the ladder](README.md)

---

## What Compose Is

Compose is a **declarative** way to describe multi-container systems.

In step 07, you ran five commands with a dozen flags. Compose replaces all of that with one file and one command.

The step-07 setup, expressed in YAML:

```yaml
services:

  serving:
    build: ./serving
    container_name: iris-api
    ports:
      - "3000:3000"
    environment:
      REDIS_URL: redis://redis:6379
    depends_on:
      - redis
    networks:
      - ml-net

  redis:
    image: redis:7-alpine
    container_name: iris-cache
    volumes:
      - redis-data:/data
    networks:
      - ml-net

networks:
  ml-net:
    driver: bridge

volumes:
  redis-data:
```

**Every line maps to a flag from step 07.**

| Step 07 command | Compose equivalent |
|-----------------|-------------------|
| `docker network create ml-net` | `networks: ml-net:` block |
| `docker volume create redis-data` | `volumes: redis-data:` block |
| `docker run --name redis --network ml-net -v redis-data:/data redis:7-alpine` | `redis:` service block |
| `docker run --network ml-net -p 3000:3000 -e REDIS_URL=... iris-api` | `serving:` service block |

Nothing new. Just **declared instead of typed**.

---

## The Training Service (Optional)

Training is a one-off job, not a long-running service. We don't want it to start with `docker compose up`.

Three consequences of step 05's lessons carry over:

1. **It writes an artifact to the host** — so it needs a volume, and `OUTPUT_DIR` to tell `train.py` where to write.
2. **It must run as your user** — otherwise `model.joblib` comes out root-owned.
3. **It needs a writable home** — otherwise `uv` tries to write its cache to `/` and gets `Permission denied`.

All three are the same flags we typed by hand. Compose just declares them.

Use a **profile** so training doesn't start with the main system:

```yaml
  training:
    build: ./training
    volumes:
      - ./training/output:/out
    environment:
      OUTPUT_DIR: /out
      HOME: /tmp
    user: "${UID:-1000}:${GID:-1000}"
    profiles: ["train"]
```

| Line | Maps to step 05's flag |
|------|-----------------------|
| `volumes: ./training/output:/out` | `-v "$PWD/output:/out"` |
| `environment: OUTPUT_DIR: /out` | `-e OUTPUT_DIR=/out` |
| `environment: HOME: /tmp` | `-e HOME=/tmp` |
| `user: "${UID:-1000}:${GID:-1000}"` | `--user "$(id -u):$(id -g)"` |
| `profiles: ["train"]` | (Compose-only — not a docker run flag) |

Three things to note:

**`${UID:-1000}`** reads the shell's `UID` variable, defaulting to `1000` if unset. On most Linux systems, `UID` is set automatically. On macOS, it's not — hence the fallback. Either way, the container runs as you.

**`HOME: /tmp` matters.** In step 05 we set it on the `docker run` command line. Compose needs the same declaration. Without it, `uv` tries to write its cache to `/.cache/uv` — the container's root filesystem, which UID 1000 can't write to. The result is:

```
error: Failed to initialize cache at `/.cache/uv`
Caused by: failed to create directory `/.cache/uv`: Permission denied (os error 13)
```

Setting `HOME: /tmp` gives `uv` a writable home. The cache lives in `/tmp/.cache/uv`, dies with the container, and no one notices. The `uv` warning about "hardlinks falling back to full copy" is also expected — the cache and the mounted output live on different filesystems, so `uv` copies instead of hardlinking. Slower, still correct.

**`profiles: ["train"]`** means this service only starts when explicitly requested. `docker compose up` ignores it. `docker compose --profile train run --rm training` runs it.

---

## The Full File

`docker-compose.yml`:

```yaml
services:

  # --- Training (one-off, opt-in via `--profile train`) --------------------
  # Produces ./training/output/model.joblib on the host.
  # Not started by `docker compose up`. Run explicitly:
  #   docker compose --profile train run --rm training
  training:
    build: ./training
    volumes:
      - ./training/output:/out
    environment:
      OUTPUT_DIR: /out
      HOME: /tmp
    user: "${UID:-1000}:${GID:-1000}"
    profiles: ["train"]

  # --- Serving API ---------------------------------------------------------
  serving:
    build: ./serving
    container_name: iris-api
    ports:
      - "3000:3000"
    environment:
      REDIS_URL: redis://redis:6379
    depends_on:
      - redis
    networks:
      - ml-net

  # --- Redis cache ---------------------------------------------------------
  redis:
    image: redis:7-alpine
    container_name: iris-cache
    volumes:
      - redis-data:/data
    networks:
      - ml-net
    # No `ports:` — Redis is intentionally not published to the host.
    # Only services on ml-net can reach it.

networks:
  ml-net:
    driver: bridge

volumes:
  redis-data:
```

**Compare the `serving` and `redis` blocks to step 07's `docker run` commands.** They're the same system. The difference is that step 07's flags are now YAML keys.

**Compare the `training` block to step 05's full command.** Same story:

| Step 05 flag | Compose key |
|--------------|-------------|
| `--rm` | implicit — `docker compose run --rm` |
| `--user "$(id -u):$(id -g)"` | `user: "${UID:-1000}:${GID:-1000}"` |
| `-v "$PWD/output:/out"` | `volumes: - ./training/output:/out` |
| `-e OUTPUT_DIR=/out` | `environment: OUTPUT_DIR: /out` |
| `-e HOME=/tmp` | `environment: HOME: /tmp` |

Compose is a **transcription** of the manual commands. Not a new concept — a new syntax.

---

## Run It

```bash
# From Lesson_2_Docker_in_Depth/

# 0. Create the output directory (bind mounts need the host path to exist)
mkdir -p training/output

# 1. Train the model (one-off, uses the "train" profile)
docker compose --profile train run --rm training

# 2. Bring up serving + redis
docker compose up --build

# 3. First call — runs the classifier, caches the result
curl -X POST http://localhost:3000/predict \
  -H "Content-Type: application/json" \
  -d '{"sepal_length":5.1,"sepal_width":3.5,"petal_length":1.4,"petal_width":0.2}'

# 4. Second call — same input, returns from cache
curl -X POST http://localhost:3000/predict \
  -H "Content-Type: application/json" \
  -d '{"sepal_length":5.1,"sepal_width":3.5,"petal_length":1.4,"petal_width":0.2}'

# 5. Check hit rate
curl http://localhost:3000/stats
```

Verify the artifact:

```bash
ls -la training/output/
# -rw-r--r-- 1 silva silva 2113 ... model.joblib
```

Owned by you. Because Compose ran the container as `user: 1000:1000`.

That's the whole system. Two containers running, wired together, reachable from the host. One file describing it.

---

## What Compose Gave You

| Without Compose | With Compose |
|-----------------|--------------|
| 5 shell commands, 3 flags each | One file, one command |
| Manual cleanup between runs | `docker compose down` |
| No version control for topology | `docker-compose.yml` in git |
| Can't reproduce on another machine | Clone + `docker compose up` |
| Root-owned files from `-v` | Declared `user:` fixes it |
| `$HOME` errors from `uv` | Declared `HOME:` fixes it |

Same pattern as `uv`. **Declare, don't type.**

---

## Compose Is Not Kubernetes

A common question at this point: "Why not just use Kubernetes?"

For one machine, Compose is the right tool. Kubernetes is:

- Multi-node
- Self-healing
- Rolling deployments
- Service mesh
- Way, way more than you need to run two containers on your laptop

**Compose scales down to one laptop. Kubernetes scales up to a fleet.** You'll meet Kubernetes in Module 5. What you learn here — services, networks, volumes, declarative topology, per-service config — is the foundation.

| Concept | Compose | Kubernetes |
|---------|---------|------------|
| Service | `services:` entry | `Deployment` |
| Network | `networks:` | `Service`, `NetworkPolicy` |
| Volume | `volumes:` | `PersistentVolumeClaim` |
| Config | `environment:` | `ConfigMap`, `Secret` |
| User | `user:` | `securityContext.runAsUser` |
| Run | `docker compose up` | `kubectl apply -f` |

Every concept here has a Kubernetes counterpart. You're learning the vocabulary.

---

## Compose Command Cheatsheet

```bash
# Start everything (detached)
docker compose up -d

# Rebuild and start
docker compose up --build

# Stop and remove containers + networks
docker compose down

# Stop and remove containers + networks + named volumes
docker compose down -v

# View logs from a service
docker compose logs -f serving

# Shell into a running service
docker compose exec serving sh

# Run a one-off command in a service
docker compose run --rm training

# Use a profile
docker compose --profile train run --rm training

# Show running services
docker compose ps

# Show the resolved config (merges profiles + env)
docker compose config
```

`docker compose config` is worth knowing — it prints what Compose actually resolves your file to, after variable substitution (`${UID:-1000}` → `1000`) and profile merging. Useful for debugging.

---

## Where You Are Now

You started with "it works on my laptop" and ended with a reproducible, multi-container system described in a single file.

Recap of the ladder:

| Step | Concept | Artifact |
|------|---------|----------|
| 01 | Virtual environments | (concept only) |
| 02 | `uv` | `pyproject.toml`, `uv.lock` |
| 03 | Training pipeline | `model.joblib` |
| 04 | Containers, from first principles | (concept only) |
| 05 | Containerized training | `iris-trainer` image, `output/model.joblib` |
| 06 | Containerized serving | `iris-api` image |
| 07 | Manual networking | `ml-net` network, `redis-data` volume |
| 08 | Compose | `docker-compose.yml` |

Every step layered on the last. Nothing skipped. And every concept here recurses:

- **Module 2** versions the `model.joblib` artifact.
- **Module 4** compresses it.
- **Module 5** scales the containers to Kubernetes.
- **Module 6** instruments them.

Docker is the substrate. You've got the substrate now.

---

## Checkpoint

You should now be able to answer:

- **What does Compose replace?** All the manual `docker network`, `docker volume`, and `docker run` commands.
- **How does the training service map to step 05's `docker run`?** `volumes:` → `-v`, `environment:` → `-e`, `user:` → `--user`.
- **Why `HOME: /tmp`?** Without it, `uv` tries to cache to `/.cache/uv`, which UID 1000 can't write to. The container fails with `Permission denied`.
- **Why `profiles: ["train"]`?** So training doesn't start with the main system — it's a one-off, not a service.
- **Why `mkdir -p training/output` before running?** Bind mounts need the host path to exist.
- **What's the relationship between Compose and Kubernetes?** Same declarative shape, different scale.

**You've finished the ladder.** Return to the [README](README.md) for the [Quick Start](README.md#quick-start) and [Where It Fits](README.md#where-it-fits).