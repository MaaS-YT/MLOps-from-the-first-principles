# Module 1 — The ML System

*Start at the end. Understand what you're building toward before you build it.*

The first question in production ML is the simplest to state and the easiest to get wrong:
once you have a trained model, how does anything actually use it?

This module answers that by building the serving layer — the last mile between a model
and a user — and then explaining the container infrastructure that makes every other
component in the system run identically across environments.

---

## Lessons

| Lesson | Topic | Problem It Solves | Guide |
|--------|-------|-------------------|-------|
| 1.1 | [Serving with FastAPI](Lesson_1_Serving_with_FastAPI/) | A trained model delivers no value until something can call it | [article.md](Lesson_1_Serving_with_FastAPI/article.md) · [FASTAPI_DOCKER_GUIDE.md](Lesson_1_Serving_with_FastAPI/FASTAPI_DOCKER_GUIDE.md) |
| 1.2 | [Docker in Depth](Lesson_2_Docker_in_Depth/) | Every component runs in a container — this is how containers actually work | [README.md](Lesson_2_Docker_in_Depth/README.md) — start here, then walk the ladder (01 → 08) |

---

## What This Module Builds

```
SERVING LAYER (Lesson 1.1)
  POST /predict → FastAPI → model.predict() → response
  Containerized: same behavior on laptop, staging server, cloud VM

INFRASTRUCTURE (Lesson 1.2)
  venv → uv → container → image → network → Compose
  Docker images, layer caching, inter-container DNS, bind mounts, named volumes
  The foundation every other component in the course runs on top of
```

Lesson 1.1 wraps the model in an API that anything can call.
Lesson 1.2 explains the container infrastructure all subsequent modules depend on — starting
from the smallest unit of isolation (a virtual environment) and building up to a multi-service
system declared in a single Compose file.

---

## How Lesson 1.2 Is Structured

Unlike 1.1, which reads as a single guide, 1.2 is a **ladder** — eight small docs meant to be read in order.

| Step | Guide | What you learn |
|------|-------|----------------|
| 01 | [Why Virtual Environments Exist](Lesson_2_Docker_in_Depth/01-virtual-env.md) | The dependency problem, from first principles |
| 02 | [Enter `uv`](Lesson_2_Docker_in_Depth/02-uv.md) | Fast, reproducible Python environments |
| 03 | [Build the Training Pipeline](Lesson_2_Docker_in_Depth/03-training-pipeline.md) | A Python training job in a venv |
| 04 | [Why Containers Exist](Lesson_2_Docker_in_Depth/04-why-container.md) | The same problem, one level up |
| 05 | [Dockerize the Training Job](Lesson_2_Docker_in_Depth/05-dockerize-training.md) | The same script, inside an image |
| 06 | [Serving — A Second Image](Lesson_2_Docker_in_Depth/06-serving.md) | A Node.js API in its own container |
| 07 | [Wiring Containers by Hand](Lesson_2_Docker_in_Depth/07-network.md) | Networks, DNS, volumes — no Compose yet |
| 08 | [Docker Compose](Lesson_2_Docker_in_Depth/08-compose.md) | The whole system declared in one file |

Start at the lesson's `README.md` — it contains the ladder table, quick start, and mental model.

---

## Where This Fits

This module builds the **Serving Layer** and establishes the **container infrastructure**
at the base of the system map. Every subsequent module — versioning, experiment tracking,
data pipelines, compression — produces artifacts that this layer eventually serves.

Open `SYSTEM_MAP.md` at the repo root to see where every lesson fits in the full system.