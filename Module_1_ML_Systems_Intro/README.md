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
| 1.2 | [Docker in Depth](Lesson_2_Docker_in_Depth/) | Every component runs in a container — this is how containers actually work | [docker_commands.md](Lesson_2_Docker_in_Depth/docker_commands.md) |

---

## What This Module Builds

```
SERVING LAYER (Lesson 1.1)
  POST /predict → FastAPI → model.predict() → response
  Containerized: same behavior on laptop, staging server, cloud VM

INFRASTRUCTURE (Lesson 1.2)
  Docker images, layer caching, Compose, inter-container networking
  The foundation every other component in the course runs on top of
```

Lesson 1.1 wraps the model in an API that anything can call.
Lesson 1.2 explains the container infrastructure all subsequent modules depend on.

---

## Where This Fits

This module builds the **Serving Layer** and establishes the **container infrastructure**
at the base of the system map. Every subsequent module — versioning, experiment tracking,
data pipelines, compression — produces artifacts that this layer eventually serves.

Open `SYSTEM_MAP.md` at the repo root to see where every lesson fits in the full system.
