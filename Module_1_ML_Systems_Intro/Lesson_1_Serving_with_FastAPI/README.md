# Lesson 1.1 — Serving with FastAPI

| | |
|---|---|
| **Problem this solves** | A trained model is a Python object in memory. Nothing outside your machine can use it. A REST API gives it a stable, language-agnostic interface. Docker makes that API run identically everywhere. |
| **Mental model** | `train.py` runs once (offline). `app.py` runs forever (online). `test.py` verifies the handoff worked. Three different jobs with different cadences — separating them is the pattern that scales to production. |
| **What the lecture demonstrates** | Training an Iris classifier → saving the model → wrapping it in FastAPI → containerizing with Docker → calling the API end-to-end |
| **Where this fits** | This is the **Serving Layer** — the last mile. Everything else in the course produces a better model for this layer to serve. |

---

## Files

| File | Purpose |
|------|---------|
| `article.md` | Start here — the "why": what production ML is and what makes it harder than training |
| `FASTAPI_DOCKER_GUIDE.md` | The "how": FastAPI mechanics, container internals, the three-file pattern |
| `train.py` | Trains the Iris classifier, saves `iris_model.pkl` |
| `app.py` | Loads the model, serves predictions via FastAPI |
| `test.py` | End-to-end verification — sends a real request, measures latency |
| `Dockerfile` | Container recipe for the API |
| `requirements.txt` | Python dependencies |

**Read order:** `article.md` → `FASTAPI_DOCKER_GUIDE.md` → code files.

---

## Quick Start

```bash
# Run locally
uv sync && source .venv/bin/activate
python train.py
uvicorn app:app --reload --port 5000

# Run in Docker
docker build -t iris-api .
docker run -p 5000:8000 iris-api

# Test
python test.py
```
