# 06 — Serving — A Second Image

> [← Previous: Dockerize the Training Job](05-dockerize-training.md) · [Next: Wiring Containers by Hand →](07-network.md)

---

## Why a Second Language?

The training job was Python. That's where sklearn lives.

Serving is different. It needs:

- Fast startup
- Concurrent request handling
- A tiny runtime — no numpy, no sklearn, no pandas

Node.js is a good fit. And it makes an important point:

> **Docker doesn't care what runs inside.**

The Dockerfile pattern from step 05 applies identically here. Same layering, same caching. Only the base image and the command change.

---

## What We're Building

A Node.js API with three endpoints:

- `POST /predict` — classify an Iris sample
- `GET /health` — liveness + cache status
- `GET /stats` — cache hit rate

It talks to Redis for caching. **Redis doesn't exist yet** — that's step 07. For now, the API runs standalone and reports `cache: unavailable`.

---

## Project Layout

```bash
mkdir -p serving && cd serving
```

```
serving/
├── Dockerfile
├── package.json
└── server.js
```

---

## Dependencies

`serving/package.json`:

```json
{
  "name": "iris-predict-api",
  "version": "1.0.0",
  "description": "Iris prediction API with Redis cache — Docker lesson demo",
  "main": "server.js",
  "scripts": {
    "start": "node server.js"
  },
  "dependencies": {
    "express": "^4.19.2",
    "redis": "^4.7.0"
  }
}
```

Same idea as `pyproject.toml`. Declares what the project needs. `npm install` generates `package-lock.json` — the Node equivalent of `uv.lock`.

```bash
npm install       # generates package-lock.json
```

Commit `package.json` **and** `package-lock.json`. Never commit `node_modules/`.

---

## The Server

`serving/server.js`:

```javascript
const express = require('express');
const { createClient } = require('redis');
const crypto = require('crypto');

const app = express();
app.use(express.json());

// REDIS_URL is injected at runtime. Falls back to localhost for direct runs.
const REDIS_URL = process.env.REDIS_URL || 'redis://localhost:6379';
const cache = createClient({ url: REDIS_URL });
cache.on('error', (err) => console.error('Redis error:', err.message));
cache.connect().catch((err) =>
    console.warn('Cache unavailable, running without cache:', err.message)
);

const stats = { cacheHits: 0, modelCalls: 0 };

// The decision-tree boundaries from step 03, re-expressed in JS.
// In a real system this would call a Python sidecar or load an ONNX model.
// For this lesson, it mirrors the tree so the demo is honest.
function classifyIris({ sepal_length, sepal_width, petal_length, petal_width }) {
    if (petal_length < 2.5) return { species: 'setosa', confidence: 1.00 };
    if (petal_length < 4.9) {
        const conf = petal_width < 1.7 ? 0.91 : 0.72;
        return { species: 'versicolor', confidence: conf };
    }
    const conf = petal_width >= 1.8 ? 0.96 : 0.78;
    return { species: 'virginica', confidence: conf };
}

app.get('/health', (req, res) => {
    res.json({ status: 'ok', cache: cache.isReady ? 'connected' : 'unavailable' });
});

app.post('/predict', async (req, res) => {
    const { sepal_length, sepal_width, petal_length, petal_width } = req.body;
    const raw = `${sepal_length},${sepal_width},${petal_length},${petal_width}`;
    const key = 'iris:' + crypto.createHash('md5').update(raw).digest('hex');

    if (cache.isReady) {
        const cached = await cache.get(key);
        if (cached) {
            stats.cacheHits++;
            return res.json({ ...JSON.parse(cached), source: 'cache' });
        }
    }

    const result = classifyIris({ sepal_length, sepal_width, petal_length, petal_width });
    if (cache.isReady) await cache.set(key, JSON.stringify(result), { EX: 300 });

    stats.modelCalls++;
    res.json({ ...result, source: 'model' });
});

app.get('/stats', (req, res) => {
    const total = stats.cacheHits + stats.modelCalls;
    const hitRate = total > 0 ? (stats.cacheHits / total).toFixed(3) : '0.000';
    res.json({ ...stats, total, hitRate });
});

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => console.log(`Server on :${PORT}`));
```

**Note the two environment variables:** `REDIS_URL` and `PORT`. Both have localhost defaults so you can run the file directly. In a container, they'll be injected — that's the whole point of step 07.

---

## The Dockerfile

`serving/Dockerfile`:

```dockerfile
FROM node:20-alpine

WORKDIR /app

# Dependencies first — cached until package.json changes
COPY package*.json ./
RUN npm install --production

# Code changes more often — copy last
COPY server.js ./

EXPOSE 3000
CMD ["node", "server.js"]
```

**Compare this to `training/Dockerfile`.** Same skeleton:

| Step | Training | Serving |
|------|----------|---------|
| Base | `python:3.11-slim` | `node:20-alpine` |
| Package manager | `uv` | `npm` |
| Deps manifest | `pyproject.toml`, `uv.lock` | `package.json`, `package-lock.json` |
| Deps install | `uv sync --frozen` | `npm install --production` |
| Code | `train.py` | `server.js` |
| Run | `uv run python train.py` | `node server.js` |

**The pattern is the language-agnostic part.** Base image, deps first, code second, entrypoint. Same everywhere.

---

## Build and Run Standalone

```bash
cd serving/

# Build
docker build -t iris-api .

# Run — Redis isn't up yet, so the API will report cache: unavailable
docker run --rm -p 3000:3000 iris-api
```

In another terminal:

```bash
curl http://localhost:3000/health
# → {"status":"ok","cache":"unavailable"}

curl -X POST http://localhost:3000/predict \
  -H "Content-Type: application/json" \
  -d '{"sepal_length":5.1,"sepal_width":3.5,"petal_length":1.4,"petal_width":0.2}'
# → {"species":"setosa","confidence":1,"source":"model"}
```

It works. But `source` will always be `"model"` — there's no cache to hit.

**We have two images. They don't know about each other.**

- `iris-trainer` — produces `model.joblib`
- `iris-api` — serves predictions, wants a Redis cache

Step 07 connects them.

---

## A Note on `--user` and Volumes

Notice the `docker run` above has **no `--user` flag and no `-v` mount**.

That's deliberate. The serving container:

- Doesn't write to the host filesystem (no `-v`)
- Doesn't need a specific UID (no `--user`)

The root-ownership papercut from step 05 only appears when **a container writes to a host-mounted directory**. The serving API reads from the network and writes to the response — both stay inside the container.

If you later add a volume (for logs, for instance), the `--user` lesson applies again. For now, the container can run as root harmlessly because it writes nothing to your host.

---

## Checkpoint

You should now be able to answer:

- **Why is the Dockerfile pattern the same across languages?** Because it's about filesystem layering, not language semantics.
- **What does `-p 3000:3000` do?** Maps host port 3000 → container port 3000.
- **Why does `cache: unavailable`?** Redis isn't running. Step 07 fixes that.
- **Why no `--user` or `-v` here?** The serving container doesn't write to the host. Ownership only matters when it does.

**Next:** [Wiring Containers by Hand →](07-network.md) — networks, DNS, volumes. No Compose yet.