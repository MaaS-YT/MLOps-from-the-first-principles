// ---------------------------------------------------------------------------
// Iris prediction API
//
// Runs standalone (fallback to localhost:6379) or inside a Docker network
// where REDIS_URL is injected by Compose and "redis" resolves via Docker DNS.
//
// This file is intentionally short. The lesson is about Docker, not ML.
// ---------------------------------------------------------------------------

const express = require('express');
const { createClient } = require('redis');
const crypto = require('crypto');

const app = express();
app.use(express.json());

// --- Redis connection --------------------------------------------------------
// REDIS_URL is injected by Docker Compose at runtime. The localhost fallback
// lets you run `node server.js` directly on your laptop without a container.
const REDIS_URL = process.env.REDIS_URL || 'redis://localhost:6379';
const cache = createClient({ url: REDIS_URL });

cache.on('error', (err) => console.error('Redis error:', err.message));

// Redis v4 client is async — connect before accepting traffic.
// If Redis is down, the API still works, just without caching.
cache.connect()
    .then(() => console.log('Cache connected:', REDIS_URL))
    .catch((err) =>
        console.warn('Cache unavailable, running without cache:', err.message)
    );

// In-process counters — reset on container restart.
// A production system would export these to Prometheus (Module 6).
const stats = { cacheHits: 0, modelCalls: 0 };

// ---------------------------------------------------------------------------
// Iris classifier
//
// Mirrors the first splits of the decision tree sklearn learned in the
// training container. In a real system this would load model.joblib via a
// Python sidecar or an ONNX runtime. For this lesson, a clean function keeps
// the demo honest and language-agnostic.
//
// Rules from the actual Iris decision boundaries:
//   petal_length < 2.5  → setosa     (100% separable)
//   petal_length < 4.9  → versicolor (minor overlap with virginica)
//   else                → virginica
// ---------------------------------------------------------------------------
function classifyIris({ sepal_length, sepal_width, petal_length, petal_width }) {
    if (petal_length < 2.5) {
        return { species: 'setosa', confidence: 1.00 };
    }
    if (petal_length < 4.9) {
        const conf = petal_width < 1.7 ? 0.91 : 0.72;
        return { species: 'versicolor', confidence: conf };
    }
    const conf = petal_width >= 1.8 ? 0.96 : 0.78;
    return { species: 'virginica', confidence: conf };
}

// ---------------------------------------------------------------------------
// Routes
// ---------------------------------------------------------------------------

// /health — the universal liveness probe.
// Returns Redis connectivity so an operator can diagnose cache failures
// without shelling into the container.
app.get('/health', (req, res) => {
    res.json({
        status: 'ok',
        cache: cache.isReady ? 'connected' : 'unavailable',
    });
});

// /predict — checks cache first. On miss: runs the classifier, writes to
// cache with a 5-minute TTL, returns the result with source="model".
// Same input on the second call returns source="cache".
app.post('/predict', async (req, res) => {
    const { sepal_length, sepal_width, petal_length, petal_width } = req.body;

    // Deterministic cache key — MD5 of the four features as a string.
    // Identical inputs always produce the same key, regardless of JSON key order.
    const raw = `${sepal_length},${sepal_width},${petal_length},${petal_width}`;
    const key = 'iris:' + crypto.createHash('md5').update(raw).digest('hex');

    // Cache lookup — Redis GET is ~0.1ms. The classifier is ~0.01ms (it's a
    // rule). In production with a real model (50–200ms), caching repeated
    // inputs matters a lot more than it does here. The lesson is the pattern.
    if (cache.isReady) {
        const cached = await cache.get(key);
        if (cached) {
            stats.cacheHits++;
            return res.json({ ...JSON.parse(cached), source: 'cache' });
        }
    }

    // Cache miss — run the classifier
    const result = classifyIris({ sepal_length, sepal_width, petal_length, petal_width });

    // Write to cache with a 5-minute TTL. EX = expire in seconds.
    if (cache.isReady) {
        await cache.set(key, JSON.stringify(result), { EX: 300 });
    }

    stats.modelCalls++;
    res.json({ ...result, source: 'model' });
});

// /stats — cache hit rate since the last container restart.
// Useful for showing the caching effect live during the demo.
app.get('/stats', (req, res) => {
    const total = stats.cacheHits + stats.modelCalls;
    const hitRate = total > 0 ? (stats.cacheHits / total).toFixed(3) : '0.000';
    res.json({ ...stats, total, hitRate });
});

// ---------------------------------------------------------------------------
// Start
// ---------------------------------------------------------------------------
const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
    console.log(`Server running on port ${PORT}`);
    console.log(`POST /predict  — classify an Iris sample (cached)`);
    console.log(`GET  /health   — liveness + cache status`);
    console.log(`GET  /stats    — cache hit rate`);
});