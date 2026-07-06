const express = require('express');
const { createClient } = require('redis');
const crypto = require('crypto');

const app  = express();
app.use(express.json());

// ---------------------------------------------------------------------------
// Redis connection
// REDIS_URL is injected by Docker Compose. Falls back to localhost for
// running the service directly outside of Docker.
// ---------------------------------------------------------------------------
const REDIS_URL = process.env.REDIS_URL || 'redis://localhost:6379';
const cache = createClient({ url: REDIS_URL });

cache.on('error', (err) => console.error('Redis error:', err.message));

// Redis v4 client is async — connect before the server starts accepting requests.
cache.connect().then(() => {
    console.log('Cache connected:', REDIS_URL);
}).catch((err) => {
    console.warn('Cache unavailable, running without cache:', err.message);
});

// In-process counters — reset on container restart
const stats = { cacheHits: 0, modelCalls: 0 };


// ---------------------------------------------------------------------------
// Iris classifier
//
// A rule-based classifier that mirrors the first splits of the decision tree
// sklearn would learn from the Iris dataset. The Docker module is about
// containers, not ML — a clean function is enough to make the demo honest.
//
// Rules derived from the actual Iris decision boundaries:
//   petal_length < 2.5          → setosa   (100% accurate on the dataset)
//   petal_length < 4.9          → versicolor (with minor overlap)
//   else                        → virginica
// ---------------------------------------------------------------------------
function classifyIris({ sepal_length, sepal_width, petal_length, petal_width }) {
    if (petal_length < 2.5) {
        return { species: 'setosa',     confidence: 1.00 };
    }
    if (petal_length < 4.9) {
        // Use petal_width to sharpen the versicolor vs virginica boundary
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
app.get('/health', async (req, res) => {
    const redisOk = cache.isReady;
    res.json({
        status:  'ok',
        cache:   redisOk ? 'connected' : 'unavailable',
    });
});


// /predict — the core endpoint.
// Checks the cache first. On miss: runs the classifier, writes to cache with
// a TTL of 5 minutes, returns the result with source="model".
// Same input on the second call returns source="cache".
app.post('/predict', async (req, res) => {
    const { sepal_length, sepal_width, petal_length, petal_width } = req.body;

    // Deterministic cache key — MD5 of the four features as a string.
    // Identical inputs always produce the same key, regardless of JSON key order.
    const raw = `${sepal_length},${sepal_width},${petal_length},${petal_width}`;
    const key = 'iris:' + crypto.createHash('md5').update(raw).digest('hex');

    // Cache lookup — Redis GET is ~0.1ms; the classifier is ~0.01ms (it's a rule).
    // In production with a real model (50–200ms), caching repeated inputs matters.
    if (cache.isReady) {
        const cached = await cache.get(key);
        if (cached) {
            stats.cacheHits++;
            return res.json({ ...JSON.parse(cached), source: 'cache' });
        }
    }

    // Cache miss — run the classifier
    const result = classifyIris({ sepal_length, sepal_width, petal_length, petal_width });

    // Write to cache with 5-minute TTL. EX = expire in seconds.
    if (cache.isReady) {
        await cache.set(key, JSON.stringify(result), { EX: 300 });
    }

    stats.modelCalls++;
    res.json({ ...result, source: 'model' });
});


// /stats — cache hit rate since last container restart.
// Useful for showing the caching effect live during the demo.
app.get('/stats', (req, res) => {
    const total   = stats.cacheHits + stats.modelCalls;
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
