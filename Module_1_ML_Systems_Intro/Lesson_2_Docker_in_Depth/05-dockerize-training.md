# 05 — Dockerize the Training Job

> [← Previous: Why Containers Exist](04_why_containers.md) · [Next: Serving — A Second Image →](06_serving.md)

---

## What We're Doing

Take `training/train.py` from step 03 — the exact same file — and run it inside a container.

The script doesn't change. Only **where it runs** changes. This is the whole point.

---

## Update `train.py` First

Before containerizing, make one change to `train.py`: let it read the output directory from an environment variable.

```python
import os
from pathlib import Path

import joblib
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier


# Where to write the artifact. Defaults to the current directory so
# `uv run train.py` on the host keeps working unchanged.
OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", "."))


def main() -> None:
    # 1. Load data
    X, y = load_iris(return_X_y=True)
    print(f"Loaded Iris: X={X.shape}, y={y.shape}")

    # 2. Train — shallow tree, deterministic
    model = DecisionTreeClassifier(max_depth=3, random_state=42)
    model.fit(X, y)

    # 3. Sanity check — a depth-3 tree can't perfectly separate versicolor
    #    from virginica. 0.973 is expected and correct.
    accuracy = model.score(X, y)
    print(f"Train accuracy: {accuracy:.3f}")

    # 4. Persist the artifact
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUTPUT_DIR / "model.joblib"
    joblib.dump(model, out)
    print(f"Saved {out} ({out.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
```

**Why this change:** when the container runs, we don't want it writing to `/app` (which is owned by root, and which won't survive the container's exit). We want it writing to a mounted directory we control. `OUTPUT_DIR` is how the host tells the container where to put the artifact.

Locally, without the env var, behavior is identical to before: `model.joblib` lands next to `train.py`.

---

## The Dockerfile

`training/Dockerfile`:

```dockerfile
FROM python:3.11-slim

# Bring in uv — the same tool you used locally
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Dependencies first — this layer is cached until pyproject.toml or uv.lock change
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

# Code changes more often — copy last
COPY train.py ./

# Run training inside the uv-managed venv
CMD ["uv", "run", "python", "train.py"]
```

Read it top to bottom. Every line maps to a concept from the previous steps.

---

## Line-by-Line

| Line | What it does | Why |
|------|--------------|-----|
| `FROM python:3.11-slim` | Base image: Debian + Python 3.11 | The "OS + interpreter" layer your venv couldn't isolate |
| `COPY --from=ghcr.io/astral-sh/uv:latest ...` | Pull `uv` from its official image | Same tool, same behavior, everywhere |
| `WORKDIR /app` | Set the working directory inside the container | All future paths are relative to this |
| `COPY pyproject.toml uv.lock ./` | Copy lockfiles *first* | Dependency layer is cached until these change |
| `RUN uv sync --frozen --no-dev` | Install exactly what the lockfile says | `--frozen` = fail if drift, don't silently re-resolve |
| `COPY train.py ./` | Copy code | Changes often → copy last → don't invalidate the cache above |
| `CMD [...]` | The command that runs when the container starts | Same as `uv run python train.py` on your laptop |

---

## The Layer Cache

This is what a Dockerfile "layer" means in practice.

Rebuild after editing `train.py`:

```bash
docker build -t iris-trainer ./training
# → CACHED  [1/6] FROM python:3.11-slim
# → CACHED  [2/6] COPY uv
# → CACHED  [3/6] WORKDIR
# → CACHED  [4/6] COPY pyproject.toml uv.lock
# → CACHED  [5/6] RUN uv sync --frozen --no-dev
# → CHANGED [6/6] COPY train.py
```

Only the last layer re-runs. `uv sync` was not executed again — it was cached.

Rebuild after editing `pyproject.toml`:

```bash
docker build -t iris-trainer ./training
# → CACHED  [1/6] FROM python:3.11-slim
# → CACHED  [2/6] COPY uv
# → CACHED  [3/6] WORKDIR
# → CHANGED [4/6] COPY pyproject.toml uv.lock
# → CHANGED [5/6] RUN uv sync --frozen --no-dev   ← re-runs
# → CHANGED [6/6] COPY train.py
```

Everything downstream of the changed layer re-runs. **Order matters, and the order is: least-changing first.**

This is why `COPY . .` at the top of a Dockerfile is a mistake. It invalidates everything below it on every edit.

---

## Build the Image

```bash
cd training/
docker build -t iris-trainer .
```

Confirm it exists:

```bash
docker images | grep iris-trainer
```

Nothing has run yet. An image is a template. Running it is a separate act.

---

## Run It — Five Versions, One Problem

Here's the same container run five ways. Each adds one flag. Each solves one problem.

### Version 1 — The Naive Run

```bash
docker run iris-trainer
```

Output:

```
Loaded Iris: X=(150, 4), y=(150,)
Train accuracy: 0.973
Saved /app/model.joblib (2113 bytes)
```

Training runs. The model trains. The artifact is written to `/app/model.joblib` — **inside the container**.

Check your host:

```bash
ls model.joblib
```

Nothing. The container's filesystem is gone. When the process exited, Docker tore down the container, and every file written inside it died with it.

The container did the work. The result is unreachable.

Also — the container lingers in a stopped state:

```bash
docker ps -a
# CONTAINER ID   IMAGE          STATUS                     NAMES
# 4a3b2c1d...    iris-trainer   Exited (0) 2 seconds ago    clever_einstein
```

Every naive `docker run` leaves one of these behind.

---

### Version 2 — Add `--rm`

```bash
docker run --rm iris-trainer
```

Same output. Same missing artifact. The only difference: the container is gone the moment it exits.

```bash
docker run --rm iris-trainer
docker ps -a
# nothing from this run
```

**`--rm` doesn't change what the container does. It changes what's left after it does it.**

Without `--rm`, dead containers pile up. Do it ten times, you have ten corpses eating disk.

---

### Version 3 — Add a Volume

```bash
docker run --rm -v "$PWD/output:/out" iris-trainer
```

Still no artifact. Why?

Because `train.py` still writes to `"."` — which inside the container resolves to `/app`. The volume mount makes `/out` *visible* inside the container, but nothing writes there. `-v` alone doesn't redirect anything; it only makes a path available.

**A volume is a door. The script has to walk through it.**

---

### Version 4 — Add the Env Var

```bash
mkdir -p output

docker run --rm \
  -v "$PWD/output:/out" \
  -e OUTPUT_DIR=/out \
  iris-trainer
```

Now:

```
Loaded Iris: X=(150, 4), y=(150,)
Train accuracy: 0.973
Saved /out/model.joblib (2113 bytes)
```

Check your host:

```bash
ls -la output/
# -rw-r--r-- 1 root root 2113 ... model.joblib
```

**The artifact survives.** But look at the ownership: `root root`.

That's the next problem. The container runs as `root` by default. Files it writes to the host come out owned by root. You can read them, but you can't overwrite or delete them without `sudo`.

---

### Version 5 — Run as Your User

```bash
docker run --rm \
  --user "$(id -u):$(id -g)" \
  -v "$PWD/output:/out" \
  -e OUTPUT_DIR=/out \
  -e HOME=/tmp \
  iris-trainer
```

Now check ownership:

```bash
ls -la output/
# -rw-r--r-- 1 silva silva 2113 ... model.joblib
```

Owned by you. Because the container ran as you.

Two new flags:

- `--user "$(id -u):$(id -g)"` — the container's process runs with your host UID:GID. Files it writes to the host come out owned by you.
- `-e HOME=/tmp` — `uv` needs a writable home. When running as a non-root user, `/root` isn't writable. `/tmp` is.

---

## The Five Commands, Side by Side

| Command | Artifact survives? | Container lingers? | Owned by you? |
|---------|--------------------|--------------------|---------------|
| `docker run iris-trainer` | ❌ | ✅ (dead) | — |
| `+ --rm` | ❌ | ❌ | — |
| `+ -v "$PWD/output:/out"` | ⚠️ only if script writes there | ❌ | ❌ (root) |
| `+ -e OUTPUT_DIR=/out` | ✅ | ❌ | ❌ (root) |
| `+ --user "$(id -u):$(id -g)" -e HOME=/tmp` | ✅ | ❌ | ✅ |

Every flag fixes one specific problem. Nothing is decoration.

---

## The Flags, Decoded

| Flag | What it does | Why you need it |
|------|--------------|-----------------|
| `--rm` | Delete the container when it exits | Prevents dead containers piling up |
| `-v host:container` | Mount a host directory into the container | Lets the container write to your host filesystem |
| `-e VAR=value` | Set an environment variable inside the container | Configures runtime behavior without rebuilding the image |
| `--user UID:GID` | Run the container process as this user | Files written to the host come out owned by you, not root |
| `-e HOME=/tmp` | Set a writable home directory | `uv` needs a writable home when running as non-root |

---

## The Full Command, Explained

```bash
docker run --rm \
  --user "$(id -u):$(id -g)" \
  -v "$PWD/output:/out" \
  -e OUTPUT_DIR=/out \
  -e HOME=/tmp \
  iris-trainer
```

| Piece | Reads as |
|-------|----------|
| `docker run` | Start a container |
| `--rm` | …and delete it when it exits |
| `--user "$(id -u):$(id -g)"` | …running as me (UID 1000, GID 1000) |
| `-v "$PWD/output:/out"` | …with the host's `output/` visible as `/out` |
| `-e OUTPUT_DIR=/out` | …and `OUTPUT_DIR` set to `/out` |
| `-e HOME=/tmp` | …and `HOME` set to `/tmp` |
| `iris-trainer` | …from this image |

---

## Why Not Bake These Into the Image?

You could put `ENV OUTPUT_DIR=/out` in the Dockerfile, and `USER appuser` for the non-root pattern. That's what production images do.

But for a lesson, keeping them on the command line makes each one **visible**. You see exactly what changes between runs, and why.

The Dockerfile should describe **what the image is** — Python, dependencies, code. The `docker run` command should describe **how this particular execution happens** — where output goes, which user runs it, what the environment looks like.

| Layer | Responsibility | Changes how often |
|-------|----------------|-------------------|
| Dockerfile | The artifact — base image, deps, code | Rarely |
| `docker run` flags | The execution — mounts, user, env | Every run |

Once you internalize this split, Compose (step 08) becomes obvious: it just **declares the run-time flags in a file** so you stop typing them.

---

## The Same Command, Two Worlds

| On your laptop | In the container |
|----------------|------------------|
| `uv sync` | `RUN uv sync --frozen --no-dev` (build time) |
| `uv run python train.py` | `CMD ["uv", "run", "python", "train.py"]` |
| Writes `model.joblib` next to `train.py` | Writes to `$OUTPUT_DIR/model.joblib` inside the container |
| — | `-v "$PWD/output:/out" -e OUTPUT_DIR=/out` makes those the same place |

The script is identical. The environment is now reproducible — not just at the Python level, but at every level above the kernel.

---

## The Artifact Contract, Updated

| Artifact | Produced by | Consumed by |
|----------|-------------|-------------|
| `model.joblib` | `iris-trainer` container | The serving API (step 06) |
| `iris-trainer` image | `docker build` | Anyone who needs to re-run training |
| `uv.lock` | `uv sync` | The `RUN uv sync` layer in the Dockerfile |

Two artifacts now: the model, and the **image that produces it**. The image is what makes training reproducible on a fresh machine — no Python install, no `uv` install, no `pyproject.toml` resolution. `docker run` and you're done.

---

## A Note on Root-Owned Files

If you ran Version 4 above (without `--user`), you now have root-owned files:

```bash
ls -la output/
# -rw-r--r-- 1 root root ... model.joblib
```

You can read them but not overwrite or delete them as `silva`. Fix:

```bash
sudo chown -R "$USER:$USER" output/
```

This is a common Docker papercut. **Every time you use `-v` with a container that runs as root, the container writes into your host directory as root.** Files come out root-owned.

The fix is always the same: run the container as your user (`--user`). Once you do, ownership matches and the problem disappears.

---

## Checkpoint

You should now be able to answer:

- **What does `FROM python:3.11-slim` give you that a venv didn't?** The interpreter and OS, not just `site-packages`.
- **Why `COPY pyproject.toml uv.lock` before `COPY train.py`?** So dependency install is cached when only the code changes.
- **What's the difference between `docker run` and `docker run --rm`?** The container is deleted on exit with `--rm`; without it, the container lingers in a stopped state.
- **Why doesn't `-v` alone extract the artifact?** The mount makes a path visible; the script must actually write there.
- **Why `--user`?** Because the container runs as root by default, and files it writes to the host end up root-owned.
- **Why `-e HOME=/tmp`?** Because `uv` needs a writable home, and non-root users can't write to `/root`.

**Next:** [Serving — A Second Image →](06_serving.md) — a Node.js API, in its own container, in a different language.