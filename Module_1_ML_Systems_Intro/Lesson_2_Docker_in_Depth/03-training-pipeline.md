# 03 — Build the Training Pipeline

> [← Previous: Enter `uv`](02-uv.md) · [Next: Why Containers Exist →](04-why-container.md)

---

## What We're Building

A Python script that trains a decision tree on the Iris dataset and writes `model.joblib`.

No Docker yet. Just a working Python job in a `uv`-managed venv. **This is the artifact that everything downstream will containerize.**

---

## Step 1 — Project Layout

```bash
mkdir -p training && cd training
```

Files we'll create:

```
training/
├── pyproject.toml     # dependencies — this project's own
├── train.py           # the training script
└── model.joblib       # produced by running train.py
```

> **Note on this repo:** `training/` is a standalone `uv` project. It has its own `pyproject.toml` and its own `uv.lock`, independent of the DDODS workspace at the repo root. Commands run from inside `training/` operate only on this project.
>
> In the DDODS workspace, `uv` may refuse to treat `training/` as its own project unless the parent `pyproject.toml` either lists it under `[tool.uv.workspace].members` or excludes it under `[tool.uv.workspace].exclude`. Try `uv lock` first; if it errors, add `training/` to the parent's `exclude` list.

---

## Step 2 — Declare Dependencies

`training/pyproject.toml`:

```toml
[project]
name = "iris-training"
version = "0.1.0"
description = "Iris training job — Docker lesson demo"
requires-python = ">=3.11"
dependencies = [
    "scikit-learn>=1.4",
    "joblib>=1.3",
]

[tool.uv]
package = false
```

Lock and sync:

```bash
uv lock       # generates training/uv.lock
uv sync       # creates training/.venv/
```

You now have a project-scoped environment and a lockfile that pins every transitive dependency. From here on, `uv run` inside `training/` uses this venv — not any inherited from the workspace.

---

## Step 3 — The Training Script

`training/train.py`:

```python
import os
from pathlib import Path

import joblib
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier


# Where to write the artifact. Defaults to the current directory so
# `uv run train.py` on the host keeps working unchanged.
# The container sets OUTPUT_DIR to control where the artifact lands.
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

Two design choices worth noting:

**`OUTPUT_DIR` from env, defaulting to `"."`.** On your laptop, the artifact lands next to `train.py`. In a container, `OUTPUT_DIR` points at a mounted directory so the artifact escapes the ephemeral filesystem. Same script, two environments. Step 05 relies on this.

**`max_depth=3`.** A shallow, interpretable tree. It does **not** perfectly fit the data — versicolor and virginica overlap, so train accuracy is 0.973, not 1.000. That's fine. The point of this lesson is Docker, not model accuracy. A slightly imperfect model is honest.

---

## Step 4 — Run It

```bash
uv run python train.py
```

Expected output:

```
Loaded Iris: X=(150, 4), y=(150,)
Train accuracy: 0.973
Saved model.joblib (2113 bytes)
```

You now have `training/model.joblib` — the artifact.

---

## Step 5 — Inspect What You Built

```bash
# Where did the environment go?
ls -la .venv/

# What's in the artifact?
ls -lh model.joblib

# What's pinned?
head -20 uv.lock
```

Three things to observe:

**`.venv/` is local.** It was created by `uv sync` in this directory, using this project's `pyproject.toml` and `uv.lock`. It does not come from the DDODS workspace.

**`model.joblib` is small** — a couple of KB. It's a shallow decision tree. A real model would be larger; the workflow is the same.

**`uv.lock` pins every dependency**, including transitive ones. Same lockfile → same environment, on any machine, forever.

---

## The Artifact Contract

Everything downstream depends on three things:

| Artifact | Produced by | Consumed by |
|----------|-------------|-------------|
| `model.joblib` | `train.py` | The serving API (step 06) |
| `uv.lock` | `uv lock` | The `RUN uv sync` layer in step 05's Dockerfile |
| `pyproject.toml` | You | `uv` |

In later modules:

- **Module 2 (DVC)** versions `model.joblib`.
- **Module 4 (Compression)** shrinks it.
- **Module 5 (Kubernetes)** ships it inside a container.

For now, it lives on your laptop. Step 05 will containerize the *process that produces it*.

---

## What's Committed vs Generated

Commit to git:

- `train.py`
- `pyproject.toml`
- `uv.lock` ← **this is the reproducibility contract**

Do **not** commit:

- `.venv/`
- `model.joblib`
- `__pycache__/`

Add a `.gitignore` in `training/`:

```gitignore
.venv/
model.joblib
__pycache__/
*.pyc
```

---

## Checkpoint

You should now be able to answer:

- **What does `uv sync` produce?** A `.venv/` directory in `training/`, populated from `uv.lock`.
- **What does `train.py` produce?** `model.joblib` — the artifact.
- **Why does `train.py` read `OUTPUT_DIR`?** So the same script can write locally (default `.`) or inside a container (mounted directory). Step 05 needs this.
- **Why is train accuracy 0.973 and not 1.000?** A `max_depth=3` tree can't perfectly separate versicolor from virginica. This is expected.
- **What's committed?** `pyproject.toml` and `uv.lock`. Not `.venv/`, not `model.joblib`.

**Next:** [Why Containers Exist →](04-why-container.md) — the same problem, one level up.