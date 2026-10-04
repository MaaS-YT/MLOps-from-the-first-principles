# 04 — The Model Registry

> [← Previous: The Tracking Server](03-the-tracking-server.md) · [Back to the ladder](README.md)

---

## The problem the registry solves

You have six runs in MLflow. Random Forest won. You've decided to ship it.

Now someone — a teammate, a service, a deployment script — needs to load it.

How do they reference it?

**Option A — by run ID.**

```python
model = mlflow.sklearn.load_model("runs:/47c45b4722b44251a8d2a4acc34a7e9b/model")
```

This works. But it's brittle:
- The run ID is opaque. `47c45b47...` tells you nothing.
- If you retrain and the new model is better, you have to change the code.
- No history. No way to say "this is version 2 of the production model."

**Option B — by file path.**

```python
model = joblib.load("models_nyc_taxi/20261004_213045/model.pkl")
```

Worse. The path is on your laptop. Nobody else can load it. No metadata attached.

**Option C — by registered name and stage.**

```python
model = mlflow.sklearn.load_model("models:/nyc_taxi_predictor/Production")
```

**This is what the registry is for.**

The name `nyc_taxi_predictor` doesn't change. The stage `Production` doesn't change. When you promote a new version to Production, the serving code picks it up without any changes.

---

## What a registered model is

A registered model is a **named, versioned pointer** to a run's model artifact.

Look at the structure in the database:

```
registered_models
  └── name: "nyc_taxi_predictor"
       │
       └── model_versions
            ├── version 1  →  run_id: 47c45b47...  (Random Forest)
            ├── version 2  →  run_id: 8a2269c3...  (tuned)
            └── version 3  →  run_id: ...          (future)

Each version also has:
  - current_stage: None | Staging | Production | Archived
  - aliases: [] or ["champion", "challenger", ...]
  - source: the artifact URI of the model
```

**The registered model is a name.** The versions are pointers to runs. The stages and aliases are labels.

**No new artifact is stored.** The model file already lives in the run's artifact directory. The registry just adds a name and a version number that points at it.

This is why registering is fast — it's just a database write, not a file copy.

---

## Registering

Two ways. Same result.

### Via the Python API

```python
registered = mlflow.register_model(
    model_uri=f"runs:/{best_run_id}/model",
    name="nyc_taxi_predictor",
)
print(registered.name)      # "nyc_taxi_predictor"
print(registered.version)   # "1"
```

`model_uri` is a pointer. `runs:/<run_id>/model` means "the artifact at path `model` inside run `<run_id>`."

MLflow looks up the run, reads its `artifact_uri`, finds the `model/` directory, and creates a new registered model entry that points at it.

### Via the UI

In the MLflow UI:
1. Click the run you want to register
2. Go to the **Artifacts** tab
3. Find the `model/` folder
4. Click **"Register model"** (top right)
5. Give it a name
6. Click **Register**

Same result. The UI is doing the same API call under the hood.

### Automatically during logging

`log_model` accepts a `registered_model_name` argument:

```python
mlflow.sklearn.log_model(
    sk_model=model,
    artifact_path='model',
    registered_model_name='nyc_taxi_predictor',  # register on the spot
)
```

If the model name doesn't exist, MLflow creates it. If it exists, MLflow adds a new version.

**Convenient for iteration.** Each new run auto-registers as a new version. But **dangerous for production** — you don't want every experimental run to bump the version number. Use it when you're iterating and want to see versions accumulate; avoid it when a run is a scratch experiment.

The lesson uses the explicit `register_model` call — cleaner, more intentional.

---

## Stages

A version can be in one of four stages. Stages are the classic MLflow way to mark lifecycle state.

| Stage | Meaning |
|-------|---------|
| **None** | Just registered. Not yet evaluated for deployment. |
| **Staging** | Under evaluation. Candidate for Production. |
| **Production** | The live model. Serving systems load by this stage. |
| **Archived** | Retired. Kept for history but not used. |

Stages are **exclusive.** A version can only be in one stage at a time. When you transition a version to Production, MLflow automatically moves any previous Production version to Archived.

### Transitioning

```python
client = MlflowClient()

client.transition_model_version_stage(
    name='nyc_taxi_predictor',
    version=1,
    stage='Staging',
)
```

Behind the scenes, MLflow updates the `current_stage` column on version 1 in the `model_versions` table.

Then to promote:

```python
client.transition_model_version_stage(
    name='nyc_taxi_predictor',
    version=1,
    stage='Production',
)
```

Version 1 is now Production. Any previous Production version is auto-archived.

### Why stages exist

**Stage transitions are the interface between ML and ops.**

The ML team's job: train models, register them, transition to Staging.

The ops team's job: evaluate Staging models, transition the winner to Production.

The serving system doesn't know about ML. It loads `models:/nyc_taxi_predictor/Production` and gets whatever's currently in that stage. When ops promotes a new version, the serving system picks it up on the next request (or restart, depending on caching).

**No coordination required between ML and ops.** The registry is the handshake.

---

## Aliases

Aliases are the modern alternative to stages. Introduced in MLflow 2.x, they're gradually replacing stages in new deployments.

An **alias** is a named pointer to a version. Unlike stages, a version can have **multiple aliases**, and the same alias can point to different versions over time.

```python
client.set_registered_model_alias(
    name='nyc_taxi_predictor',
    alias='champion',
    version=1,
)
```

Now version 1 is aliased as `champion`. Load it:

```python
model = mlflow.sklearn.load_model("models:/nyc_taxi_predictor@champion")
```

Note the syntax: `@alias` instead of `/stage`.

When a new version wins, move the alias:

```python
client.set_registered_model_alias(
    name='nyc_taxi_predictor',
    alias='champion',
    version=2,
)
```

Now `@champion` points at version 2. Any serving code that loads `@champion` picks up the new version automatically.

### Why aliases are better

**1. Multiple labels per version.** A version can be `champion` and `validated` and `eu-region` at the same time. Stages force one label.

**2. Aliases are versioned history.** When you move `champion` from v1 to v2, the move is recorded. You can see the alias history. Stages just overwrite.

**3. Aliases are names, not slots.** "champion", "challenger", "shadow", "canary", "us-east" — the vocabulary is yours. Stages have four fixed values.

**4. Aliases don't auto-archive.** When you set `champion` to v2, v1 keeps whatever other aliases it had. Stages would auto-archive v1 when v2 becomes Production.

**The pattern in practice:**

```python
# The "current production" model is always @champion
model = mlflow.sklearn.load_model("models:/nyc_taxi_predictor@champion")
```

When you want to promote a new version:

```python
client.set_registered_model_alias(
    name='nyc_taxi_predictor',
    alias='champion',
    version=new_version,
)
```

**One line.** No stage transition, no auto-archiving, no side effects.

### Stages vs aliases — which to use

**For new projects:** aliases. Modern, flexible, no auto-archive surprises.

**For legacy systems:** stages. If your serving code loads `models:/name/Production`, keep it working.

**For teaching:** both. The lesson uses both so the reader knows both patterns. In a real project, pick one and stick with it.

---

## Loading by stage and alias

The whole point of the registry is that loading doesn't require the version number.

```python
# By stage
model = mlflow.sklearn.load_model("models:/nyc_taxi_predictor/Production")

# By alias
model = mlflow.sklearn.load_model("models:/nyc_taxi_predictor@champion")

# By explicit version
model = mlflow.sklearn.load_model("models:/nyc_taxi_predictor/1")

# Latest version (whatever it is)
model = mlflow.sklearn.load_model("models:/nyc_taxi_predictor/latest")
```

**The serving code doesn't change when you promote a new model.** You update the stage or alias, and the next load picks up the new version.

This is the key property: **the model reference is decoupled from the model artifact.**

The serving system's code says "load the production model." The production model changes. The code doesn't.

---

## The artifact contract

When you load a registered model — or download it with `download_artifacts` — you get a **directory** containing multiple files, not just the model.

```
model_dir/
├── MLmodel                     ← the artifact contract
├── model.pkl                   ← the serialized model
├── conda.yaml                  ← the exact environment
├── python_env.yaml             ← the same, in pip format
├── requirements.txt            ← pip dependencies
├── input_example.json          ← a sample input
└── serving_input_example.json  ← the same, in serving format
```

**The `MLmodel` file is the important one.** It's a YAML document that describes:

```yaml
artifact_path: ...
flavors:
  python_function:
    env:
      conda: conda.yaml
      virtualenv: python_env.yaml
    loader_module: mlflow.sklearn
    model_path: model.pkl
    predict_fn: predict
    python_version: 3.12.11
  sklearn:
    pickled_model: model.pkl
    serialization_format: pickle
    sklearn_version: 1.6.1
mlflow_version: 3.2.0
run_id: 47c45b47...
signature:
  inputs: '[{"type": "tensor", ..., "shape": [-1, 4]}]'
  outputs: '[{"type": "tensor", ..., "shape": [-1]}]'
```

**This is what makes an MLflow model portable.** Any MLflow-aware system can read the `MLmodel` file and know:

- What framework the model uses (`sklearn`)
- How to deserialize it (`pickle`)
- What Python and library versions it was trained with (`3.12.11`, `sklearn 1.6.1`)
- What the input and output schemas are
- Which run produced it

**The signature** is particularly important. It declares the input shape — `[-1, 4]` means "any number of rows, 4 features." A serving system can validate incoming requests against this schema before passing them to the model.

**The environment files** — `conda.yaml`, `python_env.yaml`, `requirements.txt` — let you reconstruct the exact environment to re-run the model. If you ever need to debug a production issue, you can spin up an environment with exactly the libraries that produced the model.

**This is the difference between a model file and a model artifact.**

A `.pkl` file is just serialized Python objects. An MLflow artifact is a self-describing package with schema, environment, and lineage.

---

## What the lesson demonstrates

In `pipeline-tracked.ipynb`, cells 13–19:

1. **Register** the best run's model as `nyc_taxi_predictor` version 1
2. **Transition** to Staging, then to Production
3. **Alias** it as `champion`
4. **Load** by stage, by alias, and by version
5. **Predict** with the loaded model
6. **Download** the artifact contract to `model_dir/`

By the end, you have:

- A named model in the registry
- Version 1 in Production
- An alias pointing at it
- Working code that loads by name
- The full artifact contract on disk

**Every piece a real deployment needs.**

---

## What comes next

The registry is the interface between ML and serving.

In production, a serving system loads `models:/nyc_taxi_predictor/Production` at startup (or on each request, or on a cache expiry — depends on the system). When a new version is promoted to Production, the serving system picks it up automatically.

**The ML team changes the model. The serving team changes nothing.**

This is the payoff of the whole chain:

```
data (DVC)  →  run (MLflow)  →  artifact (MLflow)  →  registry (MLflow)  →  serving
```

Each link is a step toward making the model *loadable by name* — the property that lets ML and ops move independently.

That's rung 04.

---

## Checkpoint

You should now be able to answer:

- **What's a registered model?** A named, versioned pointer to a run's model artifact. No new file is stored.
- **What's a stage?** An exclusive label on a version: None, Staging, Production, Archived.
- **What's an alias?** A named, non-exclusive pointer to a version. Modern replacement for stages.
- **How do you load by stage vs alias?** `models:/name/Production` vs `models:/name@champion`.
- **What does the registry decouple?** The model reference from the model artifact. Serving code says "load the production model," and the model can change underneath.
- **What is the `MLmodel` file?** The artifact contract. Declares the framework, serialization, environment, and signature.
- **What's the signature for?** Input and output schema. Lets a serving system validate requests before passing them to the model.

**You've finished the ladder.** Return to the [README](README.md) for the [Quick Start](README.md#quick-start) and [Where This Fits](README.md#where-this-fits).