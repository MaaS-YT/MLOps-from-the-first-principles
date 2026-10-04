# Lesson 2.1 — Data & Model Versioning with DVC

> **Module 2 · Reproducibility** · [← Module 1: The ML System](../../Module_1_ML_Systems_Intro/) · [Lesson 2.2: MLflow →](../Lesson_2_Experiment_Tracking_MLflow/)

---

## The Problem

You trained a model on Tuesday. You got 91% accuracy.

Today is Thursday. You want to retrain it.

You open your laptop and see this:

```
Local directory
├── data.csv
├── data1.csv
├── data2.csv
└── data_final.csv
```

Which one trained the model? You don't remember. You check your git history — 40 commits, none of them mention data. You check your notebook — it just calls `pd.read_csv("data.csv")`. You check Slack — no one remembers either.

**The data link in the chain is broken.**

The model is now unreproducible — not because the code is wrong, not because the environment changed, but because you can't answer the most basic question:

> *Which data trained this model?*

That's the problem DVC solves.

---

## The Mental Model

**DVC is git for data.**

You already trust git for code. It tracks every change, lets you go back in time, and keeps a clean history. DVC does the same thing for data — but with one crucial difference: **the data doesn't live in git.**

Here's the trick:

```
In git (small, committed):         In DVC cache / remote (large, not in git):
  data.csv.dvc                       the actual data.csv
  (contains the hash)                (stored by its content hash)
```

Git versions a **small pointer file** — `.dvc`.

DVC versions the **actual data** — in a cache and a remote.

When you `git checkout` an old commit, the `.dvc` file changes. When you `dvc checkout`, DVC reads the new hash and restores the exact matching data.

Your data version is now **tied to your git commit.** Checking out a commit restores both the code *and* the data that went with it. Automatically.

The shape is the one you saw in Module 1:

```
.dvc file    →  hash + remote    →  checked-out dataset
definition   →  artifact         →  instance
Dockerfile   →  image            →  container
```

Same triune. Different link in the chain.

---

## The Ladder

Read the five rungs in order. Each assumes the previous.

| Rung | Guide | What you learn |
|------|-------|----------------|
| 01 | [The Data Problem](01-the-data-problem.md) | Why git alone can't version ML projects |
| 02 | [How DVC Works](02-how-dvc-works.md) | The pointer file, the cache, the remote |
| 03 | [The Demo Walkthrough](03-the-demo-walkthrough.md) | The working example, line by line |
| 04 | [Key Concepts](04-key-concepts.md) | Content addressing, DVC vs git vs W&B |
| 05 | [The Bigger Picture](05-the-bigger-picture.md) | Where DVC fits in the reproducibility stack |

**Start with rung 01.** Don't skip to the commands. The commands only make sense once the problem is clear.

## The Five-Step Progression

Where DVC sits in the longer arc of data versioning:

```
1. Files on disk               no versioning, overwritten, lost
2. Git for data                repo bloat, not practical beyond small files
3. DVC for data                versioned data tied to git commits       ← you are here
4. DVC + experiment tracking   link data versions to model metrics      → Lesson 2.2
5. Feature store               versioned, serving-ready features        → Module 3
```

DVC is rung three. Lesson 2.2 adds rung four. Module 3 adds rung five.

Each rung assumes the previous. Without rung three, rung four is impossible.

---

## Quick Start

Run this after walking the ladder — it's the destination, not the starting point.

```bash
cd version-project/sample

# See the current data
cat data.csv

# Look at the git history
git log --oneline
# 3d104de  added new row              ← version 2 (4 rows)
# 99ed495  config the local remote
# a164b3f  add data and track dvc     ← version 1 (3 rows)
# 72cd2ff  Add sample script
# 14d6a51  init git and dvc

# Go back to version 1
git checkout a164b3f
dvc checkout
cat data.csv
# id,value
# 1,10
# 2,20
# 3,30

# Come back to version 2
git checkout main
dvc checkout
cat data.csv
# id,value
# 1,10
# 2,20
# 3,30
# 5,50
```

That's the whole workflow. **One `git checkout`, one `dvc checkout`. Data version restored.**

---

## What You Should Be Able to Answer at the End

- **Why doesn't git work for data?** Binary files, size, no useful diffs, repo bloat.
- **What does a `.dvc` file contain?** A hash of the data, plus metadata (size, path).
- **Where does the data actually live?** In the DVC cache (local) and the remote (shared).
- **How do git and DVC stay in sync?** `git checkout` + `dvc checkout` together restore both code and data.
- **What's a DVC remote?** Storage backend — S3, GCS, SSH, or a local directory. The shared source of truth.
- **What shape does this have?** `.dvc` → hash + remote → checked-out data. Definition → artifact → instance.
- **How does DVC compare to W&B artifacts?** DVC is git-native and offline-friendly. W&B is cloud-first and integrated with runs. Both version large files; different scopes.

---

## Where This Fits

This lesson is the **data link** in the reproducibility chain.

```
data → code → environment → run → artifact → model
 ↑                    ↑       ↑        ↑       ↑
2.1                 Module 1  2.2     2.2     2.3
DVC                 Docker   MLflow  MLflow  W&B
```

- **Backward:** Module 1 gave you a container that runs identically anywhere. But the container needs the *right data* to reproduce the model. That's what this lesson provides.
- **Forward (Lesson 2.2):** Once data is versioned, you need to track *which run used which version*. That's MLflow.
- **Forward (Module 3):** Data pipelines produce versioned artifacts. DVC is the tool that versions them.
- **Forward (Module 4):** Optimized models are artifacts too. Same pattern.

> 📍 **Full chain diagram:** [`../README.md`](../README.md)

---

## Files

| File | Purpose |
|------|---------|
| `01-the-data-problem.md` … `05-the-bigger-picture.md` | The five rungs — read in order |
| [`version-project/`](version-project/) | The working git+DVC demo (a full project) |
| [`version-project/sample/data.csv`](version-project/sample/data.csv) | The versioned dataset |
| [`version-project/sample/data.csv.dvc`](version-project/sample/data.csv.dvc) | The pointer file that lives in git |
| [`version-project/sample/sample-code.py`](version-project/sample/sample-code.py) | A minimal script that reads the data |
| [`mock-remote/`](mock-remote/) | Local DVC remote (simulates S3) |
| `dvc-ladder.excalidraw` | The lesson board |

---

## Key Terms

| Term | Definition |
|------|------------|
| **DVC** | Data Version Control — versions large data alongside git |
| **`.dvc` file** | A small YAML pointer: "this data is at this hash on this remote" |
| **DVC cache** | Local content-addressed storage at `.dvc/cache/` |
| **DVC remote** | Shared storage backend — S3, GCS, SSH, or a local path |
| **Content addressing** | Files stored by their hash, not by name. Same principle as git |
| **MD5** | The hash function DVC uses by default |
| **`dvc add`** | Tell DVC to track a file — creates the `.dvc` pointer |
| **`dvc push`** | Upload data from local cache to the remote |
| **`dvc pull`** | Download data from the remote to the local cache |
| **`dvc checkout`** | Restore data to match the current `.dvc` pointer |
| **`dvc.yaml`** | Declarative pipeline definition |
| **`dvc repro`** | Rerun only the pipeline stages whose dependencies changed |

---

## Checkpoint

You should now be able to answer:

- **What does DVC version?** Data files — and, later, model artifacts.
- **What does git version?** The `.dvc` pointer files, code, and configs.
- **Why not commit the data to git?** Repo bloat. The data stays in history forever, even after deletion.
- **What's the workflow?** `dvc add` → `git add` → `git commit` → `dvc push`. Then: `git checkout` + `dvc checkout` to restore.
- **What's the artifact in this link of the chain?** The hash inside the `.dvc` file. It uniquely identifies one version of the data — forever.
- **What's the instance?** The checked-out data on disk, restored to match the current commit.

**Next:** [Lesson 2.2 — MLflow →](../Lesson_2_Experiment_Tracking_MLflow/)

---

## How to Use This Lesson

**For a first read:** open rung 01 and walk the ladder in order. Don't skip to the commands.

**For the demo:** follow the Quick Start above, then read rung 03 for the line-by-line walkthrough.

---

> *Data is like code, but heavier.*
> *Same discipline. Different tool.*
