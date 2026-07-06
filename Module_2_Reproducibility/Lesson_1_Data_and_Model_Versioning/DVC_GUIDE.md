# Lesson 2.1 — Data Versioning with DVC

> **Lesson 2.1** — Why git alone cannot version ML projects, how DVC separates data storage from data references, and how to make any training result reproducible from a single commit.

| | |
|---|---|
| **Problem this solves** | You got 91% accuracy last Tuesday. Today you can't reproduce it because you don't know which version of the data produced it. Without data versioning, model experiments are not reproducible. |
| **Mental model** | DVC works like git but for large files. The file stays in a remote store (S3, GCS, local cache). Git tracks a small `.dvc` pointer file. Checking out a git commit restores both the code and the exact data it was written for. |
| **What the lecture demonstrates** | Initializing DVC in a git repo → adding a dataset → pushing to a remote → switching between two dataset versions → seeing how code + data versions stay in sync |
| **Where this fits** | DVC is the **Versioning Layer** in the system map. It sits between the training run and the model registry — ensuring every artifact is traceable to the exact data that produced it. |

---

# Data Versioning with DVC — Concepts & Guide

Demo: `version-project/sample/` — a git+DVC project tracking two versions of a dataset.

---

## Part 1 — The Data Versioning Problem

### Git is not enough

Git is the standard tool for versioning code. It works because code files are:
- Small (kilobytes to megabytes)
- Text-based (meaningful diffs)
- Safe to store in a repository and share

Data files are the opposite:
- Large (gigabytes to terabytes)
- Binary or opaque (a diff of two CSVs tells you nothing useful)
- Dangerous to store in git (one mistake bloats the repo permanently)

If you commit a 2GB dataset to git, it stays in the history forever.
Even if you delete it later, anyone who clones the repo downloads 2GB.
Git Large File Storage (LFS) helps but still couples data to the code repository.

The deeper problem is not size — it is **traceability**.

You train a model on `data_v3.csv`. You get 91% accuracy.
A colleague trains the same model on `data_v4.csv`. They get 88%.
Why did it drop? What changed between v3 and v4?
Who changed it, and when?

Without versioning, you cannot answer these questions.
Without those answers, you cannot debug, reproduce, or audit your ML system.

---

### The solution DVC provides

DVC (Data Version Control) stores data files outside of git — in a **cache** and a **remote** —
while putting a small **pointer file** (`.dvc`) into git in their place.

```
In git (small, committed):         In DVC cache / remote (large, not committed):
  data.csv.dvc                       the actual data.csv
  (contains the hash of the data)    (stored by its content hash)
```

When you switch git branches or commits, the `.dvc` file changes.
DVC reads the new hash and pulls the corresponding data version from the cache.

Your data version is now **tied to your git commit**. Checking out a git commit
restores both the code and the data that went with it — automatically.

---

### Git + DVC together

```
git       versions code, configs, notebooks, .dvc pointer files
DVC       versions data files, model artifacts, large binaries

git commit  ──►  captures code + .dvc pointer (the hash of the current data)
dvc push    ──►  uploads the actual data to the remote
dvc pull    ──►  downloads the data matching the current git commit
```

The two tools divide responsibility cleanly:
git manages what you want to track in history.
DVC manages the large files those history entries point to.

---

### Where DVC fits in the MLOps progression

```
1. Files on disk               no versioning, overwritten, lost
2. Git for data                repo bloat, not practical beyond small files
3. DVC for data                versioned data tied to git commits    ← you are here
4. DVC + experiment tracking   link data versions to model metrics
5. Feature store               versioned, serving-ready features (Feast, next lesson)
```

---

## Part 2 — How DVC Works

### The .dvc file is a pointer

When you tell DVC to track a file, it does two things:

1. Computes the file's MD5 hash
2. Copies the file into the local cache (`.dvc/cache/`)
3. Writes a `.dvc` file containing that hash

```yaml
# data.csv.dvc  — this is what lives in git
outs:
- md5: bbd304055be67c211495c95f4e697eb2
  size: 24
  hash: md5
  path: data.csv
```

The `.dvc` file is tiny (a few lines of YAML) and goes into git.
The actual `data.csv` goes into `.gitignore` — git never sees it.

This is the core mechanism: **git tracks the hash, DVC tracks the file**.

---

### The cache

The cache lives at `.dvc/cache/` inside your project (by default).
Files are stored by their content hash — the same way git stores objects:

```
.dvc/cache/files/md5/
  bb/d304055be67c211495c95f4e697eb2   ← version 1 of data.csv
  1a/2dc46fd6ee44c2ee01cdc0e67d067d   ← version 2 of data.csv
```

Two versions of the same file coexist in the cache under different hashes.
No data is ever deleted from the cache — you can always restore any version.

---

### The remote

The cache is local. The **remote** is the shared storage your team pushes to and pulls from.
It works exactly like a git remote — same concept, different data.

```
git remote     = GitHub / GitLab (stores code)
DVC remote     = S3 / GCS / Azure Blob / local path (stores data)
```

In this demo, the remote is a local directory (`mock-remote/`) simulating cloud storage.
In production it would be an S3 bucket or similar.

```
[core]
    remote = local_remote
['remote "local_remote"']
    url = ../../../mock-remote
```

`dvc push` uploads from local cache → remote.
`dvc pull` downloads from remote → local cache → working directory.

---

### Switching between versions

This is where git and DVC work together as one system.

The demo has two commits, each with a different version of `data.csv`:

```
commit a164b3f  "add data and track data with dvc"
  data.csv.dvc → md5: bbd304...   (3 rows: id=1,2,3)

commit 3d104de  "added new row"
  data.csv.dvc → md5: 1a2dc4...   (4 rows: id=1,2,3,5)
```

To go back to version 1:
```bash
git checkout a164b3f         # .dvc file now points to md5: bbd304...
dvc checkout                 # DVC reads the .dvc file, restores data.csv to 3 rows
```

To come back to version 2:
```bash
git checkout main            # .dvc file now points to md5: 1a2dc4...
dvc checkout                 # data.csv restored to 4 rows
```

You never manually manage which CSV file is which version.
Git commit + `dvc checkout` is the complete operation.

---

## Part 3 — The Demo Walkthrough

### What the demo represents

The `version-project/sample/` directory is a minimal but complete DVC project.
Its git history tells a deliberate story:

```
14d6a51  init git and dvc         → project initialized, DVC set up
72cd2ff  Add sample script        → sample-code.py added (reads the CSV)
a164b3f  add data and track dvc   → data.csv (3 rows) tracked for the first time
99ed495  config the local remote  → mock-remote configured as DVC remote
3d104de  added new row            → data.csv updated to 4 rows, new .dvc hash recorded
```

The two meaningful commits are `a164b3f` and `3d104de`.
Each represents a dataset version. The `.dvc` file in git is the version marker.

---

### The .dvc file at each version

**Version 1** (commit `a164b3f`):
```yaml
outs:
- md5: bbd304055be67c211495c95f4e697eb2
  size: 24
  path: data.csv
```

**Version 2** (commit `3d104de`, current):
```yaml
outs:
- md5: 1a2dc46fd6ee44c2ee01cdc0e67d067d
  size: 29
  path: data.csv
```

The only thing that changed in git is the hash value in the `.dvc` file.
The data itself never touched git — it lives in the cache and the mock-remote.

---

### The sample script

```python
# sample-code.py
import pandas as pd
df = pd.read_csv('data.csv')
print(df.head())
```

This is intentionally minimal — it represents any ML script that reads the data.
The point is: the script never changes. The data changes.
DVC ensures the right data version is present whenever the script runs.

---

## Part 4 — Key Concepts

### Reproducibility via content addressing

DVC uses **content-addressed storage**: files are stored and retrieved by their hash,
not by name or path. This is the same principle git uses for its object store.

Two consequences:
- Identical files are stored only once (deduplication)
- A hash uniquely identifies exactly one version of a file — forever

If your model was trained on `md5: bbd304...`, you can always reproduce that
training environment by checking out the git commit that points to that hash.
No ambiguity. No "which version was it again?"

---

### What DVC tracks vs what git tracks

```
Track with git:
  source code (.py, .ipynb)
  configuration files (.yaml, .json)
  .dvc pointer files
  dvc.yaml pipeline definitions
  requirements, Dockerfiles

Track with DVC:
  datasets (CSV, Parquet, JSON)
  trained model artifacts (.pkl, .pt, .h5)
  large preprocessed outputs
  anything binary or large
```

A good rule: if `git diff` produces a useful, readable diff → use git.
If the file is too large or too binary for a useful diff → use DVC.

---

### DVC vs W&B / MLflow artifacts

All three version large files. The difference is scope and integration:

| | DVC | W&B Artifacts | MLflow log_model |
|---|---|---|---|
| Integration | git-native | W&B runs | MLflow runs |
| Data + models | Both | Both | Models primarily |
| Offline use | Yes (fully local) | Cloud required | Local or server |
| Best for | data versioning tied to code | experiment artifacts | model lifecycle |

DVC is the right tool when you want data versioning to be first-class in your git workflow,
independent of any specific experiment tracking platform.

---

## Part 5 — The Bigger Picture

### Why data versioning matters for the whole ML system

```
Experiment reproducibility
  "Reproduce run X" requires: same code + same data + same environment
  Git gives you the code. DVC gives you the data. Docker gives you the environment.

Debugging model degradation
  "The model was better last month" → checkout last month's git tag
  → dvc checkout → same data, same code, same model → find what changed

Auditing and compliance
  "What data was this model trained on?" → git log the .dvc file
  → exact hash → exact data → auditable answer

Team collaboration
  Everyone pulls the same data version for the same git commit
  No more "works on my machine" for data
```

### The complete reproducibility stack

```
Code version     →  git
Data version     →  DVC
Environment      →  Docker / conda
Experiments      →  MLflow / W&B
Features         →  Feast (coming next)
```

Each layer solves one problem. Together they make an ML system fully reproducible.

---

## Quick Reference

### Setup a new DVC project

```bash
git init
dvc init
git commit -m "init dvc"
```

### Track a file

```bash
dvc add data/mydata.csv          # creates data/mydata.csv.dvc, updates .gitignore
git add data/mydata.csv.dvc .gitignore
git commit -m "track dataset v1"
```

### Configure a remote

```bash
dvc remote add myremote s3://my-bucket/dvc-store
dvc remote default myremote
git add .dvc/config
git commit -m "configure dvc remote"
```

### Push and pull

```bash
dvc push      # upload data to remote
dvc pull      # download data from remote (after git clone or git checkout)
```

### Switch to a previous data version

```bash
git checkout <commit-hash>    # restores the .dvc pointer file
dvc checkout                  # restores the actual data file to match
```

### Check what's tracked

```bash
dvc status    # are local files in sync with .dvc files?
dvc diff      # what changed between commits?
```

---

## Official Documentation

- DVC concepts: https://dvc.org/doc/user-guide/concepts
- Get started: https://dvc.org/doc/start
- Data versioning: https://dvc.org/doc/start/data-management
- Remotes: https://dvc.org/doc/user-guide/data-management/remote-storage
- DVC + git: https://dvc.org/doc/user-guide/how-it-works
