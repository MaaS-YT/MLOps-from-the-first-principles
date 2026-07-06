# Lesson 2.1 — Data & Model Versioning with DVC

| | |
|---|---|
| **Problem this solves** | You got 91% accuracy last Tuesday. Today you can't reproduce it because you don't know which version of the data trained it. Without data versioning, model experiments are not reproducible. |
| **Mental model** | DVC works like git but for large files. The file stays in a remote store (S3, GCS, local cache). Git tracks a small `.dvc` pointer file. Checking out a git commit restores both the code and the exact data it was written for. |
| **What the lecture demonstrates** | Initializing DVC in a git repo → adding a dataset → pushing to a remote → switching between two dataset versions → seeing how code + data stay in sync |
| **Where this fits** | DVC is the **Versioning Layer** in the system map. Every artifact is traceable to the exact data that produced it. |

---

## Files

| File | Purpose |
|------|---------|
| `DVC_GUIDE.md` | Full guide: data versioning problem, how DVC works, demo walkthrough |
| `version-project/sample/` | Minimal git+DVC project with two committed dataset versions |
| `version-project/sample/sample/sample-code.py` | Script that reads the versioned CSV — represents any ML training script |

**Start with:** `DVC_GUIDE.md`

---

## Quick Start

```bash
cd version-project/sample

# See the current data version
cat data.csv

# Switch to the previous version
git log --oneline                     # find the first data commit
git checkout <first-data-commit>
dvc checkout                          # restores data.csv to match the .dvc pointer

# Come back to current
git checkout main
dvc checkout
```
