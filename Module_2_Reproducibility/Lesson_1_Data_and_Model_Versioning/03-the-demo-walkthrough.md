# 03 — The Demo Walkthrough

> [← Previous: How DVC Works](02-how-dvc-works.md) · [Next: Key Concepts →](04-key-concepts.md)

---

## What the demo represents

`version-project/sample/` is a minimal but complete DVC project. Its git history tells a deliberate story.

```
<hash>  init git and dvc                  → project initialized, DVC set up
<hash>  Add sample script                 → sample-code.py added (reads the CSV)
<hash>  add data and track data with dvc  → data.csv (3 rows) tracked for the first time
<hash>  config the local remote           → mock-remote configured as DVC remote
<hash>  added new row                     → data.csv updated to 4 rows, new .dvc hash recorded
```

Run `git log --oneline` inside `sample/` to see the actual hashes.

The two meaningful commits are the ones with messages **"add data and track data with dvc"** and **"added new row"**. Each represents a dataset version. **The `.dvc` file in git is the version marker.**

---

## The `.dvc` file at each version

**Version 1** — the commit with message `add data and track data with dvc`:

```yaml
outs:
- md5: bbd304055be67c211495c95f4e697eb2
  size: 24
  path: data.csv
```

**Version 2** — the commit with message `added new row` (current `HEAD`):

```yaml
outs:
- md5: 1a2dc46fd6ee44c2ee01cdc0e67d067d
  size: 29
  path: data.csv
```

The only thing that changed in git is the hash value in the `.dvc` file. The data itself never touched git — it lives in the cache and the mock-remote.

---

## The sample script

```python
# sample-code.py
import pandas as pd
df = pd.read_csv('data.csv')
print(df.head())
```

Intentionally minimal. This represents any ML script that reads data. The point: **the script never changes. The data changes.** DVC ensures the right data version is present whenever the script runs.

---

## Walk the demo

**Prerequisite:** the venv lives in `version-project/`. Every DVC command must run through it. From inside `sample/`, that means `uv run --project .. dvc <command>`.

```bash
cd version-project/sample

# What does the data look like now?
cat data.csv
```

Output:

```
id,value
1,10
2,20
3,30
5,50
```

Four rows. This is version 2.

```bash
# What's the git history?
git log --oneline
```

Output (hashes will vary):

```
<hash> added new row
<hash> config the local remote
<hash> add data and track data with dvc
<hash> Add sample script
<hash> init git and dvc
```

Now go back to version 1. Find the hash of the commit that says `add data and track data with dvc`:

```bash
git checkout <that-hash>
uv run --project .. dvc checkout
cat data.csv
```

Output:

```
id,value
1,10
2,20
3,30
```

Three rows. **Version 1 restored.** The `.dvc` file at this commit pointed to `md5: bbd304...`, and `dvc checkout` pulled that exact data from the cache.

Come back to the latest version:

```bash
git checkout master
uv run --project .. dvc checkout
cat data.csv
```

Output:

```
id,value
1,10
2,20
3,30
5,50
```

Four rows again. Version 2 restored.

---

## What just happened

You switched between two data versions — without ever managing which CSV file was which.

- **Git** handled the code and the `.dvc` pointer.
- **DVC** handled the actual data.
- **`git checkout` + `dvc checkout`** is the complete operation.

The script that reads the data — `sample-code.py` — never changed. Only the data underneath it did. And DVC guaranteed the right version was present at the right time.

---

## Quick Reference

The commands you'll use most.

### Set up a new DVC project

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

### Official documentation

- DVC concepts: https://dvc.org/doc/user-guide/concepts
- Get started: https://dvc.org/doc/start
- Data versioning: https://dvc.org/doc/start/data-management
- Remotes: https://dvc.org/doc/user-guide/data-management/remote-storage
- DVC + git: https://dvc.org/doc/user-guide/how-it-works

---

## Checkpoint

You should now be able to answer:

- **What are the two meaningful commits?** The commit with message `add data and track data with dvc` (version 1, 3 rows) and the commit with message `added new row` (version 2, 4 rows).
- **What changes in git between the two versions?** Only the hash inside `data.csv.dvc`.
- **What does `git checkout <v1-hash>` alone do?** Restores the `.dvc` pointer. The data on disk is still version 2 until you run `dvc checkout`.
- **What does `dvc checkout` do after that?** Reads the pointer and restores the matching data from the cache.
- **What's the complete operation to switch data versions?** `git checkout <commit>` + `dvc checkout`.
- **Why `uv run --project ..`?** The demo's git repo lives in `sample/`, but the venv lives one level up.

**Next:** [Key Concepts →](04-key-concepts.md)