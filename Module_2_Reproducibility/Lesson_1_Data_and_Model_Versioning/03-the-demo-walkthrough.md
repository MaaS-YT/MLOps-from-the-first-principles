# 03 — The Demo Walkthrough

> [← Previous: How DVC Works](02-how-dvc-works.md) · [Next: Key Concepts →](04-key-concepts.md)

---

## What the demo represents

`version-project/sample/` is a minimal but complete DVC project. Its git history tells a deliberate story.

```
14d6a51  init git and dvc         → project initialized, DVC set up
72cd2ff  Add sample script        → sample-code.py added (reads the CSV)
a164b3f  add data and track dvc   → data.csv (3 rows) tracked for the first time
99ed495  config the local remote  → mock-remote configured as DVC remote
3d104de  added new row            → data.csv updated to 4 rows, new .dvc hash recorded
```

The two meaningful commits are `a164b3f` and `3d104de`. Each represents a dataset version. **The `.dvc` file in git is the version marker.**

---

## The `.dvc` file at each version

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

Output:

```
3d104de added new row
99ed495 config the local remote
a164b3f add data and track dvc
72cd2ff Add sample script
14d6a51 init git and dvc
```

Now go back to version 1:

```bash
git checkout a164b3f
dvc checkout
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
git checkout main
dvc checkout
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

## Checkpoint

You should now be able to answer:

- **What are the two meaningful commits?** `a164b3f` (version 1, 3 rows) and `3d104de` (version 2, 4 rows).
- **What changes in git between the two versions?** Only the hash inside `data.csv.dvc`.
- **What does `git checkout a164b3f` alone do?** Restores the `.dvc` pointer. The data on disk is still version 2.
- **What does `dvc checkout` do after that?** Reads the pointer and restores the matching data from the cache.
- **What's the complete operation to switch data versions?** `git checkout <commit>` + `dvc checkout`.

**Next:** [Key Concepts →](04-key-concepts.md)