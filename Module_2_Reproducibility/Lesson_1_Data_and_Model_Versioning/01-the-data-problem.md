# 01 — The Data Problem

> [← Back to the ladder](README.md) · [Next: How DVC Works →](02-how-dvc-works.md)

---

## Why git alone isn't enough

Git is the standard tool for versioning code. It works because code files are:

- **Small** — kilobytes to megabytes
- **Text-based** — diffs are meaningful and readable
- **Safe to store in a repository** — one file, one commit, clean history

Data files are the opposite:

- **Large** — gigabytes to terabytes
- **Binary or opaque** — a diff of two CSVs tells you nothing useful
- **Dangerous to store in git** — one mistake bloats the repo permanently

If you commit a 2 GB dataset to git, it stays in history forever. Even if you delete it in a later commit, anyone who clones the repo still downloads 2 GB. Git LFS helps with the size, but it still couples your data to the code repository.

**The deeper problem is not size. It's traceability.**

---

## The actual failure mode

You train a model on `data_v3.csv`. You get 91% accuracy.

A colleague trains the same model on `data_v4.csv`. They get 88%.

- Why did it drop?
- What changed between v3 and v4?
- Who changed it, and when?

Without versioning, you cannot answer any of those questions. Without those answers, you cannot debug, reproduce, or audit your ML system.

This is the data link in the reproducibility chain — and it's broken.

```
data → code → environment → run → artifact → model
  ✗
```

---

## What DVC provides

DVC (Data Version Control) stores data files **outside of git** — in a cache and a remote — while putting a small **pointer file** (`.dvc`) into git in their place.

```
In git (small, committed):         In DVC cache / remote (large, not in git):
  data.csv.dvc                       the actual data.csv
  (contains the hash)                (stored by its content hash)
```

When you switch git branches or commits, the `.dvc` file changes. DVC reads the new hash and pulls the corresponding data version from the cache.

**Your data version is now tied to your git commit.** Checking out a commit restores both the code *and* the data that went with it — automatically.

---

## Git + DVC together

The two tools divide responsibility cleanly:

```
git       versions code, configs, notebooks, .dvc pointer files
DVC       versions data files, model artifacts, large binaries

git commit  ──►  captures code + .dvc pointer (the hash of the current data)
dvc push    ──►  uploads the actual data to the remote
dvc pull    ──►  downloads the data matching the current git commit
```

You never manage data versions by hand. Git commit plus `dvc checkout` is the complete operation.

---

## Where DVC fits in the progression

```
1. Files on disk               no versioning, overwritten, lost
2. Git for data                repo bloat, not practical beyond small files
3. DVC for data                versioned data tied to git commits    ← you are here
4. DVC + experiment tracking   link data versions to model metrics
5. Feature store               versioned, serving-ready features (Feast, next lesson)
```

DVC is rung three. Without it, rung four — linking data versions to model metrics — is impossible.

---

## Checkpoint

You should now be able to answer:

- **Why does git fail for data?** Size, binary format, no useful diffs, permanent repo bloat.
- **What's the deeper problem beyond size?** Traceability — which data trained which model.
- **What does DVC put in git?** A small pointer file (`.dvc`), not the data.
- **What does DVC put outside git?** The actual data, in a cache and a remote.
- **What's the payoff?** A git commit now restores both the code *and* the data. Automatically.

**Next:** [How DVC Works →](02-how-dvc-works.md)