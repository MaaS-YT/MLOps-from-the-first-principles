# 02 — How DVC Works

> [← Previous: The Data Problem](01-the-data-problem.md) · [Next: The Demo Walkthrough →](03-the-demo-walkthrough.md)

---

## The `.dvc` file is a pointer

When you tell DVC to track a file, it does three things:

1. Computes the file's MD5 hash
2. Copies the file into the local cache (`.dvc/cache/`)
3. Writes a `.dvc` file containing that hash

```yaml
# data.csv.dvc — this is what lives in git
outs:
- md5: bbd304055be67c211495c95f4e697eb2
  size: 24
  hash: md5
  path: data.csv
```

The `.dvc` file is tiny — a few lines of YAML — and goes into git.

The actual `data.csv` is added to `.gitignore`. Git never sees it.

**The core mechanism: git tracks the hash, DVC tracks the file.**

**Notice the shape:** `.dvc file` → `hash` → `checked-out data`. This is the same **definition → artifact → instance** triune you saw in Module 1 (`Dockerfile → image → container`). It'll appear again in Lesson 2.2 (script → run → re-run) and Lesson 2.3 (config → dashboard → teammate reproduces).

---

## The cache

The cache lives at `.dvc/cache/` inside your project by default. Files are stored by their content hash — the same way git stores objects.

```
.dvc/cache/files/md5/
  bb/d304055be67c211495c95f4e697eb2   ← version 1 of data.csv
  1a/2dc46fd6ee44c2ee01cdc0e67d067d   ← version 2 of data.csv
```

Two versions of the same file coexist in the cache under different hashes. Nothing is ever deleted from the cache. You can always restore any version.

---

## The remote

The cache is local. The **remote** is the shared storage your team pushes to and pulls from. It works exactly like a git remote — same concept, different data.

```
git remote     = GitHub / GitLab (stores code)
DVC remote     = S3 / GCS / Azure Blob / local path (stores data)
```

In this lesson's demo, the remote is a local directory (`mock-remote/`) simulating cloud storage. In production it would be an S3 bucket or similar.

```
[core]
    remote = local_remote
['remote "local_remote"']
    url = ../../mock-remote
```

- `dvc push` — upload from local cache → remote.
- `dvc pull` — download from remote → local cache → working directory.

---

## Switching between versions

This is where git and DVC work together as one system.

The demo has two commits, each with a different version of `data.csv`:

```
commit <hash>  "add data and track data with dvc"
  data.csv.dvc → md5: bbd304...   (3 rows: id=1, 2, 3)

commit <hash>  "added new row"
  data.csv.dvc → md5: 1a2dc4...   (4 rows: id=1, 2, 3, 5)
```

The actual hashes are dynamic — read them from `git log --oneline`.

To go back to version 1:

```bash
# Find the hash of the commit that added the data
git log --oneline
# → look for the commit with message "add data and track data with dvc"

git checkout <that-hash>   # .dvc file now points to the v1 hash
dvc checkout               # DVC reads the .dvc file, restores data.csv to 3 rows
```

To come back to version 2:

```bash
git checkout master        # .dvc file now points to the v2 hash
dvc checkout               # data.csv restored to 4 rows
```

You never manage which CSV file is which version by hand. **Git commit + `dvc checkout` is the complete operation.**

---

## Running DVC in this lesson

The `sample/` directory is a nested git repo inside the `version-project/` uv project. So every `dvc` command needs to run through the parent's venv:

```bash
# From inside version-project/sample/
uv run --project .. dvc status
uv run --project .. dvc checkout
uv run --project .. dvc push
```

The `--project ..` flag tells `uv` to use the venv at `version-project/.venv/`. Without it, `uv` walks up to the DDODS workspace and gets confused.

---

## Checkpoint

You should now be able to answer:

- **What's in a `.dvc` file?** A hash, a size, and the path — nothing else.
- **Where does the actual data live?** In the DVC cache (local) and the remote (shared).
- **What does `dvc checkout` do?** Reads the current `.dvc` file and restores the matching data.
- **What does `dvc push` do?** Uploads new data from the cache to the remote.
- **What's the triune here?** `.dvc` → hash + remote → checked-out data. Definition → artifact → instance.
- **Why `uv run --project ..`?** The demo's git repo lives in `sample/`, but the venv lives one level up.

**Next:** [The Demo Walkthrough →](03-the-demo-walkthrough.md)