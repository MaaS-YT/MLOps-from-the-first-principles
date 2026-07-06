"""
DVC versioning workflow — live demonstration script.

Shows the complete lifecycle of versioning a dataset and model artifact:
  1. Initialize DVC in a git repo
  2. Track a data file (dvc add)
  3. Commit the .dvc pointer to git
  4. Simulate a data change → commit a new version
  5. Switch between versions (dvc checkout)

Run from the repo root:
    uv run python sample/sample-code.py

Prerequisites: git repo already initialized, remote storage configured.
For local testing the script uses a local directory as the DVC remote.
"""

import subprocess
import os
import hashlib
from pathlib import Path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def run(cmd: str, cwd: Path = None, check: bool = True) -> str:
    """Run a shell command and return stdout. Prints the command first."""
    print(f"\n$ {cmd}")
    result = subprocess.run(
        cmd, shell=True, capture_output=True, text=True, cwd=cwd
    )
    if result.stdout:
        print(result.stdout.rstrip())
    if result.returncode != 0 and check:
        print(result.stderr.rstrip())
        raise RuntimeError(f"Command failed: {cmd}")
    return result.stdout.strip()


def file_hash(path: Path) -> str:
    """MD5 of a file — what DVC uses to detect changes."""
    return hashlib.md5(path.read_bytes()).hexdigest()[:8]


# ---------------------------------------------------------------------------
# Setup: scratch workspace so the demo is self-contained
# ---------------------------------------------------------------------------

WORKSPACE = Path(__file__).parent / "_dvc_demo_workspace"
WORKSPACE.mkdir(exist_ok=True)

DATA_FILE = WORKSPACE / "train.csv"
REMOTE_DIR = WORKSPACE / "_remote"   # local directory acting as DVC remote

print("=" * 60)
print("DVC VERSIONING DEMO")
print("=" * 60)


# ---------------------------------------------------------------------------
# Step 1: Initialize git + DVC
#
# DVC layers on top of git. git tracks code and metadata (.dvc files,
# .dvc/config). DVC tracks large file content separately in the cache
# and optionally pushes to a remote (S3, GCS, local dir, ...).
# ---------------------------------------------------------------------------

print("\n── STEP 1: Initialize ──────────────────────────────────────")

run("git init .", cwd=WORKSPACE)
run("git config user.email demo@example.com", cwd=WORKSPACE)
run("git config user.name Demo", cwd=WORKSPACE)
run("dvc init", cwd=WORKSPACE)

# dvc init creates:
#   .dvc/           DVC internals (cache, config, tmp)
#   .dvcignore      like .gitignore but for DVC
# These are committed to git so the repo knows DVC is in use.
run("git add .", cwd=WORKSPACE)
run('git commit -m "init: git + dvc"', cwd=WORKSPACE)

# Add a local directory as the DVC remote.
# In production this is an S3 bucket, GCS bucket, or SSH server.
REMOTE_DIR.mkdir(exist_ok=True)
run(f"dvc remote add -d localremote {REMOTE_DIR.resolve()}", cwd=WORKSPACE)
run("git add .dvc/config", cwd=WORKSPACE)
run('git commit -m "config: add local DVC remote"', cwd=WORKSPACE)


# ---------------------------------------------------------------------------
# Step 2: Create and track version 1 of the dataset
#
# `dvc add <file>` does three things:
#   a) Copies the file content into the DVC cache (.dvc/cache/), keyed by
#      its MD5 hash (content-addressable storage — same content = same key).
#   b) Creates <file>.dvc — a tiny YAML file that records the MD5 and size.
#      This is the pointer git will track instead of the raw data.
#   c) Adds <file> to .gitignore so git never accidentally tracks the data.
# ---------------------------------------------------------------------------

print("\n── STEP 2: Track data v1 ───────────────────────────────────")

DATA_FILE.write_text("feature1,feature2,label\n1.2,3.4,0\n5.6,7.8,1\n")
print(f"\nCreated {DATA_FILE.name} (hash: {file_hash(DATA_FILE)})")

run(f"dvc add train.csv", cwd=WORKSPACE)

# What the .dvc file looks like — this is what git versions:
dvc_pointer = (WORKSPACE / "train.csv.dvc").read_text()
print(f"\ntrain.csv.dvc contents:\n{dvc_pointer}")

run("git add train.csv.dvc .gitignore", cwd=WORKSPACE)
run('git commit -m "data: add training set v1 (2 rows)"', cwd=WORKSPACE)

# Push data content to the remote.
# git push sends the .dvc pointer; dvc push sends the actual bytes.
run("dvc push", cwd=WORKSPACE)

V1_COMMIT = run("git rev-parse HEAD", cwd=WORKSPACE)
print(f"\nVersion 1 commit: {V1_COMMIT[:8]}")


# ---------------------------------------------------------------------------
# Step 3: Simulate a data update — version 2
#
# The data file changes (more rows added, perhaps a new data collection run).
# `dvc add` recalculates the hash and updates the .dvc pointer file.
# git diff on train.csv.dvc shows only the MD5 change — a clean, auditable
# record that the data changed without storing the binary diff.
# ---------------------------------------------------------------------------

print("\n── STEP 3: Update data → v2 ────────────────────────────────")

DATA_FILE.write_text(
    "feature1,feature2,label\n1.2,3.4,0\n5.6,7.8,1\n"
    "9.0,1.2,0\n3.4,5.6,1\n7.8,9.0,0\n"   # 3 new rows
)
print(f"\nUpdated {DATA_FILE.name} (hash: {file_hash(DATA_FILE)})")

run("dvc add train.csv", cwd=WORKSPACE)
run("git add train.csv.dvc", cwd=WORKSPACE)
run('git commit -m "data: add training set v2 (5 rows)"', cwd=WORKSPACE)
run("dvc push", cwd=WORKSPACE)

V2_COMMIT = run("git rev-parse HEAD", cwd=WORKSPACE)
print(f"\nVersion 2 commit: {V2_COMMIT[:8]}")

# Show git log — the data lineage is now in git history
print("\ngit log (data lineage):")
run("git log --oneline", cwd=WORKSPACE)


# ---------------------------------------------------------------------------
# Step 4: Switch back to version 1
#
# This is the key reproducibility operation: any collaborator or CI job can
# reconstruct the exact dataset used to train any model in git history.
#
# `git checkout <commit> -- train.csv.dvc` restores the v1 pointer.
# `dvc checkout` reads that pointer's MD5, looks up the file in the local
# cache (or fetches from remote if not cached), and restores the data file.
#
# The data file itself never goes through git — only the pointer does.
# This is why DVC can version 10 GB datasets without bloating the git repo.
# ---------------------------------------------------------------------------

print("\n── STEP 4: Switch back to v1 ───────────────────────────────")

print(f"\nCurrent data ({DATA_FILE.stat().st_size} bytes):")
print(DATA_FILE.read_text())

run(f"git checkout {V1_COMMIT} -- train.csv.dvc", cwd=WORKSPACE)
run("dvc checkout", cwd=WORKSPACE)   # restores data from cache using pointer

print(f"\nRestored v1 data ({DATA_FILE.stat().st_size} bytes):")
print(DATA_FILE.read_text())

# Restore to v2
run("git checkout main -- train.csv.dvc", cwd=WORKSPACE)
run("dvc checkout", cwd=WORKSPACE)
print(f"\nBack on v2 ({DATA_FILE.stat().st_size} bytes)")


# ---------------------------------------------------------------------------
# Step 5: Simulate a new collaborator (dvc pull)
#
# Scenario: teammate clones the repo and needs the data.
# They have the .dvc pointer from git but not the actual file.
# `dvc pull` fetches the content from the remote using the pointer's MD5.
#
# This is the workflow that makes DVC work across a team:
#   - Data scientists push data changes: dvc push + git push
#   - Teammates get the exact version: git pull + dvc pull
# ---------------------------------------------------------------------------

print("\n── STEP 5: Simulate dvc pull (new collaborator) ────────────")

# Remove the local data file to simulate a fresh clone
DATA_FILE.unlink()
print(f"\nDeleted {DATA_FILE.name} (simulating fresh git clone)")
print(f"File exists: {DATA_FILE.exists()}")

run("dvc pull", cwd=WORKSPACE)
print(f"\nAfter dvc pull — file exists: {DATA_FILE.exists()}")
print(f"Content restored ({DATA_FILE.stat().st_size} bytes)")


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print("""
git tracks:   train.csv.dvc  (pointer — tiny, text, diffable)
DVC tracks:   train.csv       (data — never in git)
Remote holds: the actual bytes, keyed by MD5

Key commands:
  dvc add <file>     track a file, create .dvc pointer
  dvc push           upload content to remote
  dvc pull           download content from remote
  dvc checkout       restore data to match current .dvc pointers

Reproducibility workflow:
  git checkout <model-commit>   restore the code + .dvc pointer
  dvc checkout                  restore the exact training data
  python train.py               reproduce the run
""")
