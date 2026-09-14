# 01 — Why Virtual Environments Exist

> [← Back to the ladder](README.md) · [Next: Enter `uv` →](02_uv.md)

---

## The Problem

You have two Python projects on your laptop.

- **Project A** needs `numpy==1.24`
- **Project B** needs `numpy==2.0`

There is one `site-packages` directory on your machine. Two versions of the same library cannot live in it.

You install `numpy==2.0`. Project A breaks.

This is the **dependency conflict problem**. It predates Python. Every language has a version of it.

---

## The First-Principles Solution

Isolate the packages per project.

```
system Python
    │
    ├──▶ project-A/.venv/site-packages  (numpy 1.24)
    └──▶ project-B/.venv/site-packages  (numpy 2.0)
```

A **virtual environment** is a directory containing:

1. A symlink (or copy) of the Python interpreter
2. Its own `site-packages/` where packages get installed
3. A `pyvenv.cfg` telling Python "when you run from here, look in *this* `site-packages`, not the system one"

That's the whole trick. **Nothing else changes.** Same kernel, same OS libraries, same shell. Only the Python package path is redirected.

---

## The Manual Version (No Tools)

You can build a venv by hand. Understanding this makes `uv` and `python -m venv` unmagical.

```bash
# 1. Make the directory
mkdir -p myproject/.venv/lib/python3.11/site-packages

# 2. Symlink the interpreter
ln -s $(which python3.11) myproject/.venv/bin/python

# 3. Write the config
cat > myproject/.venv/pyvenv.cfg <<EOF
home = /usr/bin
include-system-site-packages = false
version = 3.11.0
EOF

# 4. Activate it
export PATH="$PWD/myproject/.venv/bin:$PATH"

# 5. Prove it
which python    # → /path/to/myproject/.venv/bin/python
python -c "import sys; print(sys.prefix)"
                # → /path/to/myproject/.venv
```

**What you just built:** a directory that redirects Python's package lookup. Nothing more.

---

## What a Virtual Environment Is *Not*

Common misconceptions — clear them now, because containers will reuse this intuition:

| Misconception | Reality |
|---------------|---------|
| "It's a sandbox" | It's a path redirect. The process runs with your full user permissions. |
| "It isolates the OS" | No. Same kernel, same libc, same everything above Python. |
| "It's like a container" | It's *one layer* of what a container isolates. Containers go further. |
| "It's a security boundary" | No. Anyone in the venv can read `/etc/passwd`. |

A venv solves **one** problem: package version conflicts. Nothing else.

---

## The Unwritten Contract

A venv is only useful if it's **reproducible**. Two developers on two machines must build the same one.

That means:

1. Dependencies must be **declared** — a list of packages and versions
2. Installation must be **deterministic** — same input, same output
3. The lock must be **committed** — so CI and prod agree

`requirements.txt` was the first attempt. It works, but it's slow and it doesn't lock transitive dependencies cleanly. That's what `uv` fixes.

---

## Checkpoint

You should now be able to answer:

- **What does a venv isolate?** Python packages.
- **What does it *not* isolate?** Everything above Python — OS, libraries, environment.
- **Why does it exist?** Package version conflicts, one machine, multiple projects.

**Next:** [Enter `uv` →](02_uv.md) — the tool that makes this fast and reproducible.