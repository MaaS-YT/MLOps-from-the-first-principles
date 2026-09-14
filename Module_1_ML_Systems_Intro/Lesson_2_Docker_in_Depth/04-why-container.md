# 04 — Why Containers Exist

> [← Previous: Build the Training Pipeline](03_training_pipeline.md) · [Next: Dockerize the Training Job →](05_dockerize_training.md)

---

## Recap: What a Venv Solves

Step 01 and 02 established:

- A venv isolates **Python packages**
- `uv` locks them for reproducibility
- The environment travels with the repo via `pyproject.toml` + `uv.lock`

**What a venv does *not* solve:**

- The Python version itself
- System libraries (`libssl`, `libgomp`, CUDA drivers)
- OS-level packages (`apt install`)
- Environment variables, config files
- Anything outside `site-packages/`

If any of those differ between your laptop and the server, the venv won't save you.

---

## The Same Problem, One Level Up

Consider a real ML training job.

It needs:

| Layer | Example |
|-------|---------|
| Python packages | `scikit-learn`, `numpy` — *solved by venv* |
| Python interpreter | 3.11, not 3.10 — *not solved by venv* |
| System libs | `libgomp.so.1` for numpy — *not solved by venv* |
| OS packages | `build-essential` for some wheels — *not solved by venv* |
| Environment | `CUDA_VISIBLE_DEVICES=0` — *not solved by venv* |

A venv isolates **one row** of that table.

The other four rows are exactly what break when you deploy. You've seen it:

> "It worked on my laptop."  
> "The server has a different version of libssl."

---

## The First-Principles Solution, Generalized

If isolating Python packages works, isolate **everything above the kernel**.

```
host kernel
    │
    ├──▶ container A   (Ubuntu + Python 3.11 + numpy 1.24 + libssl 3.0)
    └──▶ container B   (Alpine + Node 20 + express)
```

A **container** is a process that:

1. Has its own filesystem (its own `/usr`, `/lib`, `/etc`)
2. Has its own network stack
3. Has its own process tree
4. Shares the **kernel** with the host

That's the whole trick. Same shape as a venv — **one level up**.

| | Virtual environment | Container |
|---|---|---|
| Isolates | `site-packages/` | The entire userspace |
| Shares with host | Interpreter, OS, libs | Kernel only |
| Artifact | `uv.lock` + `pyproject.toml` | **Image** |
| Runtime | `uv run ...` | `docker run ...` |
| Startup | Instant | Milliseconds |
| Size | MBs | MBs–100s of MBs |

A container is **not** a VM. It shares the host kernel. It is not "a small OS running inside your OS" — it's a process with a different *view* of the filesystem, network, and process tree.

---

## How the Isolation Actually Works

Linux gives us the primitives. Nothing else.

### Namespaces — what the process *sees*

| Namespace | Isolates |
|-----------|----------|
| `mnt` | Filesystem mounts — container has its own `/` |
| `pid` | Process tree — container sees itself as PID 1 |
| `net` | Network interfaces, ports, routing |
| `uts` | Hostname, domain name |
| `user` | UID/GID mappings |
| `ipc` | Shared memory, semaphores |

### cgroups — what the process *uses*

Cap CPU, memory, I/O. `--memory=512m` is a cgroup limit, enforced by the kernel.

### The image — what the process *runs*

A filesystem, frozen. Layers stacked. The kernel mounts it as the container's `/`.

Combine: namespaces + cgroups + image = a process that believes it has the machine to itself.

**No hypervisor. No guest kernel. Just Linux, configured precisely.**

---

## The Image Is the New Contract

In step 02, the contract was `uv.lock` — same lockfile → same environment.

Now the contract is the **image** — same image → same container, on any machine with a compatible kernel.

```
Dockerfile  →  Image  →  Container
(recipe)       (artifact)  (process)
```

- **Dockerfile** — the recipe. Instructions that build the image.
- **Image** — the artifact. Read-only. Layered. Portable.
- **Container** — a running instance. Ephemeral. Isolated.

This is the exact same shape as:

```
pyproject.toml  →  uv.lock  →  .venv/
```

Definition → pinned artifact → running environment.

**You already know this pattern. Docker is that pattern applied to the whole OS.**

---

## What Containers Are *Not*

Clear the common misconceptions before step 05:

| Misconception | Reality |
|---------------|---------|
| "A container is a VM" | No hypervisor. Shares the host kernel. |
| "Containers are secure by default" | Namespaces are isolation, not a security boundary. Sandboxing needs more (gVisor, Kata, seccomp). |
| "You need a full OS in the image" | That's why Alpine exists — 5 MB base. |
| "Docker is the only runtime" | Containerd, Podman, CRI-O. Docker is the tooling layer. |

---

## Checkpoint

You should now be able to answer:

- **What does a venv isolate vs a container?** One layer of the stack vs the entire userspace above the kernel.
- **What are the three mechanisms?** Namespaces (view), cgroups (resources), image (filesystem).
- **What's the artifact?** The image — the same role `uv.lock` played for Python packages.

**Next:** [Dockerize the Training Job →](05_dockerize_training.md) — the exact script from step 03, now inside an image.