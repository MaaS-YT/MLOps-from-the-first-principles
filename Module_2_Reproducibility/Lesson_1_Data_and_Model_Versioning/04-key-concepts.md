# 04 — Key Concepts

> [← Previous: The Demo Walkthrough](03-the-demo-walkthrough.md) · [Next: The Bigger Picture →](05-the-bigger-picture.md)

---

## Content addressing

DVC uses **content-addressed storage**: files are stored and retrieved by their hash, not by name or path. This is the same principle git uses for its object store.

Two consequences:

- **Deduplication** — identical files are stored only once, no matter how many paths point to them.
- **Uniqueness** — a hash uniquely identifies exactly one version of a file, forever.

If your model was trained on `md5: bbd304...`, you can always reproduce that training environment by checking out the git commit that points to that hash. No ambiguity. No "which version was it again?"

The hash is the identity. Everything else is a reference to it.

---

## What to track with git vs DVC

A simple rule: **if `git diff` produces a useful, readable diff → use git. If the file is too large or too binary for a useful diff → use DVC.**

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

The boundary isn't perfect — a small CSV could go either way — but the rule of thumb is reliable: text is git, everything else is DVC.

---

## DVC vs W&B artifacts vs MLflow models

All three version large files. The difference is scope and integration.

| | DVC | W&B Artifacts | MLflow log_model |
|---|---|---|---|
| **Integration** | git-native | W&B runs | MLflow runs |
| **Data + models** | Both | Both | Models primarily |
| **Offline use** | Yes (fully local) | Cloud required | Local or server |
| **Lineage graph** | Implicit (git history) | Native, visual | Per-run |
| **Best for** | Data versioning tied to code | Experiment artifacts | Model lifecycle |

DVC is the right tool when you want **data versioning to be first-class in your git workflow**, independent of any specific experiment tracking platform.

W&B and MLflow are the right tools when you want **artifacts tied to specific training runs** — you'll see this in Lessons 2.2 and 2.3.

They're not competitors. They solve different problems on the same chain.

---

## The triune, again

```
.dvc file    →  hash + remote    →  checked-out data
definition   →  artifact         →  instance
```

Every link in the reproducibility chain has this shape. You saw it first in Module 1 with Docker:

```
Dockerfile   →  image            →  container
```

You'll see it twice more in this module:

```
training script  →  run + metrics     →  re-run from same inputs   (MLflow)
run config       →  dashboard + lineage → teammate reproduces       (W&B)
```

Same shape. Different link. **Learn it once, recognize it everywhere.**

---

## Checkpoint

You should now be able to answer:

- **What's content addressing?** Storing files by their hash, not their name. Same principle as git objects.
- **What's the payoff of content addressing?** A hash uniquely identifies a version of a file — forever.
- **What's the git-vs-DVC rule?** Text → git. Large or binary → DVC.
- **How is DVC different from W&B and MLflow?** DVC versions data tied to git commits. W&B and MLflow version artifacts tied to training runs.
- **What shape do all three share?** definition → artifact → instance.

**Next:** [The Bigger Picture →](05-the-bigger-picture.md)