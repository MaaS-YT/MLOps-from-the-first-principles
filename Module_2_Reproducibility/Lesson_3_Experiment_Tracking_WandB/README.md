# Lesson 2.3 — Experiment Tracking with Weights & Biases

| | |
|---|---|
| **Problem this solves** | Two problems MLflow doesn't solve out of the box: (1) your teammate can't see your local MLflow server; (2) you know which model won, but not which data version trained it. |
| **Mental model** | W&B is a cloud-native experiment tracker with one extra layer — **artifacts**. An artifact is any versioned file (dataset, model, evaluation). `use_artifact()` creates a lineage edge. The full chain — raw data → processed data → model — becomes a queryable graph. |
| **What the lecture demonstrates** | Two demos: (1) a two-stage sklearn pipeline with explicit artifact hand-off; (2) a three-stage LSTM pipeline showing real-time training curves, model checkpointing with aliases, and the full lineage graph in the W&B UI |
| **Where this fits** | W&B completes the **Training Layer** alongside MLflow. The key addition is artifact lineage — trace any production model back to the exact dataset version that produced it. |

---

## Files

| File | Purpose |
|------|---------|
| `WANDB_GUIDE.md` | Full guide: shared visibility problem, artifact concepts, LSTM time-series concepts |
| `sklearn_demo.ipynb` | Two-stage pipeline: data upload → training, with artifact hand-off |
| `ts_lstm_demo.ipynb` | Three-stage pipeline: ingest → preprocess → train LSTM, full lineage |

**Start with:** `WANDB_GUIDE.md`

---

## Prerequisites

```bash
wandb login    # free account at wandb.ai
```

Run with:
```bash
uv run --no-sync jupyter notebook sklearn_demo.ipynb
uv run --no-sync jupyter notebook ts_lstm_demo.ipynb
```
