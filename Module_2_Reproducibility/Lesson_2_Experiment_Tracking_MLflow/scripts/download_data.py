"""
Download and sample the NYC Yellow Taxi dataset.

Run this once. It produces:
    data/sample_nyc_taxi.csv    (100,000 rows, ~7 MB)

After that, the file is tracked by DVC and the notebooks pull it.

Usage:
    uv run python scripts/download_data.py
"""

import os
import sys
from pathlib import Path

import pandas as pd


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SAMPLE_SIZE = 100_000
RANDOM_STATE = 42

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
OUTPUT_PATH = DATA_DIR / "sample_nyc_taxi.csv"

KAGGLE_DATASET = "elemento/nyc-yellow-taxi-trip-data"
KAGGLE_FILE = "yellow_tripdata_2016-01.csv"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def log(msg: str) -> None:
    print(f"[download] {msg}", flush=True)


def check_kagglehub() -> None:
    try:
        import kagglehub  # noqa: F401
    except ImportError:
        log("ERROR: kagglehub is not installed.")
        log("Install it with: uv add kagglehub")
        sys.exit(1)


def download_kaggle_dataset() -> Path:
    import kagglehub

    log(f"Downloading from Kaggle: {KAGGLE_DATASET}")
    path = kagglehub.dataset_download(KAGGLE_DATASET)
    log(f"Downloaded to: {path}")
    return Path(path)


def load_and_sample(raw_dir: Path, sample_size: int) -> pd.DataFrame:
    file_path = raw_dir / KAGGLE_FILE

    if not file_path.exists():
        log(f"ERROR: expected file not found: {file_path}")
        log(f"Files in {raw_dir}: {os.listdir(raw_dir)}")
        sys.exit(1)

    log(f"Reading {file_path}")
    log(f"Sampling {sample_size:,} rows...")

    chunks = pd.read_csv(file_path, chunksize=500_000, low_memory=False)
    df1 = next(chunks)
    df2 = next(chunks)
    df = pd.concat([df1, df2], ignore_index=True)

    log(f"Loaded {len(df):,} rows total")

    if len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=RANDOM_STATE)
        log(f"Sampled down to {len(df):,} rows")

    return df


def save_csv(df: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    size_mb = output_path.stat().st_size / (1024 * 1024)
    log(f"Saved: {output_path} ({size_mb:.1f} MB)")
    log(f"Rows: {len(df):,}")
    log(f"Columns: {list(df.columns)}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    log("=" * 60)
    log("NYC Yellow Taxi - download & sample")
    log("=" * 60)

    check_kagglehub()

    if OUTPUT_PATH.exists():
        size_mb = OUTPUT_PATH.stat().st_size / (1024 * 1024)
        log(f"Sample already exists: {OUTPUT_PATH} ({size_mb:.1f} MB)")
        log("Delete it and re-run if you want a fresh download.")
        return

    raw_dir = download_kaggle_dataset()
    df = load_and_sample(raw_dir, SAMPLE_SIZE)
    save_csv(df, OUTPUT_PATH)

    log("=" * 60)
    log("Done. Next steps:")
    log("  1. dvc add data/sample_nyc_taxi.csv")
    log("  2. dvc push")
    log("  3. git add data/sample_nyc_taxi.csv.dvc .gitignore")
    log("  4. git commit -m 'Track NYC taxi sample with DVC'")
    log("=" * 60)


if __name__ == "__main__":
    main()
