import os
from pathlib import Path

import joblib
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier


OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", "."))


def main() -> None:
    X, y = load_iris(return_X_y=True)
    print(f"Loaded Iris: X={X.shape}, y={y.shape}")

    model = DecisionTreeClassifier(max_depth=3, random_state=42)
    model.fit(X, y)

    accuracy = model.score(X, y)
    print(f"Train accuracy: {accuracy:.3f}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUTPUT_DIR / "model.joblib"
    joblib.dump(model, out)
    print(f"Saved {out} ({out.stat().st_size} bytes)")


if __name__ == "__main__":
    main()