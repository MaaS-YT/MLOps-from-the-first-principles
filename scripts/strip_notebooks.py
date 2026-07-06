"""
Strip outputs and execution counts from all notebooks in the repo.

Run before committing:
    python scripts/strip_notebooks.py

Or via make:
    make clean-notebooks

Uses only nbformat (already in the project deps) — no nbconvert needed.
"""

import sys
import argparse
from pathlib import Path

try:
    import nbformat
except ImportError:
    print("nbformat not found. Run: uv pip install nbformat", file=sys.stderr)
    sys.exit(1)


def strip_notebook(path: Path) -> bool:
    """Strip outputs and execution counts from a single notebook. Returns True if changed."""
    nb = nbformat.read(path, as_version=4)
    changed = False

    for cell in nb.cells:
        if cell.cell_type != "code":
            continue

        # Clear outputs (print statements, display(), plots, error tracebacks)
        if cell.get("outputs"):
            cell["outputs"] = []
            changed = True

        # Clear execution count — removes the [In: 47] numbering
        if cell.get("execution_count") is not None:
            cell["execution_count"] = None
            changed = True

    if changed:
        nbformat.write(nb, path)

    return changed


def main():
    parser = argparse.ArgumentParser(description="Strip notebook outputs")
    parser.add_argument(
        "--check",
        action="store_true",
        help="Check only — exit 1 if any notebook has outputs (for CI)",
    )
    parser.add_argument(
        "paths",
        nargs="*",
        help="Specific notebooks to strip (default: all in repo)",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).parent.parent

    if args.paths:
        notebooks = [Path(p) for p in args.paths]
    else:
        notebooks = sorted(
            p for p in repo_root.rglob("*.ipynb")
            if ".venv" not in p.parts
            and ".ipynb_checkpoints" not in p.parts
            and "node_modules" not in p.parts
        )

    dirty = []
    cleaned = []

    for nb_path in notebooks:
        nb = nbformat.read(nb_path, as_version=4)
        has_output = any(
            cell.cell_type == "code" and (cell.get("outputs") or cell.get("execution_count"))
            for cell in nb.cells
        )

        if has_output:
            if args.check:
                dirty.append(nb_path)
            else:
                strip_notebook(nb_path)
                cleaned.append(nb_path)
                print(f"  cleaned  {nb_path.relative_to(repo_root)}")
        else:
            print(f"  ok       {nb_path.relative_to(repo_root)}")

    if args.check:
        if dirty:
            print("\nNotebooks with uncommitted outputs:")
            for p in dirty:
                print(f"  {p.relative_to(repo_root)}")
            print("\nRun `make clean-notebooks` before committing.")
            sys.exit(1)
        else:
            print("All notebooks are clean.")
    else:
        if cleaned:
            print(f"\nStripped {len(cleaned)} notebook(s). Stage them with git add.")
        else:
            print("\nAll notebooks were already clean.")


if __name__ == "__main__":
    main()
