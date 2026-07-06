.PHONY: clean-notebooks check-notebooks install-hook help

# ── Notebook hygiene ─────────────────────────────────────────────────────────

clean-notebooks:
	@echo "Stripping notebook outputs..."
	@uv run --no-sync python scripts/strip_notebooks.py
	@echo "Done. Stage the cleaned notebooks: git add -u '*.ipynb'"

check-notebooks:
	@uv run --no-sync python scripts/strip_notebooks.py --check

# ── Pre-commit hook ──────────────────────────────────────────────────────────

install-hook:
	@GIT_HOOKS=$$(git rev-parse --git-dir)/hooks; \
	 cp scripts/pre-commit $$GIT_HOOKS/pre-commit && \
	 chmod +x $$GIT_HOOKS/pre-commit && \
	 echo "Pre-commit hook installed at $$GIT_HOOKS/pre-commit"

# ── Help ─────────────────────────────────────────────────────────────────────

help:
	@echo ""
	@echo "  make clean-notebooks   Strip outputs from all notebooks (run before git add)"
	@echo "  make check-notebooks   Exit 1 if any notebook has outputs (for CI)"
	@echo "  make install-hook      Install pre-commit hook (auto-strips on every commit)"
	@echo ""
