.PHONY: install format lint test run-smoke run-final audit plots

SHELL := /bin/bash

PYTHON ?= $(shell if [ -x .venv/bin/python ]; then printf '%s\n' .venv/bin/python; elif command -v python3 >/dev/null 2>&1; then command -v python3; else command -v python; fi)
PIP ?= $(PYTHON) -m pip
PYTEST ?= $(PYTHON) -m pytest

install:
	$(PIP) install -r requirements.txt

format:
	$(PYTHON) -m black src tests run_experiments.py
	$(PYTHON) -m ruff check --fix src tests run_experiments.py

lint:
	$(PYTHON) -m ruff check src tests run_experiments.py

test:
	$(PYTEST) -q

run-smoke:
	@set -euo pipefail; \
	tmp="outputs_smoke.tmp"; backup="outputs_smoke.backup"; \
	rm -rf "$$tmp" "$$backup"; \
	$(PYTHON) run_experiments.py --data data/routerbench.csv --output-dir "$$tmp" --config config/smoke.yaml; \
	restore_backup() { \
		status=$$?; \
		if [ -e "$$backup" ] && [ ! -e outputs_smoke ]; then mv "$$backup" outputs_smoke; fi; \
		exit $$status; \
	}; \
	trap restore_backup EXIT; \
	trap 'exit 130' INT; \
	trap 'exit 143' TERM; \
	if [ -e outputs_smoke ]; then mv outputs_smoke "$$backup"; fi; \
	mv "$$tmp" outputs_smoke; \
	rm -rf "$$backup"; \
	trap - EXIT INT TERM

run-final:
	@set -euo pipefail; \
	tmp="outputs_final.tmp"; backup="outputs_final.backup"; \
	rm -rf "$$tmp" "$$backup"; \
	$(PYTHON) run_experiments.py --data data/routerbench.csv --output-dir "$$tmp" --config config/final.yaml; \
	restore_backup() { \
		status=$$?; \
		if [ -e "$$backup" ] && [ ! -e outputs_final ]; then mv "$$backup" outputs_final; fi; \
		exit $$status; \
	}; \
	trap restore_backup EXIT; \
	trap 'exit 130' INT; \
	trap 'exit 143' TERM; \
	if [ -e outputs_final ]; then mv outputs_final "$$backup"; fi; \
	mv "$$tmp" outputs_final; \
	rm -rf "$$backup"; \
	trap - EXIT INT TERM

audit:
	$(PYTHON) -m src.audit --data data/routerbench.csv --output-dir outputs_final

plots:
	$(PYTHON) run_experiments.py --output-dir outputs_final --config config/final.yaml --only-plots
