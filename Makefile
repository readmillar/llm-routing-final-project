.PHONY: install format lint test run-final audit plots

install:
	pip install -r requirements.txt

format:
	black src tests run_experiments.py
	ruff check --fix src tests run_experiments.py

lint:
	ruff check src tests run_experiments.py

test:
	pytest -q

run-final:
	python run_experiments.py --data data/routerbench.csv --output-dir outputs_final --config config/final.yaml --time-limit 600 --max-cascades 500

audit:
	python -m src.audit --data data/routerbench.csv --output-dir outputs_final

plots:
	python run_experiments.py --output-dir outputs_final --only-plots
