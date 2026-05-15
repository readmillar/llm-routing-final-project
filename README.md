# INDENG 164 LLM Routing Final Project

This project implements the locked final-project specification for:

**Cheap First, Reliable Always: A Robust Cascading Optimization Framework for LLM Routing**

The code loads `data/routerbench.csv`, preserves the observed incomplete prompt-model grid, and solves:

- A0 weighted baseline
- A1 single-shot portfolio MILP
- A2 two-stage cascade MILP
- A3 robust reliability-aware cascade MILP

## Setup

Use Python 3.13:

```bash
python3.13 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## Run

```bash
.venv/bin/python run_experiments.py --data data/routerbench.csv --output-dir outputs --time-limit 60 --max-cascades 250
```

For a faster sanity run:

```bash
.venv/bin/python run_experiments.py --data data/routerbench.csv --output-dir outputs_quick --skip-a2 --skip-a3 --time-limit 20 --max-cascades 80
```

Regenerate plots from existing CSVs:

```bash
.venv/bin/python run_experiments.py --output-dir outputs --only-plots
```

## Outputs

Tables are written to `outputs/tables/`, figures to `outputs/figures/`, and JSON solution details to `outputs/solutions/`.

Required tables include `data_summary.csv`, `model_summary.csv`, `missing_pairs.csv`, `budget_grid.csv`, `a0_results.csv`, `a1_results.csv`, `a2_results.csv`, `a3_results.csv`, `summary_comparison.csv`, `domain_quality.csv`, and `selected_model_usage.csv`.

Required figures include `cost_quality_frontier.png`, `quality_vs_pool_size.png`, `selected_model_usage.png`, `domain_performance_comparison.png`, and `robustness_heatmap.png`.

## Modeling Notes

The code uses only observed prompt-model pairs from the CSV. It does not impute missing `MMLU-Pro` rows for `deepseek-v3.1-terminus`.

Cascade success is precomputed as:

```text
R[p,a] = r[p,m1] + (1-r[p,m1]) * rho * r[p,m2]
C[p,a] = c[p,m1] + (1-r[p,m1]) * c[p,m2]
Esc[p,a] = 1-r[p,m1]
```

with `rho=0.75` by default and `r[p,m] = normalized quality[p,m]`.
