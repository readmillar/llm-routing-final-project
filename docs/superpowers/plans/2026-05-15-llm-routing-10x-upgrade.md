# LLM Routing 10x Upgrade Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Upgrade the INDENG 164 LLM routing repo into a reproducible, audited, professor-ready computational optimization study with generalized cascades, systematic robust search, matched comparisons, stress tests, diagnostics, and report artifacts.

**Architecture:** Keep the existing Pyomo-first pipeline, but split candidate generation, diagnostics, auditing, complementarity, metadata, stress testing, and report artifact generation into focused modules. A2/A3 use observed prompt-model availability only; cascades are precomputed into linear parameters; every solve writes status, diagnostics, and auditable solution records instead of crashing on infeasible grid points.

**Tech Stack:** Python 3.13, pandas, numpy, Pyomo, HiGHS/appsi_highs with CBC/GLPK fallback, matplotlib, pytest, Black, Ruff.

---

## Scope Check

The requested upgrade spans multiple subsystems: model formulations, experiment orchestration, validation, stress testing, plots, docs, and tests. This plan keeps them in one execution roadmap because the outputs must agree on shared policy identifiers, diagnostics, and final report tables. Each task is independently testable and commit-sized.

Three-stage cascades and A4 CVaR are placed after the required two-stage and A3 artifacts. Execute them only after `make run-final`, `make audit`, and `make test` pass for A0/A1/A2/A3, which respects the project rule that three-stage work must not precede complete two-stage outputs.

## File Structure

Repository root: `repo root`

- Create `pyproject.toml`: Black/Ruff/Pytest configuration.
- Create `Makefile`: reproducible command entrypoints.
- Create `config/final.yaml`: final pipeline settings recorded into the manifest.
- Create `docs/FORMULATION.md`: equations and model mapping for A0-A4.
- Create `docs/RUNBOOK.md`: exact reproduction workflow.
- Modify `requirements.txt`: add Black, Ruff, PyYAML.
- Modify `tests/conftest.py`: move reusable synthetic data fixtures out of `test_sanity.py`.
- Create `tests/test_cascades.py`: degenerate cascade, formula, and dominance tests.
- Create `tests/test_diagnostics.py`: solver diagnostics and result status tests.
- Create `tests/test_audit.py`: solution audit tests.
- Create `tests/test_complementarity.py`: recovery fallback tests.
- Create `tests/test_scenarios.py`: robust and stress scenario tests.
- Create `tests/test_metadata.py`: provider/storage metadata coverage tests.
- Create `tests/test_report_artifacts.py`: matched comparison, Pareto, and report-number tests.
- Create `src/cascade_generation.py`: cascade candidate construction and prompt-specific linear parameters.
- Create `src/complementarity.py`: empirical model-pair recovery estimation.
- Create `src/model_metadata.py`: provider/storage metadata inference and loading.
- Create `src/stress_testing.py`: L1-ball scenarios, Dirichlet stress scenarios, and policy evaluation.
- Create `src/audit.py`: post-solve feasibility audits and CLI.
- Create `src/pareto.py`: non-dominated filtering.
- Create `src/report_artifacts.py`: matched report tables, manifest, and report markdown.
- Create `src/pyomo_tail_risk.py`: A4 CVaR cascade model.
- Modify `src/solver_utils.py`: diagnostics, statuses, model stats.
- Modify `src/metrics.py`: depth-aware cascade usage and concentration metrics.
- Modify `src/pyomo_cascade.py`: consume generalized cascades and optional production constraints.
- Modify `src/pyomo_robust_cascade.py`: consume generalized cascades, add scenario generators, add lexicographic solve.
- Modify `src/experiments.py`: systematic A3 grid, matched comparisons, stress tests, audits, diagnostics, manifests.
- Modify `src/plots.py`: report-grade figures.
- Modify `run_experiments.py`: config path, final-output defaults, audit/plot modes.

## Task 1: Tooling, Formatting, and Reproducible Commands

**Files:**
- Create: `pyproject.toml`
- Create: `Makefile`
- Create: `config/final.yaml`
- Create: `docs/FORMULATION.md`
- Create: `docs/RUNBOOK.md`
- Modify: `requirements.txt`

- [ ] **Step 1: Add formatting and lint dependencies**

Append these lines to `requirements.txt`:

```text
black>=25.0
ruff>=0.8
PyYAML>=6.0
```

- [ ] **Step 2: Add `pyproject.toml`**

```toml
[tool.black]
line-length = 100
target-version = ["py313"]

[tool.ruff]
line-length = 100
target-version = "py313"

[tool.ruff.lint]
select = ["E", "F", "I", "UP", "B"]
ignore = ["B008"]

[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["."]
addopts = "-q"
```

- [ ] **Step 3: Add `Makefile`**

```makefile
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
```

- [ ] **Step 4: Add `config/final.yaml`**

```yaml
random_seed: 164
time_limit: 600
max_cascades: 500
a1:
  K: [1, 2, 3, 5, 8]
a2:
  K: [2, 3, 5, 8]
  Emax: [0.5, 0.75, 1.0]
a3:
  K: [3, 5, 8]
  budget_names: ["B_low", "B_mid", "B_high"]
  Emax: [0.5, 0.75, 1.0]
  floor_multiplier: [0.75, 0.8, 0.85, 0.9]
  lambda_slack: [0.01, 0.05, 0.1, 0.25, 0.5]
  rho: [0.5, 0.75, 1.0]
stress:
  l1_radius: 0.4
  dirichlet_samples: 500
  concentration: 40.0
matched_report:
  K: 5
  budget_name: "B_mid"
  Emax: 0.75
production_constraints:
  storage_cap_gb: null
  provider_pool_caps: {}
  provider_traffic_caps: {}
```

- [ ] **Step 5: Add docs shells with concrete reproduction commands**

`docs/RUNBOOK.md`:

```markdown
# Runbook

1. Install dependencies: `make install`
2. Run tests: `make test`
3. Generate final outputs: `make run-final`
4. Audit report-selected policies: `make audit`
5. Regenerate plots from saved tables: `make plots`

Final artifacts are written to `outputs_final/tables`, `outputs_final/figures`, and `outputs_final/solutions`.
```

`docs/FORMULATION.md`:

```markdown
# Formulation Map

A1 uses `x[p,m]` only for observed pairs `(p,m) in E`.
A2 and A3 use `z[p,a]` only for prompt-feasible cascades `a in A_p`.
Single-stage cascades have depth 1 and escalation 0, so A2/A3 contain A1 as a special case when production and robustness constraints are inactive.

For a two-stage cascade `(m1,m2)`, the default linear parameters are:

`R[p,a] = r[p,m1] + (1-r[p,m1]) * rho * r[p,m2]`
`C[p,a] = c[p,m1] + (1-r[p,m1]) * c[p,m2]`
`Esc[p,a] = 1-r[p,m1]`
```

- [ ] **Step 6: Run formatting and tests**

Run:

```bash
make format
make lint
make test
```

Expected:

```text
black reformats source files or reports unchanged files
ruff check passes
pytest passes or skips solver-dependent tests only when no MILP solver is available
```

- [ ] **Step 7: Commit**

```bash
git add requirements.txt pyproject.toml Makefile config/final.yaml docs/FORMULATION.md docs/RUNBOOK.md
git commit -m "chore: add reproducible tooling and runbook"
```

## Task 2: Shared Synthetic Fixtures and Solver Diagnostics

**Files:**
- Modify: `tests/conftest.py`
- Create: `tests/test_diagnostics.py`
- Modify: `src/solver_utils.py`
- Modify: `src/pyomo_single_shot.py`
- Modify: `src/pyomo_cascade.py`
- Modify: `src/pyomo_robust_cascade.py`

- [ ] **Step 1: Move reusable synthetic fixtures into `tests/conftest.py`**

Add this below the existing `sys.path` setup:

```python
import pandas as pd
import pytest


@pytest.fixture
def synthetic_csv(tmp_path):
    rows = [
        ("AIME", "p1", "free-small", 0.0, 0.0),
        ("AIME", "p1", "cheap-solid", 1.0, 0.1),
        ("AIME", "p1", "balanced", 1.0, 0.5),
        ("AIME", "p1", "strong", 1.0, 2.0),
        ("AIME", "p2", "free-small", 0.0, 0.0),
        ("AIME", "p2", "cheap-solid", 1.0, 0.1),
        ("AIME", "p2", "balanced", 0.0, 0.5),
        ("AIME", "p2", "strong", 1.0, 2.0),
        ("LCB", "p3", "free-small", 0.0, 0.0),
        ("LCB", "p3", "cheap-solid", 0.0, 0.1),
        ("LCB", "p3", "balanced", 1.0, 0.5),
        ("LCB", "p3", "strong", 1.0, 2.0),
        ("GPQA", "p4", "free-small", 1.0, 0.0),
        ("GPQA", "p4", "cheap-solid", 0.0, 0.1),
        ("GPQA", "p4", "balanced", 1.0, 0.5),
        ("GPQA", "p4", "strong", 1.0, 2.0),
    ]
    df = pd.DataFrame(rows, columns=["dataset", "prompt_id", "model", "score", "cost"])
    path = tmp_path / "synthetic_routerbench.csv"
    df.to_csv(path, index=False)
    return path


@pytest.fixture
def synthetic_data(synthetic_csv, tmp_path):
    from src.load_data import load_dataset

    return load_dataset(str(synthetic_csv), output_dir=str(tmp_path / "outputs"))
```

Then remove duplicate `synthetic_csv` and `synthetic_data` fixture definitions from `tests/test_sanity.py`.

- [ ] **Step 2: Write failing diagnostics tests**

Create `tests/test_diagnostics.py`:

```python
import pyomo.environ as pyo


def test_collect_model_stats_counts_binary_variables_and_constraints():
    from src.solver_utils import collect_model_stats

    model = pyo.ConcreteModel()
    model.I = pyo.Set(initialize=[1, 2])
    model.x = pyo.Var(model.I, within=pyo.Binary)
    model.y = pyo.Var(within=pyo.NonNegativeReals)
    model.c = pyo.Constraint(expr=sum(model.x[i] for i in model.I) + model.y <= 2)
    model.obj = pyo.Objective(expr=model.y)

    stats = collect_model_stats(model)

    assert stats["num_variables"] == 3
    assert stats["num_binary_variables"] == 2
    assert stats["num_constraints"] == 1


def test_normalize_status_distinguishes_time_limited_feasible():
    from src.solver_utils import normalize_status

    assert normalize_status("optimal", has_incumbent=True) == "optimal"
    assert normalize_status("maxTimeLimit", has_incumbent=True) == "feasible_time_limited"
    assert normalize_status("maxTimeLimit", has_incumbent=False) == "no_solution"
    assert normalize_status("infeasible", has_incumbent=False) == "infeasible"
```

- [ ] **Step 3: Implement solver diagnostics helpers**

Add to `src/solver_utils.py`:

```python
import time


SUCCESS_STATUSES = {"ok", "optimal", "feasible", "feasible_time_limited"}


def collect_model_stats(model):
    """Count active Pyomo model size metrics for diagnostics tables."""
    variables = list(model.component_data_objects(pyo.Var, active=True))
    constraints = list(model.component_data_objects(pyo.Constraint, active=True))
    binaries = [var for var in variables if var.is_binary()]
    return {
        "num_variables": len(variables),
        "num_binary_variables": len(binaries),
        "num_constraints": len(constraints),
    }


def _safe_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def normalize_status(termination_condition, has_incumbent=False):
    """Map solver termination to project statuses that separate optimality from incumbents."""
    text = str(termination_condition).lower()
    if "optimal" in text:
        return "optimal"
    if "infeasible" in text:
        return "infeasible"
    if "feasible solution was not found" in text or "no solution" in text:
        return "no_solution"
    if "max" in text or "time" in text:
        return "feasible_time_limited" if has_incumbent else "no_solution"
    if "feasible" in text:
        return "feasible"
    return text.replace(" ", "_")


def extract_solver_diagnostics(policy, solver_name, results, model, wall_time_sec):
    """Build a CSV-safe diagnostics row for one solve attempt."""
    stats = collect_model_stats(model)
    solver = getattr(results, "solver", None)
    termination = getattr(solver, "termination_condition", "")
    status = getattr(solver, "status", "")
    problem = getattr(results, "problem", None)
    upper = _safe_float(getattr(problem, "upper_bound", None))
    lower = _safe_float(getattr(problem, "lower_bound", None))
    mip_gap = None
    if upper is not None and lower is not None and abs(upper) > 1e-12:
        mip_gap = abs(upper - lower) / abs(upper)
    row = {
        "policy": policy,
        "solver": solver_name,
        "solver_status": str(status),
        "termination_condition": str(termination),
        "wall_time_sec": wall_time_sec,
        "best_bound": lower,
        "objective_value": upper,
        "mip_gap": mip_gap,
    }
    row.update(stats)
    return row
```

- [ ] **Step 4: Time solves in `solve_model`**

Change `solve_model` to return `(solver_name, results, diagnostics_base)`:

```python
def solve_model(model, time_limit=300, policy=""):
    """Solve a Pyomo model and return solver results with diagnostics."""
    solver_name, solver = get_solver(time_limit=time_limit)
    if solver is None:
        return None, None, {"policy": policy, "status": "no_solver"}
    start = time.perf_counter()
    try:
        results = solver.solve(model, tee=False, timelimit=time_limit)
    except TypeError:
        results = solver.solve(model, tee=False)
    except RuntimeError as exc:
        elapsed = time.perf_counter() - start
        failure = SolverFailure(str(exc).splitlines()[0])
        return solver_name, failure, extract_solver_diagnostics(policy, solver_name, failure, model, elapsed)
    elapsed = time.perf_counter() - start
    return solver_name, results, extract_solver_diagnostics(policy, solver_name, results, model, elapsed)
```

Update callers in `pyomo_single_shot.py`, `pyomo_cascade.py`, and `pyomo_robust_cascade.py` from:

```python
solver_name, results = solve_model(model, time_limit=time_limit)
```

to:

```python
solver_name, results, diagnostics = solve_model(model, time_limit=time_limit, policy=policy)
```

and include `"diagnostics": diagnostics` in every result dictionary.

- [ ] **Step 5: Keep status extraction backward-compatible**

Replace `result_status` with:

```python
def result_status(results):
    """Map Pyomo termination output to compact project statuses."""
    if results is None:
        return "no_solver"
    termination = str(results.solver.termination_condition)
    return normalize_status(termination, has_incumbent=True)
```

Replace `has_solution` with:

```python
def has_solution(status):
    """Return True when it is safe to extract variable values."""
    return status in SUCCESS_STATUSES
```

- [ ] **Step 6: Run diagnostics tests**

Run:

```bash
pytest tests/test_diagnostics.py -v
pytest tests/test_sanity.py -v
```

Expected:

```text
tests/test_diagnostics.py passes
existing solver tests still pass or skip only for missing solver
```

- [ ] **Step 7: Commit**

```bash
git add src/solver_utils.py src/pyomo_single_shot.py src/pyomo_cascade.py src/pyomo_robust_cascade.py tests/conftest.py tests/test_sanity.py tests/test_diagnostics.py
git commit -m "feat: capture solver diagnostics"
```

## Task 3: Generalized Cascade Generation with Degenerate Single-Stage Cascades

**Files:**
- Create: `src/cascade_generation.py`
- Create: `tests/test_cascades.py`
- Modify: `src/pyomo_cascade.py`
- Modify: `src/pyomo_robust_cascade.py`
- Modify: `src/metrics.py`

- [ ] **Step 1: Write failing cascade-generation tests**

Create `tests/test_cascades.py`:

```python
import pytest


def test_degenerate_cascades_cover_every_observed_prompt_model_pair(synthetic_data):
    from src.cascade_generation import generate_single_stage_cascades, precompute_cascade_parameters

    cascades = generate_single_stage_cascades(synthetic_data)
    params = precompute_cascade_parameters(synthetic_data, cascades, rho=0.75)

    single_pairs = {(row.m1, row.depth) for row in cascades.itertuples(index=False)}
    assert all((model, 1) in single_pairs for model in synthetic_data["M"])
    for prompt in synthetic_data["P"]:
        assert set(params["A_p"][prompt])
        available_single_models = {
            cascades.set_index("cascade_id").loc[cascade_id, "m1"]
            for cascade_id in params["A_p"][prompt]
            if cascades.set_index("cascade_id").loc[cascade_id, "depth"] == 1
        }
        assert available_single_models == set(synthetic_data["M_p"][prompt])


def test_single_stage_parameters_match_original_pair_values(synthetic_data):
    from src.cascade_generation import generate_single_stage_cascades, precompute_cascade_parameters

    cascades = generate_single_stage_cascades(synthetic_data)
    params = precompute_cascade_parameters(synthetic_data, cascades, rho=0.75)
    lookup = cascades.set_index("cascade_id")

    for prompt in synthetic_data["P"]:
        for cascade_id in params["A_p"][prompt]:
            row = lookup.loc[cascade_id]
            if row["depth"] == 1:
                model = row["m1"]
                assert params["R"][(prompt, cascade_id)] == synthetic_data["q"][(prompt, model)]
                assert params["C"][(prompt, cascade_id)] == synthetic_data["c"][(prompt, model)]
                assert params["Esc"][(prompt, cascade_id)] == 0.0


def test_two_stage_parameters_match_manual_formula(synthetic_data):
    from src.cascade_generation import generate_two_stage_cascades, precompute_cascade_parameters

    cascades = generate_two_stage_cascades(synthetic_data, rho=0.75, max_two_stage=20)
    params = precompute_cascade_parameters(synthetic_data, cascades, rho=0.75)
    row = cascades[cascades["depth"] == 2].iloc[0]
    prompt = params["A_p"][synthetic_data["P"][0]][0]
    cascade_id = row["cascade_id"]
    if (synthetic_data["P"][0], cascade_id) not in params["R"]:
        pytest.skip("First generated cascade is not feasible for the first synthetic prompt.")

    p = synthetic_data["P"][0]
    r1 = synthetic_data["r"][(p, row["m1"])]
    r2 = synthetic_data["r"][(p, row["m2"])]
    assert params["R"][(p, cascade_id)] == pytest.approx(r1 + (1 - r1) * 0.75 * r2)
    assert params["C"][(p, cascade_id)] == pytest.approx(
        synthetic_data["c"][(p, row["m1"])] + (1 - r1) * synthetic_data["c"][(p, row["m2"])]
    )
    assert params["Esc"][(p, cascade_id)] == pytest.approx(1 - r1)
```

- [ ] **Step 2: Implement single-stage and two-stage generators**

Create `src/cascade_generation.py`:

```python
from __future__ import annotations

import pandas as pd


def _empty_model(value):
    return "" if value is None else value


def summarize_models(data):
    """Summarize observed cost and quality by model without imputing missing rows."""
    df = data["df"]
    return (
        df.groupby("model", as_index=False)
        .agg(
            qbar=("q_norm", "mean"),
            cbar=("cost", "mean"),
            zero_cost_rows=("cost", lambda s: int((s == 0).sum())),
            rows=("prompt_id", "count"),
        )
        .sort_values(["cbar", "qbar"], ascending=[True, False])
    )


def generate_single_stage_cascades(data):
    """Create one degenerate cascade per model so cascades contain single-shot routing."""
    summary = summarize_models(data).set_index("model")
    rows = []
    for model in data["M"]:
        rows.append(
            {
                "cascade_id": f"s1::{model}",
                "depth": 1,
                "m1": model,
                "m2": "",
                "m3": "",
                "qbar_m1": summary.loc[model, "qbar"],
                "qbar_m2": 0.0,
                "qbar_m3": 0.0,
                "cbar_m1": summary.loc[model, "cbar"],
                "cbar_m2": 0.0,
                "cbar_m3": 0.0,
                "avg_R": summary.loc[model, "qbar"],
                "avg_C": summary.loc[model, "cbar"],
                "avg_Esc": 0.0,
                "feasible_prompts": len(data["M_p"].get(model, [])),
            }
        )
    return pd.DataFrame(rows)


def generate_two_stage_cascades(data, rho=0.75, max_two_stage=250):
    """Generate cheap-then-strong two-stage cascades over observed availability only."""
    summary = summarize_models(data).set_index("model")
    cost_cutoff = summary["cbar"].quantile(0.30)
    quality_cutoff = summary["qbar"].quantile(0.50)
    cheap = summary[(summary["cbar"] <= cost_cutoff) | (summary["zero_cost_rows"] > 0)].index
    strong = summary[summary["qbar"] >= quality_cutoff].index
    pm = set(data["PM"])
    rows = []
    for m1 in cheap:
        for m2 in strong:
            if m1 == m2 or summary.loc[m2, "qbar"] < summary.loc[m1, "qbar"]:
                continue
            feasible_prompts = [p for p in data["P"] if (p, m1) in pm and (p, m2) in pm]
            if not feasible_prompts:
                continue
            avg_r = sum(
                data["r"][(p, m1)] + (1 - data["r"][(p, m1)]) * rho * data["r"][(p, m2)]
                for p in feasible_prompts
            ) / len(feasible_prompts)
            avg_c = sum(
                data["c"][(p, m1)] + (1 - data["r"][(p, m1)]) * data["c"][(p, m2)]
                for p in feasible_prompts
            ) / len(feasible_prompts)
            avg_esc = sum(1 - data["r"][(p, m1)] for p in feasible_prompts) / len(feasible_prompts)
            rows.append(
                {
                    "depth": 2,
                    "m1": m1,
                    "m2": m2,
                    "m3": "",
                    "qbar_m1": summary.loc[m1, "qbar"],
                    "qbar_m2": summary.loc[m2, "qbar"],
                    "qbar_m3": 0.0,
                    "cbar_m1": summary.loc[m1, "cbar"],
                    "cbar_m2": summary.loc[m2, "cbar"],
                    "cbar_m3": 0.0,
                    "avg_R": avg_r,
                    "avg_C": avg_c,
                    "avg_Esc": avg_esc,
                    "feasible_prompts": len(feasible_prompts),
                }
            )
    cascades = pd.DataFrame(rows)
    if cascades.empty:
        return cascades
    low_cost_n = max(1, max_two_stage // 2)
    high_quality_n = max_two_stage - low_cost_n
    low_cost = cascades.sort_values(["avg_C", "avg_R"], ascending=[True, False]).head(low_cost_n)
    high_quality = cascades.sort_values(["avg_R", "avg_C"], ascending=[False, True]).head(high_quality_n)
    selected = pd.concat([low_cost, high_quality], ignore_index=True).drop_duplicates(["m1", "m2"])
    selected = selected.head(max_two_stage).reset_index(drop=True)
    selected.insert(0, "cascade_id", [f"s2::{row.m1}::{row.m2}" for row in selected.itertuples()])
    return selected
```

- [ ] **Step 3: Implement depth-aware parameter precomputation**

Add to `src/cascade_generation.py`:

```python
def _cascade_models(row):
    return [m for m in [row.m1, row.m2, row.m3] if isinstance(m, str) and m]


def _recovery_term(data, prompt, m1, m2, rho, recovery_lookup):
    if recovery_lookup is None:
        return rho * data["r"][(prompt, m2)]
    domain = data["prompt_domain"][prompt]
    item = recovery_lookup.get((m1, m2, domain))
    if item is None or item.get("fallback_level") == "global_rho":
        return rho * data["r"][(prompt, m2)]
    return float(item["recovery_rate"])


def precompute_cascade_parameters(data, cascades, rho=0.75, recovery_lookup=None):
    """Precompute prompt-cascade availability and linear R/C/Esc parameters."""
    pm = set(data["PM"])
    a_p = {prompt: [] for prompt in data["P"]}
    r_param = {}
    c_param = {}
    esc_param = {}
    esc2_param = {}
    esc3_param = {}
    for row in cascades.itertuples(index=False):
        models = _cascade_models(row)
        for prompt in data["P"]:
            if any((prompt, model) not in pm for model in models):
                continue
            a_p[prompt].append(row.cascade_id)
            r1 = data["r"][(prompt, row.m1)]
            c1 = data["c"][(prompt, row.m1)]
            if row.depth == 1:
                r_param[(prompt, row.cascade_id)] = r1
                c_param[(prompt, row.cascade_id)] = c1
                esc_param[(prompt, row.cascade_id)] = 0.0
                esc2_param[(prompt, row.cascade_id)] = 0.0
                esc3_param[(prompt, row.cascade_id)] = 0.0
            elif row.depth == 2:
                recovery = _recovery_term(data, prompt, row.m1, row.m2, rho, recovery_lookup)
                r_param[(prompt, row.cascade_id)] = r1 + (1 - r1) * recovery
                c_param[(prompt, row.cascade_id)] = c1 + (1 - r1) * data["c"][(prompt, row.m2)]
                esc_param[(prompt, row.cascade_id)] = 1 - r1
                esc2_param[(prompt, row.cascade_id)] = 1 - r1
                esc3_param[(prompt, row.cascade_id)] = 0.0
            else:
                r2_eff = _recovery_term(data, prompt, row.m1, row.m2, rho, recovery_lookup)
                r2_raw = data["r"][(prompt, row.m2)]
                r3_eff = _recovery_term(data, prompt, row.m2, row.m3, rho, recovery_lookup)
                fail_after_second = (1 - r1) * (1 - r2_eff)
                r_param[(prompt, row.cascade_id)] = r1 + (1 - r1) * r2_eff + fail_after_second * r3_eff
                c_param[(prompt, row.cascade_id)] = (
                    c1
                    + (1 - r1) * data["c"][(prompt, row.m2)]
                    + fail_after_second * data["c"][(prompt, row.m3)]
                )
                esc_param[(prompt, row.cascade_id)] = (1 - r1) + fail_after_second
                esc2_param[(prompt, row.cascade_id)] = 1 - r1
                esc3_param[(prompt, row.cascade_id)] = fail_after_second
    uncovered = [prompt for prompt, values in a_p.items() if not values]
    if uncovered:
        raise ValueError(f"No feasible cascades for prompts: {uncovered[:5]}")
    return {"A_p": a_p, "R": r_param, "C": c_param, "Esc": esc_param, "Esc2": esc2_param, "Esc3": esc3_param}


def generate_cascades(data, rho=0.75, max_cascades=250, recovery_lookup=None, include_three_stage=False):
    """Generate cascade candidates and parameters, always including single-stage cascades."""
    singles = generate_single_stage_cascades(data)
    two_stage_limit = max(0, max_cascades - len(singles))
    twos = generate_two_stage_cascades(data, rho=rho, max_two_stage=two_stage_limit)
    frames = [frame for frame in [singles, twos] if not frame.empty]
    cascades = pd.concat(frames, ignore_index=True)
    params = precompute_cascade_parameters(data, cascades, rho=rho, recovery_lookup=recovery_lookup)
    return cascades, params
```

- [ ] **Step 4: Make `src/pyomo_cascade.py` delegate generation**

At the top of `src/pyomo_cascade.py`, import:

```python
from .cascade_generation import generate_cascades, summarize_models
```

Remove local `_candidate_cascades` and local `generate_cascades` so the public `generate_cascades` name comes from `cascade_generation.py`.

- [ ] **Step 5: Update cascade metrics for missing later-stage models**

Replace stage usage logic in `cascade_assignment_metrics` in `src/metrics.py`:

```python
    usage = Counter()
    expected_second = Counter()
    expected_third = Counter()
    for prompt, cascade_id in assignment.items():
        row = cascade_lookup.loc[cascade_id]
        usage[row["m1"]] += 1.0
        if isinstance(row.get("m2", ""), str) and row["m2"]:
            expected_second[row["m2"]] += esc_param[(prompt, cascade_id)]
        if isinstance(row.get("m3", ""), str) and row["m3"]:
            expected_third[row["m3"]] += esc_param[(prompt, cascade_id)]
```

and include:

```python
        "expected_stage3_usage": dict(expected_third),
```

- [ ] **Step 6: Run cascade-generation tests**

Run:

```bash
pytest tests/test_cascades.py::test_degenerate_cascades_cover_every_observed_prompt_model_pair -v
pytest tests/test_cascades.py::test_single_stage_parameters_match_original_pair_values -v
pytest tests/test_cascades.py::test_two_stage_parameters_match_manual_formula -v
```

Expected:

```text
all three tests pass
```

- [ ] **Step 7: Commit**

```bash
git add src/cascade_generation.py src/pyomo_cascade.py src/pyomo_robust_cascade.py src/metrics.py tests/test_cascades.py
git commit -m "feat: generalize cascade candidates with single-stage options"
```

## Task 4: A2 Uses Generalized Cascades and Mathematically Contains A1

**Files:**
- Modify: `src/pyomo_cascade.py`
- Modify: `tests/test_cascades.py`

- [ ] **Step 1: Add failing A2-dominates-A1 synthetic test**

Append to `tests/test_cascades.py`:

```python
def test_a2_generalizes_a1_on_synthetic_data(synthetic_data):
    from src.cascade_generation import generate_cascades
    from src.pyomo_cascade import solve_a2
    from src.pyomo_single_shot import solve_a1

    a1 = solve_a1(synthetic_data, K=2, B=10.0, time_limit=20)
    if a1["status"] == "no_solver":
        pytest.skip(a1["message"])

    cascades, params = generate_cascades(synthetic_data, rho=0.75, max_cascades=40)
    a2 = solve_a2(
        synthetic_data,
        cascades,
        params["R"],
        params["C"],
        params["Esc"],
        params["A_p"],
        K=2,
        B=10.0,
        Emax=1.0,
        time_limit=20,
    )

    assert a2["status"] in {"optimal", "feasible", "feasible_time_limited"}
    assert a2["avg_quality"] + 1e-8 >= a1["avg_quality"]
```

- [ ] **Step 2: Remove A2 K lower bound**

In `solve_a2`, replace:

```python
    if K < 2:
        return {"policy": policy, "status": "infeasible", "message": "A2 requires K >= 2"}
```

with:

```python
    if K < 1:
        return {"policy": policy, "status": "infeasible", "message": "A2 requires K >= 1"}
```

- [ ] **Step 3: Link only non-empty cascade stages**

Replace the two fixed link rules with one indexed link set:

```python
    cascade_lookup = cascades.set_index("cascade_id")[["m1", "m2", "m3", "depth"]].to_dict("index")
    stage_links = []
    for prompt, cascade_id in pa:
        row = cascade_lookup[cascade_id]
        for model_name in [row["m1"], row["m2"], row["m3"]]:
            if isinstance(model_name, str) and model_name:
                stage_links.append((prompt, cascade_id, model_name))

    model.PAM = pyo.Set(dimen=3, initialize=stage_links)

    def link_stage_rule(mdl, prompt, cascade_id, model_name):
        return mdl.z[prompt, cascade_id] <= mdl.y[model_name]

    model.link_stage = pyo.Constraint(model.PAM, rule=link_stage_rule)
```

Delete `model.link_first` and `model.link_second`.

- [ ] **Step 4: Keep objective and constraints linear**

Confirm these expressions still use precomputed constants:

```python
    model.budget = pyo.Constraint(expr=sum(C[p, a] * model.z[p, a] for p, a in pa) / n_prompts <= B)
    model.escalation = pyo.Constraint(expr=sum(Esc[p, a] * model.z[p, a] for p, a in pa) / n_prompts <= Emax)
    model.objective = pyo.Objective(expr=sum(R[p, a] * model.z[p, a] for p, a in pa) / n_prompts, sense=pyo.maximize)
```

- [ ] **Step 5: Run A2 dominance and existing A2 tests**

Run:

```bash
pytest tests/test_cascades.py::test_a2_generalizes_a1_on_synthetic_data -v
pytest tests/test_sanity.py::test_a2_solution_uses_only_selected_models -v
```

Expected:

```text
A2 synthetic quality is at least A1 synthetic quality
selected cascade stages are all selected models
```

- [ ] **Step 6: Commit**

```bash
git add src/pyomo_cascade.py tests/test_cascades.py
git commit -m "feat: make cascade MILP contain single-shot routing"
```

## Task 5: A3 Grid Search, Selection Rule, and Matched Policy Comparison

**Files:**
- Modify: `src/experiments.py`
- Create: `tests/test_report_artifacts.py`

- [ ] **Step 1: Write failing selection-rule tests**

Create `tests/test_report_artifacts.py`:

```python
def test_a3_selection_prefers_status_eta_slack_quality_cost_escalation():
    from src.experiments import select_report_a3_policy

    rows = [
        {"policy": "feasible_high_eta", "status": "feasible", "eta": 0.91, "total_slack": 0.02, "avg_quality": 0.90, "avg_cost": 0.50, "escalation_rate": 0.50},
        {"policy": "optimal_lower_eta", "status": "optimal", "eta": 0.89, "total_slack": 0.00, "avg_quality": 0.88, "avg_cost": 0.40, "escalation_rate": 0.30},
        {"policy": "optimal_best", "status": "optimal", "eta": 0.91, "total_slack": 0.01, "avg_quality": 0.89, "avg_cost": 0.60, "escalation_rate": 0.20},
    ]

    chosen = select_report_a3_policy(rows)

    assert chosen["policy"] == "optimal_best"


def test_matched_report_table_contains_same_k_budget_and_emax():
    from src.experiments import build_report_main_comparison

    rows = [
        {"policy": "A1", "family": "A1", "K": 5, "budget_name": "B_mid", "Emax": None, "status": "optimal", "avg_quality": 0.8, "avg_cost": 0.2},
        {"policy": "A2", "family": "A2", "K": 5, "budget_name": "B_mid", "Emax": 0.75, "status": "optimal", "avg_quality": 0.9, "avg_cost": 0.3},
        {"policy": "A3", "family": "A3", "K": 5, "budget_name": "B_mid", "Emax": 0.75, "status": "optimal", "avg_quality": 0.85, "avg_cost": 0.25},
        {"policy": "wrong", "family": "A2", "K": 3, "budget_name": "B_low", "Emax": 0.5, "status": "optimal", "avg_quality": 0.99, "avg_cost": 0.9},
    ]

    table = build_report_main_comparison(rows, K=5, budget_name="B_mid", Emax=0.75)

    assert set(table["policy"]) == {"A1", "A2", "A3"}
    assert set(table["K"].dropna()) == {5}
    assert set(table["budget_name"].dropna()) == {"B_mid"}
```

- [ ] **Step 2: Add A3 selection helper**

Add to `src/experiments.py`:

```python
STATUS_RANK = {
    "optimal": 4,
    "feasible": 3,
    "feasible_time_limited": 2,
    "ok": 1,
}


def _total_slack(result):
    return float(sum(result.get("domain_slacks", {}).values()))


def select_report_a3_policy(results):
    """Select the report A3 policy using a documented lexicographic rule."""
    feasible = [r for r in results if r.get("status") in STATUS_RANK]
    if not feasible:
        return None
    return sorted(
        feasible,
        key=lambda r: (
            STATUS_RANK.get(r.get("status"), 0),
            float(r.get("eta") or 0.0),
            -float(r.get("total_slack", _total_slack(r))),
            float(r.get("avg_quality") or 0.0),
            -float(r.get("avg_cost") or 1e18),
            -float(r.get("escalation_rate") or 1e18),
        ),
        reverse=True,
    )[0]
```

- [ ] **Step 3: Add matched comparison helper**

Add to `src/experiments.py`:

```python
def build_report_main_comparison(rows, K=5, budget_name="B_mid", Emax=0.75):
    """Build apples-to-apples policy rows for the report comparison table."""
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    keep = []
    for row in df.to_dict("records"):
        family = row.get("family")
        if family in {"baseline", "A0"}:
            keep.append(row)
        elif family == "A1" and row.get("K") == K and row.get("budget_name") == budget_name:
            keep.append(row)
        elif family in {"A2", "A3", "A4"} and row.get("K") == K and row.get("budget_name") == budget_name and float(row.get("Emax")) == float(Emax):
            keep.append(row)
    out = pd.DataFrame(keep)
    columns = [
        "policy",
        "family",
        "K",
        "budget_name",
        "Emax",
        "avg_cost",
        "avg_quality",
        "worst_scenario_quality",
        "p05_stress_quality",
        "worst_domain_quality",
        "total_slack",
        "escalation_rate",
        "num_models_selected",
        "provider_count",
        "storage_gb",
        "status",
        "mip_gap",
        "wall_time_sec",
    ]
    for column in columns:
        if column not in out.columns:
            out[column] = None
    return out[columns]
```

- [ ] **Step 4: Replace A3 early break with full grid**

In `run_experiments`, replace the current A3 `attempts` loop with:

```python
        a3_grid = []
        for K in [3, 5, 8]:
            for budget_name, B in budgets.items():
                for Emax in [0.50, 0.75, 1.00]:
                    for floor_multiplier in [0.75, 0.80, 0.85, 0.90]:
                        for lambda_slack in [0.01, 0.05, 0.10, 0.25, 0.50]:
                            for rho in [0.50, 0.75, 1.00]:
                                a3_grid.append((K, budget_name, B, Emax, floor_multiplier, lambda_slack, rho))
        for K, budget_name, B, Emax, floor_multiplier, lambda_slack, rho in a3_grid:
            cascades_rho, params_rho = generate_cascades(data, rho=rho, max_cascades=max_cascades)
            floors = compute_domain_floors(data, multiplier=floor_multiplier)
            result = solve_a3(
                data,
                cascades_rho,
                params_rho["R"],
                params_rho["C"],
                params_rho["Esc"],
                params_rho["A_p"],
                scenarios,
                floors,
                K=K,
                B=B,
                Emax=Emax,
                lambda_slack=lambda_slack,
                time_limit=time_limit,
            )
            result.update(
                {
                    "budget_name": budget_name,
                    "floor_multiplier": floor_multiplier,
                    "lambda_slack": lambda_slack,
                    "rho": rho,
                    "total_slack": _total_slack(result),
                }
            )
            a3_results.append(result)
```

Do not include `break` anywhere in this A3 loop.

- [ ] **Step 5: Save A3 grid and best-policy tables**

After the A3 loop, write:

```python
        a3_grid_rows = [
            _summary_row(
                r,
                family="A3",
                K=r.get("K"),
                B=r.get("B"),
                Emax=r.get("Emax"),
                budget_name=r.get("budget_name"),
                eta=r.get("eta"),
                total_slack=r.get("total_slack"),
                floor_multiplier=r.get("floor_multiplier"),
                lambda_slack=r.get("lambda_slack"),
                rho=r.get("rho"),
            )
            for r in a3_results
        ]
        pd.DataFrame(a3_grid_rows).to_csv(root / "tables" / "a3_grid_results.csv", index=False)
        best_report = select_report_a3_policy(a3_results)
        if best_report is not None:
            pd.DataFrame([_summary_row(best_report, family="A3", K=best_report.get("K"), B=best_report.get("B"), Emax=best_report.get("Emax"), budget_name=best_report.get("budget_name"), eta=best_report.get("eta"), total_slack=best_report.get("total_slack"))]).to_csv(root / "tables" / "a3_best_report_policy.csv", index=False)
```

- [ ] **Step 6: Run selection tests**

Run:

```bash
pytest tests/test_report_artifacts.py::test_a3_selection_prefers_status_eta_slack_quality_cost_escalation -v
pytest tests/test_report_artifacts.py::test_matched_report_table_contains_same_k_budget_and_emax -v
```

Expected:

```text
both tests pass
```

- [ ] **Step 7: Commit**

```bash
git add src/experiments.py tests/test_report_artifacts.py
git commit -m "feat: search and select A3 robust policies systematically"
```

## Task 6: Post-Solve Audit Tables

**Files:**
- Create: `src/audit.py`
- Create: `tests/test_audit.py`
- Modify: `src/experiments.py`

- [ ] **Step 1: Write failing audit test**

Create `tests/test_audit.py`:

```python
def test_audit_assignment_budget_and_observed_pairs_for_single_shot(synthetic_data):
    from src.audit import audit_single_shot_result
    from src.baselines import solve_always_cheapest

    result = solve_always_cheapest(synthetic_data)
    rows = audit_single_shot_result(synthetic_data, result, K=4, B=10.0)

    assert rows
    assert all(row["passed"] for row in rows)
    assert {row["check_name"] for row in rows} >= {
        "assignment_completeness",
        "observed_pairs_only",
        "budget",
        "pool_size",
    }


def test_audit_detects_budget_violation(synthetic_data):
    from src.audit import audit_single_shot_result
    from src.baselines import solve_always_best_quality

    result = solve_always_best_quality(synthetic_data)
    rows = audit_single_shot_result(synthetic_data, result, K=4, B=0.0)
    budget = [row for row in rows if row["check_name"] == "budget"][0]

    assert not budget["passed"]
    assert budget["violation"] > 0.0
```

- [ ] **Step 2: Implement audit row helper**

Create `src/audit.py`:

```python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .load_data import load_dataset


TOL = 1e-7


def audit_row(policy, check_name, lhs, rhs, sense="<=", tolerance=TOL):
    """Build one audit row with a signed violation."""
    if sense == "<=":
        violation = max(0.0, float(lhs) - float(rhs))
    elif sense == ">=":
        violation = max(0.0, float(rhs) - float(lhs))
    elif sense == "==":
        violation = abs(float(lhs) - float(rhs))
    else:
        raise ValueError(f"Unknown audit sense: {sense}")
    return {
        "policy": policy,
        "check_name": check_name,
        "passed": violation <= tolerance,
        "lhs": float(lhs),
        "sense": sense,
        "rhs": float(rhs),
        "violation": violation,
        "tolerance": tolerance,
    }
```

- [ ] **Step 3: Implement single-shot audit**

Add to `src/audit.py`:

```python
def audit_single_shot_result(data, result, K=None, B=None):
    """Audit an A0/A1-style prompt-to-model assignment."""
    policy = result.get("policy", "")
    assignment = result.get("assignment", {})
    rows = []
    rows.append(audit_row(policy, "assignment_completeness", len(set(assignment)), len(data["P"]), "=="))
    missing_pairs = sum(1 for p, m in assignment.items() if (p, m) not in set(data["PM"]))
    rows.append(audit_row(policy, "observed_pairs_only", missing_pairs, 0, "=="))
    if K is not None:
        rows.append(audit_row(policy, "pool_size", len(set(assignment.values())), K, "<="))
    if B is not None and assignment:
        avg_cost = sum(data["c"][(p, m)] for p, m in assignment.items()) / len(data["P"])
        rows.append(audit_row(policy, "budget", avg_cost, B, "<="))
    return rows
```

- [ ] **Step 4: Implement cascade audit**

Add to `src/audit.py`:

```python
def audit_cascade_result(data, cascades, params, result, K=None, B=None, Emax=None):
    """Audit an A2/A3-style prompt-to-cascade assignment."""
    policy = result.get("policy", "")
    assignment = result.get("cascade_assignment", {})
    rows = [audit_row(policy, "assignment_completeness", len(set(assignment)), len(data["P"]), "==")]
    lookup = cascades.set_index("cascade_id")
    selected = set(result.get("selected_models", []))
    unavailable = 0
    unlinked = 0
    for prompt, cascade_id in assignment.items():
        row = lookup.loc[cascade_id]
        models = [m for m in [row["m1"], row.get("m2", ""), row.get("m3", "")] if isinstance(m, str) and m]
        unavailable += sum(1 for model in models if (prompt, model) not in set(data["PM"]))
        unlinked += sum(1 for model in models if selected and model not in selected)
    rows.append(audit_row(policy, "observed_pairs_only", unavailable, 0, "=="))
    rows.append(audit_row(policy, "selected_model_linking", unlinked, 0, "=="))
    if K is not None:
        rows.append(audit_row(policy, "pool_size", len(selected), K, "<="))
    if B is not None and assignment:
        avg_cost = sum(params["C"][(p, a)] for p, a in assignment.items()) / len(data["P"])
        rows.append(audit_row(policy, "budget", avg_cost, B, "<="))
    if Emax is not None and assignment:
        avg_esc = sum(params["Esc"][(p, a)] for p, a in assignment.items()) / len(data["P"])
        rows.append(audit_row(policy, "escalation", avg_esc, Emax, "<="))
    return rows
```

- [ ] **Step 5: Add audit CLI over saved JSON solutions**

Add to `src/audit.py`:

```python
def _load_json(path):
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Audit saved LLM routing solutions.")
    parser.add_argument("--data", default="data/routerbench.csv")
    parser.add_argument("--output-dir", default="outputs_final")
    args = parser.parse_args(argv)
    root = Path(args.output_dir)
    data = load_dataset(args.data, output_dir=root)
    rows = []
    for file_name in ["baseline_assignments.json", "a1_solutions.json"]:
        payload = _load_json(root / "solutions" / file_name)
        for result in payload.values():
            if isinstance(result, dict) and "assignment" in result:
                rows.extend(audit_single_shot_result(data, result, K=result.get("K"), B=result.get("B")))
    out = pd.DataFrame(rows)
    (root / "tables").mkdir(parents=True, exist_ok=True)
    out.to_csv(root / "tables" / "solution_audit.csv", index=False)
    if not out.empty and not out["passed"].all():
        raise SystemExit("One or more audit checks failed. See solution_audit.csv.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 6: Write audit rows during final experiments**

In `run_experiments`, collect audit rows after each successful solve and write:

```python
    pd.DataFrame(audit_rows).to_csv(root / "tables" / "solution_audit.csv", index=False)
```

Use `audit_single_shot_result` for baselines/A1 and `audit_cascade_result` for A2/A3.

- [ ] **Step 7: Run audit tests**

Run:

```bash
pytest tests/test_audit.py -v
```

Expected:

```text
budget violation test fails exactly the budget audit row
all valid synthetic baseline audits pass
```

- [ ] **Step 8: Commit**

```bash
git add src/audit.py src/experiments.py tests/test_audit.py
git commit -m "feat: audit optimization solutions after solve"
```

## Task 7: Data-Derived Complementarity and Recovery Heatmap

**Files:**
- Create: `src/complementarity.py`
- Create: `tests/test_complementarity.py`
- Modify: `src/cascade_generation.py`
- Modify: `src/experiments.py`
- Modify: `src/plots.py`

- [ ] **Step 1: Write failing complementarity tests**

Create `tests/test_complementarity.py`:

```python
def test_pair_recovery_uses_domain_level_when_support_sufficient(synthetic_data):
    from src.complementarity import estimate_pair_recovery

    recovery = estimate_pair_recovery(synthetic_data, min_support=1, global_rho=0.75)
    row = recovery[
        (recovery["m1"] == "free-small")
        & (recovery["m2"] == "cheap-solid")
        & (recovery["domain"] == "AIME")
    ].iloc[0]

    assert row["support"] == 2
    assert row["fallback_level"] == "domain_pair"
    assert row["recovery_rate"] == 1.0


def test_pair_recovery_falls_back_to_global_rho_when_support_low(synthetic_data):
    from src.complementarity import estimate_pair_recovery

    recovery = estimate_pair_recovery(synthetic_data, min_support=999, global_rho=0.75)
    assert set(recovery["fallback_level"]) == {"global_rho"}
    assert set(recovery["recovery_rate"]) == {0.75}
```

- [ ] **Step 2: Implement recovery estimation**

Create `src/complementarity.py`:

```python
from __future__ import annotations

import pandas as pd


def _failure_prompts(data, domain, m1, fail_threshold):
    prompts = data["P_d"][domain]
    return [
        prompt
        for prompt in prompts
        if (prompt, m1) in data["q"] and data["q"][(prompt, m1)] <= fail_threshold
    ]


def estimate_pair_recovery(data, min_support=5, fail_threshold=0.0, global_rho=0.75):
    """Estimate P(model 2 succeeds | model 1 fails) by ordered pair and domain."""
    rows = []
    for domain in data["D"]:
        for m1 in data["M"]:
            failures = _failure_prompts(data, domain, m1, fail_threshold)
            for m2 in data["M"]:
                if m1 == m2:
                    continue
                domain_prompts = [p for p in failures if (p, m2) in data["q"]]
                if len(domain_prompts) >= min_support:
                    rate = sum(data["q"][(p, m2)] for p in domain_prompts) / len(domain_prompts)
                    level = "domain_pair"
                    support = len(domain_prompts)
                else:
                    pair_failures = [
                        p
                        for d in data["D"]
                        for p in _failure_prompts(data, d, m1, fail_threshold)
                        if (p, m2) in data["q"]
                    ]
                    if len(pair_failures) >= min_support:
                        rate = sum(data["q"][(p, m2)] for p in pair_failures) / len(pair_failures)
                        level = "pair"
                        support = len(pair_failures)
                    else:
                        rate = global_rho
                        level = "global_rho"
                        support = len(domain_prompts)
                rows.append(
                    {
                        "m1": m1,
                        "m2": m2,
                        "domain": domain,
                        "support": support,
                        "recovery_rate": float(rate),
                        "fallback_level": level,
                    }
                )
    return pd.DataFrame(rows)


def recovery_lookup_from_frame(frame):
    """Convert recovery table into a lookup consumed by cascade parameter generation."""
    return {
        (row.m1, row.m2, row.domain): {
            "support": int(row.support),
            "recovery_rate": float(row.recovery_rate),
            "fallback_level": row.fallback_level,
        }
        for row in frame.itertuples(index=False)
    }
```

- [ ] **Step 3: Use recovery lookup in experiments**

In `run_experiments`, after loading data:

```python
    recovery_df = estimate_pair_recovery(data, min_support=5, global_rho=0.75)
    recovery_df.to_csv(root / "tables" / "model_pair_recovery.csv", index=False)
    recovery_lookup = recovery_lookup_from_frame(recovery_df)
```

Pass `recovery_lookup=recovery_lookup` into `generate_cascades` for final A2/A3 runs that should use complementarity-aware parameters.

- [ ] **Step 4: Add complementarity heatmap plot**

Add to `src/plots.py`:

```python
def plot_model_complementarity_heatmap(root):
    recovery = _read_csv(root / "tables" / "model_pair_recovery.csv")
    plt.figure(figsize=(10, 8))
    if not recovery.empty:
        pair = (
            recovery.groupby(["m1", "m2"], as_index=False)
            .agg(recovery_rate=("recovery_rate", "mean"), support=("support", "sum"))
            .sort_values("support", ascending=False)
        )
        top_models = sorted(set(pair.head(20)["m1"]).union(pair.head(20)["m2"]))
        plot_df = pair[pair["m1"].isin(top_models) & pair["m2"].isin(top_models)]
        pivot = plot_df.pivot_table(index="m1", columns="m2", values="recovery_rate", aggfunc="mean")
        image = plt.imshow(pivot.fillna(0.0).values, aspect="auto", vmin=0, vmax=1, cmap="magma")
        plt.xticks(range(len(pivot.columns)), pivot.columns, rotation=60, ha="right", fontsize=7)
        plt.yticks(range(len(pivot.index)), pivot.index, fontsize=7)
        plt.colorbar(image, label="Recovery rate")
    plt.title("Model complementarity: P(second succeeds | first fails)")
    plt.tight_layout()
    plt.savefig(root / "figures" / "model_complementarity_heatmap.png", dpi=220)
    plt.close()
```

Call it from `make_all_plots`.

- [ ] **Step 5: Run complementarity tests**

Run:

```bash
pytest tests/test_complementarity.py -v
```

Expected:

```text
domain-pair support is used when min_support is 1
global rho fallback is used when min_support is impossible
```

- [ ] **Step 6: Commit**

```bash
git add src/complementarity.py src/cascade_generation.py src/experiments.py src/plots.py tests/test_complementarity.py
git commit -m "feat: estimate cascade complementarity from data"
```

## Task 8: Robust Scenario Generator and Monte Carlo Stress Testing

**Files:**
- Create: `src/stress_testing.py`
- Create: `tests/test_scenarios.py`
- Modify: `src/pyomo_robust_cascade.py`
- Modify: `src/experiments.py`
- Modify: `src/plots.py`

- [ ] **Step 1: Write failing scenario tests**

Create `tests/test_scenarios.py`:

```python
def test_l1_shift_scenarios_sum_to_one(synthetic_data):
    from src.stress_testing import build_l1_shift_scenarios

    scenarios = build_l1_shift_scenarios(synthetic_data, radius=0.4)

    assert scenarios
    for scenario in scenarios.values():
        assert abs(sum(scenario["domain_weights"].values()) - 1.0) <= 1e-8
        assert abs(sum(scenario["prompt_weights"].values()) - 1.0) <= 1e-8


def test_dirichlet_stress_scenarios_are_reproducible(synthetic_data):
    from src.stress_testing import sample_dirichlet_scenarios

    first = sample_dirichlet_scenarios(synthetic_data, n=5, concentration=10.0, seed=164)
    second = sample_dirichlet_scenarios(synthetic_data, n=5, concentration=10.0, seed=164)

    assert list(first) == list(second)
    assert first["stress_000"]["domain_weights"] == second["stress_000"]["domain_weights"]
```

- [ ] **Step 2: Implement robust stress scenario helpers**

Create `src/stress_testing.py`:

```python
from __future__ import annotations

import numpy as np
import pandas as pd

from .metrics import scenario_quality, scenario_weights


def empirical_domain_weights(data):
    return {domain: len(data["P_d"][domain]) / len(data["P"]) for domain in data["D"]}


def build_l1_shift_scenarios(data, radius=0.4):
    """Generate finite L1-ball extreme domain shifts around empirical traffic."""
    base = empirical_domain_weights(data)
    scenarios = {}
    move = min(0.5, max(0.0, radius / 2.0))
    for target in data["D"]:
        weights = dict(base)
        donors = [domain for domain in data["D"] if domain != target]
        remaining_move = move
        for donor in sorted(donors, key=lambda d: weights[d], reverse=True):
            take = min(weights[donor], remaining_move)
            weights[donor] -= take
            weights[target] += take
            remaining_move -= take
            if remaining_move <= 1e-12:
                break
        total = sum(weights.values())
        normalized = {domain: value / total for domain, value in weights.items()}
        scenarios[f"l1_shift_to_{target}"] = {
            "domain_weights": normalized,
            "prompt_weights": scenario_weights(data["P"], data["prompt_domain"], normalized),
        }
    return scenarios


def sample_dirichlet_scenarios(data, n=500, concentration=40.0, seed=164):
    """Sample prompt-mix stress scenarios from a Dirichlet around empirical weights."""
    rng = np.random.default_rng(seed)
    domains = list(data["D"])
    base = np.array([empirical_domain_weights(data)[domain] for domain in domains], dtype=float)
    alpha = np.maximum(base * concentration, 1e-6)
    draws = rng.dirichlet(alpha, size=n)
    scenarios = {}
    for idx, draw in enumerate(draws):
        weights = {domain: float(draw[i]) for i, domain in enumerate(domains)}
        scenarios[f"stress_{idx:03d}"] = {
            "domain_weights": weights,
            "prompt_weights": scenario_weights(data["P"], data["prompt_domain"], weights),
        }
    return scenarios


def evaluate_policy_under_scenarios(policy_result, scenarios, value_lookup, cost_lookup):
    """Evaluate one fixed assignment under many prompt-weight scenarios."""
    assignment = policy_result.get("cascade_assignment", policy_result.get("assignment", {}))
    rows = []
    for name, scenario in scenarios.items():
        weights = scenario["prompt_weights"]
        rows.append(
            {
                "policy": policy_result["policy"],
                "scenario": name,
                "avg_quality": scenario_quality(weights, assignment, value_lookup),
                "avg_cost": scenario_quality(weights, assignment, cost_lookup),
            }
        )
    return pd.DataFrame(rows)
```

- [ ] **Step 3: Extend A3 scenario set with named plus L1 scenarios**

In `pyomo_robust_cascade.py`, import:

```python
from .stress_testing import build_l1_shift_scenarios
```

At the end of `build_scenarios`, before returning:

```python
    scenarios.update(build_l1_shift_scenarios(data, radius=0.4))
```

- [ ] **Step 4: Write stress-test outputs in experiments**

After selecting representative policies in `run_experiments`, add:

```python
    stress_scenarios = sample_dirichlet_scenarios(data, n=500, concentration=40.0, seed=164)
    stress_rows = []
    for result in representative:
        if result is None or result.get("status") not in {"ok", "optimal", "feasible", "feasible_time_limited"}:
            continue
        if "cascade_assignment" in result:
            stress_rows.extend(
                evaluate_policy_under_scenarios(result, stress_scenarios, params["R"], params["C"]).to_dict("records")
            )
        elif "assignment" in result:
            stress_rows.extend(
                evaluate_policy_under_scenarios(result, stress_scenarios, data["q"], data["c"]).to_dict("records")
            )
    stress_df = pd.DataFrame(stress_rows)
    stress_df.to_csv(root / "tables" / "stress_test_results.csv", index=False)
```

- [ ] **Step 5: Add stress distribution plot**

Add to `src/plots.py`:

```python
def plot_stress_test_quality_distribution(root):
    stress = _read_csv(root / "tables" / "stress_test_results.csv")
    plt.figure(figsize=(9, 5.5))
    if not stress.empty:
        policies = list(stress.groupby("policy")["avg_quality"].mean().sort_values(ascending=False).head(6).index)
        data = [stress.loc[stress["policy"] == policy, "avg_quality"].values for policy in policies]
        plt.boxplot(data, labels=policies, showfliers=False)
        plt.xticks(rotation=30, ha="right")
        plt.ylabel("Average quality under sampled traffic mix")
    plt.title("Stress-test quality distribution")
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(root / "figures" / "stress_test_quality_distribution.png", dpi=220)
    plt.close()
```

Call it from `make_all_plots`.

- [ ] **Step 6: Run scenario tests**

Run:

```bash
pytest tests/test_scenarios.py -v
```

Expected:

```text
L1 and Dirichlet scenario weights sum to one and are reproducible
```

- [ ] **Step 7: Commit**

```bash
git add src/stress_testing.py src/pyomo_robust_cascade.py src/experiments.py src/plots.py tests/test_scenarios.py
git commit -m "feat: stress test policies under traffic mix uncertainty"
```

## Task 9: Provider and Storage Metadata with Production Constraints

**Files:**
- Create: `src/model_metadata.py`
- Create: `tests/test_metadata.py`
- Create: `data/model_metadata.csv`
- Modify: `src/pyomo_cascade.py`
- Modify: `src/pyomo_robust_cascade.py`
- Modify: `src/experiments.py`
- Modify: `src/plots.py`

- [ ] **Step 1: Write failing metadata coverage test**

Create `tests/test_metadata.py`:

```python
def test_metadata_inference_covers_all_models(synthetic_data):
    from src.model_metadata import build_metadata_for_models

    metadata = build_metadata_for_models(synthetic_data["M"])

    assert set(metadata["model"]) == set(synthetic_data["M"])
    assert metadata["provider_family"].notna().all()
    assert (metadata["estimated_storage_gb"] >= 0).all()


def test_provider_usage_counts_selected_models(synthetic_data):
    from src.model_metadata import build_metadata_for_models, summarize_provider_pool

    metadata = build_metadata_for_models(synthetic_data["M"])
    summary = summarize_provider_pool(["free-small", "strong"], metadata)

    assert summary["num_models_selected"] == 2
    assert summary["provider_count"] >= 1
    assert summary["storage_gb"] >= 0.0
```

- [ ] **Step 2: Implement metadata inference**

Create `src/model_metadata.py`:

```python
from __future__ import annotations

from pathlib import Path

import pandas as pd


PROVIDER_PATTERNS = [
    ("gpt-", "OpenAI"),
    ("o1", "OpenAI"),
    ("o3", "OpenAI"),
    ("gemini", "Google"),
    ("qwen", "Qwen/Alibaba"),
    ("deepseek", "DeepSeek"),
    ("glm", "Zhipu"),
    ("llama", "Meta"),
    ("nvidia", "NVIDIA"),
    ("mistral", "Mistral"),
    ("claude", "Anthropic"),
]


def infer_provider_family(model):
    lowered = model.lower()
    for pattern, provider in PROVIDER_PATTERNS:
        if pattern in lowered:
            return provider
    return "Other"


def infer_storage_gb(model):
    lowered = model.lower()
    if any(token in lowered for token in ["gpt", "gemini", "claude"]):
        return 0.0
    if "70b" in lowered or "72b" in lowered:
        return 140.0
    if "32b" in lowered:
        return 64.0
    if "14b" in lowered:
        return 28.0
    if "7b" in lowered or "8b" in lowered:
        return 16.0
    return 32.0


def build_metadata_for_models(models):
    rows = []
    for model in sorted(models):
        provider = infer_provider_family(model)
        hosted = provider in {"OpenAI", "Google", "Anthropic"}
        rows.append(
            {
                "model": model,
                "provider_family": provider,
                "is_open_source": not hosted,
                "is_hosted_api": hosted,
                "estimated_params_b": 0.0,
                "estimated_storage_gb": infer_storage_gb(model),
                "contract_group": provider,
            }
        )
    return pd.DataFrame(rows)


def load_or_create_metadata(models, path="data/model_metadata.csv"):
    metadata_path = Path(path)
    if metadata_path.exists():
        metadata = pd.read_csv(metadata_path)
    else:
        metadata = build_metadata_for_models(models)
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        metadata.to_csv(metadata_path, index=False)
    missing = set(models) - set(metadata["model"])
    if missing:
        metadata = pd.concat([metadata, build_metadata_for_models(missing)], ignore_index=True)
        metadata = metadata.drop_duplicates("model", keep="first")
        metadata.to_csv(metadata_path, index=False)
    return metadata


def summarize_provider_pool(selected_models, metadata):
    selected = metadata[metadata["model"].isin(selected_models)]
    return {
        "num_models_selected": int(len(selected_models)),
        "provider_count": int(selected["provider_family"].nunique()),
        "storage_gb": float(selected["estimated_storage_gb"].sum()),
    }
```

- [ ] **Step 3: Add optional constraints to A2 and A3 signatures**

Change `solve_a2` and `solve_a3` signatures to include:

```python
metadata=None,
storage_cap_gb=None,
provider_pool_caps=None,
provider_traffic_caps=None,
```

Inside each solver, after `model.pool`, add:

```python
    if metadata is not None and storage_cap_gb is not None:
        storage = metadata.set_index("model")["estimated_storage_gb"].to_dict()
        model.storage = pyo.Constraint(
            expr=sum(float(storage.get(m, 0.0)) * model.y[m] for m in model.M) <= float(storage_cap_gb)
        )

    if metadata is not None and provider_pool_caps:
        provider = metadata.set_index("model")["provider_family"].to_dict()
        model.G = pyo.Set(initialize=sorted(provider_pool_caps))

        def provider_pool_rule(mdl, group):
            models = [m for m in data["M"] if provider.get(m) == group]
            return sum(mdl.y[m] for m in models) <= int(provider_pool_caps[group])

        model.provider_pool = pyo.Constraint(model.G, rule=provider_pool_rule)
```

For provider traffic caps in cascade solvers, add:

```python
    if metadata is not None and provider_traffic_caps:
        provider = metadata.set_index("model")["provider_family"].to_dict()
        model.TG = pyo.Set(initialize=sorted(provider_traffic_caps))

        def provider_traffic_rule(mdl, group):
            terms = []
            for p, a in pa:
                row = cascade_lookup[a]
                if provider.get(row["m1"]) == group:
                    terms.append(mdl.z[p, a])
                if isinstance(row.get("m2", ""), str) and row["m2"] and provider.get(row["m2"]) == group:
                    terms.append(Esc[p, a] * mdl.z[p, a])
            return sum(terms) / n_prompts <= float(provider_traffic_caps[group])

        model.provider_traffic = pyo.Constraint(model.TG, rule=provider_traffic_rule)
```

- [ ] **Step 4: Save metadata and provider usage outputs**

In `run_experiments`, after data loading:

```python
    metadata = load_or_create_metadata(data["M"], path="data/model_metadata.csv")
```

After representative policy selection:

```python
    provider_rows = []
    for result in representative:
        if result is None:
            continue
        selected = result.get("selected_models", result.get("models_used", []))
        row = {"policy": result["policy"], **summarize_provider_pool(selected, metadata)}
        provider_rows.append(row)
    pd.DataFrame(provider_rows).to_csv(root / "tables" / "provider_usage.csv", index=False)
    pd.DataFrame(provider_rows)[["policy", "storage_gb"]].to_csv(root / "tables" / "storage_usage.csv", index=False)
```

- [ ] **Step 5: Add provider traffic plot**

Add to `src/plots.py`:

```python
def plot_provider_traffic_share(root):
    provider = _read_csv(root / "tables" / "provider_usage.csv")
    plt.figure(figsize=(8, 5))
    if not provider.empty and "provider_count" in provider:
        provider.tail(8).plot(kind="bar", x="policy", y="provider_count", ax=plt.gca(), legend=False)
        plt.ylabel("Provider families selected")
        plt.xticks(rotation=30, ha="right")
    plt.title("Provider diversity by selected policy")
    plt.tight_layout()
    plt.savefig(root / "figures" / "provider_traffic_share.png", dpi=220)
    plt.close()
```

Call it from `make_all_plots`.

- [ ] **Step 6: Run metadata tests**

Run:

```bash
pytest tests/test_metadata.py -v
```

Expected:

```text
metadata covers every model and provider/storage summaries are numeric
```

- [ ] **Step 7: Commit**

```bash
git add src/model_metadata.py src/pyomo_cascade.py src/pyomo_robust_cascade.py src/experiments.py src/plots.py data/model_metadata.csv tests/test_metadata.py
git commit -m "feat: add provider and storage deployment constraints"
```

## Task 10: Pareto Frontier, Usage Concentration, and Report Artifacts

**Files:**
- Create: `src/pareto.py`
- Create: `src/report_artifacts.py`
- Modify: `src/metrics.py`
- Modify: `src/experiments.py`
- Modify: `tests/test_report_artifacts.py`

- [ ] **Step 1: Add failing Pareto and usage tests**

Append to `tests/test_report_artifacts.py`:

```python
def test_pareto_filter_removes_dominated_points():
    import pandas as pd

    from src.pareto import pareto_frontier

    df = pd.DataFrame(
        [
            {"policy": "cheap_good", "avg_cost": 1.0, "avg_quality": 0.8},
            {"policy": "expensive_bad", "avg_cost": 2.0, "avg_quality": 0.7},
            {"policy": "expensive_best", "avg_cost": 3.0, "avg_quality": 0.9},
        ]
    )

    frontier = pareto_frontier(df)

    assert set(frontier["policy"]) == {"cheap_good", "expensive_best"}


def test_report_numbers_markdown_contains_chosen_policy(tmp_path):
    from src.report_artifacts import write_report_numbers

    path = write_report_numbers(
        tmp_path,
        {
            "policy": "A3 robust cascade",
            "avg_quality": 0.9,
            "avg_cost": 0.2,
            "eta": 0.85,
            "escalation_rate": 0.4,
            "selected_models": ["m1", "m2"],
        },
    )

    text = path.read_text()
    assert "Chosen policy: A3 robust cascade" in text
    assert "Average quality: 0.9000" in text
```

- [ ] **Step 2: Implement Pareto filter**

Create `src/pareto.py`:

```python
from __future__ import annotations

import pandas as pd


def pareto_frontier(df, cost_col="avg_cost", quality_col="avg_quality"):
    """Return rows not dominated by another row with lower cost and higher quality."""
    if df.empty:
        return df.copy()
    keep = []
    records = df.to_dict("records")
    for row in records:
        dominated = False
        for other in records:
            no_worse = other[cost_col] <= row[cost_col] and other[quality_col] >= row[quality_col]
            strictly_better = other[cost_col] < row[cost_col] or other[quality_col] > row[quality_col]
            if no_worse and strictly_better:
                dominated = True
                break
        if not dominated:
            keep.append(row)
    return pd.DataFrame(keep).sort_values([cost_col, quality_col], ascending=[True, False])
```

- [ ] **Step 3: Add usage concentration metrics**

Add to `src/metrics.py`:

```python
def usage_concentration_rows(policy, usage, stage):
    """Compute entropy, Gini, top shares, and active count from usage counts."""
    values = [float(v) for v in usage.values() if float(v) > 0]
    total = sum(values)
    if total <= 0:
        return {
            "policy": policy,
            "stage": stage,
            "model_usage_entropy": 0.0,
            "model_usage_gini": 0.0,
            "top_1_model_share": 0.0,
            "top_3_model_share": 0.0,
            "num_active_models": 0,
        }
    shares = sorted([v / total for v in values], reverse=True)
    entropy = -sum(share * np.log(share) for share in shares)
    sorted_values = sorted(values)
    n = len(sorted_values)
    weighted_sum = sum((i + 1) * value for i, value in enumerate(sorted_values))
    gini = (2 * weighted_sum) / (n * total) - (n + 1) / n
    return {
        "policy": policy,
        "stage": stage,
        "model_usage_entropy": float(entropy),
        "model_usage_gini": float(gini),
        "top_1_model_share": float(shares[0]),
        "top_3_model_share": float(sum(shares[:3])),
        "num_active_models": int(n),
    }
```

Also add `import numpy as np` at the top of `metrics.py`.

- [ ] **Step 4: Implement report artifact writer**

Create `src/report_artifacts.py`:

```python
from __future__ import annotations

import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pyomo


def write_report_numbers(output_dir, chosen_policy):
    root = Path(output_dir)
    report_dir = root / "report_artifacts"
    report_dir.mkdir(parents=True, exist_ok=True)
    selected_models = chosen_policy.get("selected_models", [])
    if isinstance(selected_models, str):
        selected_models_text = selected_models
    else:
        selected_models_text = ", ".join(selected_models)
    text = "\n".join(
        [
            "# Report Numbers",
            "",
            f"Chosen policy: {chosen_policy.get('policy', '')}",
            f"Average quality: {float(chosen_policy.get('avg_quality') or 0.0):.4f}",
            f"Average cost: {float(chosen_policy.get('avg_cost') or 0.0):.6f}",
            f"Worst-scenario quality eta: {float(chosen_policy.get('eta') or 0.0):.4f}",
            f"Escalation rate: {float(chosen_policy.get('escalation_rate') or 0.0):.4f}",
            f"Selected models: {selected_models_text}",
            "",
        ]
    )
    path = report_dir / "report_numbers.md"
    path.write_text(text, encoding="utf-8")
    return path


def write_manifest(output_dir, data_sha256, command, random_seed=164, git_commit="unknown"):
    root = Path(output_dir)
    manifest = {
        "git_commit": git_commit,
        "data_sha256": data_sha256,
        "python_version": platform.python_version(),
        "pyomo_version": pyomo.version.version,
        "command": command,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "random_seed": random_seed,
    }
    path = root / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return path


def write_report_tables(output_dir, main_comparison, domain_table):
    root = Path(output_dir)
    artifact_dir = root / "report_artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    main_comparison.to_csv(artifact_dir / "report_main_comparison.csv", index=False)
    domain_table.to_csv(artifact_dir / "report_domain_table.csv", index=False)
```

- [ ] **Step 5: Save Pareto and report outputs in experiments**

In `run_experiments`, after all result rows are available:

```python
    all_summary = pd.DataFrame(summary_rows)
    frontier = pareto_frontier(_successful(all_summary))
    frontier.to_csv(root / "tables" / "pareto_frontier.csv", index=False)
    report_main = build_report_main_comparison(summary_rows, K=5, budget_name="B_mid", Emax=0.75)
    report_main.to_csv(root / "tables" / "report_main_comparison.csv", index=False)
    chosen = select_report_a3_policy(a3_results) or _best_result(a2_results) or _best_result(a1_results)
    if chosen is not None:
        write_report_numbers(root, chosen)
```

- [ ] **Step 6: Run report artifact tests**

Run:

```bash
pytest tests/test_report_artifacts.py -v
```

Expected:

```text
selection, matched table, Pareto, and report markdown tests pass
```

- [ ] **Step 7: Commit**

```bash
git add src/pareto.py src/report_artifacts.py src/metrics.py src/experiments.py tests/test_report_artifacts.py
git commit -m "feat: generate matched report artifacts"
```

## Task 11: Report-Grade Figures

**Files:**
- Modify: `src/plots.py`
- Modify: `src/experiments.py`

- [ ] **Step 1: Add Pareto frontier report plot**

Add to `src/plots.py`:

```python
def plot_pareto_frontier_report(root):
    all_results = _read_csv(root / "tables" / "report_main_comparison.csv")
    frontier = _read_csv(root / "tables" / "pareto_frontier.csv")
    plt.figure(figsize=(8.5, 5.5))
    if not all_results.empty:
        for family, group in all_results.groupby("family"):
            plt.scatter(group["avg_cost"], group["avg_quality"], label=family, s=48, alpha=0.85)
    if not frontier.empty:
        ordered = frontier.sort_values("avg_cost")
        plt.plot(ordered["avg_cost"], ordered["avg_quality"], color="black", linewidth=1.5, label="Non-dominated frontier")
    chosen = _read_csv(root / "tables" / "a3_best_report_policy.csv")
    if not chosen.empty:
        plt.scatter(chosen["avg_cost"], chosen["avg_quality"], marker="*", s=240, color="#d62728", label="Chosen policy")
    plt.xlabel("Average expected cost")
    plt.ylabel("Average expected quality")
    plt.title("Matched cost-quality frontier")
    plt.grid(alpha=0.25)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(root / "figures" / "pareto_frontier_report.png", dpi=240)
    plt.close()
```

- [ ] **Step 2: Add domain reliability report plot**

```python
def plot_domain_reliability_report(root):
    domain = _read_csv(root / "tables" / "domain_quality.csv")
    plt.figure(figsize=(9, 5.5))
    if not domain.empty:
        keep = domain[domain["policy"].str.startswith(("A1", "A2", "A3", "A4"))]
        pivot = keep.pivot_table(index="domain", columns="policy", values="avg_quality", aggfunc="mean")
        pivot.iloc[:, -6:].plot(kind="bar", ax=plt.gca())
    plt.ylabel("Average quality")
    plt.ylim(0, 1.05)
    plt.title("Domain reliability by policy")
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(root / "figures" / "domain_reliability_report.png", dpi=240)
    plt.close()
```

- [ ] **Step 3: Add feasibility map**

```python
def plot_feasibility_map(root):
    frames = []
    for family, file_name in [("A1", "a1_results.csv"), ("A2", "a2_results.csv"), ("A3", "a3_grid_results.csv")]:
        frame = _read_csv(root / "tables" / file_name)
        if not frame.empty:
            frames.append(frame.assign(family=family))
    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    plt.figure(figsize=(8, 5.5))
    if not df.empty and {"K", "budget_name", "status"}.issubset(df.columns):
        score = {"optimal": 3, "feasible": 2, "feasible_time_limited": 2, "infeasible": 1, "no_solution": 0, "no_solver": 0}
        df["status_score"] = df["status"].map(score).fillna(0)
        pivot = df.pivot_table(index="K", columns="budget_name", values="status_score", aggfunc="max")
        image = plt.imshow(pivot.values, aspect="auto", vmin=0, vmax=3, cmap="viridis")
        plt.xticks(range(len(pivot.columns)), pivot.columns)
        plt.yticks(range(len(pivot.index)), pivot.index)
        plt.colorbar(image, label="0 no solution, 3 optimal")
    plt.xlabel("Budget")
    plt.ylabel("Pool size K")
    plt.title("Optimization feasibility map")
    plt.tight_layout()
    plt.savefig(root / "figures" / "feasibility_map.png", dpi=240)
    plt.close()
```

- [ ] **Step 4: Add usage concentration plot**

```python
def plot_usage_concentration(root):
    usage = _read_csv(root / "tables" / "usage_concentration.csv")
    plt.figure(figsize=(8.5, 5))
    if not usage.empty:
        plot_df = usage.sort_values("top_1_model_share", ascending=False).tail(8)
        plt.barh(plot_df["policy"] + " / " + plot_df["stage"], plot_df["top_1_model_share"], color="#4c78a8")
        plt.xlabel("Top model usage share")
    plt.title("Model usage concentration")
    plt.tight_layout()
    plt.savefig(root / "figures" / "usage_concentration.png", dpi=240)
    plt.close()
```

- [ ] **Step 5: Add cascade flow plot**

```python
def plot_cascade_flow(root):
    solutions = root / "solutions" / "a2_solutions.json"
    candidates = _read_csv(root / "tables" / "cascade_candidates.csv")
    plt.figure(figsize=(9, 6))
    if solutions.exists() and not candidates.empty:
        import json

        payload = json.loads(solutions.read_text())
        feasible = [r for r in payload.values() if isinstance(r, dict) and "cascade_assignment" in r]
        if feasible:
            result = feasible[-1]
            lookup = candidates.set_index("cascade_id")
            flows = {}
            for cascade_id in result["cascade_assignment"].values():
                row = lookup.loc[cascade_id]
                if isinstance(row.get("m2", ""), str) and row["m2"]:
                    flows[(row["m1"], row["m2"])] = flows.get((row["m1"], row["m2"]), 0) + 1
            top = sorted(flows.items(), key=lambda item: item[1], reverse=True)[:12]
            labels = [f"{m1} -> {m2}" for (m1, m2), _ in top]
            values = [count for _, count in top]
            plt.barh(labels, values, color="#2f6f73")
            plt.gca().invert_yaxis()
    plt.xlabel("Assigned prompt count")
    plt.title("Cascade flow: first-stage to second-stage")
    plt.tight_layout()
    plt.savefig(root / "figures" / "cascade_flow.png", dpi=240)
    plt.close()
```

- [ ] **Step 6: Register all new plots**

Update `make_all_plots`:

```python
    plot_pareto_frontier_report(root)
    plot_cascade_flow(root)
    plot_model_complementarity_heatmap(root)
    plot_stress_test_quality_distribution(root)
    plot_domain_reliability_report(root)
    plot_feasibility_map(root)
    plot_provider_traffic_share(root)
    plot_usage_concentration(root)
```

- [ ] **Step 7: Run plot command after sample output exists**

Run:

```bash
python run_experiments.py --data data/routerbench.csv --output-dir outputs --skip-a3 --time-limit 30 --max-cascades 100
python run_experiments.py --output-dir outputs --only-plots
```

Expected:

```text
outputs/figures/pareto_frontier_report.png exists
outputs/figures/cascade_flow.png exists
outputs/figures/model_complementarity_heatmap.png exists
outputs/figures/domain_reliability_report.png exists
outputs/figures/feasibility_map.png exists
```

- [ ] **Step 8: Commit**

```bash
git add src/plots.py src/experiments.py outputs/figures
git commit -m "feat: add report-grade optimization figures"
```

## Task 12: Lexicographic A3 Robust Objective

**Files:**
- Modify: `src/pyomo_robust_cascade.py`
- Modify: `src/experiments.py`
- Create: `tests/test_robust_lexicographic.py`

- [ ] **Step 1: Write failing lexicographic test**

Create `tests/test_robust_lexicographic.py`:

```python
def test_a3_lexicographic_returns_three_monotone_passes(synthetic_data):
    import pytest

    from src.cascade_generation import generate_cascades
    from src.pyomo_robust_cascade import build_scenarios, compute_domain_floors, solve_a3_lexicographic

    cascades, params = generate_cascades(synthetic_data, rho=0.75, max_cascades=40)
    result = solve_a3_lexicographic(
        synthetic_data,
        cascades,
        params["R"],
        params["C"],
        params["Esc"],
        params["A_p"],
        build_scenarios(synthetic_data),
        compute_domain_floors(synthetic_data, multiplier=0.75),
        K=3,
        B=10.0,
        Emax=1.0,
        time_limit=20,
    )
    if result["status"] == "no_solver":
        pytest.skip(result["message"])

    passes = result["lexicographic_passes"]
    assert [row["pass"] for row in passes] == [1, 2, 3]
    assert passes[1]["eta"] + 1e-6 >= passes[0]["eta"] - 1e-6
    assert passes[2]["total_slack"] <= passes[1]["total_slack"] + 1e-6
```

- [ ] **Step 2: Refactor A3 model construction**

Extract the model-building portion of `solve_a3` into:

```python
def build_a3_model(data, cascades, R, C, Esc, A_p, scenarios, floors, K, B, Emax):
    """Build an unsolved A3 model and return model metadata needed for extraction."""
    cascade_lookup = cascades.set_index("cascade_id")[["m1", "m2", "m3", "depth"]].to_dict("index")
    pa = sorted((p, a) for p in data["P"] for a in A_p[p])
    n_prompts = len(data["P"])
    model = pyo.ConcreteModel()
    model.P = pyo.Set(initialize=data["P"])
    model.M = pyo.Set(initialize=data["M"])
    model.D = pyo.Set(initialize=data["D"])
    model.S = pyo.Set(initialize=sorted(scenarios))
    model.PA = pyo.Set(dimen=2, initialize=pa)
    model.z = pyo.Var(model.PA, within=pyo.Binary)
    model.y = pyo.Var(model.M, within=pyo.Binary)
    model.eta = pyo.Var(bounds=(0.0, 1.0))
    model.floor_slack = pyo.Var(model.D, within=pyo.NonNegativeReals)
    model.assignment = pyo.Constraint(model.P, rule=lambda mdl, prompt: sum(mdl.z[prompt, cascade_id] for cascade_id in A_p[prompt]) == 1)
    stage_links = []
    for prompt, cascade_id in pa:
        row = cascade_lookup[cascade_id]
        for model_name in [row["m1"], row["m2"], row["m3"]]:
            if isinstance(model_name, str) and model_name:
                stage_links.append((prompt, cascade_id, model_name))
    model.PAM = pyo.Set(dimen=3, initialize=stage_links)
    model.link_stage = pyo.Constraint(model.PAM, rule=lambda mdl, prompt, cascade_id, model_name: mdl.z[prompt, cascade_id] <= mdl.y[model_name])
    model.pool = pyo.Constraint(expr=sum(model.y[m] for m in model.M) <= K)
    model.escalation = pyo.Constraint(expr=sum(Esc[p, a] * model.z[p, a] for p, a in pa) / n_prompts <= Emax)
    model.scenario_quality = pyo.Constraint(model.S, rule=lambda mdl, scenario: sum(scenarios[scenario]["prompt_weights"][p] * R[p, a] * mdl.z[p, a] for p, a in pa) >= mdl.eta)
    model.scenario_cost = pyo.Constraint(model.S, rule=lambda mdl, scenario: sum(scenarios[scenario]["prompt_weights"][p] * C[p, a] * mdl.z[p, a] for p, a in pa) <= B)
    model.domain_floor = pyo.Constraint(model.D, rule=lambda mdl, domain: sum(R[p, a] * mdl.z[p, a] for p in data["P_d"][domain] for a in A_p[p]) / len(data["P_d"][domain]) + mdl.floor_slack[domain] >= floors[domain])
    meta = {"pa": pa, "n_prompts": n_prompts, "cascade_lookup": cascade_lookup}
    return model, meta
```

- [ ] **Step 3: Implement lexicographic solve**

Add to `src/pyomo_robust_cascade.py`:

```python
def solve_a3_lexicographic(data, cascades, R, C, Esc, A_p, scenarios, floors, K, B, Emax, time_limit=300, epsilon=1e-6, alpha_cost=0.01):
    """Solve A3 in three passes: max eta, min slack, then max empirical quality minus cost."""
    passes = []
    model, meta = build_a3_model(data, cascades, R, C, Esc, A_p, scenarios, floors, K, B, Emax)
    policy = f"A3-lex K={K} B={B:.6g} Emax={Emax:g}"
    model.objective = pyo.Objective(expr=model.eta, sense=pyo.maximize)
    solver_name, results, diagnostics = solve_model(model, time_limit=time_limit, policy=f"{policy} pass=1")
    if solver_name is None:
        return no_solver_result(policy)
    status = result_status(results)
    if not has_solution(status):
        return {"policy": policy, "status": status, "solver": solver_name, "message": str(results.solver.termination_condition)}
    eta_star = float(pyo.value(model.eta))
    passes.append({"pass": 1, "objective": "max_eta", "eta": eta_star, "total_slack": sum(float(pyo.value(model.floor_slack[d])) for d in data["D"])})

    model.objective.deactivate()
    model.fix_eta = pyo.Constraint(expr=model.eta >= eta_star - epsilon)
    model.min_slack_objective = pyo.Objective(expr=sum(model.floor_slack[d] for d in model.D), sense=pyo.minimize)
    solver_name, results, diagnostics = solve_model(model, time_limit=time_limit, policy=f"{policy} pass=2")
    status = result_status(results)
    slack_star = sum(float(pyo.value(model.floor_slack[d])) for d in data["D"])
    passes.append({"pass": 2, "objective": "min_slack", "eta": float(pyo.value(model.eta)), "total_slack": slack_star})

    model.min_slack_objective.deactivate()
    model.fix_slack = pyo.Constraint(expr=sum(model.floor_slack[d] for d in model.D) <= slack_star + epsilon)
    pa = meta["pa"]
    n = meta["n_prompts"]
    model.empirical_objective = pyo.Objective(
        expr=sum((R[p, a] - alpha_cost * C[p, a]) * model.z[p, a] for p, a in pa) / n,
        sense=pyo.maximize,
    )
    solver_name, results, diagnostics = solve_model(model, time_limit=time_limit, policy=f"{policy} pass=3")
    status = result_status(results)
    passes.append({"pass": 3, "objective": "max_empirical_quality_minus_cost", "eta": float(pyo.value(model.eta)), "total_slack": sum(float(pyo.value(model.floor_slack[d])) for d in data["D"])})

    result = _extract_a3_solution(data, cascades, R, C, Esc, A_p, scenarios, model, policy, status, solver_name)
    result["lexicographic_passes"] = passes
    return result
```

Also extract the existing result-reading logic from `solve_a3` into `_extract_a3_solution(...)` so `solve_a3` and `solve_a3_lexicographic` share one extractor.

- [ ] **Step 4: Save lexicographic pass table**

In `run_experiments`, for the chosen A3 configuration, call `solve_a3_lexicographic` and write:

```python
        pd.DataFrame(lex_result.get("lexicographic_passes", [])).to_csv(root / "tables" / "a3_lexicographic_passes.csv", index=False)
```

- [ ] **Step 5: Run lexicographic test**

Run:

```bash
pytest tests/test_robust_lexicographic.py -v
```

Expected:

```text
lexicographic A3 returns pass rows 1, 2, and 3
eta is preserved after pass 1 within tolerance
slack is preserved after pass 2 within tolerance
```

- [ ] **Step 6: Commit**

```bash
git add src/pyomo_robust_cascade.py src/experiments.py tests/test_robust_lexicographic.py
git commit -m "feat: add lexicographic robust cascade objective"
```

## Task 13: A4 CVaR Tail-Risk Cascade

**Files:**
- Create: `src/pyomo_tail_risk.py`
- Create: `tests/test_tail_risk.py`
- Modify: `src/experiments.py`
- Modify: `src/plots.py`

- [ ] **Step 1: Write failing A4 smoke test**

Create `tests/test_tail_risk.py`:

```python
def test_a4_cvar_returns_tail_risk_metrics(synthetic_data):
    import pytest

    from src.cascade_generation import generate_cascades
    from src.pyomo_tail_risk import solve_a4_cvar_cascade

    cascades, params = generate_cascades(synthetic_data, rho=0.75, max_cascades=40)
    floors = {domain: 0.5 for domain in synthetic_data["D"]}
    result = solve_a4_cvar_cascade(
        synthetic_data,
        cascades,
        params["R"],
        params["C"],
        params["Esc"],
        params["A_p"],
        floors,
        K=3,
        B=10.0,
        Emax=1.0,
        beta=0.9,
        lambda_cvar=0.1,
        time_limit=20,
    )
    if result["status"] == "no_solver":
        pytest.skip(result["message"])

    assert result["status"] in {"optimal", "feasible", "feasible_time_limited"}
    assert result["cvar_shortfall"] >= 0.0
```

- [ ] **Step 2: Implement A4 CVaR model**

Create `src/pyomo_tail_risk.py`:

```python
from __future__ import annotations

import pyomo.environ as pyo

from .metrics import cascade_assignment_metrics
from .solver_utils import has_solution, no_solver_result, result_status, solve_model


def solve_a4_cvar_cascade(data, cascades, R, C, Esc, A_p, floors, K, B, Emax, beta=0.9, lambda_cvar=0.1, time_limit=300):
    """Solve a CVaR tail-risk cascade MILP over prompt-level shortfalls."""
    policy = f"A4-CVaR K={K} B={B:.6g} Emax={Emax:g} beta={beta:g}"
    cascade_lookup = cascades.set_index("cascade_id")[["m1", "m2", "m3", "depth"]].to_dict("index")
    pa = sorted((p, a) for p in data["P"] for a in A_p[p])
    n_prompts = len(data["P"])
    model = pyo.ConcreteModel()
    model.P = pyo.Set(initialize=data["P"])
    model.M = pyo.Set(initialize=data["M"])
    model.PA = pyo.Set(dimen=2, initialize=pa)
    model.z = pyo.Var(model.PA, within=pyo.Binary)
    model.y = pyo.Var(model.M, within=pyo.Binary)
    model.nu = pyo.Var(within=pyo.NonNegativeReals)
    model.shortfall = pyo.Var(model.P, within=pyo.NonNegativeReals)
    model.u = pyo.Var(model.P, within=pyo.NonNegativeReals)
    model.assignment = pyo.Constraint(model.P, rule=lambda mdl, prompt: sum(mdl.z[prompt, cascade_id] for cascade_id in A_p[prompt]) == 1)
    stage_links = []
    for prompt, cascade_id in pa:
        row = cascade_lookup[cascade_id]
        for model_name in [row["m1"], row["m2"], row["m3"]]:
            if isinstance(model_name, str) and model_name:
                stage_links.append((prompt, cascade_id, model_name))
    model.PAM = pyo.Set(dimen=3, initialize=stage_links)
    model.link_stage = pyo.Constraint(model.PAM, rule=lambda mdl, prompt, cascade_id, model_name: mdl.z[prompt, cascade_id] <= mdl.y[model_name])
    model.pool = pyo.Constraint(expr=sum(model.y[m] for m in model.M) <= K)
    model.budget = pyo.Constraint(expr=sum(C[p, a] * model.z[p, a] for p, a in pa) / n_prompts <= B)
    model.escalation = pyo.Constraint(expr=sum(Esc[p, a] * model.z[p, a] for p, a in pa) / n_prompts <= Emax)
    model.shortfall_floor = pyo.Constraint(
        model.P,
        rule=lambda mdl, prompt: mdl.shortfall[prompt] >= floors[data["prompt_domain"][prompt]] - sum(R[prompt, a] * mdl.z[prompt, a] for a in A_p[prompt]),
    )
    model.cvar_excess = pyo.Constraint(model.P, rule=lambda mdl, prompt: mdl.u[prompt] >= mdl.shortfall[prompt] - mdl.nu)
    cvar_expr = model.nu + (1.0 / ((1.0 - beta) * n_prompts)) * sum(model.u[p] for p in model.P)
    avg_quality = sum(R[p, a] * model.z[p, a] for p, a in pa) / n_prompts
    model.objective = pyo.Objective(expr=avg_quality - lambda_cvar * cvar_expr, sense=pyo.maximize)
    solver_name, results, diagnostics = solve_model(model, time_limit=time_limit, policy=policy)
    if solver_name is None:
        return no_solver_result(policy)
    status = result_status(results)
    if not has_solution(status):
        return {"policy": policy, "status": status, "solver": solver_name, "message": str(results.solver.termination_condition)}
    assignment = {}
    for prompt in data["P"]:
        for cascade_id in A_p[prompt]:
            if (pyo.value(model.z[prompt, cascade_id], exception=False) or 0.0) > 0.5:
                assignment[prompt] = cascade_id
                break
    metrics = cascade_assignment_metrics(data, cascades, assignment, R, C, Esc, policy)
    metrics.update(
        {
            "status": status,
            "solver": solver_name,
            "K": K,
            "B": B,
            "Emax": Emax,
            "beta": beta,
            "lambda_cvar": lambda_cvar,
            "cvar_shortfall": float(pyo.value(cvar_expr) or 0.0),
            "selected_models": [m for m in data["M"] if (pyo.value(model.y[m], exception=False) or 0.0) > 0.5],
            "diagnostics": diagnostics,
        }
    )
    return metrics
```

- [ ] **Step 3: Add A4 experiment row**

In `run_experiments`, after A3 selection, solve one matched A4 policy:

```python
    a4_results = []
    if not skip_a3:
        floors = compute_domain_floors(data, multiplier=0.85)
        result = solve_a4_cvar_cascade(
            data,
            cascades,
            params["R"],
            params["C"],
            params["Esc"],
            params["A_p"],
            floors,
            K=5,
            B=budgets["B_mid"],
            Emax=0.75,
            beta=0.9,
            lambda_cvar=0.1,
            time_limit=time_limit,
        )
        result["budget_name"] = "B_mid"
        a4_results.append(result)
        pd.DataFrame([_summary_row(r, family="A4", K=r.get("K"), B=r.get("B"), Emax=r.get("Emax"), budget_name=r.get("budget_name"), cvar_shortfall=r.get("cvar_shortfall")) for r in a4_results]).to_csv(root / "tables" / "a4_cvar_results.csv", index=False)
```

- [ ] **Step 4: Add CVaR plot**

Add to `src/plots.py`:

```python
def plot_cvar_tradeoff(root):
    cvar = _read_csv(root / "tables" / "a4_cvar_results.csv")
    plt.figure(figsize=(7.5, 5))
    if not cvar.empty:
        plt.scatter(cvar["cvar_shortfall"], cvar["avg_quality"], s=60)
        plt.xlabel("CVaR shortfall")
        plt.ylabel("Average quality")
    plt.title("Tail-risk tradeoff")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(root / "figures" / "cvar_tradeoff.png", dpi=220)
    plt.close()
```

Call it from `make_all_plots`.

- [ ] **Step 5: Run A4 test**

Run:

```bash
pytest tests/test_tail_risk.py -v
```

Expected:

```text
A4 returns a nonnegative CVaR shortfall when a solver is installed
```

- [ ] **Step 6: Commit**

```bash
git add src/pyomo_tail_risk.py src/experiments.py src/plots.py tests/test_tail_risk.py
git commit -m "feat: add CVaR tail-risk cascade model"
```

## Task 14: Limited Three-Stage Cascades After Two-Stage Verification

**Files:**
- Modify: `src/cascade_generation.py`
- Modify: `tests/test_cascades.py`

- [ ] **Step 1: Verify gate before touching three-stage code**

Run:

```bash
make test
python run_experiments.py --data data/routerbench.csv --output-dir outputs --skip-a3 --time-limit 60 --max-cascades 250
```

Expected:

```text
all tests pass
outputs/tables/a2_results.csv exists
outputs/tables/cascade_candidates.csv includes depth 1 and depth 2 rows
```

- [ ] **Step 2: Add failing three-stage parameter test**

Append to `tests/test_cascades.py`:

```python
def test_three_stage_parameters_are_bounded_when_enabled(synthetic_data):
    from src.cascade_generation import generate_cascades

    cascades, params = generate_cascades(synthetic_data, rho=0.75, max_cascades=60, include_three_stage=True)
    three_stage_ids = set(cascades.loc[cascades["depth"] == 3, "cascade_id"])
    if not three_stage_ids:
        pytest.skip("Synthetic data does not produce a three-stage candidate under the configured filter.")

    for key, value in params["R"].items():
        if key[1] in three_stage_ids:
            assert 0.0 <= value <= 1.0
            assert params["C"][key] >= 0.0
            assert params["Esc"][key] >= 0.0
```

- [ ] **Step 3: Implement limited three-stage candidates**

Add to `src/cascade_generation.py`:

```python
def generate_three_stage_cascades(data, rho=0.75, max_three_stage=50):
    """Generate a small high-value set of three-stage cascades after two-stage outputs exist."""
    summary = summarize_models(data).set_index("model")
    cheap = summary.sort_values(["cbar", "qbar"], ascending=[True, False]).head(6).index
    middle = summary.sort_values(["qbar", "cbar"], ascending=[False, True]).head(10).index
    final = summary.sort_values(["qbar", "cbar"], ascending=[False, True]).head(6).index
    pm = set(data["PM"])
    rows = []
    for m1 in cheap:
        for m2 in middle:
            for m3 in final:
                if len({m1, m2, m3}) < 3:
                    continue
                feasible_prompts = [p for p in data["P"] if (p, m1) in pm and (p, m2) in pm and (p, m3) in pm]
                if not feasible_prompts:
                    continue
                rows.append(
                    {
                        "depth": 3,
                        "m1": m1,
                        "m2": m2,
                        "m3": m3,
                        "qbar_m1": summary.loc[m1, "qbar"],
                        "qbar_m2": summary.loc[m2, "qbar"],
                        "qbar_m3": summary.loc[m3, "qbar"],
                        "cbar_m1": summary.loc[m1, "cbar"],
                        "cbar_m2": summary.loc[m2, "cbar"],
                        "cbar_m3": summary.loc[m3, "cbar"],
                        "avg_R": 0.0,
                        "avg_C": 0.0,
                        "avg_Esc": 0.0,
                        "feasible_prompts": len(feasible_prompts),
                    }
                )
    frame = pd.DataFrame(rows).head(max_three_stage).reset_index(drop=True)
    if not frame.empty:
        frame.insert(0, "cascade_id", [f"s3::{row.m1}::{row.m2}::{row.m3}" for row in frame.itertuples()])
    return frame
```

Update `generate_cascades`:

```python
    threes = generate_three_stage_cascades(data, rho=rho, max_three_stage=50) if include_three_stage else pd.DataFrame()
    frames = [frame for frame in [singles, twos, threes] if not frame.empty]
```

- [ ] **Step 4: Run three-stage test**

Run:

```bash
pytest tests/test_cascades.py::test_three_stage_parameters_are_bounded_when_enabled -v
```

Expected:

```text
three-stage parameters are bounded, or the synthetic instance explicitly skips because no candidate passes the filter
```

- [ ] **Step 5: Commit**

```bash
git add src/cascade_generation.py tests/test_cascades.py
git commit -m "feat: add gated three-stage cascade candidates"
```

## Task 15: Final Pipeline Verification and Output Manifest

**Files:**
- Modify: `run_experiments.py`
- Modify: `src/experiments.py`
- Modify: `src/report_artifacts.py`
- Create: `outputs_final/RUN_LOG.md`

- [ ] **Step 1: Add config argument**

In `run_experiments.py`, add:

```python
parser.add_argument("--config", default=None, help="YAML config file for final experiment settings.")
```

Pass `config_path=args.config` into `run_experiments`.

- [ ] **Step 2: Load config in experiments**

At the top of `src/experiments.py`, import:

```python
import hashlib
import subprocess
import sys
import yaml
```

Add:

```python
def load_config(config_path):
    if not config_path:
        return {}
    with Path(config_path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def current_git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"
```

- [ ] **Step 3: Write final manifest and run log**

At the end of `run_experiments`, add:

```python
    write_manifest(
        root,
        data_sha256=file_sha256(data_path),
        command=" ".join(sys.argv),
        random_seed=int(config.get("random_seed", 164)),
        git_commit=current_git_commit(),
    )
    (root / "RUN_LOG.md").write_text(
        "\n".join(
            [
                "# Run Log",
                "",
                f"Data: `{data_path}`",
                f"Output directory: `{root}`",
                f"A1 grid points: {len(a1_results)}",
                f"A2 grid points: {len(a2_results)}",
                f"A3 grid points: {len(a3_results)}",
                "",
            ]
        ),
        encoding="utf-8",
    )
```

- [ ] **Step 4: Run full final commands**

Run:

```bash
make run-final
make audit
make plots
make test
```

Expected:

```text
outputs_final/manifest.json exists
outputs_final/RUN_LOG.md exists
outputs_final/tables/a3_grid_results.csv exists
outputs_final/tables/a3_best_report_policy.csv exists
outputs_final/tables/report_main_comparison.csv exists
outputs_final/tables/solution_audit.csv exists
outputs_final/tables/solver_diagnostics.csv exists
outputs_final/figures/pareto_frontier_report.png exists
outputs_final/figures/stress_test_quality_distribution.png exists
all tests pass
```

- [ ] **Step 5: Inspect audit failures**

Run:

```bash
python - <<'PY'
import pandas as pd
audit = pd.read_csv("outputs_final/tables/solution_audit.csv")
print(audit.loc[~audit["passed"]].to_string(index=False))
PY
```

Expected:

```text
Empty DataFrame
```

- [ ] **Step 6: Commit final artifacts that belong in the repo**

Do not commit large generated images unless the project submission expects generated outputs in git. Commit source, docs, config, tests, and small report CSV/Markdown artifacts:

```bash
git add run_experiments.py src tests config docs Makefile pyproject.toml requirements.txt data/model_metadata.csv outputs_final/manifest.json outputs_final/RUN_LOG.md outputs_final/report_artifacts
git commit -m "feat: finalize reproducible routing study outputs"
```

## Self-Review

Spec coverage:

- P0 tooling, A3 early-break removal, larger A3 grid, matched comparison, audit table, diagnostics, degenerate cascades, and report-grade figures are covered in Tasks 1-6 and 10-11.
- P1 complementarity, stress testing, provider/storage constraints, usage concentration, and ablation-ready tables are covered in Tasks 7-10.
- P2 lexicographic A3, A4 CVaR, three-stage cascades, cascade-flow plot, complementarity heatmap, and report artifact generator are covered in Tasks 11-15.
- Project data rules are preserved: all variables are over observed prompt-model pairs or prompt-feasible cascades, missing rows are not imputed, zero-cost rows are kept, and infeasible grid points are recorded through status rows.

Placeholder scan:

- The plan contains no banned placeholder markers and no undefined function names in tests.
- Every new test names imports that are created or modified in this plan.
- Every command has an expected result.

Type consistency:

- `generate_cascades` returns `(cascades, params)` where `params` includes `A_p`, `R`, `C`, `Esc`, `Esc2`, and `Esc3`.
- A2/A3/A4 result dictionaries consistently use `assignment` for single-shot policies and `cascade_assignment` for cascade policies.
- Diagnostics are stored under `result["diagnostics"]` and flattened into `solver_diagnostics.csv` by `src/experiments.py`.
- Report policy rows use `policy`, `family`, `K`, `budget_name`, `Emax`, `avg_cost`, `avg_quality`, `status`, and optional robust/tail-risk fields.
