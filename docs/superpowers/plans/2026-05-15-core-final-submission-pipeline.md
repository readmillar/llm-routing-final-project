# Core Final Submission Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the default final submission path a fast, reproducible, atomic A0-A3 pipeline while keeping optional extension modules in the repository behind explicit config flags.

**Architecture:** `Makefile` owns the public commands, writes final artifacts to temporary output directories first, and promotes them only after successful runs. `config/final.yaml` and `config/smoke.yaml` own all A1/A2/A3 grids, solver limits, cascade constants, and feature flags. `src/experiments.py` remains the orchestration layer, but it must read grids from config and leave A4, empirical recovery, stress tests, three-stage cascades, provider/storage constraints, and lexicographic A3 disabled unless a config explicitly enables them.

**Tech Stack:** Python, Pyomo, pandas, numpy, matplotlib, pytest, Make, YAML.

---

## Scope Check

Do not delete optional modules. Keep working extras in the repository.

The default submission path is the A0-A3 report path:

- A0 weighted baseline
- A1 single-shot portfolio MILP
- A2 two-stage stochastic cascade MILP
- A3 robust reliability-aware cascade MILP using scenario-weighted SAA and domain slacks
- Solver diagnostics
- Solution audit
- Report comparison tables
- Required figures

These extensions remain available only behind config flags and are disabled by default:

- A4 CVaR
- Empirical recovery
- Stress tests
- Three-stage cascades
- Provider/storage constraints
- Lexicographic A3

Final acceptance commands:

```bash
make test
make run-smoke
make run-final
make audit
make plots
make test
```

All six commands must pass.

---

## Current State

Current files that matter:

- `Makefile` has `make run-final`, `make audit`, `make test`, and `make plots`, but no `run-smoke` target and no atomic output promotion.
- `Makefile` currently prefers `python`; the required preference order is `.venv/bin/python`, then `python3`, then `python`.
- `config/final.yaml` exists but currently describes a heavier-than-default final profile.
- `config/smoke.yaml` does not exist.
- `run_experiments.py` accepts `--config`, but CLI defaults for `--time-limit` and `--max-cascades` currently override config values.
- `src/experiments.py` loads config but still has hardcoded A1/A2/A3 orchestration grids, always estimates empirical recovery, always enables A4, and always runs stress tests.
- `src/experiments.py` currently writes canonical repo table names such as `a0_results.csv`, `a1_results.csv`, `a2_results.csv`, `a3_results.csv`, `a3_grid_results.csv`, `model_pair_recovery.csv`, and `stress_test_results.csv`.
- `src/plots.py` expects existing repo table names.
- `src/audit.py` writes `solution_audit.csv` and tolerates optional A4 solution files when present.

---

## Target File Structure

Modify these files:

```text
Makefile
config/final.yaml
run_experiments.py
src/experiments.py
tests/test_final_manifest.py
docs/RUNBOOK.md
```

Create this file:

```text
config/smoke.yaml
```

Do not delete optional model, plotting, stress-testing, metadata, provider, storage, or A4 code.

---

## Task 1: Make Make Targets Atomic And Add Smoke Config

**Files:**

- Modify: `Makefile`
- Modify: `config/final.yaml`
- Create: `config/smoke.yaml`

**Purpose:** Public commands use the right Python interpreter, `run-final` and `run-smoke` promote output directories only after success, and both final and smoke runs are driven entirely by YAML config.

### Steps

- [ ] Replace the Makefile Python selector with `.venv/bin/python`, then `python3`, then `python`.
- [ ] Add `run-smoke`.
- [ ] Make `run-smoke` write to `outputs_smoke.tmp` and rename to `outputs_smoke` only after the experiment command succeeds.
- [ ] Make `run-final` write to `outputs_final.tmp` and rename to `outputs_final` only after the experiment command succeeds.
- [ ] Keep an existing `outputs_final` directory untouched if the experiment command fails.
- [ ] Replace `config/final.yaml` with the core final profile below.
- [ ] Create `config/smoke.yaml` with the tiny smoke profile below.
- [ ] Run `make test`.

### Makefile Replacement

Replace the command section with:

```make
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
	if [ -e outputs_smoke ]; then mv outputs_smoke "$$backup"; fi; \
	if mv "$$tmp" outputs_smoke; then \
		rm -rf "$$backup"; \
	else \
		if [ -e "$$backup" ]; then mv "$$backup" outputs_smoke; fi; \
		exit 1; \
	fi

run-final:
	@set -euo pipefail; \
	tmp="outputs_final.tmp"; backup="outputs_final.backup"; \
	rm -rf "$$tmp" "$$backup"; \
	$(PYTHON) run_experiments.py --data data/routerbench.csv --output-dir "$$tmp" --config config/final.yaml; \
	if [ -e outputs_final ]; then mv outputs_final "$$backup"; fi; \
	if mv "$$tmp" outputs_final; then \
		rm -rf "$$backup"; \
	else \
		if [ -e "$$backup" ]; then mv "$$backup" outputs_final; fi; \
		exit 1; \
	fi

audit:
	$(PYTHON) -m src.audit --data data/routerbench.csv --output-dir outputs_final

plots:
	$(PYTHON) run_experiments.py --output-dir outputs_final --config config/final.yaml --only-plots
```

### `config/final.yaml`

Replace the file with:

```yaml
random_seed: 164
profile: core_final

time_limit: 45
max_cascades: 250
base_rho: 0.75

features:
  empirical_recovery: false
  stress_tests: false
  a4_cvar: false
  three_stage: false
  lexicographic_a3: false
  provider_storage_constraints: false

a1:
  K: [1, 2, 3, 5]
  budget_names: ["B_low", "B_mid", "B_high"]

a2:
  K: [2, 3, 5]
  budget_names: ["B_low", "B_mid", "B_high"]
  Emax: [0.5, 0.75, 1.0]

a3:
  K: [5]
  budget_names: ["B_mid"]
  Emax: [0.75]
  floor_multiplier: [0.85]
  lambda_slack: [0.1]
  rho: [0.75]

matched_report:
  K: 5
  budget_name: "B_mid"
  Emax: 0.75
  floor_multiplier: 0.85
  lambda_slack: 0.1
  rho: 0.75

stress:
  dirichlet_samples: 0
  concentration: 40.0

production_constraints:
  storage_cap_gb: null
  provider_pool_caps: {}
  provider_traffic_caps: {}
```

### `config/smoke.yaml`

Create the file with:

```yaml
random_seed: 164
profile: smoke

time_limit: 20
max_cascades: 40
base_rho: 0.75

features:
  empirical_recovery: false
  stress_tests: false
  a4_cvar: false
  three_stage: false
  lexicographic_a3: false
  provider_storage_constraints: false

a1:
  K: [1]
  budget_names: ["B_mid"]

a2:
  K: [2]
  budget_names: ["B_mid"]
  Emax: [1.0]

a3:
  K: [2]
  budget_names: ["B_mid"]
  Emax: [1.0]
  floor_multiplier: [0.75]
  lambda_slack: [0.1]
  rho: [0.75]

matched_report:
  K: 2
  budget_name: "B_mid"
  Emax: 1.0
  floor_multiplier: 0.75
  lambda_slack: 0.1
  rho: 0.75

stress:
  dirichlet_samples: 0
  concentration: 40.0

production_constraints:
  storage_cap_gb: null
  provider_pool_caps: {}
  provider_traffic_caps: {}
```

### Verification

Run:

```bash
make test
```

Expected result:

- Tests run through `.venv/bin/python` when the venv exists.
- No `python: No such file or directory` failure.

---

## Task 2: Add Config Helpers With Required Grid Values

**Files:**

- Modify: `src/experiments.py`
- Modify: `tests/test_final_manifest.py`

**Purpose:** Make A1/A2/A3 grid construction config-driven. The orchestration path must not contain hardcoded A1/A2/A3 grid lists.

### Steps

- [ ] Add default feature flags near the existing config helpers.
- [ ] Add stable empty-output column constants for disabled optional outputs.
- [ ] Add config helpers for merged config, feature flags, required lists, solver limits, base rho, and budget selection.
- [ ] Add grid-builder helpers for A1/A2/A3 that read only from config.
- [ ] Replace `should_run_a4(skip_a3=False)` with `should_run_a4(config)`.
- [ ] Replace `needs_cascade_candidates(skip_a2=False, skip_a3=False)` with a version that receives `run_a4`.
- [ ] Add tests that prove `config/final.yaml` and `config/smoke.yaml` define explicit grids.

### Constants

Add after `RUN_LOG_SCHEMA_VERSION`:

```python
DEFAULT_FEATURES = {
    "empirical_recovery": False,
    "stress_tests": False,
    "a4_cvar": False,
    "three_stage": False,
    "lexicographic_a3": False,
    "provider_storage_constraints": False,
}

RECOVERY_COLUMNS = [
    "m1",
    "m2",
    "domain",
    "support",
    "recovery_rate",
    "fallback_level",
]

STRESS_RESULT_COLUMNS = [
    "policy",
    "scenario",
    "avg_quality",
    "avg_cost",
    "grid_id",
    "policy_label",
    "rho",
]

A4_RESULT_COLUMNS = [
    "policy",
    "status",
    "objective",
    "avg_quality",
    "avg_cost",
    "K",
    "B",
    "budget_name",
    "Emax",
    "beta",
    "lambda_cvar",
    "cvar_shortfall",
]
```

If `A4_RESULT_COLUMNS` already exists, update the existing constant instead of creating a duplicate.

### Config Helper Code

Add near `load_config`:

```python
def merged_experiment_config(config_path: str | Path | None) -> dict:
    """Load experiment config and fill feature defaults."""
    if config_path is None:
        config_path = Path("config/final.yaml")
    config = load_config(config_path)
    config["features"] = {
        **DEFAULT_FEATURES,
        **dict(config.get("features") or {}),
    }
    return config


def config_feature(config: dict, name: str) -> bool:
    """Return an experiment feature flag value."""
    return bool((config.get("features") or {}).get(name, DEFAULT_FEATURES.get(name, False)))


def config_section(config: dict, section: str) -> dict:
    """Return a named config section as a mapping."""
    value = config.get(section) or {}
    if not isinstance(value, dict):
        raise ValueError(f"Config section {section!r} must be a mapping.")
    return value


def required_config_list(config: dict, section: str, key: str) -> list:
    """Return a required list-valued config key."""
    value = config_section(config, section).get(key)
    if value is None:
        raise ValueError(f"Missing required config value: {section}.{key}")
    if isinstance(value, list):
        return list(value)
    if isinstance(value, tuple):
        return list(value)
    return [value]


def configured_time_limit(config: dict, cli_value: float | None) -> float:
    """CLI solver limits override config; config overrides code defaults."""
    if cli_value is not None:
        return float(cli_value)
    return float(config.get("time_limit", 60.0))


def configured_max_cascades(config: dict, cli_value: int | None) -> int:
    """CLI cascade limits override config; config overrides code defaults."""
    if cli_value is not None:
        return int(cli_value)
    return int(config.get("max_cascades", 250))


def configured_base_rho(config: dict) -> float:
    """Return the base two-stage recovery factor used for candidate generation."""
    return float(config.get("base_rho", 0.75))
```

Add after `budget_scenarios`:

```python
def configured_budget_items(config: dict, section: str, budgets: dict[str, float]) -> list[tuple[str, float]]:
    """Return budget name/value pairs requested by a model-family config section."""
    names = required_config_list(config, section, "budget_names")
    missing = [name for name in names if name not in budgets]
    if missing:
        raise ValueError(f"{section}.budget_names contains unknown budgets: {missing}")
    return [(name, budgets[name]) for name in names]


def build_a1_grid(config: dict, budgets: dict[str, float]) -> list[tuple[int, str, float]]:
    """Return configured A1 grid tuples: K, budget_name, budget."""
    return [
        (int(K), budget_name, float(budget))
        for K in required_config_list(config, "a1", "K")
        for budget_name, budget in configured_budget_items(config, "a1", budgets)
    ]


def build_a2_grid(config: dict, budgets: dict[str, float]) -> list[tuple[int, str, float, float]]:
    """Return configured A2 grid tuples: K, budget_name, budget, Emax."""
    return [
        (int(K), budget_name, float(budget), float(Emax))
        for K in required_config_list(config, "a2", "K")
        for budget_name, budget in configured_budget_items(config, "a2", budgets)
        for Emax in required_config_list(config, "a2", "Emax")
    ]


def build_a3_grid(config: dict, budgets: dict[str, float]) -> list[tuple[int, str, float, float, float, float, float]]:
    """Return configured A3 grid tuples: K, budget_name, budget, Emax, floor multiplier, slack penalty, rho."""
    return [
        (
            int(K),
            budget_name,
            float(budget),
            float(Emax),
            float(floor_multiplier),
            float(lambda_slack),
            float(rho),
        )
        for K in required_config_list(config, "a3", "K")
        for budget_name, budget in configured_budget_items(config, "a3", budgets)
        for Emax in required_config_list(config, "a3", "Emax")
        for floor_multiplier in required_config_list(config, "a3", "floor_multiplier")
        for lambda_slack in required_config_list(config, "a3", "lambda_slack")
        for rho in required_config_list(config, "a3", "rho")
    ]
```

Replace the current A4/cascade helpers with:

```python
def should_run_a4(config: dict) -> bool:
    """A4 is an optional extension and is disabled in default configs."""
    return config_feature(config, "a4_cvar")


def needs_cascade_candidates(skip_a2: bool = False, skip_a3: bool = False, run_a4: bool = False) -> bool:
    """Return whether any enabled model family needs cascade candidates."""
    return (not skip_a2) or (not skip_a3) or run_a4
```

### Tests

Append to `tests/test_final_manifest.py`:

```python
def test_default_configs_disable_extensions():
    from src.experiments import config_feature, merged_experiment_config

    for path in ["config/final.yaml", "config/smoke.yaml"]:
        config = merged_experiment_config(path)
        assert config_feature(config, "a4_cvar") is False
        assert config_feature(config, "stress_tests") is False
        assert config_feature(config, "empirical_recovery") is False
        assert config_feature(config, "three_stage") is False
        assert config_feature(config, "provider_storage_constraints") is False
        assert config_feature(config, "lexicographic_a3") is False


def test_final_config_a3_grid_is_core_sized():
    from src.experiments import build_a3_grid, merged_experiment_config

    config = merged_experiment_config("config/final.yaml")
    budgets = {"B_low": 0.01, "B_mid": 0.02, "B_high": 0.03}

    assert build_a3_grid(config, budgets) == [(5, "B_mid", 0.02, 0.75, 0.85, 0.1, 0.75)]


def test_smoke_config_a3_grid_is_tiny():
    from src.experiments import build_a3_grid, merged_experiment_config

    config = merged_experiment_config("config/smoke.yaml")
    budgets = {"B_low": 0.01, "B_mid": 0.02, "B_high": 0.03}

    assert build_a3_grid(config, budgets) == [(2, "B_mid", 0.02, 1.0, 0.75, 0.1, 0.75)]


def test_grid_builders_require_explicit_config_values():
    from src.experiments import build_a1_grid

    budgets = {"B_mid": 0.02}
    config = {"a1": {"budget_names": ["B_mid"]}}

    with pytest.raises(ValueError, match="a1.K"):
        build_a1_grid(config, budgets)
```

If `pytest` is not imported at the top of `tests/test_final_manifest.py`, add:

```python
import pytest
```

### Verification

Run:

```bash
make test
```

Expected result:

- Config-helper tests pass.
- The tests fail if an A1/A2/A3 config section omits its required grid values.

---

## Task 3: Wire Orchestration To Config And Define `base_rho`

**Files:**

- Modify: `run_experiments.py`
- Modify: `src/experiments.py`

**Purpose:** Make the execution path truly config-driven and remove undefined `rho` usage before cascade candidate generation.

### Steps

- [ ] In `run_experiments.py`, change `--time-limit` and `--max-cascades` defaults to `None`.
- [ ] In `src/experiments.py`, change `run_experiments` defaults for `time_limit` and `max_cascades` to `None`.
- [ ] Load config with `merged_experiment_config`.
- [ ] Compute `base_rho` before any cascade candidate generation.
- [ ] Use `base_rho` for cascade candidate generation.
- [ ] Execute empirical recovery only when `features.empirical_recovery` is true.
- [ ] Replace hardcoded A1/A2/A3 loop headers with `build_a1_grid`, `build_a2_grid`, and `build_a3_grid`.
- [ ] Execute A4 only when `features.a4_cvar` is true.
- [ ] Execute stress tests only when `features.stress_tests` is true.
- [ ] Execute lexicographic A3 only when `features.lexicographic_a3` is true.
- [ ] Keep disabled optional outputs as stable empty CSVs using existing repo table names.

### CLI Changes

In `run_experiments.py`, change:

```python
parser.add_argument("--time-limit", type=float, default=None)
parser.add_argument("--max-cascades", type=int, default=None)
```

### Orchestrator Signature

Change `src/experiments.py`:

```python
def run_experiments(
    data_path: str = "data/routerbench.csv",
    output_dir: str = "outputs",
    config_path: str | Path | None = None,
    skip_a1: bool = False,
    skip_a2: bool = False,
    skip_a3: bool = False,
    time_limit: float | None = None,
    max_cascades: int | None = None,
) -> dict:
```

### Config Load Block

Replace the config-load block with:

```python
config = merged_experiment_config(config_path)
time_limit = configured_time_limit(config, time_limit)
max_cascades = configured_max_cascades(config, max_cascades)
base_rho = configured_base_rho(config)
run_a4 = should_run_a4(config)
```

### Empirical Recovery Block

Use the existing table name `model_pair_recovery.csv`:

```python
if config_feature(config, "empirical_recovery"):
    recovery_df = estimate_pair_recovery(data)
    recovery_lookup = recovery_lookup_from_frame(recovery_df)
else:
    recovery_df = pd.DataFrame(columns=RECOVERY_COLUMNS)
    recovery_lookup = None
recovery_df.to_csv(root / "tables" / "model_pair_recovery.csv", index=False)
```

### Cascade Candidate Block

Use `base_rho`, not an undefined `rho`:

```python
cascades = pd.DataFrame()
params = {"A_p": {}, "R": {}, "C": {}, "Esc": {}}
if needs_cascade_candidates(skip_a2=skip_a2, skip_a3=skip_a3, run_a4=run_a4):
    cascades, params = generate_cascades(
        data,
        rho=base_rho,
        max_cascades=max_cascades,
        recovery_lookup=recovery_lookup,
    )
    cascades.to_csv(root / "tables" / "cascade_candidates.csv", index=False)
```

### Grid Loop Headers

Replace A1 loop header:

```python
for K, budget_name, budget in build_a1_grid(config, budgets):
```

Replace A2 loop header:

```python
for K, budget_name, budget, Emax in build_a2_grid(config, budgets):
```

Replace A3 loop header:

```python
for K, budget_name, budget, Emax, floor_multiplier, lambda_slack, rho_scenario in build_a3_grid(config, budgets):
```

Inside the A3 loop, use `rho_scenario` consistently when building rho-specific cascades:

```python
if rho_scenario not in rho_cascades:
    rho_cascades[rho_scenario] = generate_cascades(
        data,
        rho=rho_scenario,
        max_cascades=max_cascades,
        recovery_lookup=recovery_lookup,
    )
cascades_rho, params_rho = rho_cascades[rho_scenario]
```

Use this in A3 metadata rows:

```python
"rho": rho_scenario,
```

### Optional Output Gates

Use existing repo table names for disabled outputs:

```python
a4_results = []
if run_a4:
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
    if _has_assignment_result(result, "cascade_assignment"):
        audit_rows.extend(
            audit_cascade_result(
                data,
                cascades,
                params,
                result,
                K=result.get("K"),
                B=result.get("B"),
                Emax=result.get("Emax"),
            )
        )
    _record_solution_tables(
        a4_results,
        data,
        scenarios,
        params["R"],
        params["C"],
        output_rows=table_rows,
    )
write_json(root / "solutions" / "a4_solutions.json", {r["policy"]: r for r in a4_results})
pd.DataFrame(
    [
        _summary_row(
            r,
            family="A4",
            K=r.get("K"),
            B=r.get("B"),
            Emax=r.get("Emax"),
            budget_name=r.get("budget_name"),
            beta=r.get("beta"),
            lambda_cvar=r.get("lambda_cvar"),
            cvar_shortfall=r.get("cvar_shortfall"),
        )
        for r in a4_results
    ],
    columns=A4_RESULT_COLUMNS,
).to_csv(root / "tables" / "a4_cvar_results.csv", index=False)
```

```python
if config_feature(config, "stress_tests"):
    stress_config = config_section(config, "stress")
    stress_scenarios = sample_dirichlet_scenarios(
        data,
        n=int(stress_config.get("dirichlet_samples", 500)),
        concentration=float(stress_config.get("concentration", 40.0)),
        seed=int(config.get("random_seed", 164)),
    )
    stress_rows = []
    for result in representative:
        if result is None or result.get("status") not in {"ok", "optimal", "feasible", "feasible_time_limited"}:
            continue
        if "cascade_assignment" in result:
            stress_rho = result.get("rho", base_rho)
            params_rho = generate_cascades(
                data,
                rho=stress_rho,
                max_cascades=max_cascades,
                recovery_lookup=recovery_lookup,
            )[1]
            stress_rows.extend(
                evaluate_policy_under_scenarios(
                    result,
                    stress_scenarios,
                    params_rho["R"],
                    params_rho["C"],
                ).to_dict("records")
            )
        elif "assignment" in result:
            stress_rows.extend(
                evaluate_policy_under_scenarios(
                    result,
                    stress_scenarios,
                    data["q"],
                    data["c"],
                ).to_dict("records")
            )
    stress_results = pd.DataFrame(stress_rows)
else:
    stress_results = pd.DataFrame(columns=STRESS_RESULT_COLUMNS)
stress_results.to_csv(root / "tables" / "stress_test_results.csv", index=False)
```

For lexicographic A3, keep the existing solve call intact and wrap only the condition:

```python
if config_feature(config, "lexicographic_a3") and best_report is not None:
    lex_rho = best_report.get("rho", base_rho)
    if lex_rho not in rho_cascades:
        rho_cascades[lex_rho] = generate_cascades(
            data,
            rho=lex_rho,
            max_cascades=max_cascades,
            recovery_lookup=recovery_lookup,
        )
    cascades_rho, params_rho = rho_cascades[lex_rho]
    lex_result = solve_a3_lexicographic(
        data,
        cascades_rho,
        params_rho["R"],
        params_rho["C"],
        params_rho["Esc"],
        params_rho["A_p"],
        scenarios,
        compute_domain_floors(data, multiplier=best_report.get("floor_multiplier", 0.75)),
        K=best_report["K"],
        B=best_report["B"],
        Emax=best_report["Emax"],
        time_limit=time_limit,
    )
else:
    lex_result = {"status": "disabled", "lexicographic_passes": []}
```

### Verification

Run:

```bash
make test
rg -n "for K in \\[|for Emax in \\[|for floor_multiplier in \\[|rho=rho" src/experiments.py
```

Expected result:

- Tests pass.
- `rg` finds no hardcoded A1/A2/A3 grid loops in `run_experiments`.
- `rg` finds no `rho=rho` call before cascade candidate generation.

---

## Task 4: Preserve Existing Table Names And Add Aliases Where Helpful

**Files:**

- Modify: `src/experiments.py`
- Modify: `tests/test_final_manifest.py`

**Purpose:** Keep table names compatible with existing plotting, audit, and report code while allowing compatibility aliases for newer grid-result names.

### Steps

- [ ] Keep canonical table names already used by the repo.
- [ ] Do not rename `a0_results.csv`, `a1_results.csv`, `a2_results.csv`, `a3_results.csv`, `a3_grid_results.csv`, `model_pair_recovery.csv`, or `stress_test_results.csv`.
- [ ] Add aliases for `a1_grid_results.csv` and `a2_grid_results.csv`.
- [ ] Update tests to assert canonical names and aliases exist.

### Code

Add near the table constants in `src/experiments.py`:

```python
TABLE_ALIASES = {
    "a1_results.csv": ["a1_grid_results.csv"],
    "a2_results.csv": ["a2_grid_results.csv"],
}


def write_table_with_aliases(frame: pd.DataFrame, path: Path) -> None:
    """Write a canonical CSV and any compatibility aliases."""
    frame.to_csv(path, index=False)
    for alias in TABLE_ALIASES.get(path.name, []):
        frame.to_csv(path.with_name(alias), index=False)
```

Replace only the A1 and A2 result table writes:

```python
write_table_with_aliases(pd.DataFrame(a1_rows), root / "tables" / "a1_results.csv")
```

```python
write_table_with_aliases(pd.DataFrame(a2_rows), root / "tables" / "a2_results.csv")
```

Keep A3 writes as the existing canonical pair:

```python
pd.DataFrame(a3_grid_rows).to_csv(root / "tables" / "a3_grid_results.csv", index=False)
pd.DataFrame(a3_rows).to_csv(root / "tables" / "a3_results.csv", index=False)
```

### Verification

Run:

```bash
make test
```

Expected result:

- Existing plot code still reads canonical table names.
- Tests can also assert `a1_grid_results.csv` and `a2_grid_results.csv` exist as aliases.

---

## Task 5: Add Four-Domain Integration Test

**Files:**

- Modify: `tests/test_final_manifest.py`

**Purpose:** Protect the config-driven core path with a realistic tiny fixture that exercises all four domains and A3 scenario logic.

### Steps

- [ ] Add a four-domain RouterBench-like fixture.
- [ ] Run `run_experiments` on the tiny fixture with a tiny config written inside the test.
- [ ] Assert required canonical tables exist.
- [ ] Assert A1/A2 aliases exist.
- [ ] Assert disabled optional tables exist and are empty.
- [ ] Assert A3 domain slack output includes AIME, GPQA, LCB, and MMLU-Pro.
- [ ] Assert A3 scenario metrics are non-empty.

### Imports

If missing, add:

```python
import pandas as pd
import pytest
```

### Test Code

Append:

```python
@pytest.fixture
def four_domain_routerbench_csv(tmp_path):
    rows = []
    row_id = 0
    domains = ["AIME", "GPQA", "LCB", "MMLU-Pro"]
    models = ["free", "cheap", "balanced", "strong"]
    costs = {"free": 0.0, "cheap": 0.0001, "balanced": 0.0002, "strong": 0.0004}
    for domain_idx, domain in enumerate(domains):
        for prompt_idx in range(2):
            prompt_id = f"{domain}-{prompt_idx}"
            for model_idx, model in enumerate(models):
                rows.append(
                    {
                        "row_id": row_id,
                        "dataset": domain,
                        "prompt_id": prompt_id,
                        "index": prompt_idx,
                        "model": model,
                        "score": int(model == "strong" or (domain_idx + prompt_idx + model_idx) % 3 == 0),
                        "cost": costs[model],
                        "prompt_tokens": 100 + prompt_idx,
                        "completion_tokens": 20 + model_idx,
                    }
                )
                row_id += 1
    path = tmp_path / "routerbench.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def test_core_config_writes_required_artifacts_for_four_domains(four_domain_routerbench_csv, tmp_path):
    from src.experiments import run_experiments

    config_path = tmp_path / "core.yaml"
    config_path.write_text(
        """
random_seed: 164
profile: test_core
time_limit: 20
max_cascades: 30
base_rho: 0.75
features:
  empirical_recovery: false
  stress_tests: false
  a4_cvar: false
  three_stage: false
  lexicographic_a3: false
  provider_storage_constraints: false
a1:
  K: [1, 2]
  budget_names: ["B_mid"]
a2:
  K: [2]
  budget_names: ["B_mid"]
  Emax: [1.0]
a3:
  K: [2]
  budget_names: ["B_mid"]
  Emax: [1.0]
  floor_multiplier: [0.75]
  lambda_slack: [0.1]
  rho: [0.75]
matched_report:
  K: 2
  budget_name: "B_mid"
  Emax: 1.0
  floor_multiplier: 0.75
  lambda_slack: 0.1
  rho: 0.75
stress:
  dirichlet_samples: 0
  concentration: 40.0
production_constraints:
  storage_cap_gb: null
  provider_pool_caps: {}
  provider_traffic_caps: {}
""",
        encoding="utf-8",
    )

    output_dir = tmp_path / "outputs"
    run_experiments(
        data_path=str(four_domain_routerbench_csv),
        output_dir=str(output_dir),
        config_path=config_path,
    )

    required_tables = [
        "a0_results.csv",
        "a1_results.csv",
        "a1_grid_results.csv",
        "a2_results.csv",
        "a2_grid_results.csv",
        "a3_results.csv",
        "a3_grid_results.csv",
        "a3_domain_slacks.csv",
        "a3_scenario_metrics.csv",
        "model_pair_recovery.csv",
        "report_main_comparison.csv",
        "solver_diagnostics.csv",
        "stress_test_results.csv",
        "a4_cvar_results.csv",
    ]
    for name in required_tables:
        assert (output_dir / "tables" / name).exists(), name

    assert (output_dir / "manifest.json").exists()
    assert (output_dir / "RUN_LOG.md").exists()
    assert pd.read_csv(output_dir / "tables" / "model_pair_recovery.csv").empty
    assert pd.read_csv(output_dir / "tables" / "stress_test_results.csv").empty
    assert pd.read_csv(output_dir / "tables" / "a4_cvar_results.csv").empty

    slacks = pd.read_csv(output_dir / "tables" / "a3_domain_slacks.csv")
    assert set(slacks["domain"]) == {"AIME", "GPQA", "LCB", "MMLU-Pro"}

    scenario_metrics = pd.read_csv(output_dir / "tables" / "a3_scenario_metrics.csv")
    assert not scenario_metrics.empty
```

### Verification

Run:

```bash
make test
```

Expected result:

- The tiny integration test passes.
- A3 domain slack and scenario metric artifacts are exercised across all four domains.

---

## Task 6: Document Default Path And Extension Flags

**Files:**

- Modify: `docs/RUNBOOK.md`

**Purpose:** Make the default submission path and extension policy clear.

### Steps

- [ ] Add a "Default Submission Path" section.
- [ ] Include the exact acceptance commands.
- [ ] State that `run-final` writes `outputs_final.tmp` first and promotes only after success.
- [ ] List disabled-by-default extension flags.

### Documentation Patch

Add near the top of `docs/RUNBOOK.md`:

```markdown
## Default Submission Path

The default final run is the A0-A3 report path.

Run:

    make test
    make run-smoke
    make run-final
    make audit
    make plots
    make test

`make run-smoke` writes `outputs_smoke.tmp` first and promotes it to `outputs_smoke` only after success.

`make run-final` writes `outputs_final.tmp` first and promotes it to `outputs_final` only after success. If the experiment command fails, the existing `outputs_final` directory is left in place.

The default path includes A0, A1, A2, A3, solver diagnostics, solution audit, report comparison tables, and required figures.

The following extensions remain in the repository but are disabled by default in `config/final.yaml` and `config/smoke.yaml`:

- `features.empirical_recovery`
- `features.a4_cvar`
- `features.three_stage`
- `features.lexicographic_a3`
- `features.provider_storage_constraints`
- `features.stress_tests`

Only include extension outputs in the final report when the extension is stable, audited, and easy to explain in the 4-5 page narrative.
```

### Verification

Run:

```bash
make test
```

Expected result:

- Documentation change does not affect tests.

---

## Task 7: Run Final Acceptance Commands

**Files:**

- Source/config/docs/tests changed above
- Generated `outputs_smoke/`
- Generated `outputs_final/`

**Purpose:** Prove the default final path works end-to-end without requiring disabled extensions.

### Steps

- [ ] Run the full acceptance sequence.
- [ ] Do not manually delete `outputs_final` before `make run-final`; the Makefile target handles temporary output and promotion.
- [ ] Inspect final tables and figures.
- [ ] Confirm no temporary output directory remains after successful promotion.

### Commands

Run:

```bash
make test
make run-smoke
make run-final
make audit
make plots
make test
```

Inspect:

```bash
ls outputs_final/tables
ls outputs_final/figures
test ! -d outputs_final.tmp
test ! -d outputs_smoke.tmp
sed -n '1,80p' outputs_final/RUN_LOG.md
```

Required canonical tables:

```text
outputs_final/tables/a0_results.csv
outputs_final/tables/a1_results.csv
outputs_final/tables/a2_results.csv
outputs_final/tables/a3_results.csv
outputs_final/tables/a3_grid_results.csv
outputs_final/tables/report_main_comparison.csv
outputs_final/tables/solver_diagnostics.csv
outputs_final/tables/solution_audit.csv
```

Compatibility aliases:

```text
outputs_final/tables/a1_grid_results.csv
outputs_final/tables/a2_grid_results.csv
```

Disabled optional tables should exist and be empty:

```text
outputs_final/tables/a4_cvar_results.csv
outputs_final/tables/model_pair_recovery.csv
outputs_final/tables/stress_test_results.csv
```

### Verification

Expected result:

- `make test` passes before and after runs.
- `make run-smoke` completes and promotes `outputs_smoke.tmp` to `outputs_smoke`.
- `make run-final` completes and promotes `outputs_final.tmp` to `outputs_final`.
- `make audit` writes `outputs_final/tables/solution_audit.csv`.
- `make plots` completes against `outputs_final`.
- `outputs_final/tables/report_main_comparison.csv` contains A0, A1, A2, and A3 rows.
- `outputs_final/tables/solver_diagnostics.csv` includes solver metadata for A1/A2/A3.

---

## Task 8: Commit Focused Changes

**Files:**

- `Makefile`
- `config/final.yaml`
- `config/smoke.yaml`
- `run_experiments.py`
- `src/experiments.py`
- `tests/test_final_manifest.py`
- `docs/RUNBOOK.md`
- selected verified `outputs_final/` artifacts if they are part of the submission package

**Purpose:** Preserve the implementation in one reviewable commit without staging unrelated files.

### Steps

- [ ] Check worktree status.
- [ ] Review source/config/test/docs diff.
- [ ] Stage only files from this plan plus verified final artifacts requested for submission.
- [ ] Do not stage unrelated pre-existing untracked files.
- [ ] Commit with a focused message.

### Commands

Run:

```bash
git status --short
git diff -- Makefile config/final.yaml config/smoke.yaml run_experiments.py src/experiments.py tests/test_final_manifest.py docs/RUNBOOK.md
```

Stage source/config/test/docs:

```bash
git add Makefile config/final.yaml config/smoke.yaml run_experiments.py src/experiments.py tests/test_final_manifest.py docs/RUNBOOK.md
```

Stage verified final artifacts only when they are required for submission:

```bash
git add outputs_final/manifest.json outputs_final/RUN_LOG.md outputs_final/report_artifacts outputs_final/tables/report_main_comparison.csv outputs_final/tables/solver_diagnostics.csv outputs_final/tables/solution_audit.csv outputs_final/figures
```

Commit:

```bash
git commit -m "feat: make final pipeline core A0-A3 by default"
```

### Verification

Run:

```bash
git status --short
git show --stat --oneline HEAD
```

Expected result:

- Commit contains focused source/config/test/docs changes and selected verified artifacts.
- Optional modules remain in the repository.
- Unrelated untracked files remain unstaged.

---

## Self-Review Checklist

Before reporting completion:

- [ ] No optional module was deleted.
- [ ] A4, empirical recovery, stress tests, three-stage cascades, provider/storage constraints, and lexicographic A3 are disabled by default.
- [ ] `make run-smoke` writes `outputs_smoke.tmp` and promotes only after success.
- [ ] `make run-final` writes `outputs_final.tmp` and promotes only after success.
- [ ] Makefile Python preference is `.venv/bin/python`, then `python3`, then `python`.
- [ ] `src/experiments.py` A1/A2/A3 orchestration grids are read from config.
- [ ] `base_rho` is defined before cascade candidate generation.
- [ ] Canonical existing table names remain available.
- [ ] Compatibility aliases exist for `a1_grid_results.csv` and `a2_grid_results.csv`.
- [ ] Four-domain integration test covers A3 domain slack and scenario metric output.
- [ ] `make test`, `make run-smoke`, `make run-final`, `make audit`, `make plots`, and `make test` all pass.
