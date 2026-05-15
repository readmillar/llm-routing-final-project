# LLM Routing Project Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a reproducible Pyomo pipeline for the INDENG 164 final project that loads the locked `routerbench.csv`, solves A0/A1/A2/A3 routing policies, and writes report-ready tables and figures.

**Architecture:** Keep optimization logic in focused Python modules under `src/` and keep the starter notebook as reference only. The pipeline uses observed prompt-model pairs only, precomputes cascade parameters to keep all Pyomo models linear MILPs, records infeasible grid points instead of crashing, and writes every generated artifact under `outputs/`.

**Tech Stack:** Python 3.13, pandas, numpy, Pyomo, highspy/HiGHS with CBC/GLPK fallback, matplotlib, pytest.

---

## Scope Check

The PRD describes one coherent subsystem: a robust reliability-aware LLM routing optimizer. The project must preserve the incomplete availability grid in the real CSV: 7,860 observed rows, 240 prompts, 33 models, 4 domains, and 60 missing pairs for `MMLU-Pro` x `deepseek-v3.1-terminus`.

Because the implementation spans multiple modules and exceeds 200 lines, execution should proceed in small module-sized edits with tests and sanity checks after each milestone.

## File Structure

- Create `/Users/alexandermillar/INDENG 164/requirements.txt`: installable runtime dependencies.
- Create `/Users/alexandermillar/INDENG 164/AGENTS.md`: project-specific locked-CSV build rules.
- Create `/Users/alexandermillar/INDENG 164/data/routerbench.csv`: canonical input copied from Downloads.
- Create `/Users/alexandermillar/INDENG 164/src/__init__.py`: package marker.
- Create `/Users/alexandermillar/INDENG 164/src/load_data.py`: column detection, CSV loading, canonical long-format conversion.
- Create `/Users/alexandermillar/INDENG 164/src/preprocessing.py`: validation, `PM`, `M_p`, `P_d`, `q`, `r`, `c`, missing-pair summaries.
- Create `/Users/alexandermillar/INDENG 164/src/metrics.py`: assignment metrics, domain metrics, scenario weights, selected-model usage.
- Create `/Users/alexandermillar/INDENG 164/src/baselines.py`: always-cheapest, always-best-quality, A0 weighted routing.
- Create `/Users/alexandermillar/INDENG 164/src/solver_utils.py`: Pyomo solver fallback and JSON output helpers.
- Create `/Users/alexandermillar/INDENG 164/src/pyomo_single_shot.py`: A1 single-shot portfolio MILP.
- Create `/Users/alexandermillar/INDENG 164/src/pyomo_cascade.py`: cascade generation and A2 cascade MILP.
- Create `/Users/alexandermillar/INDENG 164/src/pyomo_robust_cascade.py`: robust scenarios, domain floors, A3 robust cascade MILP.
- Create `/Users/alexandermillar/INDENG 164/src/experiments.py`: budget grid, model grids, result tables, solution JSON.
- Create `/Users/alexandermillar/INDENG 164/src/plots.py`: five required matplotlib figures.
- Create `/Users/alexandermillar/INDENG 164/run_experiments.py`: CLI entrypoint.
- Create `/Users/alexandermillar/INDENG 164/tests/test_sanity.py`: synthetic-data regression tests with comments on every test case.
- Create `/Users/alexandermillar/INDENG 164/README.md`: setup, run, output, and report workflow.

## Tasks

### Task 1: Skeleton and Locked Data

**Files:**
- Create: `/Users/alexandermillar/INDENG 164/requirements.txt`
- Create: `/Users/alexandermillar/INDENG 164/AGENTS.md`
- Create: `/Users/alexandermillar/INDENG 164/data/routerbench.csv`

- [ ] **Step 1: Add dependencies**

```text
pandas>=2.2
numpy>=2.0
pyomo>=6.8
highspy>=1.10
matplotlib>=3.9
pytest>=8.0
```

- [ ] **Step 2: Copy locked data**

Run: `mkdir -p data outputs/tables outputs/figures outputs/solutions src tests && cp /Users/alexandermillar/Downloads/routerbench.csv data/routerbench.csv`

Expected: `data/routerbench.csv` exists with 7,860 rows.

- [ ] **Step 3: Add project AGENTS rules**

Use the locked AGENTS text from `/Users/alexandermillar/Downloads/AGENTS_LLM_Routing_Project_v2.1_STARTER_CSV_LOCKED.md`.

### Task 2: Data Loading and Preprocessing

**Files:**
- Create: `/Users/alexandermillar/INDENG 164/src/load_data.py`
- Create: `/Users/alexandermillar/INDENG 164/src/preprocessing.py`
- Test: `/Users/alexandermillar/INDENG 164/tests/test_sanity.py`

- [ ] **Step 1: Write failing tests**

Add tests for normalization, zero-cost preservation, missing locked columns, and known missing-pair detection. Each test must include a docstring describing the covered case.

- [ ] **Step 2: Implement loader**

Implement `load_raw_data`, `detect_columns`, `standardize_long_format`, and `load_dataset`. Exact canonical columns: `prompt_id`, `domain`, `model`, `quality`, `cost`.

- [ ] **Step 3: Implement preprocessing**

Implement `normalize_quality`, `validate_data`, `build_sets_and_matrices`, and `write_data_summaries`. The returned data dictionary must include `P`, `M`, `D`, `PM`, `M_p`, `P_d`, `prompt_domain`, `q`, `r`, `c`, `df`, and `missing_pairs`.

- [ ] **Step 4: Verify**

Run: `pytest tests/test_sanity.py -v`

Expected: preprocessing tests pass.

### Task 3: Metrics and Baselines

**Files:**
- Create: `/Users/alexandermillar/INDENG 164/src/metrics.py`
- Create: `/Users/alexandermillar/INDENG 164/src/baselines.py`
- Test: `/Users/alexandermillar/INDENG 164/tests/test_sanity.py`

- [ ] **Step 1: Write failing tests**

Add tests that always-cheapest, always-best-quality, and weighted A0 assign every prompt exactly once and that scenario prompt weights sum to one.

- [ ] **Step 2: Implement metrics**

Implement `assignment_metrics`, `cascade_assignment_metrics`, `domain_quality_rows`, `scenario_weights`, `scenario_quality`, and `usage_rows`.

- [ ] **Step 3: Implement baselines**

Implement `solve_always_cheapest`, `solve_always_best_quality`, `solve_weighted_baseline`, and `run_weighted_baselines` over prompt-specific `M_p`.

- [ ] **Step 4: Verify**

Run: `pytest tests/test_sanity.py -v`

Expected: all non-solver tests pass.

### Task 4: A1 Single-Shot MILP

**Files:**
- Create: `/Users/alexandermillar/INDENG 164/src/solver_utils.py`
- Create: `/Users/alexandermillar/INDENG 164/src/pyomo_single_shot.py`
- Test: `/Users/alexandermillar/INDENG 164/tests/test_sanity.py`

- [ ] **Step 1: Write failing test**

Add a synthetic-data test that A1 selects at most `K` models, assigns every prompt, and routes only through selected models.

- [ ] **Step 2: Implement solver fallback**

Try `appsi_highs`, `highs`, `cbc`, then `glpk`; apply a time limit when supported.

- [ ] **Step 3: Implement A1**

Create `y[m]` for all models and `x[p,m]` only for observed `PM`. Maximize average quality subject to assignment, pool-size, linkage, and average-cost budget constraints.

- [ ] **Step 4: Verify**

Run: `pytest tests/test_sanity.py::test_a1_solution_assigns_each_prompt_and_respects_selected_pool -v`

Expected: pass when HiGHS is installed; skip with a clear reason when no solver is available.

### Task 5: A2 Cascade MILP

**Files:**
- Create: `/Users/alexandermillar/INDENG 164/src/pyomo_cascade.py`
- Test: `/Users/alexandermillar/INDENG 164/tests/test_sanity.py`

- [ ] **Step 1: Write failing tests**

Add tests for cascade parameter bounds and selected-model linkage.

- [ ] **Step 2: Implement cascade generation**

Cheap first-stage models are bottom 30% by average cost or have any zero-cost rows. Strong second-stage models are top 50% by average quality. Keep `m1 != m2`, `qbar[m2] >= qbar[m1]`, and prompt feasibility only when both model rows exist.

- [ ] **Step 3: Implement A2**

Create `z[p,a]` only for feasible prompt-cascade pairs. Maximize average `R` subject to one cascade per prompt, selected-model linkage, pool-size cap, average expected cost budget, and average escalation cap.

- [ ] **Step 4: Verify**

Run: `pytest tests/test_sanity.py::test_cascades_have_expected_parameters tests/test_sanity.py::test_a2_solution_uses_only_selected_models -v`

Expected: pass with a solver installed.

### Task 6: A3 Robust Cascade MILP

**Files:**
- Create: `/Users/alexandermillar/INDENG 164/src/pyomo_robust_cascade.py`
- Test: `/Users/alexandermillar/INDENG 164/tests/test_sanity.py`

- [ ] **Step 1: Write failing tests**

Add tests that robust scenarios produce unit-sum prompt weights and A3 returns `eta`, scenario metrics, and domain slacks.

- [ ] **Step 2: Implement scenarios and floors**

Implement empirical, balanced, coding-heavy, math-heavy, and knowledge-heavy domain weights. Compute `tau[d] = 0.90 * mean_p max_m q[p,m]` for each domain.

- [ ] **Step 3: Implement A3**

Extend A2 with `eta`, nonnegative domain slack `s[d]`, scenario quality lower bounds, scenario cost budgets, and domain floor constraints. Objective: maximize `eta - lambda_slack * sum_d s[d]`.

- [ ] **Step 4: Verify**

Run: `pytest tests/test_sanity.py::test_a3_returns_robust_metrics -v`

Expected: pass with a solver installed.

### Task 7: Experiments, Plots, and CLI

**Files:**
- Create: `/Users/alexandermillar/INDENG 164/src/experiments.py`
- Create: `/Users/alexandermillar/INDENG 164/src/plots.py`
- Create: `/Users/alexandermillar/INDENG 164/run_experiments.py`
- Create: `/Users/alexandermillar/INDENG 164/README.md`

- [ ] **Step 1: Implement experiments**

Compute the data-derived budget grid, run A0, A1 grid `K=[1,2,3,5,8]`, A2 grid `K=[2,3,5]`, `Emax=[1.0,0.75,0.5]`, and A3 final `K=3`. Save required CSVs and JSON solution files under `outputs/`.

- [ ] **Step 2: Implement plots**

Generate `cost_quality_frontier.png`, `quality_vs_pool_size.png`, `selected_model_usage.png`, `domain_performance_comparison.png`, and `robustness_heatmap.png` from output CSVs.

- [ ] **Step 3: Implement CLI**

Support `--data`, `--output-dir`, `--skip-a1`, `--skip-a2`, `--skip-a3`, `--only-plots`, `--time-limit`, and `--max-cascades`.

- [ ] **Step 4: Verify full project**

Run: `python run_experiments.py --data data/routerbench.csv --output-dir outputs --time-limit 60 --max-cascades 250`

Expected: all required tables and figures are created; infeasible or time-limited grid points are recorded in CSVs instead of crashing.

## Self-Review

- Spec coverage: P0-1 through P0-9 are covered by Tasks 1-7. P1-2 is included through usage summaries; P1-1 and P1-3 remain optional and are not needed for the locked final build.
- Placeholder scan: the plan has no `TBD`, `TODO`, or undefined task outputs.
- Type consistency: all modules pass the same data dictionary containing `P`, `M`, `D`, `PM`, `M_p`, `P_d`, `prompt_domain`, `q`, `r`, `c`, and `df`.
