# AGENTS.md — LLM Routing Optimization Project

You are working on an INDENG 164 final project. Build a reproducible Pyomo pipeline for LLM routing using the real starter CSV.

## Source data

Use `data/routerbench.csv`.

Exact columns:
`row_id`, `dataset`, `prompt_id`, `index`, `model`, `score`, `cost`, `prompt_tokens`, `completion_tokens`.

Map:
- `dataset` → domain
- `score` → quality
- `cost` → cost

Facts to preserve:
- 7,860 observed rows
- 240 prompts
- 33 models
- 4 domains: AIME, GPQA, LCB, MMLU-Pro
- 60 missing prompt-model pairs
- Missing pairs are all MMLU-Pro × `deepseek-v3.1-terminus`
- `score` is binary `{0,1}` and already normalized
- cost is nonnegative and includes zero-cost rows/models

## Availability rule

Do not assume a complete prompt-model grid. Do not impute missing pairs. Use:

- `E = {(p,m): row exists}`
- `M_p = {m: (p,m) in E}`
- `A_p = {a=(m1,m2): (p,m1) in E and (p,m2) in E}`

All assignment variables must be defined only over available pairs/cascades.

## Core models

- A0 weighted baseline over `M_p`.
- A1 single-shot portfolio MILP with `x[p,m]` only for `(p,m) in E`.
- A2 two-stage cascade MILP with `z[p,a]` only for `a in A_p`.
- A3 robust reliability-aware cascade MILP using scenario-weighted SAA and domain slacks.

## Cascade assumptions

Use `q[p,m] = score[p,m]` and `r[p,m] = q[p,m]`.
Use recovery factor `rho = 0.75`.

For cascade `a=(m1,m2)`:

```text
R[p,a]   = r[p,m1] + (1-r[p,m1]) * rho * r[p,m2]
C[p,a]   = c[p,m1] + (1-r[p,m1]) * c[p,m2]
Esc[p,a] = 1-r[p,m1]
```

## Implementation rules

Do:
- Implement in small steps.
- Run sanity checks after each module.
- Save outputs to `outputs/tables`, `outputs/figures`, `outputs/solutions`.
- Keep all optimization formulations linear MILPs.
- Use Pyomo, pandas, numpy, matplotlib.
- Prefer appsi_highs/highs, then cbc, then glpk.
- Preserve zero-cost models and costs.
- Write docstrings and clear comments for report reuse.
- Record infeasible grid points instead of crashing the pipeline.

Do not:
- Add embeddings, learned classifiers, API calls, dashboards, or custom solvers.
- Implement three-stage cascades unless all required two-stage outputs are complete.
- Hide logic in notebooks only.
- Drop `deepseek-v3.1-terminus` globally just because MMLU-Pro rows are missing.
- Fill missing scores/costs with zero.

## Critical report facts

- Stochasticity enters through first-stage cascade success/failure.
- Multi-stage decision enters through escalation to a second model.
- Robustness enters through scenario-weighted SAA over prompt distributions.
- Realistic constraints include pool size, budget, escalation cap, and domain quality floors.
