# Codex Prompt Book v2.1 — Starter CSV Locked
## LLM Routing Optimization Project

Use this file as the build guide for Codex. The PRD is `PRD_LLM_Routing_Project_v2.1_STARTER_CSV_LOCKED.md`.

---

## Master Prompt

```text
You are building a Pyomo optimization project for INDENG 164.

Project title:
Cheap First, Reliable Always: A Robust Cascading Optimization Framework for LLM Routing

Goal:
Build a reproducible Python/Pyomo pipeline that selects a small LLM model pool and routes 240 prompts across 33 candidate models using four policy variants:
A0 weighted baseline, A1 single-shot portfolio MILP, A2 two-stage stochastic cascade MILP, and A3 robust reliability-aware cascade MILP.

Locked data:
Use data/routerbench.csv as the canonical starter dataset.
Actual columns:
row_id,dataset,prompt_id,index,model,score,cost,prompt_tokens,completion_tokens

Column mapping:
- domain = dataset
- quality = score
- prompt_id = prompt_id
- model = model
- cost = cost

Observed data profile:
- 7,860 rows
- 240 prompts
- 33 models overall
- 4 domains: AIME, GPQA, LCB, MMLU-Pro
- 60 prompts per domain
- score is binary and already in [0,1]
- several zero-cost/open-source models
- deepseek-v3.1-terminus is missing for all 60 MMLU-Pro prompts

Critical data rule:
Do not assume a full 240 x 33 grid. Build availability sets:
- E = observed prompt-model pairs
- M_p = models available for prompt p
- A_p = cascades whose two models are both available for prompt p
Do not impute missing values.

Locked scope:
- Two-stage cascades only.
- No embeddings or learned classifiers.
- No live API calls.
- Use Pyomo + pandas + numpy + matplotlib.
- Try solvers in this order: appsi_highs, highs, cbc, glpk.
- Write outputs to outputs/tables, outputs/figures, outputs/solutions.

Models:
A0: For alpha in [0.1,0.5,1,2,5,10], route each prompt to argmin over m in M_p of cost[p,m] - alpha*quality[p,m].

A1: Binary y[m] for all models and x[p,m] only for observed pairs (p,m) in E. Maximize average quality subject to one available model per prompt, x[p,m] <= y[m], sum y <= K, average cost <= B.

A2: Generate candidate cascades a=(m1,m2), cheap first-stage and strong second-stage. For each prompt p, only allow cascade a if both models are in M_p. Precompute:
R[p,a] = r[p,m1] + (1-r[p,m1]) * rho * r[p,m2], with rho=0.75
C[p,a] = c[p,m1] + (1-r[p,m1]) * c[p,m2]
Esc[p,a] = 1-r[p,m1]
Binary y[m] and z[p,a] only for a in A_p. Maximize average R subject to one available cascade per prompt, z[p,a] <= y[m] for both cascade models, sum y <= K, average C <= B, average Esc <= Emax.

A3: Extend A2 with robust scenarios over domain weights. Add eta and slack s[d]. Maximize eta - lambda_slack*sum s[d]. Add scenario_quality[s] >= eta, scenario_cost[s] <= B, and domain_quality[d] + s[d] >= tau[d].

Required outputs:
- data_summary.csv, model_summary.csv, missing_pairs.csv, budget_grid.csv
- a0_results.csv, a1_results.csv, a2_results.csv, a3_results.csv
- summary_comparison.csv, domain_quality.csv, selected_model_usage.csv
- cost_quality_frontier.png
- quality_vs_pool_size.png
- selected_model_usage.png
- domain_performance_comparison.png
- robustness_heatmap.png

Build order:
1. load/preprocess data
2. baselines
3. A1
4. cascade generation
5. A2
6. A3
7. plots
8. summary tables

Do not jump ahead. After each step, run a sanity check and save outputs.
```

---

## Step Prompts

### 1. Data loader

```text
Build src/load_data.py and src/preprocessing.py. Load data/routerbench.csv. Detect columns but default to prompt_id=prompt_id, domain=dataset, model=model, quality=score, cost=cost. Validate 7,860 rows, 240 prompts, 33 models overall, 4 domains, 60 prompts per domain, zero duplicate prompt-model pairs, and 60 missing prompt-model pairs. Save missing_pairs.csv showing that deepseek-v3.1-terminus is absent for MMLU-Pro. Build E, M_p, P_m, and prompt_domain. Score is already in [0,1], so set q=score.astype(float) and r=q for observed pairs. Preserve zero-cost models. Save data_summary.csv and model_summary.csv.
```

### 2. Baselines

```text
Build src/baselines.py. Implement always_cheapest, always_best_quality, and A0 weighted baseline for alpha in [0.1,0.5,1,2,5,10]. For each prompt, optimize only over m in M_p. Each policy should return assignments, avg_cost, avg_quality, domain_quality, and model usage counts. Save baseline_extremes.csv and a0_results.csv.
```

### 3. A1 MILP

```text
Build src/solver_utils.py and src/pyomo_single_shot.py. Implement get_solver with appsi_highs/highs/cbc/glpk fallback. Implement solve_a1(data,K,B): binary y[m] for all models and x[p,m] only for (p,m) in E. Objective: maximize average quality. Constraints: sum_{m in M_p} x[p,m] = 1 for every prompt, x[p,m] <= y[m], sum_m y[m] <= K, average cost <= B. Return selected models, assignment, avg quality/cost, domain quality, status, and solver name. Add infeasibility handling.
```

### 4. Cascade generation

```text
In src/pyomo_cascade.py, implement summarize_models and generate_cascades. Compute qbar/cbar over each model's observed rows. Cheap first-stage models are bottom 30% average cost or zero-cost. Strong second-stage models are top 50% average quality. Allowed global cascades require m1 != m2 and qbar[m2] >= qbar[m1]. Then build prompt-specific A_p by retaining only cascades where m1 and m2 are both available for p. Precompute R, C, Esc with rho=0.75 only for (p,a) where a in A_p. Cap global cascades to max_cascades=250 if needed and save cascade_candidates.csv.
```

### 5. A2 MILP

```text
In src/pyomo_cascade.py, implement solve_a2(data,cascades,R,C,Esc,K,B,Emax,rho). Binary y[m] and z[p,a] only for a in A_p. Objective: maximize average R. Constraints: sum_{a in A_p} z[p,a] = 1 for every prompt, z[p,a] <= y[m1] and z[p,a] <= y[m2], sum_m y[m] <= K, average C <= B, average Esc <= Emax. Return selected_models, cascade_assignment, avg_quality, avg_cost, escalation_rate, domain_quality, stage usage counts, and status.
```

### 6. A3 robust MILP

```text
Build src/pyomo_robust_cascade.py. Implement build_scenarios, compute_domain_floors, and solve_a3. Use empirical, balanced, coding_heavy, math_heavy, and knowledge_heavy domain weights. Convert domain weights to prompt weights w[s,p]. Extend A2 using z[p,a] only for a in A_p. Add eta and s[d]. Objective: maximize eta - lambda_slack*sum(s[d]). Add scenario_quality[s] >= eta, scenario_cost[s] <= B, and domain_quality[d] + s[d] >= tau[d]. Return eta, scenario metrics, domain slacks, selected models, assignments, and status.
```

### 7. Experiments

```text
Build src/experiments.py. Compute the data-derived budget grid using available model sets M_p: always-cheapest is mean_p min_{m in M_p} c[p,m]; quality-oracle cost is mean_p min cost among models tied for max quality for each prompt. Run A0, A1 grid K=[1,2,3,5,8] x B=[low,mid,high], A2 grid K=[2,3,5] x B=[low,mid,high] x Emax=[1.0,0.75,0.5], and A3 final K=3 at selected budget. Save all result tables and JSON solutions.
```

### 8. Plots

```text
Build src/plots.py using matplotlib only. Generate cost_quality_frontier.png, quality_vs_pool_size.png, selected_model_usage.png, domain_performance_comparison.png, and robustness_heatmap.png. Read outputs/tables CSVs and save figures to outputs/figures.
```

### 9. Orchestration and README

```text
Build run_experiments.py with CLI flags --data, --output-dir, --skip-a1, --skip-a2, --skip-a3, --only-plots, --time-limit, --max-cascades. Default --data should be data/routerbench.csv. Add README.md explaining setup, install, running experiments, output files, and report workflow.
```

---

## Repo AGENTS.md text

Use `AGENTS_LLM_Routing_Project_v2.1_STARTER_CSV_LOCKED.md` as the root `AGENTS.md` in the Codex repo.
