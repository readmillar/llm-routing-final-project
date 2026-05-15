# PRD v2.0 — LOCKED BUILD SPEC
## Cheap First, Reliable Always: A Robust Cascading Optimization Framework for LLM Routing

**Course:** INDENG 164 — Final Project  
**Author:** Read Millar  
**Final due:** Friday, May 15, 2026, 11:59pm America/Los_Angeles  
**Document version:** v2.1 — data-locked, starter CSV resolved  
**Status:** Finalized for Codex/build agent implementation with `routerbench.csv` as the locked input data  
**Primary deliverable:** 4–5 page report + Pyomo code + figures/tables  
**Build order:** A0 → A1 → A2 → A3 → plots/tables → report

---

## 0. Scope Lock

This project will implement one coherent optimization system rather than three separate ideas:

> **A robust, reliability-aware two-stage cascading router for selecting a small LLM model pool and routing prompts under cost, quality, uncertainty, and deployment constraints.**

The final report will compare four policies:

| Policy | Role in project | Required? |
|---|---|---:|
| A0 weighted baseline | Starter-code comparison policy | Yes |
| A1 single-shot portfolio router | Deterministic MILP baseline with model-pool selection | Yes |
| A2 two-stage cascade | Core stochastic, multi-stage innovation | Yes |
| A3 robust reliability-aware cascade | Final recommended production policy | Yes, unless solver failure forces fallback |

### Locked decisions

1. **Use two-stage cascades only.** Do not implement three-stage cascades unless all required work is already complete.
2. **Use `r[p,m] = normalized quality[p,m]` as the model success probability.** This is a modeling assumption and must be disclosed in the report.
3. **Keep cascade success linear by precomputing expected cascade quality and cost.** The Pyomo models remain MILPs.
4. **Use scenario-based robust SAA, not Wasserstein DRO or full CVaR.** This is rigorous enough for the rubric and buildable before the deadline.
5. **Use A3 as the final “recommended policy.”** A2 is the minimum fallback if A3 times out.
6. **Do not build a learned prompt classifier, frontend, dashboard, API caller, or real-time router.** The deliverable is an optimization report with reproducible code.


### Locked starter data — resolved from the course link

Use the downloaded starter benchmark file as the canonical input:

```text
data/routerbench.csv
```

Actual columns:

```text
row_id, dataset, prompt_id, index, model, score, cost, prompt_tokens, completion_tokens
```

Column mapping:

| Project field | Actual CSV column | Notes |
|---|---|---|
| prompt id | `prompt_id` | Use this as the prompt key. Do **not** use `row_id`; it is a prompt-model composite. |
| domain | `dataset` | Values are `AIME`, `GPQA`, `LCB`, `MMLU-Pro`. |
| model | `model` | 33 unique models. |
| quality | `score` | Binary 0/1 correctness; already in `[0,1]`. |
| cost | `cost` | Nonnegative decimal cost; preserve zero-cost rows. |
| token metadata | `prompt_tokens`, `completion_tokens` | Available for optional diagnostics only; not required by the locked model. |

Actual data profile:

| Metric | Value |
|---|---:|
| Rows | 7,860 |
| Unique prompts | 240 |
| Unique models | 33 |
| Domains | 4 |
| Prompts per domain | 60 each |
| Score values | 0/1 |
| Cost range | 0 to 0.480387 |
| Zero-cost rows | 2,177 |
| Models with any zero-cost row | 16 |

Important missingness note: the file is **not** a complete `240 × 33 = 7,920` grid. All 60 `MMLU-Pro` prompts are missing the model `deepseek-v3.1-terminus`, so the valid prompt-model pair count is 7,860. The implementation must define an available-pair set `PM = {(p,m): row exists}`, prompt-specific model sets `M_p`, and prompt-specific cascade sets `A_p`. Do not impute the missing rows and do not assert that every prompt has all 33 models.

Initial budget profile from the starter data:

| Policy | Avg. cost | Avg. quality |
|---|---:|---:|
| Always cheapest | 0.000000 | 0.862500 |
| Always best quality, tie-break by min cost | 0.001680502 | 0.983333 |

Default budget grid, recomputed by code but safe as initial values:

```python
B_low  = 0.0004201255
B_mid  = 0.0008402511
B_high = 0.0012603766
```

### Fallback decision tree

If time or solver capacity fails:

1. Keep A0, A1, A2, all core plots, and the report.
2. If A3 is infeasible or slow, reduce A3 to 3 scenarios: empirical, coding-heavy, math-heavy.
3. If A3 still fails, solve A2 and evaluate its quality under the 5 scenarios without optimizing over them. In the report, say robustness was evaluated out-of-sample rather than optimized.
4. Do **not** drop A2. A2 is the multi-stage stochastic centerpiece.

---

## 1. Source Map and Research Basis

This PRD is grounded in four categories of sources.

| Source | What it contributes to this PRD |
|---|---|
| INDENG 164 project prompt | Rubric, deadlines, 33-model/240-prompt dataset, stochastic/multi-stage requirement, report requirements, examples of realistic constraints |
| Existing PRD v1.0 uploaded by the user | Initial A0–A3 architecture, build sequence, acceptance criteria, file structure, and Codex prompts |
| LLMRouterBench paper | Research motivation for LLM routing, model complementarity, performance-cost routing, 33-model benchmark context, diminishing returns from larger ensembles |
| Official implementation docs | Codex as the build agent; Pyomo/HiGHS as the MILP implementation stack; vLLM as context for local open-source model deployment assumptions |

### Key rubric facts to satisfy

The assignment requires a 4–5 page comprehensive project report, Pyomo or equivalent code, visualizations, discussion/recommendations, and a personal takeaway. The report’s largest grading component is the optimization model section, worth 10/20 points. A simple deterministic model is capped at 15/20, so the project must clearly show stochasticity, multi-stage decisions, and realistic constraints.

The assignment data contains cost and performance scores for **33 models** and **240 prompts** across **AIME, LCB, GPQA, and MMLU-Pro**, including some zero-cost open-source models. The model must curate a small model pool and define a routing policy for incoming prompts.

The assignment explicitly encourages: subset selection, prompt assignment, stochastic routing, cascading policies that escalate if earlier models fail, slack variables, robustness under reweighted SAA, budget/performance tradeoffs, pool-size limits, storage/provider constraints, fairness/minimum quality guarantees, and worst-case performance.

---

## 2. Final Project Thesis

### Working title

**Cheap First, Reliable Always: A Robust Cascading Optimization Framework for LLM Routing**

### One-sentence thesis

A cheap-first two-stage cascade can reduce expected inference cost relative to single-shot routing while preserving quality, and robust domain-aware constraints prevent the router from overfitting to the empirical prompt mix or sacrificing hard benchmarks.

### Report-ready problem framing

An agentic AI company similar to Cursor must route incoming prompts to a small pool of LLMs. Always using the highest-quality model is expensive; always using the cheapest model is unreliable. The company therefore needs an optimization policy that selects a manageable model pool and routes prompts in a way that balances expected quality, expected cost, robustness, and reliability.

The proposed policy starts with a low-cost first-stage model and escalates to a stronger model only if the first model is estimated to fail. This produces a stochastic multi-stage decision: first-stage model success is random, and the second-stage cost is incurred only under failure. The final model also protects against prompt-distribution shifts by optimizing worst-scenario quality across empirical, coding-heavy, math-heavy, knowledge-heavy, and balanced prompt mixes.

---

## 3. Research Questions

Use exactly these two research questions in the final report.

### RQ1 — Cost-quality benefit of cascading

**Can a two-stage cascading LLM router reduce expected cost relative to single-shot routing while maintaining comparable expected quality?**

Required comparisons:

- Always-cheapest model policy
- Always-best-quality model policy
- A0 starter weighted baseline
- A1 single-shot optimized portfolio router
- A2 two-stage cascading router
- A3 robust reliability-aware cascading router

Primary evidence:

- Cost-quality frontier
- Average cost and quality table
- Escalation-rate table

### RQ2 — Robustness and reliability under deployment constraints

**How do robustness and reliability constraints change the selected model pool, routing behavior, and domain-level performance?**

Required analyses:

- Quality versus model-pool size `K`
- Selected models and usage concentration
- Per-domain quality across AIME, LCB, GPQA, MMLU-Pro
- Worst-scenario quality under reweighted prompt distributions
- Domain-floor slack values in A3

Primary evidence:

- Quality-vs-K plot
- Selected-model usage chart
- Domain performance comparison
- Robustness heatmap
- A3 slack table

---

## 4. Product Requirements

### 4.1 Must-have requirements

| ID | Requirement | Acceptance criteria |
|---|---|---|
| P0-1 | Load and validate dataset | Confirms 240 prompts, 33 models, 4 domains, 7,860 available prompt-model rows, and the known MMLU-Pro missing model; constructs `PM`, `M_p`, and `A_p` without imputing missing pairs |
| P0-2 | Produce normalized quality and success probability | `q_norm` and `r` are in `[0,1]`; `r[p,m] = q_norm[p,m]` documented |
| P0-3 | A0 weighted baseline | Results for α ∈ `{0.1, 0.5, 1, 2, 5, 10}` saved to CSV |
| P0-4 | A1 single-shot MILP | Solves grid over K and budgets; saves selected models, routing, cost, quality, domain metrics |
| P0-5 | A2 two-stage cascade MILP | Solves grid over K, budgets, and Emax; saves expected cost, quality, escalation rate, model usage |
| P0-6 | A3 robust cascade MILP | Solves at least one final configuration; saves η, scenario metrics, domain slacks |
| P0-7 | Produce five report figures | Saves cost-quality frontier, quality-vs-K, selected model usage, domain performance, robustness heatmap |
| P0-8 | Produce report tables | Saves summary policy table, domain quality table, selected models table, scenario table |
| P0-9 | Produce final report-ready outputs | All generated files live under `outputs/`; README explains how to run pipeline |

### 4.2 Nice-to-have requirements

| ID | Requirement | Acceptance criteria |
|---|---|---|
| P1-1 | Sensitivity on cascade recovery factor | Run A2 for recovery factor ρ ∈ `{0.75, 1.0}` if time allows |
| P1-2 | Provider-family usage summary | Parse model names into provider families and report share by family |
| P1-3 | Sankey-style cascade flow | Optional static flow table or Sankey plot showing first-stage to second-stage transitions |
| P1-4 | Robustness evaluation for every policy | Heatmap includes A0/A1/A2/A3 rather than only A2/A3 |

### 4.3 Explicit non-goals

- No full Wasserstein or φ-divergence DRO.
- No learned prompt classifier.
- No embeddings.
- No live API calls.
- No latency modeling unless latency is already in the dataset.
- No frontend.
- No custom solver algorithms.
- No three-stage cascade unless all required outputs are already complete.

---

## 5. Data Contract

### 5.1 Locked input

The starter-code data is now resolved. Use:

```text
data/routerbench.csv
```

The source file is long format and should not need wide-to-long conversion.

| Actual column | Meaning | Required? | Implementation note |
|---|---|---:|---|
| `row_id` | Composite row identifier | No | Looks like `prompt_id|model`; do not use as prompt key. |
| `dataset` | Benchmark/domain | Yes | Rename or map to `domain`. |
| `prompt_id` | Prompt identifier | Yes | This is the true prompt key. |
| `index` | Within-dataset numeric index | No | Not globally unique; do not use as prompt key. |
| `model` | Model name | Yes | 33 unique models. |
| `score` | Quality/correctness | Yes | Binary 0/1; use as `q` directly after float cast. |
| `cost` | Prompt-model cost | Yes | Nonnegative; preserve zeros. |
| `prompt_tokens` | Input-token count | Optional | Diagnostics only. |
| `completion_tokens` | Output-token count | Optional | Diagnostics only. |

Actual shape: 7,860 rows, 240 prompts, 33 models, 4 domains.

The course assignment describes 33 models and 240 prompts, but the provided CSV has one missing model-domain block: `deepseek-v3.1-terminus` is missing for all 60 `MMLU-Pro` prompts. Therefore, code must operate on available pairs rather than assuming a complete Cartesian product.

### 5.2 Required set construction

Build these sets explicitly:

```python
P = sorted(df["prompt_id"].unique())
M = sorted(df["model"].unique())
D = sorted(df["dataset"].unique())
PM = set(zip(df["prompt_id"], df["model"]))
M_p[p] = sorted(df.loc[df.prompt_id == p, "model"].unique())
P_d[d] = sorted(df.loc[df.dataset == d, "prompt_id"].unique())
```

For cascades:

```python
A = global candidate cascade list, e.g. [(m1, m2), ...]
A_p[p] = [a for a in A if (p, a[0]) in PM and (p, a[1]) in PM]
```

All A1 variables `x[p,m]` should only be created for `(p,m) in PM`. All A2/A3 variables `z[p,a]` should only be created for `a in A_p[p]`.

### 5.3 Column detection rules

The loader should first look for the exact locked columns above. It may still support aliases for robustness:

```python
COLUMN_CANDIDATES = {
    "prompt_id": ["prompt_id", "prompt", "question_id", "qid", "id"],
    "domain": ["dataset", "domain", "benchmark", "task"],
    "model": ["model", "model_name", "llm", "system"],
    "quality": ["score", "quality", "performance", "accuracy", "correct", "reward"],
    "cost": ["cost", "avg_cost", "price", "dollar_cost", "api_cost"],
}
```

If detection fails, raise a clear error and print available columns.

### 5.4 Preprocessing rules

1. Use `prompt_id` as the prompt key.
2. Map `dataset` to `domain` internally if desired.
3. Use `score` as `quality`; it is already binary 0/1.
4. Cast `score` to float and set `q_norm = score`.
5. Set `r[p,m] = q_norm[p,m]`.
6. Preserve zero-cost rows exactly.
7. Do not fill or impute missing `(prompt, model)` pairs.
8. Warn, but do not fail, when a prompt has 32 instead of 33 models because this is known and expected for `MMLU-Pro`.
9. Cache the cleaned dataframe as `data/preprocessed.parquet` if parquet support is available; otherwise use `data/preprocessed.csv`.

### 5.5 Derived data to save

| File | Contents |
|---|---|
| `outputs/tables/data_summary.csv` | Row counts, prompt/model/domain counts, score range, cost range, zero-cost count, missing-pair summary |
| `outputs/tables/model_summary.csv` | Rows, average quality, average cost, zero-cost row count by model |
| `outputs/tables/budget_grid.csv` | `B_low`, `B_mid`, `B_high`, derivation from always-cheapest and always-best-quality policies |
| `outputs/tables/cascade_candidates.csv` | Cascade ID, m1, m2, average R, average C, average escalation probability, number of prompts for which cascade is feasible |

## 6. Budget, Parameter, and Scenario Defaults

### 6.1 Budget grid

Do not hard-code dollar budgets. Derive them from the data.

Compute:

```python
always_cheapest_cost = mean_p min_m c[p,m]
always_best_quality_cost = mean_p c[p, argmax_m q[p,m]]
```

Then define:

```python
B_low  = always_cheapest_cost + 0.25 * (always_best_quality_cost - always_cheapest_cost)
B_mid  = always_cheapest_cost + 0.50 * (always_best_quality_cost - always_cheapest_cost)
B_high = always_cheapest_cost + 0.75 * (always_best_quality_cost - always_cheapest_cost)
```

If any budget is infeasible for A1 or A2, automatically increase it to the minimum feasible cost plus a small tolerance and record this adjustment in `budget_grid.csv`.

### 6.2 Model-pool sizes

| Model | K grid |
|---|---|
| A1 | `{1, 2, 3, 5, 8}` |
| A2 | `{2, 3, 5}` |
| A3 | `{3}` required, `{5}` optional |

The final report should recommend either `K=3` or `K=5` depending on the observed cost-quality frontier.

### 6.3 Escalation cap

Run A2 with:

```python
Emax_values = [1.0, 0.75, 0.50]
```

Interpretation:

- `Emax = 1.0`: no effective cap.
- `Emax = 0.75`: moderate production control.
- `Emax = 0.50`: aggressive control on expensive second-stage invocations.

A3 default:

```python
Emax = 0.75
```

If infeasible, relax to `1.0` and disclose.

### 6.4 Cascade recovery factor

The simple independent-failure cascade formula can overstate gains if model errors are correlated. To make the model more defensible, include a configurable recovery factor `rho`:

```python
rho = 0.75  # default
```

For a two-stage cascade `a = (m1, m2)`:

```text
R[p,a] = r[p,m1] + (1 - r[p,m1]) * rho * r[p,m2]
C[p,a] = c[p,m1] + (1 - r[p,m1]) * c[p,m2]
Esc[p,a] = 1 - r[p,m1]
```

If time allows, run sensitivity for `rho = 1.0` to show the idealized independent-recovery case. The report should state that `rho < 1` represents correlated failures and imperfect recovery.

### 6.5 Robust prompt-mix scenarios

Derive empirical domain weights from data. If the dataset is exactly balanced, empirical and balanced will be identical; in that case keep only one of them in A3 but still evaluate both labels in the heatmap if useful.

| Scenario | AIME | LCB | GPQA | MMLU-Pro | Purpose |
|---|---:|---:|---:|---:|---|
| empirical | data-derived | data-derived | data-derived | data-derived | Actual sample distribution |
| balanced | 0.25 | 0.25 | 0.25 | 0.25 | Equal benchmark weighting |
| coding_heavy | 0.10 | 0.55 | 0.15 | 0.20 | Cursor-like coding company |
| math_heavy | 0.55 | 0.10 | 0.15 | 0.20 | Tutoring/math assistant |
| knowledge_heavy | 0.10 | 0.10 | 0.40 | 0.40 | Research/knowledge assistant |

Convert scenario domain weights to prompt weights:

```text
w[s,p] = pi[s, domain(p)] / |P_domain(p)|
```

Verify:

```python
abs(sum_p w[s,p] - 1.0) <= 1e-8
```

### 6.6 Domain quality floors

Use data-derived floors. For each domain `d`, compute the always-best domain quality:

```python
oracle_domain_quality[d] = mean_{p in P_d} max_m q[p,m]
```

Default:

```python
tau[d] = 0.90 * oracle_domain_quality[d]
```

Because these floors may be ambitious, A3 includes nonnegative slack `s[d]` and penalizes slack in the objective.

Default slack penalty:

```python
lambda_slack = 0.10
```

If slacks are too large, try `lambda_slack = 0.25`. If A3 sacrifices robust quality too much, try `lambda_slack = 0.05`.

---

## 7. Mathematical Formulation

### 7.1 Sets and parameters

| Symbol | Type | Meaning |
|---|---|---|
| `P` | set | Prompts |
| `M` | set | Candidate models |
| `D` | set | Benchmark domains |
| `P_d` | set | Prompts in domain `d` |
| `PM` | set | Available prompt-model pairs `(p,m)` present in `routerbench.csv` |
| `M_p` | set | Available models for prompt `p` |
| `A` | set | Global candidate two-stage cascades `a=(m1,m2)` |
| `A_p` | set | Cascades feasible for prompt `p`, requiring both `(p,m1)` and `(p,m2)` in `PM` |
| `S` | set | Robust prompt-mix scenarios |
| `q[p,m] ∈ [0,1]` | parameter | Binary quality/correctness from `score`, defined only for `(p,m) in PM` |
| `c[p,m] ≥ 0` | parameter | Cost, defined only for `(p,m) in PM` |
| `r[p,m] ∈ [0,1]` | parameter | Success probability, set equal to `q[p,m]` |
| `R[p,a] ∈ [0,1]` | parameter | Expected cascade quality, defined only for `a in A_p[p]` |
| `C[p,a] ≥ 0` | parameter | Expected cascade cost, defined only for `a in A_p[p]` |
| `Esc[p,a] ∈ [0,1]` | parameter | Escalation probability, defined only for `a in A_p[p]` |
| `w[s,p] ≥ 0` | parameter | Scenario prompt weight |
| `K` | parameter | Pool-size cap |
| `B` | parameter | Average expected cost budget |
| `Emax` | parameter | Maximum average expected escalation rate |
| `tau[d]` | parameter | Domain quality floor |
| `lambda_slack` | parameter | Domain slack penalty |

### 7.2 Decision variables

| Variable | Domain | Meaning |
|---|---|---|
| `y[m]` | binary | 1 if model `m` is selected in the pool |
| `x[p,m]` | binary | 1 if prompt `p` is assigned to available model `m` in A1; created only for `(p,m) in PM` |
| `z[p,a]` | binary | 1 if prompt `p` is assigned to feasible cascade `a` in A2/A3; created only for `a in A_p[p]` |
| `eta` | continuous `[0,1]` | Worst-scenario expected quality in A3 |
| `s[d]` | continuous `≥0` | Slack on domain floor in A3 |

### 7.3 A0 — weighted baseline

For each α:

```text
f_alpha(p) = argmin_{m in M_p[p]} c[p,m] - alpha * q[p,m]
```

Outputs:

- average cost
- average quality
- domain quality
- model usage counts

A0 is not the final model. It exists to show the starter-code frontier.

### 7.4 A1 — single-shot portfolio router

```text
maximize    (1/|P|) sum_p sum_{m in M_p[p]} q[p,m] x[p,m]

subject to  sum_{m in M_p[p]} x[p,m] = 1                       for all p
            x[p,m] <= y[m]                                    for all (p,m) in PM
            sum_m y[m] <= K
            (1/|P|) sum_p sum_{m in M_p[p]} c[p,m] x[p,m] <= B
            x[p,m], y[m] binary
```

Purpose:

- Shows value of constrained optimization over the weighted baseline.
- Establishes selected model pools and diminishing returns in `K`.

### 7.5 A2 — two-stage cascading router

For cascade `a=(m1,m2)`, precompute:

```text
R[p,a]   = r[p,m1] + (1 - r[p,m1]) * rho * r[p,m2]    # only if a in A_p[p]
C[p,a]   = c[p,m1] + (1 - r[p,m1]) * c[p,m2]                  # only if a in A_p[p]
Esc[p,a] = 1 - r[p,m1]                                        # only if a in A_p[p]
```

Then solve:

```text
maximize    (1/|P|) sum_p sum_{a in A_p[p]} R[p,a] z[p,a]

subject to  sum_{a in A_p[p]} z[p,a] = 1                       for all p
            z[p,a] <= y[m]                                    for all p, a in A_p[p], m in a
            sum_m y[m] <= K
            (1/|P|) sum_p sum_{a in A_p[p]} C[p,a] z[p,a] <= B
            (1/|P|) sum_p sum_{a in A_p[p]} Esc[p,a] z[p,a] <= Emax
            z[p,a], y[m] binary
```

Why this is stochastic and multi-stage:

- First-stage success is modeled as a Bernoulli event with probability `r[p,m1]`.
- The second-stage model is invoked only if the first-stage model fails.
- Expected cost and expected quality are precomputed, keeping the optimization linear.

### 7.6 A3 — robust reliability-aware cascade

A3 extends A2 by optimizing worst-scenario quality and adding domain floor slacks.

```text
maximize    eta - lambda_slack * sum_d s[d]

subject to  all A2 constraints

            sum_p w[s,p] sum_{a in A_p[p]} R[p,a] z[p,a] >= eta     for all s
            sum_p w[s,p] sum_{a in A_p[p]} C[p,a] z[p,a] <= B       for all s
            (1/|P_d|) sum_{p in P_d} sum_{a in A_p[p]} R[p,a] z[p,a]
                + s[d] >= tau[d]                              for all d
            eta in [0,1]
            s[d] >= 0
            z[p,a], y[m] binary
```

Population-form language for report:

> Let the incoming prompt be drawn from an unknown distribution over prompt types. The production objective is to maximize expected cascade quality under the worst plausible prompt distribution, subject to an expected cost budget, pool-size limit, and escalation cap.

Sample-average approximation language for report:

> Because the true prompt distribution is unknown, we approximate the expectation using the 240 observed prompts. Robustness is modeled by reweighting the empirical sample across domain-level scenarios such as coding-heavy, math-heavy, and knowledge-heavy traffic.

---

## 8. Cascade Candidate Generation

Do not enumerate all `33 × 32 = 1056` possible ordered pairs unless it solves quickly. Generate a focused but diverse set.

### 8.1 Model summaries

For each model `m`:

```python
qbar[m] = mean_p q[p,m]
cbar[m] = mean_p c[p,m]
zero_cost[m] = all_or_average_cost_is_zero
quality_per_cost[m] = qbar[m] / max(cbar[m], epsilon)
```

### 8.2 Candidate rules

First-stage candidates:

```python
cheap_models = models where cbar[m] <= percentile(cbar, 30) or zero_cost[m]
```

Second-stage candidates:

```python
strong_models = models where qbar[m] >= percentile(qbar, 50)
```

Allowed cascade `(m1,m2)` if:

```text
m1 != m2
m1 in cheap_models
m2 in strong_models
qbar[m2] >= qbar[m1]
```

If `|A| > 250`, keep the top 250 by:

```text
score(a) = mean_p R[p,a] - beta * mean_p C[p,a]
```

with `beta` chosen so quality and cost are on similar normalized scales. Simpler fallback: keep top 250 by average `R[p,a] / max(C[p,a], epsilon)` while forcing at least one cascade per cheap first-stage model.

### 8.3 Safety checks

- `|A|` should be between 50 and 300.
- Every prompt should have at least one feasible cascade in `A_p[p]`.
- `deepseek-v3.1-terminus` can appear in cascades for AIME/GPQA/LCB but not for MMLU-Pro prompts unless rows are added; the code should handle this through `A_p`.
- Every selected cascade should use models that can be selected under `K`.
- Record average `R`, average `C`, and average `Esc` for each cascade.

---

## 9. Experiments to Run

### 9.1 Baseline policies

| Policy | Description | Output table |
|---|---|---|
| always_cheapest | For each prompt, choose model with minimum cost | `baseline_extremes.csv` |
| always_best_quality | For each prompt, choose model with maximum quality | `baseline_extremes.csv` |
| A0 weighted baseline | Choose `argmin cost - alpha*quality` | `a0_results.csv` |

### 9.2 A1 grid

Run:

```python
K_values = [1, 2, 3, 5, 8]
B_values = [B_low, B_mid, B_high]
```

Save 15 rows to `a1_results.csv`.

### 9.3 A2 grid

Run:

```python
K_values = [2, 3, 5]
B_values = [B_low, B_mid, B_high]
Emax_values = [1.0, 0.75, 0.50]
rho = 0.75
```

Save 27 rows to `a2_results.csv`. If solve time is too long, reduce to:

```python
Emax_values = [1.0, 0.75]
```

### 9.4 A3 final runs

Run:

```python
K = 3
B = B_mid or B_high, whichever produces a feasible A2 solution with strong quality
Emax = 0.75
rho = 0.75
lambda_slack = 0.10
scenarios = all 5 scenarios, dropping duplicate empirical/balanced if identical
```

Optional extra A3:

```python
K = 5
same B, Emax, rho
```

### 9.5 Policy selection rule for final recommendation

Choose the final recommended policy as:

1. A3 if feasible and scenario quality is competitive with A2.
2. Otherwise A2 with the best cost-quality tradeoff and robustness evaluated out-of-sample.

For the final summary table, use one representative configuration per policy:

- A0: α point closest to A2’s average cost.
- A1: best quality at same `K` and budget as A2/A3.
- A2: best non-dominated cascade configuration.
- A3: final robust configuration.

---

## 10. Output Tables

All tables go in `outputs/tables/`.

| File | Required columns |
|---|---|
| `data_summary.csv` | num_prompts, num_models, num_domains, q_min, q_max, cost_min, cost_max, zero_cost_models |
| `model_summary.csv` | model, provider_guess, avg_quality, avg_cost, zero_cost_flag, quality_rank, cost_rank |
| `budget_grid.csv` | B_label, B_value, derivation, feasible_A1, feasible_A2 |
| `baseline_extremes.csv` | policy, avg_cost, avg_quality, models_used |
| `a0_results.csv` | alpha, avg_cost, avg_quality, models_used, domain qualities |
| `a1_results.csv` | K, B_label, B_value, status, objective, avg_cost, avg_quality, selected_models, models_used, domain qualities |
| `a2_results.csv` | K, B_label, B_value, Emax, rho, status, objective, avg_cost, avg_quality, escalation_rate, selected_models, top_cascades, domain qualities |
| `a3_results.csv` | K, B_label, B_value, Emax, rho, lambda_slack, status, eta, avg_cost, empirical_quality, selected_models, total_slack |
| `a3_scenario_quality.csv` | policy, scenario, quality, cost |
| `domain_quality.csv` | policy, domain, quality, tau, slack |
| `selected_model_usage.csv` | policy, model, selected, first_stage_count, second_stage_expected_count, total_usage |
| `summary_comparison.csv` | policy, K, avg_cost, avg_quality, worst_scenario_quality, escalation_rate, selected_models |

---

## 11. Figures

All figures go in `outputs/figures/`.

### Figure 1 — `cost_quality_frontier.png`

- X-axis: average expected cost.
- Y-axis: average expected quality.
- Series: always-cheapest, always-best, A0, A1, A2, A3.
- Mark A3 final recommendation with a star or label.
- Use this as the main result figure.

### Figure 2 — `quality_vs_pool_size.png`

- X-axis: `K`.
- Y-axis: quality.
- Series: A1 and A2 at comparable budget.
- Annotate diminishing returns around `K=3` or `K=5` if visible.

### Figure 3 — `selected_model_usage.png`

- Bar chart of selected models in A2 or A3.
- Show first-stage usage and expected second-stage usage.
- Sort by total usage.

### Figure 4 — `domain_performance_comparison.png`

- Grouped bars by domain.
- Policies: A0, A1, A2, A3.
- Add domain floor line if readable.

### Figure 5 — `robustness_heatmap.png`

- Rows: scenarios.
- Columns: policies.
- Cell value: scenario-weighted quality.
- Annotate cells numerically.

---

## 12. Implementation Architecture

### 12.1 File structure

```text
llm-routing-project/
├── data/
│   └── original_dataset.csv
├── src/
│   ├── load_data.py
│   ├── preprocessing.py
│   ├── metrics.py
│   ├── baselines.py
│   ├── solver_utils.py
│   ├── pyomo_single_shot.py
│   ├── pyomo_cascade.py
│   ├── pyomo_robust_cascade.py
│   ├── experiments.py
│   └── plots.py
├── outputs/
│   ├── tables/
│   ├── figures/
│   └── solutions/
├── report/
│   ├── report.md or report.tex
│   └── figures/
├── tests/
│   └── test_sanity.py
├── run_experiments.py
├── requirements.txt
├── AGENTS.md
└── README.md
```

### 12.2 Python package requirements

```text
pandas
numpy
pyomo
highspy
matplotlib
pyarrow
pytest
```

Optional if available:

```text
plotly
```

Do not require commercial solvers.

### 12.3 Solver priority

Try solvers in this order:

1. `appsi_highs`
2. `highs`
3. `cbc`
4. `glpk`

Implement:

```python
def get_solver(time_limit=300):
    for name in ["appsi_highs", "highs", "cbc", "glpk"]:
        try:
            solver = SolverFactory(name)
            if solver.available(exception_flag=False):
                # Set time limit where supported.
                return solver, name
        except Exception:
            continue
    raise RuntimeError("No MILP solver available. Install highspy or GLPK/CBC.")
```

### 12.4 Coding standards

- Every solver function returns a plain Python dictionary.
- Every experiment writes a CSV even if some runs are infeasible.
- Every infeasible run records `status="infeasible"` instead of crashing the full pipeline.
- All random choices, if any, use a fixed seed.
- No hidden notebooks-only logic. If using a notebook, it should call the same functions.

---

## 13. Function-Level Build Spec

### `src/load_data.py`

Required functions:

```python
def load_raw_data(path: str) -> pd.DataFrame: ...
def detect_columns(df: pd.DataFrame) -> dict: ...
def standardize_long_format(df: pd.DataFrame, columns: dict) -> pd.DataFrame: ...
```

### `src/preprocessing.py`

Required functions:

```python
def normalize_quality(df: pd.DataFrame) -> pd.DataFrame: ...
def validate_data(df: pd.DataFrame) -> dict: ...
def build_sets_and_matrices(df: pd.DataFrame) -> dict: ...
def write_data_summaries(data: dict, output_dir: str) -> None: ...
```

Return `data` dictionary:

```python
{
    "df": cleaned_df,
    "P": list_of_prompts,
    "M": list_of_models,
    "D": list_of_domains,
    "prompt_domain": dict,
    "q": dict_or_numpy_array,
    "c": dict_or_numpy_array,
    "r": dict_or_numpy_array,
}
```

### `src/metrics.py`

Required functions:

```python
def average_quality(assignments, q, weights=None): ...
def average_cost(assignments, c, weights=None): ...
def domain_quality(assignments, q, prompt_domain): ...
def scenario_weights(P, prompt_domain, scenario_domain_weights): ...
def evaluate_policy_under_scenarios(policy_assignments, scenarios): ...
```

### `src/baselines.py`

Required functions:

```python
def solve_always_cheapest(data): ...
def solve_always_best_quality(data): ...
def solve_weighted_baseline(data, alpha: float): ...
def run_a0_grid(data, alphas, output_dir): ...
```

### `src/solver_utils.py`

Required functions:

```python
def get_solver(time_limit: int = 300): ...
def extract_solver_status(results) -> str: ...
def safe_value(var, default=None): ...
def write_json(obj, path): ...
```

### `src/pyomo_single_shot.py`

Required function:

```python
def solve_a1(data, K: int, B: float, time_limit: int = 300) -> dict: ...
```

Return keys:

```python
status, objective, avg_quality, avg_cost, selected_models, assignment,
domain_quality, solver_name, solve_seconds
```

### `src/pyomo_cascade.py`

Required functions:

```python
def summarize_models(data) -> pd.DataFrame: ...
def generate_cascades(data, rho=0.75, max_cascades=250) -> tuple[pd.DataFrame, dict]: ...
def solve_a2(data, cascades, R, C, Esc, K, B, Emax, rho=0.75, time_limit=300) -> dict: ...
```

Return keys for A2:

```python
status, objective, avg_quality, avg_cost, escalation_rate, selected_models,
cascade_assignment, cascade_usage, first_stage_usage, second_stage_expected_usage,
domain_quality, solver_name, solve_seconds
```

### `src/pyomo_robust_cascade.py`

Required functions:

```python
def build_scenarios(data) -> dict: ...
def compute_domain_floors(data, multiplier=0.90) -> dict: ...
def solve_a3(data, cascades, R, C, Esc, K, B, Emax, scenarios, tau, lambda_slack=0.10, time_limit=300) -> dict: ...
```

Return keys:

```python
status, objective, eta, avg_quality, avg_cost, scenario_quality, scenario_cost,
domain_quality, domain_slack, selected_models, cascade_assignment,
escalation_rate, solver_name, solve_seconds
```

### `src/experiments.py`

Required functions:

```python
def compute_budget_grid(data) -> pd.DataFrame: ...
def run_all_baselines(data, config): ...
def run_a1_experiments(data, config): ...
def run_a2_experiments(data, config): ...
def run_a3_experiments(data, config): ...
def build_summary_tables(output_dir): ...
```

### `src/plots.py`

Required functions:

```python
def plot_cost_quality_frontier(output_dir): ...
def plot_quality_vs_pool_size(output_dir): ...
def plot_selected_model_usage(output_dir, policy="A3"): ...
def plot_domain_performance(output_dir): ...
def plot_robustness_heatmap(output_dir): ...
def make_all_plots(output_dir): ...
```

### `run_experiments.py`

CLI behavior:

```bash
python run_experiments.py --data data/original_dataset.csv --all
python run_experiments.py --data data/original_dataset.csv --skip-a3
python run_experiments.py --data data/original_dataset.csv --only-plots
```

Required CLI flags:

```text
--data
--output-dir
--skip-a1
--skip-a2
--skip-a3
--only-plots
--time-limit
--max-cascades
```

---

## 14. Testing and Validation

### 14.1 Sanity tests

Create `tests/test_sanity.py` with tests for:

1. `q_norm` is in `[0,1]`.
2. Cost is nonnegative.
3. Scenario weights sum to 1.
4. Every A1 assignment assigns each prompt to exactly one model.
5. Every A2/A3 assignment assigns each prompt to exactly one cascade.
6. A2 expected cost and escalation metrics match manual recomputation.
7. Selected cascades only use selected models.

### 14.2 Feasibility checks

Before solving A1/A2, compute a lower bound on minimum feasible cost:

- A1 minimum cost: average over prompts of cheapest model among all models.
- A2 minimum cost: average over prompts of cheapest expected cascade.

If `B` is below the lower bound, mark as infeasible without calling solver.

### 14.3 Report integrity checks

Before writing the report, confirm:

- No `TBD` values remain in summary tables.
- All five plot files exist.
- A3 has either solved or a fallback robustness evaluation exists.
- The report’s final recommendation matches a row in `summary_comparison.csv`.

---

## 15. Report Plan

The report should be 4–5 pages. Write it after the results exist.

### Page 1 — Introduction

Include:

- Cursor-style agentic AI setting.
- Cost-quality tradeoff.
- Why always-best and always-cheapest fail.
- Two research questions.
- Why stochastic multi-stage optimization fits.

### Pages 2–3 — Optimization model

Include:

- Notation table.
- A1 single-shot MILP.
- A2 cascade MILP.
- A3 robust reliability-aware cascade.
- Explicit explanation of stochasticity and multi-stage decisions.
- Population form and sample-average approximation.
- Explanation that `R[p,a]` and `C[p,a]` are precomputed, so the model remains linear.

### Page 4 — Results

Include:

- Cost-quality frontier.
- Summary comparison table.
- Domain performance table or chart.
- Results narrative for RQ1 and RQ2.

### Page 5 — Discussion, recommendations, takeaway

Include:

- Business recommendation.
- Practical deployment interpretation.
- Limitations.
- Extensions.
- Personal takeaway.

### Report paragraph template

Use this as the opening of the model section:

> Let `P` be the set of prompts, `M` the set of candidate models, and `D` the set of benchmark domains. The company first selects a small subset of models, represented by binary variables `y_m`, and then routes each prompt either to a single model or to a two-stage cascade. In the cascading formulation, the first model succeeds with estimated probability `r_pm`; if it fails, the request escalates to a second model. This gives an expected cascade quality `R_pa` and expected cost `C_pa`, both precomputed from the data. The resulting optimization remains a mixed-integer linear program.

---

## 16. Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|---|---:|---:|---|
| A3 MILP too slow | Medium | High | Reduce `max_cascades`; use 3 scenarios; solve only K=3/B_high; fallback to A2 + scenario evaluation |
| Cascade quality looks unrealistically high | Medium | Medium | Use recovery factor `rho=0.75`; disclose independence/correlation limitation |
| Budget grid infeasible | Medium | Medium | Compute feasibility lower bounds and auto-adjust budgets |
| A2 not better than A1 | Low/Medium | Medium | Present as result; emphasize reliability/operational interpretability; compare under Emax and scenarios |
| Dataset format differs from expectation | Medium | High | Implement robust column detection and wide-to-long conversion |
| Report exceeds 5 pages | High | Low | Put only A2/A3 equations in main body; move A0/A1 detail to appendix or code comments |
| Codex changes too much at once | Medium | High | Use step prompts; require tests after each module |
| Solver unavailable | Medium | High | Install `highspy`; fallback to CBC/GLPK; use smaller grids |

---

## 17. Definition of Done

The project is done when all of the following are true:

- [ ] Data loader validates the dataset and writes `data_summary.csv`.
- [ ] A0, A1, A2 run successfully.
- [ ] A3 runs successfully or the documented fallback is complete.
- [ ] Five figures exist in `outputs/figures/`.
- [ ] Summary tables exist in `outputs/tables/`.
- [ ] Selected final policy is clearly identified.
- [ ] Report is 4–5 pages.
- [ ] Report includes stochastic/multi-stage explanation.
- [ ] Report includes at least two realistic constraints: pool size, budget, escalation cap, and/or domain floors.
- [ ] Report includes Pyomo implementation summary and optimal solutions.
- [ ] README explains how to rerun the pipeline.
- [ ] Submission uploaded to Gradescope before the deadline.

---

## 18. One-Page Codex Master Prompt

Paste this into Codex first.

```text
You are building a Pyomo optimization project for INDENG 164.

Project title:
Cheap First, Reliable Always: A Robust Cascading Optimization Framework for LLM Routing

Goal:
Build a reproducible Python/Pyomo pipeline that selects a small LLM model pool and routes 240 prompts across 33 models using four policy variants:
A0 weighted baseline, A1 single-shot portfolio MILP, A2 two-stage stochastic cascade MILP, and A3 robust reliability-aware cascade MILP.

Data:
A CSV with 240 prompts × 33 models across AIME, LCB, GPQA, and MMLU-Pro. It has or can be converted to columns: prompt_id, domain/benchmark, model, quality/score, cost. Some models have zero cost. Normalize quality to [0,1] and set r[p,m] = normalized quality.

Locked scope:
- Two-stage cascades only.
- No embeddings or learned classifiers.
- No live API calls.
- Use Pyomo + pandas + numpy + matplotlib.
- Try solvers in this order: appsi_highs, highs, cbc, glpk.
- Write outputs to outputs/tables, outputs/figures, outputs/solutions.

Models:
A0: For alpha in [0.1,0.5,1,2,5,10], route each prompt to argmin_m cost[p,m] - alpha*quality[p,m].

A1: Binary y[m] and x[p,m]. Maximize average quality subject to one model per prompt, x[p,m] <= y[m], sum y <= K, average cost <= B.

A2: Generate candidate cascades a=(m1,m2), cheap first-stage and strong second-stage. Precompute:
R[p,a] = r[p,m1] + (1-r[p,m1]) * rho * r[p,m2], with rho=0.75
C[p,a] = c[p,m1] + (1-r[p,m1]) * c[p,m2]
Esc[p,a] = 1-r[p,m1]
Binary y[m], z[p,a]. Maximize average R subject to one cascade per prompt, z[p,a] <= y[m] for models in cascade, sum y <= K, average C <= B, average Esc <= Emax.

A3: Extend A2 with robust scenarios over domain weights. Add eta and slack s[d]. Maximize eta - lambda_slack*sum s[d]. Add scenario_quality[s] >= eta, scenario_cost[s] <= B, and domain_quality[d] + s[d] >= tau[d].

Required outputs:
- data_summary.csv, model_summary.csv, budget_grid.csv
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

## 19. Stepwise Codex Prompts

Use these if Codex struggles with the master prompt.

### Prompt 1 — data loader

```text
Build src/load_data.py and src/preprocessing.py for the LLM routing project. Load a CSV, detect prompt_id/domain/model/quality/cost columns, convert wide to long if needed, validate 240 prompts × 33 models if possible, normalize quality to [0,1], set r=q_norm, preserve zero-cost models, and save data_summary.csv and model_summary.csv. Return a data dictionary with df, P, M, D, prompt_domain, q, c, r.
```

### Prompt 2 — baselines

```text
Build src/baselines.py. Implement always_cheapest, always_best_quality, and A0 weighted baseline for alpha in [0.1,0.5,1,2,5,10]. Each policy should return assignments, avg_cost, avg_quality, domain_quality, and model usage counts. Save baseline_extremes.csv and a0_results.csv.
```

### Prompt 3 — A1 MILP

```text
Build src/solver_utils.py and src/pyomo_single_shot.py. Implement get_solver with appsi_highs/highs/cbc/glpk fallback. Implement solve_a1(data,K,B): binary y[m], x[p,m], maximize avg quality, one model per prompt, x<=y, sum y<=K, avg cost<=B. Return selected models, assignment, avg quality/cost, domain quality, status, solver name. Add infeasibility handling.
```

### Prompt 4 — cascade generation

```text
In src/pyomo_cascade.py, implement summarize_models and generate_cascades. Cheap first-stage models are bottom 30% average cost or zero-cost. Strong second-stage models are top 50% average quality. Allowed cascades require m1 != m2 and qbar[m2] >= qbar[m1]. Precompute R, C, Esc with rho=0.75. Cap to max_cascades=250 if needed. Save cascade_candidates.csv.
```

### Prompt 5 — A2 MILP

```text
In src/pyomo_cascade.py, implement solve_a2(data,cascades,R,C,Esc,K,B,Emax,rho). Binary y[m], z[p,a]. Maximize avg R. Constraints: one cascade per prompt, z<=y for both cascade models, sum y<=K, avg C<=B, avg Esc<=Emax. Return selected_models, cascade_assignment, avg_quality, avg_cost, escalation_rate, domain_quality, stage usage counts, status.
```

### Prompt 6 — A3 robust MILP

```text
Build src/pyomo_robust_cascade.py. Implement build_scenarios, compute_domain_floors, and solve_a3. Use empirical, balanced, coding_heavy, math_heavy, knowledge_heavy domain weights. Convert domain weights to prompt weights. Extend A2 with eta and s[d]. Maximize eta - lambda_slack*sum(s[d]). Add scenario_quality>=eta, scenario_cost<=B, domain_quality+s>=tau. Return eta, scenario metrics, domain slacks, selected models, assignments, status.
```

### Prompt 7 — experiments

```text
Build src/experiments.py. Compute data-derived budget grid from always-cheapest and always-best-quality costs. Run A0, A1 grid K=[1,2,3,5,8] x B=[low,mid,high], A2 grid K=[2,3,5] x B=[low,mid,high] x Emax=[1.0,0.75,0.5], and A3 final K=3 at selected budget. Save all result tables and JSON solutions.
```

### Prompt 8 — plots

```text
Build src/plots.py using matplotlib only. Generate cost_quality_frontier.png, quality_vs_pool_size.png, selected_model_usage.png, domain_performance_comparison.png, robustness_heatmap.png. Read outputs/tables CSVs and save figures to outputs/figures.
```

### Prompt 9 — orchestration and README

```text
Build run_experiments.py with CLI flags --data, --output-dir, --skip-a1, --skip-a2, --skip-a3, --only-plots, --time-limit, --max-cascades. Add README.md explaining setup, install, running experiments, output files, and report workflow.
```

---

## 20. AGENTS.md Content for Codex Repository

Create an `AGENTS.md` file with this content:

```text
# AGENTS.md — LLM Routing Optimization Project

You are working on an INDENG 164 final project. The goal is to build a reproducible Pyomo pipeline for LLM routing.

Do:
- Implement in small steps.
- Run sanity checks after each module.
- Save outputs to outputs/tables, outputs/figures, outputs/solutions.
- Keep all optimization formulations linear MILPs.
- Use Pyomo, pandas, numpy, matplotlib.
- Prefer appsi_highs/highs, then cbc, then glpk.
- Preserve zero-cost models.
- Normalize quality to [0,1] and set r=q_norm.
- Write docstrings and clear comments for report reuse.

Do not:
- Add embeddings, learned classifiers, API calls, dashboards, or custom solvers.
- Implement three-stage cascades unless all required two-stage outputs are complete.
- Hide logic in notebooks only.
- Crash the whole pipeline on one infeasible run; record status and continue.

Core models:
- A0 weighted baseline.
- A1 single-shot portfolio MILP.
- A2 two-stage cascade MILP.
- A3 robust reliability-aware cascade MILP.

Critical report facts:
- Stochasticity enters through first-stage cascade success/failure.
- Multi-stage decision enters through escalation to a second model.
- Robustness enters through scenario-weighted SAA over prompt distributions.
- Realistic constraints include pool size, budget, escalation cap, and domain quality floors.
```

---

## 21. Final Submission Checklist

Use this on May 15 before uploading.

### Code and data

- [ ] `run_experiments.py` runs from a clean terminal.
- [ ] README has install and run instructions.
- [ ] Output folders contain expected CSVs/PNGs.
- [ ] Solver status recorded for each experiment.
- [ ] Infeasible runs are explained, not hidden.

### Report

- [ ] 4–5 pages.
- [ ] Two research questions stated on page 1.
- [ ] Notation table included.
- [ ] A2 and A3 equations included.
- [ ] Population form and SAA form explained.
- [ ] Stochasticity and multi-stage decision clearly labeled.
- [ ] At least two realistic constraints discussed.
- [ ] Cost-quality frontier included.
- [ ] Domain performance included.
- [ ] Robustness or scenario analysis included.
- [ ] Business recommendation included.
- [ ] Personal takeaway included.

### Narrative

- [ ] Final recommendation is a specific policy, not vague.
- [ ] Numbers in text match tables.
- [ ] Limitations are honest: quality-as-probability, recovery factor assumption, no latency data, finite sample.
- [ ] No unsupported claims about real company internals.
- [ ] All TBD placeholders removed.

---

End of PRD v2.0.
