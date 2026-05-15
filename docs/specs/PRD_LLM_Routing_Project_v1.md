# PRD / Build Spec
## Cheap First, Reliable Always: A Robust Cascading Optimization Framework for LLM Routing

**Course:** INDENG 164 — Final Project
**Author:** Read Millar
**Final due:** Friday, May 15, 2026, 11:59pm
**Document version:** v1.0 — build spec to hand to Codex before writing the report
**Status:** Approved scope. Build A0–A2 first, A3 second, report last.

---

## 0. Executive Summary

This PRD specifies a Pyomo-based optimization pipeline for LLM routing under the INDENG 164 project rubric. The grader awards 10/20 points for the optimization model and explicitly caps deterministic models at 15/20, so the project's center of gravity is the **stochastic, multi-stage cascading router with robustness constraints** (Model A3). All other artifacts (baselines, plots, report) exist to frame and justify that model.

The build proceeds in four model versions of increasing complexity (A0 → A3), each producing the inputs needed for the next. The MVP (A0+A1+A2 + 3 plots + 4-page report) is sufficient for a strong submission; the stretch goals (A3 + robustness heatmap + domain fairness slack) push the grade ceiling.

**Critical rule:** Do not start by writing the report. Build the model → produce results → write the report around the actual numbers.

---

## 1. Problem Statement

An agentic AI company (modeled on Cursor) must route incoming user prompts to one of 33 candidate LLMs, drawn from a benchmark of 240 prompts across AIME (math), LCB (code), GPQA, and MMLU-Pro (knowledge). Two naïve strategies fail:

- **Always use the strongest model** → highest quality but cost-prohibitive at production scale (Claude 4.6 Opus costs ~100× a small open-source model per token).
- **Always use the cheapest model** → low cost but unreliable on hard prompts (AIME, GPQA), producing customer-visible failures.

The starter code's weighted baseline `f(p) = argmin_m C(p,m) − α·Q(p,m)` exposes the cost-quality trade-off but provides no operational guarantees: no pool-size limit, no budget cap, no robustness to prompt-mix shift, no escalation logic, and no fairness across benchmark domains. A production router needs all of these.

**The cost of not solving this well:** in the report this manifests as a low grade (deterministic models capped at 15/20). In the simulated business context, it manifests as either a 5–10× higher inference bill than necessary, or a sharp drop in customer-perceived quality on hard prompts that the cheap model cannot handle.

---

## 2. Goals

Each goal is tied to a measurable artifact the grader can verify.

| # | Goal | Measurement |
|---|------|-------------|
| G1 | Produce a stochastic, multi-stage optimization model that clears the 15-point deterministic ceiling | Model A3 has scenario variables, cascade (multi-stage) decisions, and slack penalties |
| G2 | Quantify the cost reduction of cascading vs. single-shot routing at comparable quality | Cost-quality frontier plot shows A2 strictly Pareto-improving or non-dominated relative to A1 |
| G3 | Quantify robustness gain under prompt-mix shift | Robustness heatmap: A3 worst-scenario quality ≥ A2 worst-scenario quality |
| G4 | Demonstrate diminishing returns in pool size K | Quality-vs-K curve flattens by K = 5 |
| G5 | Deliver a 4–5 page report + Pyomo code + 5 plots + 2 summary tables by the May 15 deadline | Gradescope upload confirmed |

**User goal (the analyst persona):** select a small, defensible model pool and routing policy with a clear cost/quality story.
**Business goal (the simulated company):** a production-deployable policy that holds up under traffic-mix shift.
**Course goal (the actual user):** maximize the 20-point rubric, specifically the 10-point model section.

---

## 3. Non-Goals

Explicit out-of-scope items, with the reason for each. These exist to prevent scope creep over the final 24 hours.

1. **Full CVaR / distributionally robust optimization with Wasserstein ambiguity sets.** Out of scope — robust scenarios across 5 prompt distributions plus domain slack already satisfies "robustness under different ambiguity sets" from the rubric. CVaR adds modeling complexity that won't earn marginal points if A3 already works.
2. **Learning-based prompt classifiers / embedding routers.** Out of scope — the rubric is about *optimization*, not ML. Use `r[p,m] = normalized quality` as the success probability directly.
3. **Real API-call latency or rate-limit modeling.** Out of scope — no data in the provided dataset to support it. Mention as a limitation.
4. **More than 3-stage cascades.** Out of scope — combinatorial explosion (33³ candidates), and marginal gain over 2-stage is small for 240 prompts. Implement 2-stage; mention 3-stage as a stretch only if A2 solves fast.
5. **Re-deriving the data from the original Li et al. (2026) benchmark.** Out of scope — use the starter-code CSV as ground truth.
6. **Building a frontend / interactive dashboard.** Out of scope — the deliverable is a static report.
7. **Solver-level innovations (custom branch-and-bound, Benders).** Out of scope — HiGHS / CBC / GLPK via Pyomo is sufficient for problem sizes here.

---

## 4. Research Questions

The two questions the report will answer end-to-end. Both map directly onto rubric language ("pool size," "diversity," "robustness," "production deployment").

### RQ1
**Can a cascading LLM router reduce expected cost relative to single-shot routing while maintaining comparable expected quality?**

Comparisons:
- Always-best-model
- Starter-code weighted baseline (A0)
- Single-shot optimized router (A1)
- Two-stage cascade (A2)

### RQ2
**How do robustness and reliability constraints change the selected model pool, routing decisions, and domain-level performance?**

Analyses:
- Pool size K vs. quality
- Which models survive in the optimal pool
- Cost-quality frontier shift between A2 and A3
- Per-domain quality (AIME / LCB / GPQA / MMLU-Pro)
- Worst-case prompt-mix scenario quality
- Whether any one benchmark is sacrificed by an unconstrained model

---

## 5. User Stories

Three personas: the analyst (us, building this), the grader (the evaluator), and the simulated business user.

**Analyst stories (P0):**
- As the analyst, I want a single `run_experiments.py` entry point so I can rerun the full pipeline after any data or parameter change without re-deriving anything.
- As the analyst, I want every experiment to save its inputs, outputs, and solver status to CSV so I can reproduce numbers in the report.
- As the analyst, I want the cascade candidate generator to filter to ≤ ~200 cascades so the MILP solves in under 5 minutes on a laptop.

**Grader stories (P0):**
- As the grader, I want to see clearly defined sets, parameters, decision variables, and constraints with notation that matches the formulation in the report.
- As the grader, I want to see at least one stochastic element and at least one multi-stage decision in the optimization model.
- As the grader, I want to see an ablation: at least one hyperparameter (α, K, B, or scenario weights) varied across a meaningful range, with results plotted.

**Simulated business stories (P1):**
- As a deployment engineer, I want pool size K to be a tunable knob so I can match it to my team's monitoring budget.
- As a finance owner, I want the budget B to enforce expected cost, not just realized cost, so the policy is contract-friendly.
- As a quality lead, I want a per-domain quality floor so AIME doesn't collapse to satisfy an average target driven by easy MMLU-Pro prompts.

**Edge cases (P0):**
- Empty model pool (K = 0): infeasible, surface clearly.
- Budget B below cost of cheapest cascade for some prompt: infeasible, surface clearly.
- Quality column already in [0,1]: skip normalization.
- Some models have zero cost (open source): preserve as-is, do not divide-by-zero anywhere.

---

## 6. Optimization Model Architecture (the 10-point centerpiece)

This is the section the grader weighs most. Every formulation below maps to a Python module listed in §10.

### 6.1 Notation (single notation table — used by all four models)

| Symbol | Type | Meaning |
|---|---|---|
| $P$ | set | Prompts (240) |
| $M$ | set | Candidate models (33) |
| $D$ | set | Domains {AIME, LCB, GPQA, MMLU-Pro} |
| $P_d \subseteq P$ | set | Prompts in domain $d$ |
| $A$ | set | Candidate cascades (tuples of 2 models from $M$) |
| $S$ | set | Prompt-distribution scenarios |
| $q_{pm} \in [0,1]$ | parameter | Normalized quality of model $m$ on prompt $p$ |
| $c_{pm} \ge 0$ | parameter | Cost of model $m$ on prompt $p$ |
| $r_{pm} \in [0,1]$ | parameter | Estimated success probability of $m$ on $p$ (set $r_{pm} = q_{pm}$ after normalization) |
| $R_{pa}$ | parameter | Expected cascade success probability (derived) |
| $C_{pa}$ | parameter | Expected cascade cost (derived) |
| $w_{sp} \ge 0$ | parameter | Weight of prompt $p$ under scenario $s$, with $\sum_p w_{sp} = 1$ |
| $K \in \mathbb{Z}_+$ | parameter | Pool-size cap |
| $B > 0$ | parameter | Average-cost budget |
| $E_{\max} \in [0,1]$ | parameter | Maximum allowed expected escalation rate |
| $\tau_d \in [0,1]$ | parameter | Per-domain quality floor |
| $\lambda_{\text{slack}} \ge 0$ | parameter | Penalty weight for domain slack |
| $y_m \in \{0,1\}$ | decision | =1 iff model $m$ is in the selected pool |
| $x_{pm} \in \{0,1\}$ | decision (A1) | =1 iff prompt $p$ is routed to model $m$ |
| $z_{pa} \in \{0,1\}$ | decision (A2, A3) | =1 iff prompt $p$ uses cascade $a$ |
| $\eta \in [0,1]$ | decision (A3) | Worst-scenario expected quality |
| $s_d \ge 0$ | decision (A3) | Slack on domain $d$'s quality floor |

### 6.2 Model A0 — starter-code weighted baseline (deterministic, single-stage)

No optimization variables — pointwise rule per prompt. Used only as a reference point on plots.
$$f_\alpha(p) = \arg\min_{m \in M} \; c_{pm} - \alpha \cdot q_{pm}, \qquad \alpha \in \{0.1, 0.5, 1, 2, 5, 10\}.$$

Outputs per α: average cost, average quality, set of models used, per-domain quality.

### 6.3 Model A1 — single-shot portfolio router (deterministic, single-stage with pool selection)

$$\max_{x, y} \; \frac{1}{|P|} \sum_{p \in P} \sum_{m \in M} q_{pm} x_{pm}$$

subject to
$$\sum_{m \in M} x_{pm} = 1 \quad \forall p \in P,$$
$$x_{pm} \le y_m \quad \forall p \in P, \, m \in M,$$
$$\sum_{m \in M} y_m \le K,$$
$$\frac{1}{|P|} \sum_{p,m} c_{pm} x_{pm} \le B,$$
$$x_{pm}, y_m \in \{0,1\}.$$

Run grid: $K \in \{1,2,3,5,8\}$ × $B \in$ {percentiles of A0 cost grid}. This answers RQ1's "pool size" sub-question and gives the second point on the cost-quality frontier.

### 6.4 Model A2 — two-stage cascading router (multi-stage stochastic — *core innovation*)

**Stochastic element:** for each cascade $a = (m_1, m_2)$, the outcome of the first stage is a Bernoulli draw with success probability $r_{pm_1}$. Whether the second stage is invoked is itself a random variable, which is what makes this a multi-stage problem.

Precomputed parameters:
$$R_{p,(m_1,m_2)} = 1 - (1 - r_{pm_1})(1 - r_{pm_2}),$$
$$C_{p,(m_1,m_2)} = c_{pm_1} + (1 - r_{pm_1}) \, c_{pm_2}.$$

Cascade candidate generation rules (configurable; keeps $|A| \approx 150{-}250$):
- $m_1$ in the bottom-25th-percentile of average cost OR open-source ($c_{pm_1} = 0$).
- $m_2$ in the top-50th-percentile of average quality, with $\bar q_{m_2} \ge \bar q_{m_1}$.
- Optional: keep top-N models by quality-per-dollar to bound combinatorics.

Formulation:
$$\max_{y,z} \; \frac{1}{|P|} \sum_{p \in P} \sum_{a \in A} R_{pa} z_{pa}$$

subject to
$$\sum_{a \in A} z_{pa} = 1 \quad \forall p,$$
$$z_{pa} \le y_m \quad \forall p, \, \forall a \ni m,$$
$$\sum_{m \in M} y_m \le K,$$
$$\frac{1}{|P|} \sum_{p,a} C_{pa} z_{pa} \le B,$$
$$\frac{1}{|P|} \sum_{p, a=(m_1,m_2)} z_{pa}(1 - r_{pm_1}) \le E_{\max} \quad \text{(expected escalation rate)},$$
$$z_{pa}, y_m \in \{0,1\}.$$

The escalation-rate constraint is the second "additional realistic constraint" required by the rubric. The cascade itself satisfies the "multi-stage decisions" rubric requirement.

### 6.5 Model A3 — robust reliability-aware cascade (multi-stage stochastic + robust + fair)

Extends A2 by replacing the average-quality objective with a worst-scenario formulation and adding domain fairness slacks.

Scenarios (5 total):

| Scenario | AIME | LCB | GPQA | MMLU-Pro |
|---|---|---|---|---|
| empirical | 0.25 | 0.25 | 0.25 | 0.25 |
| coding_heavy | 0.10 | 0.55 | 0.15 | 0.20 |
| math_heavy | 0.55 | 0.10 | 0.15 | 0.20 |
| knowledge_heavy | 0.10 | 0.10 | 0.40 | 0.40 |
| balanced | 0.25 | 0.25 | 0.25 | 0.25 |

(Empirical and balanced are intentionally similar — keep both for clarity; if needed collapse to 4.)

Derive prompt weights: $w_{sp} = \frac{\pi_{s,d(p)}}{|P_{d(p)}|}$ for prompt $p$ in domain $d(p)$, scenario $s$ with domain probability $\pi_{s,d}$.

$$\max_{y,z,\eta,s} \; \eta \;-\; \lambda_{\text{slack}} \sum_{d \in D} s_d$$

subject to all A2 constraints plus
$$\sum_{p \in P} w_{sp} \sum_{a \in A} R_{pa} z_{pa} \;\ge\; \eta \quad \forall s \in S,$$
$$\sum_{p \in P} w_{sp} \sum_{a \in A} C_{pa} z_{pa} \;\le\; B \quad \forall s \in S,$$
$$\frac{1}{|P_d|} \sum_{p \in P_d} \sum_{a \in A} R_{pa} z_{pa} + s_d \;\ge\; \tau_d \quad \forall d \in D,$$
$$\eta \in [0,1], \; s_d \ge 0.$$

**Linearity:** $R_{pa}$ and $C_{pa}$ are *parameters* (precomputed), so the model stays a MILP. No bilinear terms.

**Sample-average approximation form:** the population-form statement is "maximize the worst expected quality over an ambiguity set of prompt distributions." The SAA form replaces the true distribution with the empirical-sample dataset reweighted by $w_{sp}$. This is the form we solve. The report should state both.

---

## 7. Requirements

### 7.1 Must-Have (P0)

| ID | Requirement | Acceptance criteria |
|---|---|---|
| P0-1 | Data loader produces $q, c, r$ dicts and $P, M, D$ sets | Unit test: loader on starter CSV returns 240 prompts × 33 models, no nulls, $q \in [0,1]$ after normalization |
| P0-2 | A0 baseline runs for the 6 α values | 6 rows in `outputs/tables/a0_results.csv` |
| P0-3 | A1 Pyomo model solves for $K \in \{1,2,3,5,8\}$ at 3 budget levels | 15 rows in `outputs/tables/a1_results.csv` with status="ok" |
| P0-4 | Cascade candidates generated with $|A| \in [100, 300]$ | `outputs/tables/cascade_candidates.csv` with R, C columns |
| P0-5 | A2 Pyomo model solves for $K \in \{2,3,5\}$ at 3 budget levels | 9 rows with selected models, cascade usage, escalation rate |
| P0-6 | Cost-quality frontier plot includes A0, A1, A2 | `outputs/figures/cost_quality_frontier.png` exists |
| P0-7 | Quality-vs-K plot for A1 and A2 | `outputs/figures/quality_vs_pool_size.png` |
| P0-8 | Per-domain quality table for at least one (K,B) per policy | `outputs/tables/domain_quality.csv` with 4 rows |
| P0-9 | 4-page report draft with all required sections | `report.pdf` with intro, model, results, discussion, takeaway |

### 7.2 Should-Have (P1)

| ID | Requirement | Acceptance criteria |
|---|---|---|
| P1-1 | A3 robust cascade solves for 5 scenarios at one (K,B) | `outputs/tables/a3_results.csv` with η value |
| P1-2 | Robustness heatmap (scenarios × policies) | `outputs/figures/robustness_heatmap.png` |
| P1-3 | Selected-model usage bar chart | `outputs/figures/selected_model_usage.png` |
| P1-4 | Domain performance comparison plot | `outputs/figures/domain_performance_comparison.png` |
| P1-5 | Storage or provider constraint added (rubric "two realistic constraints") | Constraint in A2 or A3 referenced in report |

### 7.3 Could-Have (P2)

- 3-stage cascade variant
- Quadratic budget-overrun penalty: $(\max\{0, \bar C - B\})^2$
- Sankey diagram of cascade flow
- Coefficient-of-variation analysis on quality across solver reruns

### 7.4 Won't-Have (this version)

- Full Wasserstein DRO
- Learned classifier for $r_{pm}$
- Real-time online routing
- Multi-objective Pareto enumeration via ε-constraint sweep beyond the 3-budget grid

---

## 8. Data Requirements

**Input:** the starter-code CSV with at least the columns `prompt_id`, `domain` (or `benchmark`), `model`, `quality`, `cost`. Dataset shape: 240 prompts × 33 models = 7,920 rows (one row per (prompt, model) pair) or wide format.

**Preprocessing checks:**
- Confirm 33 unique models and 240 unique prompts.
- Confirm 4 unique domains with prompt counts roughly balanced (likely 60 each).
- Quality range: detect whether it is already $[0,1]$, $[0,100]$, or unscaled accuracy. Min-max normalize per-prompt if necessary; otherwise normalize globally.
- Cost: confirm some models have cost 0 (open-source). Do NOT clip.
- Save `data/preprocessed.parquet` for fast reload.

**Derived quantities cached on disk:**
- `q[p,m]`, `c[p,m]`, `r[p,m]` as 240×33 NumPy arrays
- Cascade candidate table `A`
- Precomputed $R_{pa}, C_{pa}$ for all $(p, a)$

---

## 9. Implementation Plan (build order, not write-up order)

Hard rule: complete each step before starting the next. Do not write the report until step 7.

1. **Load + preprocess data** (1 hour) → outputs/preprocessed.parquet, sanity-check printouts.
2. **A0 baseline** (1 hour) → outputs/tables/a0_results.csv + first frontier dot.
3. **A1 single-shot Pyomo** (2 hours) → outputs/tables/a1_results.csv, frontier curve.
4. **Cascade generation** (1 hour) → outputs/tables/cascade_candidates.csv with size sanity check.
5. **A2 cascade Pyomo** (3 hours) → outputs/tables/a2_results.csv, second frontier curve, escalation-rate plot input.
6. **A3 robust extension** (2 hours, can defer to stretch) → outputs/tables/a3_results.csv, robustness heatmap input.
7. **Plots** (2 hours) → all 5 PNGs in outputs/figures.
8. **Tables for report** (1 hour) → summary comparison + domain table.
9. **Report writing** (3–4 hours) → model section first, results second, intro third, discussion fourth, takeaway last.
10. **Final polish** (1 hour) → cross-check numbers, regenerate plots if needed, attach code.

Total: ~17 hours of work. Two work sessions should fit before the May 15 deadline.

---

## 10. File Structure

```
INDENG 164/
├── data/
│   └── original_dataset.csv          # starter code data
├── src/
│   ├── load_data.py                  # parse CSV, build P, M, D
│   ├── preprocessing.py              # normalize quality, derive r
│   ├── baselines.py                  # A0 weighted rule
│   ├── pyomo_single_shot.py          # A1
│   ├── pyomo_cascade.py              # A2 + cascade candidate generation
│   ├── pyomo_robust_cascade.py       # A3
│   ├── experiments.py                # grid runners, result tables
│   └── plots.py                      # all 5 figures
├── outputs/
│   ├── tables/                       # all CSV results
│   ├── figures/                      # all PNG plots
│   └── solutions/                    # selected models, routing dicts (JSON)
├── report/
│   ├── report.tex                    # or report.md → PDF
│   └── figures/                      # symlink or copy from outputs/figures
├── run_experiments.py                # one-call pipeline
├── requirements.txt
└── README.md
```

Fall-back single-notebook layout (if time gets very tight): `llm_routing_project.ipynb` + `report.pdf` + `outputs/`.

---

## 11. Codex Build Prompts

Use these prompts in order. Each is self-contained so Codex can be re-prompted at any step without losing context.

### 11.1 Master prompt (paste this first only if Codex handles ~3000 tokens cleanly)

```
You are helping me build a Pyomo-based optimization project for INDENG 164.

Project title:
Cheap First, Reliable Always: A Robust Cascading Optimization Framework for LLM Routing

Context:
33 candidate LLMs evaluated on 240 prompts across AIME (math), LCB (code), GPQA, MMLU-Pro
(knowledge). Goal: select a small model pool and a routing policy that balances cost,
quality, robustness, and reliability.

Build in stages, four models:
  A0 — weighted baseline (no optimization)
  A1 — single-shot portfolio router (MILP)
  A2 — two-stage cascading router (MILP, multi-stage stochastic via expected R, C)
  A3 — robust reliability-aware cascade (MILP with worst-scenario eta and domain slacks)

Use Pyomo + pandas + matplotlib only. Solver: try HiGHS first, fall back to CBC then GLPK.
Match the file structure in the PRD §10. Save tables to outputs/tables and figures to
outputs/figures. Add docstrings to every function. Code must be readable and report-friendly.

Build A0, A1, then A2, then A3, then plots — in that order. Do not jump ahead.
```

### 11.2 Step prompts (use one at a time if the master prompt is too big)

**Prompt 1 — data loading:**
```
Build src/load_data.py and src/preprocessing.py. Load the starter CSV. Detect column
names for prompt_id, domain, model, quality, cost. Build sets P, M, D and dicts
q[p,m], c[p,m], r[p,m] (with r = quality after min-max normalization to [0,1]).
Print summary stats: |P|, |M|, prompts per domain, quality range, cost range, count of
zero-cost models. Save preprocessed parquet for reload.
```

**Prompt 2 — A0 baseline:**
```
Build src/baselines.py. For each alpha in [0.1, 0.5, 1, 2, 5, 10], assign each prompt p
to argmin_m (cost[p,m] - alpha * quality[p,m]). Save avg cost, avg quality, model usage
counts, and per-domain quality to outputs/tables/a0_results.csv.
```

**Prompt 3 — A1 single-shot Pyomo:**
```
Build src/pyomo_single_shot.py with a function solve_a1(K, B) returning a dict with
selected_models, assignment, obj_quality, avg_cost, domain_quality, status. Use binary
y[m] and x[p,m]. Objective: maximize average quality. Constraints: sum_m x[p,m]=1,
x[p,m] <= y[m], sum_m y[m] <= K, avg cost <= B. Run grid K in {1,2,3,5,8} x 3 budgets,
save to outputs/tables/a1_results.csv.
```

**Prompt 4 — cascade generation:**
```
Build cascade candidate generation in src/pyomo_cascade.py. Cheap m1: cost percentile
<= 25 OR zero-cost. Strong m2: average quality percentile >= 50, and qbar[m2] >= qbar[m1].
Optional top-N filter to cap |A| at ~250. Precompute R[p,a] and C[p,a] for all (p,a)
and save cascade_candidates.csv.
```

**Prompt 5 — A2 cascade Pyomo:**
```
Add solve_a2(K, B, Emax) to src/pyomo_cascade.py. Binary y[m], z[p,a]. Objective:
maximize avg R. Constraints: sum_a z[p,a]=1, z[p,a] <= y[m] for every m in a,
sum_m y[m] <= K, avg C <= B, expected escalation rate <= Emax. Run K in {2,3,5} x 3
budgets x Emax in {1.0, 0.5}, save to outputs/tables/a2_results.csv with cascade and
model-stage usage.
```

**Prompt 6 — A3 robust:**
```
Build src/pyomo_robust_cascade.py with solve_a3(K, B, scenarios, tau, lambda_slack).
Take A2 model, add variable eta >= 0 and slack s[d] >= 0. Add constraints:
  scenario_quality[s] >= eta for every s,
  scenario_cost[s] <= B for every s,
  domain_quality[d] + s[d] >= tau[d] for every d.
Objective: maximize eta - lambda_slack * sum(s[d]). Save scenario- and domain-level
metrics to outputs/tables/a3_results.csv.
```

**Prompt 7 — plots:**
```
Build src/plots.py. Matplotlib only. Produce:
  1) cost_quality_frontier.png (A0, A1, A2, A3 markers/lines)
  2) quality_vs_pool_size.png (A1 and A2 curves over K)
  3) selected_model_usage.png (bar chart of model usage in A2 or A3)
  4) domain_performance_comparison.png (grouped bars, 4 policies x 4 domains)
  5) robustness_heatmap.png (rows: scenarios, cols: policies, cells: avg quality)
Save all to outputs/figures.
```

**Prompt 8 — orchestration:**
```
Build run_experiments.py that calls load -> preprocess -> A0 -> A1 -> A2 -> A3 -> plots
in order. Each step should be skippable via CLI flag. Add try/except with informative
error messages if a solver is missing or a result table cannot be read.
```

---

## 12. Success Metrics

### Leading (verifiable immediately on a finished run)
- All P0 acceptance criteria from §7.1 pass.
- A2 produces at least one (K, B) point that Pareto-dominates the A0 frontier.
- A3 worst-scenario quality $\eta$ is within 5% of A2 empirical-scenario quality.
- Solver returns `optimal` status for every grid cell (no timeouts at the chosen grid).

### Lagging (post-submission)
- Gradescope score on the project ≥ 17/20.
- Bonus 2 points missed (May 8 early deadline already past) — accepted.
- Self-assessed clarity: a peer can read the model section and reproduce the formulation without questions.

**Measurement window:** all leading metrics measured before report submission on May 15.

---

## 13. Plot Specifications

| # | File | X | Y | Series | Annotation |
|---|---|---|---|---|---|
| 1 | cost_quality_frontier.png | avg cost | avg quality | A0 line + A1 dots + A2 dots + A3 star | Pareto front overlay |
| 2 | quality_vs_pool_size.png | K | quality | A1 vs A2 lines | Mark diminishing-return knee |
| 3 | selected_model_usage.png | model name | # prompts using as $m_1$ or $m_2$ | stacked: $m_1$ vs $m_2$ | Sorted by frequency |
| 4 | domain_performance_comparison.png | domain | quality | grouped bars: A0, A1, A2, A3 | Add τ_d threshold line |
| 5 | robustness_heatmap.png | policy | scenario | heatmap cell color = quality | Annotate cells with numbers |

---

## 14. Report Structure (5 pages max — write last)

**Page 1 — Introduction:** company setting (Cursor-style), why naïve always-best and always-cheap fail, the two research questions, why stochastic multi-stage suits the problem.

**Pages 2–3 — Optimization Model:** notation table from §6.1, then A1 → A2 → A3 formulations in order. Population form first, SAA form after. Highlight where stochasticity enters (cascade Bernoulli) and where multi-stage enters (escalation).

**Page 4 — Results:** cost-quality frontier figure + summary table + per-domain table. Two paragraphs of analysis answering RQ1 and RQ2.

**Page 5 — Discussion + Takeaway:** business recommendation (cheap-first cascade with K=3–5 and per-domain floor), limitations (quality-as-probability, no latency data, 240 sampled prompts), one-paragraph personal takeaway.

Summary comparison table to anchor the discussion:

| Policy | K | Avg cost | Avg quality | Worst-scenario quality | Escalation rate |
|---|---|---|---|---|---|
| A0 baseline (best α) | — | TBD | TBD | TBD | N/A |
| A1 single-shot | 3 | TBD | TBD | TBD | N/A |
| A2 cascade | 3 | TBD | TBD | TBD | TBD |
| A3 robust cascade | 3 | TBD | TBD | TBD | TBD |

Per-domain comparison:

| Policy | AIME | LCB | GPQA | MMLU-Pro |
|---|---|---|---|---|
| A0 baseline | TBD | TBD | TBD | TBD |
| A1 | TBD | TBD | TBD | TBD |
| A2 | TBD | TBD | TBD | TBD |
| A3 | TBD | TBD | TBD | TBD |

---

## 15. Timeline (today is May 14, 2026 — submission May 15, 11:59pm)

| Block | Hours | Deliverable |
|---|---|---|
| Today afternoon | 4 | Data load + A0 + A1 + first frontier numbers |
| Today evening | 4 | Cascade generation + A2 + 3 plots |
| Tomorrow morning | 4 | A3 robust + remaining plots + tables |
| Tomorrow afternoon | 4 | Report draft (model + results sections first) |
| Tomorrow evening | 2 | Polish, intro/discussion/takeaway, Gradescope upload |

If anything slips, drop A3 (P1) before dropping any P0. The MVP without A3 still earns the multi-stage points via A2's cascade.

---

## 16. Open Questions

| # | Question | Owner | Blocking? |
|---|---|---|---|
| Q1 | Is quality in the starter CSV already in [0,1]? | data inspection | Yes, blocks preprocessing |
| Q2 | What is the realistic budget range — what does the average A0 cost look like at α = 1? | A0 run | Yes, sets B grid |
| Q3 | Which open-source MILP solver is installed locally (HiGHS, CBC, GLPK)? | env check | Yes, blocks A1+ |
| Q4 | Is a 2-point scenario "ambiguity set" sufficient, or should we have all 5? | course rubric reading | No, 5 is safe |
| Q5 | Do we need a 3-stage cascade for the rubric's "multi-stage decisions"? | course rubric reading | No, 2-stage cascade is multi-stage |

Q1–Q3 resolve in the first hour of work. Q4–Q5 already resolved in this PRD; revisit only if time permits.

---

## 17. Risks

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| MILP solver times out on A3 with 5 scenarios | Medium | High (would lose robustness story) | Reduce |A| via top-N filter; relax to 3 scenarios |
| Quality scale is per-benchmark and globally min-max wrecks signal | Medium | High | Normalize per-prompt or per-domain instead, compare |
| A2 cascade is worse than A1 in some grid cells (counter-intuitive) | Low | Medium | This is fine — show both, explain in report |
| Report runs over 5 pages | High | Low | Move notation table to appendix; tighten model section prose |
| Plots look amateurish | Medium | Medium | Use matplotlib style sheet; consistent colors per policy |
| Codex generates broken Pyomo syntax | Medium | High | Run each module immediately after generation, don't batch |

---

## 18. Personal Takeaway (template — fill after running experiments)

The most surprising finding was likely [TBD: e.g., "that a two-stage cascade with just K=3 models captured ~95% of the always-best-model quality at ~30% of the cost"]. The optimization machinery turned a vague intuition ("cheap-first, escalate when needed") into a quantified policy I could defend with numbers. The robustness layer felt overengineered at first but paid off once I plotted [TBD] — without the domain floor, A2 quietly sacrificed AIME to optimize average quality.

---

## 19. Definition of Done

- [ ] All P0 acceptance criteria pass.
- [ ] All 5 plots generated and visually checked.
- [ ] Both summary tables filled with real numbers (no TBDs).
- [ ] Report ≤ 5 pages, ≥ 4 pages.
- [ ] Code attached or linked, README explains how to run.
- [ ] Gradescope submission confirmed before May 15, 11:59pm.

---

*End of PRD. Hand the Codex prompts in §11 to your build agent; come back here only to update the "Open Questions" and "Risks" sections as the build progresses.*
