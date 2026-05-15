import pandas as pd
import pyomo.environ as pyo

from .metrics import cascade_assignment_metrics
from .solver_utils import (
    has_solution,
    no_solver_result,
    pre_solve_diagnostics,
    result_status,
    solve_model,
)


def summarize_models(data):
    """Summarize observed cost and quality by model."""
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


def _candidate_cascades(data, rho):
    summary = summarize_models(data).set_index("model")
    cost_cutoff = summary["cbar"].quantile(0.30)
    quality_cutoff = summary["qbar"].quantile(0.50)
    cheap = summary[(summary["cbar"] <= cost_cutoff) | (summary["zero_cost_rows"] > 0)].index
    strong = summary[summary["qbar"] >= quality_cutoff].index
    rows = []
    pm = set(data["PM"])
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
                    "m1": m1,
                    "m2": m2,
                    "qbar_m1": summary.loc[m1, "qbar"],
                    "qbar_m2": summary.loc[m2, "qbar"],
                    "cbar_m1": summary.loc[m1, "cbar"],
                    "cbar_m2": summary.loc[m2, "cbar"],
                    "avg_R": avg_r,
                    "avg_C": avg_c,
                    "avg_Esc": avg_esc,
                    "feasible_prompts": len(feasible_prompts),
                }
            )
    if not rows:
        raise ValueError("No feasible cascade candidates generated")
    return pd.DataFrame(rows)


def generate_cascades(data, rho=0.75, max_cascades=250):
    """Generate feasible two-stage cascades and prompt-specific parameters."""
    cascades = _candidate_cascades(data, rho)
    if len(cascades) > max_cascades:
        low_cost_n = max(1, max_cascades // 2)
        high_quality_n = max_cascades - low_cost_n
        low_cost = cascades.sort_values(["avg_C", "avg_R"], ascending=[True, False]).head(
            low_cost_n
        )
        high_quality = cascades.sort_values(
            ["avg_R", "avg_C", "feasible_prompts"],
            ascending=[False, True, False],
        ).head(high_quality_n)
        cascades = pd.concat([low_cost, high_quality], ignore_index=True).drop_duplicates(
            ["m1", "m2"]
        )
        if len(cascades) < max_cascades:
            fill = _candidate_cascades(data, rho).sort_values(
                ["avg_C", "avg_R"], ascending=[True, False]
            )
            cascades = (
                pd.concat([cascades, fill], ignore_index=True)
                .drop_duplicates(["m1", "m2"])
                .head(max_cascades)
            )
    cascades = cascades.reset_index(drop=True)
    cascades.insert(0, "cascade_id", [f"c{i:03d}" for i in range(len(cascades))])

    pm = set(data["PM"])
    a_p = {prompt: [] for prompt in data["P"]}
    r_param = {}
    c_param = {}
    esc_param = {}
    for row in cascades.itertuples(index=False):
        for prompt in data["P"]:
            if (prompt, row.m1) not in pm or (prompt, row.m2) not in pm:
                continue
            a_p[prompt].append(row.cascade_id)
            r1 = data["r"][(prompt, row.m1)]
            r2 = data["r"][(prompt, row.m2)]
            r_param[(prompt, row.cascade_id)] = r1 + (1 - r1) * rho * r2
            c_param[(prompt, row.cascade_id)] = (
                data["c"][(prompt, row.m1)] + (1 - r1) * data["c"][(prompt, row.m2)]
            )
            esc_param[(prompt, row.cascade_id)] = 1 - r1
    uncovered = [prompt for prompt, values in a_p.items() if not values]
    if uncovered:
        raise ValueError(f"No feasible cascades for prompts: {uncovered[:5]}")
    return cascades, {"A_p": a_p, "R": r_param, "C": c_param, "Esc": esc_param}


def solve_a2(data, cascades, R, C, Esc, A_p, K, B, Emax, time_limit=300):
    """Solve A2 two-stage cascade MILP."""
    policy = f"A2 K={K} B={B:.6g} Emax={Emax:g}"
    if K < 2:
        message = "A2 requires K >= 2"
        return {
            "policy": policy,
            "status": "infeasible",
            "message": message,
            "diagnostics": pre_solve_diagnostics(policy, "infeasible", message),
        }
    cascade_lookup = cascades.set_index("cascade_id")[["m1", "m2"]].to_dict("index")
    pa = sorted((p, a) for p in data["P"] for a in A_p[p])
    n_prompts = len(data["P"])

    model = pyo.ConcreteModel()
    model.P = pyo.Set(initialize=data["P"])
    model.M = pyo.Set(initialize=data["M"])
    model.PA = pyo.Set(dimen=2, initialize=pa)
    model.z = pyo.Var(model.PA, within=pyo.Binary)
    model.y = pyo.Var(model.M, within=pyo.Binary)

    def assignment_rule(mdl, prompt):
        return sum(mdl.z[prompt, cascade_id] for cascade_id in A_p[prompt]) == 1

    def link_first_rule(mdl, prompt, cascade_id):
        return mdl.z[prompt, cascade_id] <= mdl.y[cascade_lookup[cascade_id]["m1"]]

    def link_second_rule(mdl, prompt, cascade_id):
        return mdl.z[prompt, cascade_id] <= mdl.y[cascade_lookup[cascade_id]["m2"]]

    model.assignment = pyo.Constraint(model.P, rule=assignment_rule)
    model.link_first = pyo.Constraint(model.PA, rule=link_first_rule)
    model.link_second = pyo.Constraint(model.PA, rule=link_second_rule)
    model.pool = pyo.Constraint(expr=sum(model.y[m] for m in model.M) <= K)
    model.budget = pyo.Constraint(expr=sum(C[p, a] * model.z[p, a] for p, a in pa) / n_prompts <= B)
    model.escalation = pyo.Constraint(
        expr=sum(Esc[p, a] * model.z[p, a] for p, a in pa) / n_prompts <= Emax
    )
    model.objective = pyo.Objective(
        expr=sum(R[p, a] * model.z[p, a] for p, a in pa) / n_prompts, sense=pyo.maximize
    )

    solver_name, results, diagnostics = solve_model(model, time_limit=time_limit, policy=policy)
    if solver_name is None:
        return no_solver_result(policy, diagnostics)
    status = result_status(results)
    if not has_solution(status):
        return {
            "policy": policy,
            "status": status,
            "solver": solver_name,
            "message": str(results.solver.termination_condition),
            "diagnostics": diagnostics,
        }

    assignment = {}
    for prompt in data["P"]:
        for cascade_id in A_p[prompt]:
            value = pyo.value(model.z[prompt, cascade_id], exception=False)
            if value is not None and value > 0.5:
                assignment[prompt] = cascade_id
                break
    if set(assignment) != set(data["P"]):
        return {
            "policy": policy,
            "status": status,
            "solver": solver_name,
            "message": "Solver stopped before loading a complete incumbent solution",
            "diagnostics": diagnostics,
        }
    metrics = cascade_assignment_metrics(data, cascades, assignment, R, C, Esc, policy)
    metrics.update(
        {
            "status": "feasible" if status == "feasible_time_limited" else status,
            "solver": solver_name,
            "diagnostics": diagnostics,
            "K": K,
            "B": B,
            "Emax": Emax,
            "selected_models": [
                m for m in data["M"] if (pyo.value(model.y[m], exception=False) or 0.0) > 0.5
            ],
        }
    )
    return metrics
