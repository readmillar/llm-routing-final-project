from __future__ import annotations

import math

import pyomo.environ as pyo

from .metrics import cascade_assignment_metrics
from .solver_utils import (
    has_solution,
    no_solver_result,
    pre_solve_diagnostics,
    result_status,
    solve_model,
)


def _termination_message(results):
    return str(results.solver.termination_condition)


def _invalid_result(policy, message):
    return {
        "policy": policy,
        "status": "invalid",
        "message": message,
        "diagnostics": pre_solve_diagnostics(policy, "invalid", message),
    }


def _infeasible_result(policy, message):
    return {
        "policy": policy,
        "status": "infeasible",
        "message": message,
        "diagnostics": pre_solve_diagnostics(policy, "infeasible", message),
    }


def _finite_float(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def solve_a4_cvar_cascade(
    data,
    cascades,
    R,
    C,
    Esc,
    A_p,
    floors,
    K,
    B,
    Emax,
    beta=0.9,
    lambda_cvar=0.1,
    time_limit=300,
):
    """Solve a CVaR tail-risk cascade MILP over prompt-level quality shortfalls."""
    beta_value = _finite_float(beta)
    beta_label = f"{beta_value:g}" if beta_value is not None else str(beta)
    policy = f"A4-CVaR K={K} B={B:.6g} Emax={Emax:g} beta={beta_label}"
    if beta_value is None or not 0.0 < beta_value < 1.0:
        return _invalid_result(policy, "A4 CVaR requires 0 < beta < 1")
    lambda_value = _finite_float(lambda_cvar)
    if lambda_value is None or lambda_value <= 0.0:
        return _invalid_result(policy, "A4 CVaR requires lambda_cvar > 0")
    if K < 1:
        return _infeasible_result(policy, "A4 requires K >= 1")

    cascade_frame = cascades.set_index("cascade_id").copy()
    if "m3" not in cascade_frame.columns:
        cascade_frame["m3"] = ""
    if "depth" not in cascade_frame.columns:
        cascade_frame["depth"] = 2
    cascade_lookup = cascade_frame[["m1", "m2", "m3", "depth"]].to_dict("index")

    pa = sorted((p, a) for p in data["P"] for a in A_p[p])
    stage_links = []
    for prompt, cascade_id in pa:
        row = cascade_lookup[cascade_id]
        for model_name in [row["m1"], row["m2"], row["m3"]]:
            if isinstance(model_name, str) and model_name:
                stage_links.append((prompt, cascade_id, model_name))
    n_prompts = len(data["P"])

    model = pyo.ConcreteModel()
    model.P = pyo.Set(initialize=data["P"])
    model.M = pyo.Set(initialize=data["M"])
    model.PA = pyo.Set(dimen=2, initialize=pa)
    model.PAM = pyo.Set(dimen=3, initialize=stage_links)
    model.z = pyo.Var(model.PA, within=pyo.Binary)
    model.y = pyo.Var(model.M, within=pyo.Binary)
    model.nu = pyo.Var(within=pyo.NonNegativeReals)
    model.shortfall = pyo.Var(model.P, within=pyo.NonNegativeReals)
    model.u = pyo.Var(model.P, within=pyo.NonNegativeReals)

    def assignment_rule(mdl, prompt):
        return sum(mdl.z[prompt, cascade_id] for cascade_id in A_p[prompt]) == 1

    def link_stage_rule(mdl, prompt, cascade_id, model_name):
        return mdl.z[prompt, cascade_id] <= mdl.y[model_name]

    def shortfall_floor_rule(mdl, prompt):
        floor = floors[data["prompt_domain"][prompt]]
        quality = sum(
            R[prompt, cascade_id] * mdl.z[prompt, cascade_id] for cascade_id in A_p[prompt]
        )
        return mdl.shortfall[prompt] >= floor - quality

    def cvar_excess_rule(mdl, prompt):
        return mdl.u[prompt] >= mdl.shortfall[prompt] - mdl.nu

    model.assignment = pyo.Constraint(model.P, rule=assignment_rule)
    model.link_stage = pyo.Constraint(model.PAM, rule=link_stage_rule)
    model.pool = pyo.Constraint(expr=sum(model.y[m] for m in model.M) <= K)
    model.budget = pyo.Constraint(expr=sum(C[p, a] * model.z[p, a] for p, a in pa) / n_prompts <= B)
    model.escalation = pyo.Constraint(
        expr=sum(Esc[p, a] * model.z[p, a] for p, a in pa) / n_prompts <= Emax
    )
    model.shortfall_floor = pyo.Constraint(model.P, rule=shortfall_floor_rule)
    model.cvar_excess = pyo.Constraint(model.P, rule=cvar_excess_rule)

    cvar_expr = model.nu + (1.0 / ((1.0 - beta_value) * n_prompts)) * sum(
        model.u[p] for p in model.P
    )
    avg_quality = sum(R[p, a] * model.z[p, a] for p, a in pa) / n_prompts
    model.objective = pyo.Objective(expr=avg_quality - lambda_value * cvar_expr, sense=pyo.maximize)

    solver_name, results, diagnostics = solve_model(model, time_limit=time_limit, policy=policy)
    if solver_name is None:
        return no_solver_result(policy, diagnostics)
    status = result_status(results)
    if not has_solution(status):
        return {
            "policy": policy,
            "status": status,
            "solver": solver_name,
            "message": _termination_message(results),
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
            "beta": beta_value,
            "lambda_cvar": lambda_value,
            "cvar_shortfall": float(pyo.value(cvar_expr, exception=False) or 0.0),
            "selected_models": [
                m for m in data["M"] if (pyo.value(model.y[m], exception=False) or 0.0) > 0.5
            ],
        }
    )
    return metrics
