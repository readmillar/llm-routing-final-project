import pyomo.environ as pyo

from .cascade_generation import generate_cascades, summarize_models  # noqa: F401
from .metrics import cascade_assignment_metrics
from .solver_utils import (
    has_solution,
    no_solver_result,
    pre_solve_diagnostics,
    result_status,
    solve_model,
)


def solve_a2(data, cascades, R, C, Esc, A_p, K, B, Emax, time_limit=300):
    """Solve A2 two-stage cascade MILP."""
    policy = f"A2 K={K} B={B:.6g} Emax={Emax:g}"
    if K < 1:
        message = "A2 requires K >= 1"
        return {
            "policy": policy,
            "status": "infeasible",
            "message": message,
            "diagnostics": pre_solve_diagnostics(policy, "infeasible", message),
        }
    cascade_lookup = cascades.set_index("cascade_id")[["m1", "m2", "m3", "depth"]].to_dict("index")
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

    def assignment_rule(mdl, prompt):
        return sum(mdl.z[prompt, cascade_id] for cascade_id in A_p[prompt]) == 1

    def link_stage_rule(mdl, prompt, cascade_id, model_name):
        return mdl.z[prompt, cascade_id] <= mdl.y[model_name]

    model.assignment = pyo.Constraint(model.P, rule=assignment_rule)
    model.link_stage = pyo.Constraint(model.PAM, rule=link_stage_rule)
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
