import pyomo.environ as pyo

from .metrics import assignment_metrics
from .solver_utils import has_solution, no_solver_result, result_status, solve_model


def _minimum_assignment_cost(data):
    return sum(min(data["c"][(p, m)] for m in data["M_p"][p]) for p in data["P"]) / len(data["P"])


def solve_a1(data, K, B, time_limit=300):
    """Solve A1 single-shot portfolio MILP over observed prompt-model pairs."""
    policy = f"A1 K={K} B={B:.6g}"
    if K <= 0:
        return {"policy": policy, "status": "infeasible", "message": "K must be positive"}
    if B + 1e-12 < _minimum_assignment_cost(data):
        return {"policy": policy, "status": "infeasible", "message": "Budget below cheapest assignment"}

    model = pyo.ConcreteModel()
    model.P = pyo.Set(initialize=data["P"])
    model.M = pyo.Set(initialize=data["M"])
    model.PM = pyo.Set(dimen=2, initialize=data["PM"])
    model.x = pyo.Var(model.PM, within=pyo.Binary)
    model.y = pyo.Var(model.M, within=pyo.Binary)

    n_prompts = len(data["P"])

    def assignment_rule(mdl, prompt):
        return sum(mdl.x[prompt, model_name] for model_name in data["M_p"][prompt]) == 1

    def link_rule(mdl, prompt, model_name):
        return mdl.x[prompt, model_name] <= mdl.y[model_name]

    def budget_rule(mdl):
        return (
            sum(data["c"][(p, m)] * mdl.x[p, m] for p, m in data["PM"]) / n_prompts
            <= B
        )

    model.assignment = pyo.Constraint(model.P, rule=assignment_rule)
    model.link = pyo.Constraint(model.PM, rule=link_rule)
    model.pool = pyo.Constraint(expr=sum(model.y[m] for m in model.M) <= K)
    model.budget = pyo.Constraint(rule=budget_rule)
    model.objective = pyo.Objective(
        expr=sum(data["q"][(p, m)] * model.x[p, m] for p, m in data["PM"]) / n_prompts,
        sense=pyo.maximize,
    )

    solver_name, results = solve_model(model, time_limit=time_limit)
    if solver_name is None:
        return no_solver_result(policy)
    status = result_status(results)
    if not has_solution(status):
        return {
            "policy": policy,
            "status": status,
            "solver": solver_name,
            "message": str(results.solver.termination_condition),
        }

    assignment = {}
    for prompt in data["P"]:
        for model_name in data["M_p"][prompt]:
            value = pyo.value(model.x[prompt, model_name], exception=False)
            if value is not None and value > 0.5:
                assignment[prompt] = model_name
                break
    if set(assignment) != set(data["P"]):
        return {
            "policy": policy,
            "status": status,
            "solver": solver_name,
            "message": "Solver stopped before loading a complete incumbent solution",
        }
    metrics = assignment_metrics(data, assignment, policy)
    metrics.update(
        {
            "status": "feasible" if status == "time_limited" else status,
            "solver": solver_name,
            "K": K,
            "B": B,
            "selected_models": [m for m in data["M"] if (pyo.value(model.y[m], exception=False) or 0.0) > 0.5],
        }
    )
    return metrics
