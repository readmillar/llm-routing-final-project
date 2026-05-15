import pyomo.environ as pyo

from .metrics import scenario_quality, scenario_weights
from .metrics import cascade_assignment_metrics
from .solver_utils import has_solution, no_solver_result, result_status, solve_model


def _normalize_domain_weights(data, weights):
    present = {d: float(weights.get(d, 0.0)) for d in data["D"]}
    total = sum(present.values())
    if total <= 0:
        return {d: 1.0 / len(data["D"]) for d in data["D"]}
    return {d: value / total for d, value in present.items()}


def build_scenarios(data):
    """Build robust prompt-mix scenarios and prompt weights."""
    empirical = {d: len(data["P_d"][d]) / len(data["P"]) for d in data["D"]}
    raw = {
        "empirical": empirical,
        "balanced": {d: 1.0 / len(data["D"]) for d in data["D"]},
        "coding_heavy": {"AIME": 0.10, "LCB": 0.55, "GPQA": 0.15, "MMLU-Pro": 0.20},
        "math_heavy": {"AIME": 0.55, "LCB": 0.10, "GPQA": 0.15, "MMLU-Pro": 0.20},
        "knowledge_heavy": {"AIME": 0.10, "LCB": 0.10, "GPQA": 0.40, "MMLU-Pro": 0.40},
    }
    scenarios = {}
    for name, domain_weights in raw.items():
        normalized = _normalize_domain_weights(data, domain_weights)
        scenarios[name] = {
            "domain_weights": normalized,
            "prompt_weights": scenario_weights(data["P"], data["prompt_domain"], normalized),
        }
    return scenarios


def compute_domain_floors(data, multiplier=0.90):
    """Compute domain floors as a multiplier of the per-domain quality oracle."""
    floors = {}
    for domain, prompts in data["P_d"].items():
        oracle = sum(max(data["q"][(p, m)] for m in data["M_p"][p]) for p in prompts) / len(prompts)
        floors[domain] = multiplier * oracle
    return floors


def solve_a3(data, cascades, R, C, Esc, A_p, scenarios, floors, K, B, Emax, lambda_slack=0.10, time_limit=300):
    """Solve A3 robust reliability-aware cascade MILP."""
    policy = f"A3 K={K} B={B:.6g} Emax={Emax:g}"
    if K < 2:
        return {"policy": policy, "status": "infeasible", "message": "A3 requires K >= 2"}

    cascade_lookup = cascades.set_index("cascade_id")[["m1", "m2"]].to_dict("index")
    pa = sorted((p, a) for p in data["P"] for a in A_p[p])
    n_prompts = len(data["P"])

    model = pyo.ConcreteModel()
    model.P = pyo.Set(initialize=data["P"])
    model.M = pyo.Set(initialize=data["M"])
    model.D = pyo.Set(initialize=data["D"])
    model.S = pyo.Set(initialize=sorted(scenarios))
    model.PA = pyo.Set(dimen=2, initialize=pa)
    model.z = pyo.Var(model.PA, within=pyo.Binary)
    model.y = pyo.Var(model.M, within=pyo.Binary)
    model.eta = pyo.Var(bounds=(0.0, 1.0))
    model.floor_slack = pyo.Var(model.D, within=pyo.NonNegativeReals)

    def assignment_rule(mdl, prompt):
        return sum(mdl.z[prompt, cascade_id] for cascade_id in A_p[prompt]) == 1

    def link_first_rule(mdl, prompt, cascade_id):
        return mdl.z[prompt, cascade_id] <= mdl.y[cascade_lookup[cascade_id]["m1"]]

    def link_second_rule(mdl, prompt, cascade_id):
        return mdl.z[prompt, cascade_id] <= mdl.y[cascade_lookup[cascade_id]["m2"]]

    def scenario_quality_rule(mdl, scenario):
        weights = scenarios[scenario]["prompt_weights"]
        return sum(weights[p] * R[p, a] * mdl.z[p, a] for p, a in pa) >= mdl.eta

    def scenario_cost_rule(mdl, scenario):
        weights = scenarios[scenario]["prompt_weights"]
        return sum(weights[p] * C[p, a] * mdl.z[p, a] for p, a in pa) <= B

    def domain_floor_rule(mdl, domain):
        prompts = data["P_d"][domain]
        return (
            sum(R[p, a] * mdl.z[p, a] for p in prompts for a in A_p[p]) / len(prompts)
            + mdl.floor_slack[domain]
            >= floors[domain]
        )

    model.assignment = pyo.Constraint(model.P, rule=assignment_rule)
    model.link_first = pyo.Constraint(model.PA, rule=link_first_rule)
    model.link_second = pyo.Constraint(model.PA, rule=link_second_rule)
    model.pool = pyo.Constraint(expr=sum(model.y[m] for m in model.M) <= K)
    model.escalation = pyo.Constraint(expr=sum(Esc[p, a] * model.z[p, a] for p, a in pa) / n_prompts <= Emax)
    model.scenario_quality = pyo.Constraint(model.S, rule=scenario_quality_rule)
    model.scenario_cost = pyo.Constraint(model.S, rule=scenario_cost_rule)
    model.domain_floor = pyo.Constraint(model.D, rule=domain_floor_rule)
    model.objective = pyo.Objective(expr=model.eta - lambda_slack * sum(model.floor_slack[d] for d in model.D), sense=pyo.maximize)

    solver_name, results = solve_model(model, time_limit=time_limit)
    if solver_name is None:
        return no_solver_result(policy)
    status = result_status(results)
    if not has_solution(status):
        return {"policy": policy, "status": status, "solver": solver_name, "message": str(results.solver.termination_condition)}

    assignment = {}
    for prompt in data["P"]:
        for cascade_id in A_p[prompt]:
            value = pyo.value(model.z[prompt, cascade_id], exception=False)
            if value is not None and value > 0.5:
                assignment[prompt] = cascade_id
                break
    if set(assignment) != set(data["P"]):
        return {"policy": policy, "status": status, "solver": solver_name, "message": "Solver stopped before loading a complete incumbent solution"}
    base = cascade_assignment_metrics(data, cascades, assignment, R, C, Esc, policy)
    scenario_metrics = {}
    for name, scenario in scenarios.items():
        weights = scenario["prompt_weights"]
        scenario_metrics[name] = {
            "avg_quality": scenario_quality(weights, assignment, R),
            "avg_cost": scenario_quality(weights, assignment, C),
        }
    base.update(
        {
            "policy": policy,
            "status": "feasible" if status == "time_limited" else status,
            "solver": solver_name,
            "K": K,
            "B": B,
            "Emax": Emax,
            "eta": float(pyo.value(model.eta, exception=False) or 0.0),
            "selected_models": [m for m in data["M"] if (pyo.value(model.y[m], exception=False) or 0.0) > 0.5],
            "cascade_assignment": assignment,
            "scenario_metrics": scenario_metrics,
            "domain_slacks": {d: float(pyo.value(model.floor_slack[d], exception=False) or 0.0) for d in data["D"]},
            "lambda_slack": lambda_slack,
        }
    )
    return base
