import pyomo.environ as pyo

from .metrics import cascade_assignment_metrics, scenario_quality, scenario_weights
from .model_metadata import validate_metadata_covers_models
from .solver_utils import (
    has_solution,
    no_solver_result,
    pre_solve_diagnostics,
    result_status,
    solve_model,
)
from .stress_testing import build_l1_shift_scenarios


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
    scenarios.update(build_l1_shift_scenarios(data, radius=0.4))
    return scenarios


def compute_domain_floors(data, multiplier=0.90):
    """Compute domain floors as a multiplier of the per-domain quality oracle."""
    floors = {}
    for domain, prompts in data["P_d"].items():
        oracle = sum(max(data["q"][(p, m)] for m in data["M_p"][p]) for p in prompts) / len(prompts)
        floors[domain] = multiplier * oracle
    return floors


def build_a3_model(
    data,
    cascades,
    R,
    C,
    Esc,
    A_p,
    scenarios,
    floors,
    K,
    B,
    Emax,
    metadata=None,
    storage_cap_gb=None,
    provider_pool_caps=None,
    provider_traffic_caps=None,
):
    """Build an unsolved A3 model and return metadata needed for extraction."""
    if metadata is not None:
        validate_metadata_covers_models(metadata, data["M"])
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
    model.D = pyo.Set(initialize=data["D"])
    model.S = pyo.Set(initialize=sorted(scenarios))
    model.PA = pyo.Set(dimen=2, initialize=pa)
    model.PAM = pyo.Set(dimen=3, initialize=stage_links)
    model.z = pyo.Var(model.PA, within=pyo.Binary)
    model.y = pyo.Var(model.M, within=pyo.Binary)
    model.eta = pyo.Var(bounds=(0.0, 1.0))
    model.floor_slack = pyo.Var(model.D, within=pyo.NonNegativeReals)

    def assignment_rule(mdl, prompt):
        return sum(mdl.z[prompt, cascade_id] for cascade_id in A_p[prompt]) == 1

    def link_stage_rule(mdl, prompt, cascade_id, model_name):
        return mdl.z[prompt, cascade_id] <= mdl.y[model_name]

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
    model.link_stage = pyo.Constraint(model.PAM, rule=link_stage_rule)
    model.pool = pyo.Constraint(expr=sum(model.y[m] for m in model.M) <= K)
    if metadata is not None and storage_cap_gb is not None:
        storage = metadata.set_index("model")["estimated_storage_gb"].to_dict()
        model.storage = pyo.Constraint(
            expr=sum(float(storage[m]) * model.y[m] for m in model.M) <= float(storage_cap_gb)
        )
    if metadata is not None and provider_pool_caps:
        provider = metadata.set_index("model")["provider_family"].to_dict()
        model.G = pyo.Set(initialize=sorted(provider_pool_caps))

        def provider_pool_rule(mdl, group):
            models = [m for m in data["M"] if provider[m] == group]
            if not models:
                return pyo.Constraint.Feasible
            return sum(mdl.y[m] for m in models) <= int(provider_pool_caps[group])

        model.provider_pool = pyo.Constraint(model.G, rule=provider_pool_rule)
    if metadata is not None and provider_traffic_caps:
        provider = metadata.set_index("model")["provider_family"].to_dict()
        model.TG = pyo.Set(initialize=sorted(provider_traffic_caps))

        def provider_traffic_rule(mdl, scenario, group):
            weights = scenarios[scenario]["prompt_weights"]
            terms = []
            for p, a in pa:
                row = cascade_lookup[a]
                if provider[row["m1"]] == group:
                    terms.append(weights[p] * mdl.z[p, a])
                if (
                    isinstance(row.get("m2", ""), str)
                    and row["m2"]
                    and provider[row["m2"]] == group
                ):
                    terms.append(weights[p] * Esc[p, a] * mdl.z[p, a])
                # Do not count m3 traffic until cascade generation exposes a proper
                # reach-to-stage-3 probability. Esc is only first-stage escalation.
            if not terms:
                return pyo.Constraint.Feasible
            return sum(terms) <= float(provider_traffic_caps[group])

        model.provider_traffic = pyo.Constraint(model.S, model.TG, rule=provider_traffic_rule)
    model.escalation = pyo.Constraint(
        expr=sum(Esc[p, a] * model.z[p, a] for p, a in pa) / n_prompts <= Emax
    )
    model.scenario_quality = pyo.Constraint(model.S, rule=scenario_quality_rule)
    model.scenario_cost = pyo.Constraint(model.S, rule=scenario_cost_rule)
    model.domain_floor = pyo.Constraint(model.D, rule=domain_floor_rule)
    meta = {"pa": pa, "n_prompts": n_prompts, "cascade_lookup": cascade_lookup}
    return model, meta


def _a3_infeasible_k_result(policy):
    """Return the standard pre-solve infeasible result for undersized A3 pools."""
    message = "A3 requires K >= 2"
    return {
        "policy": policy,
        "status": "infeasible",
        "message": message,
        "diagnostics": pre_solve_diagnostics(policy, "infeasible", message),
    }


def _termination_message(results):
    return str(results.solver.termination_condition)


def _diagnostic_value(diagnostics, key):
    if not diagnostics:
        return None
    return diagnostics.get(key)


def _a3_total_slack(model, domains):
    values = []
    for domain in domains:
        value = pyo.value(model.floor_slack[domain], exception=False)
        if value is None:
            return None
        values.append(float(value))
    return sum(values)


def _lex_pass_row(
    pass_number,
    objective,
    status,
    solver_name,
    diagnostics,
    eta=None,
    total_slack=None,
    message=None,
):
    """Build a CSV-safe row describing one lexicographic solve pass."""
    return {
        "pass": pass_number,
        "objective": objective,
        "status": status,
        "solver": solver_name,
        "eta": eta,
        "total_slack": total_slack,
        "message": message,
        "mip_gap": _diagnostic_value(diagnostics, "mip_gap"),
        "termination_condition": _diagnostic_value(diagnostics, "termination_condition"),
        "wall_time_sec": _diagnostic_value(diagnostics, "wall_time_sec"),
        "best_bound": _diagnostic_value(diagnostics, "best_bound"),
        "objective_value": _diagnostic_value(diagnostics, "objective_value"),
    }


def _lex_incomplete_result(
    policy,
    pass_number,
    status,
    solver_name,
    results,
    diagnostics,
    passes,
):
    """Return a clear result when a required lexicographic pass is not optimal."""
    message = (
        f"Lexicographic pass {pass_number} ended with status {status}; "
        "optimality is required before proceeding."
    )
    if results is not None:
        message = f"{message} Termination: {_termination_message(results)}"
    return {
        "policy": policy,
        "status": "lexicographic_incomplete",
        "failed_pass": pass_number,
        "failed_status": status,
        "solver": solver_name,
        "message": message,
        "diagnostics": diagnostics,
        "lexicographic_passes": passes,
    }


def _extract_a3_solution(
    data,
    cascades,
    R,
    C,
    Esc,
    A_p,
    scenarios,
    model,
    policy,
    status,
    solver_name,
    diagnostics=None,
    lambda_slack=None,
):
    """Read a solved A3 model into the public result dictionary."""
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
            "status": "feasible" if status == "feasible_time_limited" else status,
            "solver": solver_name,
            "diagnostics": diagnostics,
            "eta": float(pyo.value(model.eta, exception=False) or 0.0),
            "selected_models": [
                m for m in data["M"] if (pyo.value(model.y[m], exception=False) or 0.0) > 0.5
            ],
            "cascade_assignment": assignment,
            "scenario_metrics": scenario_metrics,
            "domain_slacks": {
                d: float(pyo.value(model.floor_slack[d], exception=False) or 0.0) for d in data["D"]
            },
        }
    )
    if lambda_slack is not None:
        base["lambda_slack"] = lambda_slack
    return base


def solve_a3(
    data,
    cascades,
    R,
    C,
    Esc,
    A_p,
    scenarios,
    floors,
    K,
    B,
    Emax,
    lambda_slack=0.10,
    time_limit=300,
    metadata=None,
    storage_cap_gb=None,
    provider_pool_caps=None,
    provider_traffic_caps=None,
):
    """Solve A3 robust reliability-aware cascade MILP."""
    policy = f"A3 K={K} B={B:.6g} Emax={Emax:g}"
    if K < 2:
        return _a3_infeasible_k_result(policy)

    model, _meta = build_a3_model(
        data,
        cascades,
        R,
        C,
        Esc,
        A_p,
        scenarios,
        floors,
        K,
        B,
        Emax,
        metadata=metadata,
        storage_cap_gb=storage_cap_gb,
        provider_pool_caps=provider_pool_caps,
        provider_traffic_caps=provider_traffic_caps,
    )
    model.objective = pyo.Objective(
        expr=model.eta - lambda_slack * sum(model.floor_slack[d] for d in model.D),
        sense=pyo.maximize,
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
            "message": _termination_message(results),
            "diagnostics": diagnostics,
        }

    result = _extract_a3_solution(
        data,
        cascades,
        R,
        C,
        Esc,
        A_p,
        scenarios,
        model,
        policy,
        status,
        solver_name,
        diagnostics,
        lambda_slack=lambda_slack,
    )
    result.update({"K": K, "B": B, "Emax": Emax})
    return result


def solve_a3_lexicographic(
    data,
    cascades,
    R,
    C,
    Esc,
    A_p,
    scenarios,
    floors,
    K,
    B,
    Emax,
    time_limit=300,
    epsilon=1e-6,
    alpha_cost=0.01,
    metadata=None,
    storage_cap_gb=None,
    provider_pool_caps=None,
    provider_traffic_caps=None,
):
    """Solve A3 in three passes: max eta, min slack, then max quality minus cost."""
    policy = f"A3-lex K={K} B={B:.6g} Emax={Emax:g}"
    if K < 2:
        return _a3_infeasible_k_result(policy)

    passes = []
    model, meta = build_a3_model(
        data,
        cascades,
        R,
        C,
        Esc,
        A_p,
        scenarios,
        floors,
        K,
        B,
        Emax,
        metadata=metadata,
        storage_cap_gb=storage_cap_gb,
        provider_pool_caps=provider_pool_caps,
        provider_traffic_caps=provider_traffic_caps,
    )

    model.objective = pyo.Objective(expr=model.eta, sense=pyo.maximize)
    solver_name, results, diagnostics = solve_model(
        model, time_limit=time_limit, policy=f"{policy} pass=1"
    )
    if solver_name is None:
        result = no_solver_result(policy, diagnostics)
        result["lexicographic_passes"] = [
            _lex_pass_row(
                1, "max_eta", "no_solver", solver_name, diagnostics, message=result["message"]
            )
        ]
        return result
    status = result_status(results)
    if not has_solution(status):
        passes.append(
            _lex_pass_row(
                1,
                "max_eta",
                status,
                solver_name,
                diagnostics,
                message=_termination_message(results),
            )
        )
        return _lex_incomplete_result(policy, 1, status, solver_name, results, diagnostics, passes)
    eta_value = float(pyo.value(model.eta))
    passes.append(
        _lex_pass_row(
            1,
            "max_eta",
            status,
            solver_name,
            diagnostics,
            eta=eta_value,
            total_slack=_a3_total_slack(model, data["D"]),
            message=_termination_message(results),
        )
    )
    if status != "optimal":
        return _lex_incomplete_result(policy, 1, status, solver_name, results, diagnostics, passes)
    eta_star = eta_value

    model.objective.deactivate()
    model.fix_eta = pyo.Constraint(expr=model.eta >= eta_star - epsilon)
    model.min_slack_objective = pyo.Objective(
        expr=sum(model.floor_slack[d] for d in model.D), sense=pyo.minimize
    )
    solver_name, results, diagnostics = solve_model(
        model, time_limit=time_limit, policy=f"{policy} pass=2"
    )
    if solver_name is None:
        result = no_solver_result(policy, diagnostics)
        passes.append(
            _lex_pass_row(
                2, "min_slack", "no_solver", solver_name, diagnostics, message=result["message"]
            )
        )
        result["lexicographic_passes"] = passes
        return result
    status = result_status(results)
    if not has_solution(status):
        passes.append(
            _lex_pass_row(
                2,
                "min_slack",
                status,
                solver_name,
                diagnostics,
                message=_termination_message(results),
            )
        )
        return _lex_incomplete_result(policy, 2, status, solver_name, results, diagnostics, passes)
    slack_star = _a3_total_slack(model, data["D"])
    passes.append(
        _lex_pass_row(
            2,
            "min_slack",
            status,
            solver_name,
            diagnostics,
            eta=float(pyo.value(model.eta)),
            total_slack=slack_star,
            message=_termination_message(results),
        )
    )
    if status != "optimal":
        return _lex_incomplete_result(policy, 2, status, solver_name, results, diagnostics, passes)

    model.min_slack_objective.deactivate()
    model.fix_slack = pyo.Constraint(
        expr=sum(model.floor_slack[d] for d in model.D) <= slack_star + epsilon
    )
    pa = meta["pa"]
    n_prompts = meta["n_prompts"]
    model.empirical_objective = pyo.Objective(
        expr=sum((R[p, a] - alpha_cost * C[p, a]) * model.z[p, a] for p, a in pa) / n_prompts,
        sense=pyo.maximize,
    )
    solver_name, results, diagnostics = solve_model(
        model, time_limit=time_limit, policy=f"{policy} pass=3"
    )
    if solver_name is None:
        result = no_solver_result(policy, diagnostics)
        passes.append(
            _lex_pass_row(
                3,
                "max_empirical_quality_minus_cost",
                "no_solver",
                solver_name,
                diagnostics,
                message=result["message"],
            )
        )
        result["lexicographic_passes"] = passes
        return result
    status = result_status(results)
    if not has_solution(status):
        passes.append(
            _lex_pass_row(
                3,
                "max_empirical_quality_minus_cost",
                status,
                solver_name,
                diagnostics,
                message=_termination_message(results),
            )
        )
        return {
            "policy": policy,
            "status": status,
            "solver": solver_name,
            "message": _termination_message(results),
            "diagnostics": diagnostics,
            "lexicographic_passes": passes,
        }
    passes.append(
        _lex_pass_row(
            3,
            "max_empirical_quality_minus_cost",
            status,
            solver_name,
            diagnostics,
            eta=float(pyo.value(model.eta)),
            total_slack=_a3_total_slack(model, data["D"]),
            message=_termination_message(results),
        )
    )

    result = _extract_a3_solution(
        data,
        cascades,
        R,
        C,
        Esc,
        A_p,
        scenarios,
        model,
        policy,
        status,
        solver_name,
        diagnostics,
    )
    result.update({"K": K, "B": B, "Emax": Emax, "status": status, "lexicographic_passes": passes})
    return result
