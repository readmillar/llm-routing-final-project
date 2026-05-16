from __future__ import annotations

import numpy as np
import pandas as pd

from .metrics import scenario_quality, scenario_weights


def empirical_domain_weights(data):
    """Return the empirical prompt share for each domain."""
    return {domain: len(data["P_d"][domain]) / len(data["P"]) for domain in data["D"]}


def build_l1_shift_scenarios(data, radius=0.4):
    """Generate finite L1-ball extreme domain shifts around empirical traffic."""
    base = empirical_domain_weights(data)
    scenarios = {}
    move = min(0.5, max(0.0, radius / 2.0))
    for target in data["D"]:
        weights = dict(base)
        donors = [domain for domain in data["D"] if domain != target]
        remaining_move = move
        for donor in sorted(donors, key=lambda d: weights[d], reverse=True):
            take = min(weights[donor], remaining_move)
            weights[donor] -= take
            weights[target] += take
            remaining_move -= take
            if remaining_move <= 1e-12:
                break
        total = sum(weights.values())
        normalized = {domain: value / total for domain, value in weights.items()}
        scenarios[f"l1_shift_to_{target}"] = {
            "domain_weights": normalized,
            "prompt_weights": scenario_weights(data["P"], data["prompt_domain"], normalized),
        }
    return scenarios


def sample_dirichlet_scenarios(data, n=500, concentration=40.0, seed=164):
    """Sample prompt-mix stress scenarios from a Dirichlet around empirical weights."""
    rng = np.random.default_rng(seed)
    domains = list(data["D"])
    base_weights = empirical_domain_weights(data)
    base = np.array([base_weights[domain] for domain in domains], dtype=float)
    alpha = np.maximum(base * concentration, 1e-6)
    draws = rng.dirichlet(alpha, size=n)
    scenarios = {}
    for idx, draw in enumerate(draws):
        weights = {domain: float(draw[i]) for i, domain in enumerate(domains)}
        scenarios[f"stress_{idx:03d}"] = {
            "domain_weights": weights,
            "prompt_weights": scenario_weights(data["P"], data["prompt_domain"], weights),
        }
    return scenarios


def evaluate_policy_under_scenarios(policy_result, scenarios, value_lookup, cost_lookup):
    """Evaluate one fixed assignment under many prompt-weight scenarios."""
    assignment = policy_result.get("cascade_assignment", policy_result.get("assignment", {}))
    rows = []
    for name, scenario in scenarios.items():
        weights = scenario["prompt_weights"]
        row = {
            "policy": policy_result["policy"],
            "scenario": name,
            "avg_quality": scenario_quality(weights, assignment, value_lookup),
            "avg_cost": scenario_quality(weights, assignment, cost_lookup),
        }
        for column in ["grid_id", "policy_label", "rho"]:
            if column in policy_result:
                row[column] = policy_result[column]
        rows.append(row)
    return pd.DataFrame(rows)
