from collections import Counter


def assignment_metrics(data, assignment, policy):
    """Summarize a single-shot prompt-to-model assignment."""
    n = len(data["P"])
    total_q = sum(data["q"][(p, m)] for p, m in assignment.items())
    total_c = sum(data["c"][(p, m)] for p, m in assignment.items())
    domain_quality = {}
    domain_cost = {}
    for domain, prompts in data["P_d"].items():
        domain_quality[domain] = sum(data["q"][(p, assignment[p])] for p in prompts) / len(prompts)
        domain_cost[domain] = sum(data["c"][(p, assignment[p])] for p in prompts) / len(prompts)
    return {
        "policy": policy,
        "assignment": dict(assignment),
        "avg_quality": total_q / n,
        "avg_cost": total_c / n,
        "domain_quality": domain_quality,
        "domain_cost": domain_cost,
        "model_usage": dict(Counter(assignment.values())),
        "models_used": sorted(set(assignment.values())),
    }


def cascade_assignment_metrics(data, cascades, assignment, r_param, c_param, esc_param, policy):
    """Summarize a prompt-to-cascade assignment using precomputed parameters."""
    n = len(data["P"])
    cascade_lookup = cascades.set_index("cascade_id")
    total_q = sum(r_param[(p, a)] for p, a in assignment.items())
    total_c = sum(c_param[(p, a)] for p, a in assignment.items())
    total_esc = sum(esc_param[(p, a)] for p, a in assignment.items())
    domain_quality = {}
    domain_cost = {}
    for domain, prompts in data["P_d"].items():
        domain_quality[domain] = sum(r_param[(p, assignment[p])] for p in prompts) / len(prompts)
        domain_cost[domain] = sum(c_param[(p, assignment[p])] for p in prompts) / len(prompts)
    usage = Counter()
    expected_second = Counter()
    for prompt, cascade_id in assignment.items():
        row = cascade_lookup.loc[cascade_id]
        usage[row["m1"]] += 1.0
        expected_second[row["m2"]] += esc_param[(prompt, cascade_id)]
    return {
        "policy": policy,
        "cascade_assignment": dict(assignment),
        "avg_quality": total_q / n,
        "avg_cost": total_c / n,
        "escalation_rate": total_esc / n,
        "domain_quality": domain_quality,
        "domain_cost": domain_cost,
        "stage1_usage": dict(usage),
        "expected_stage2_usage": dict(expected_second),
    }


def scenario_weights(prompts, prompt_domain, domain_weights):
    """Convert domain weights to prompt weights that sum to one."""
    by_domain = {}
    for prompt in prompts:
        by_domain.setdefault(prompt_domain[prompt], []).append(prompt)
    weights = {}
    for domain, domain_prompts in by_domain.items():
        share = float(domain_weights.get(domain, 0.0))
        if domain_prompts:
            for prompt in domain_prompts:
                weights[prompt] = share / len(domain_prompts)
    total = sum(weights.values())
    if total <= 0:
        raise ValueError("Scenario weights sum to zero")
    return {prompt: value / total for prompt, value in weights.items()}


def scenario_quality(weights, assignment, value_lookup):
    """Evaluate a single-shot or cascade assignment under prompt weights."""
    return sum(weights[p] * value_lookup[(p, assignment[p])] for p in weights)


def domain_quality_rows(policy, domain_quality):
    """Flatten a domain-quality dict into table rows."""
    return [
        {"policy": policy, "domain": domain, "avg_quality": quality}
        for domain, quality in sorted(domain_quality.items())
    ]


def usage_rows(policy, usage, total, stage="single"):
    """Flatten model usage counts into table rows."""
    rows = []
    for model, count in sorted(usage.items(), key=lambda item: (-item[1], item[0])):
        rows.append(
            {
                "policy": policy,
                "stage": stage,
                "model": model,
                "usage_count": float(count),
                "usage_share": float(count) / total if total else 0.0,
            }
        )
    return rows


def records_from_result(result, extra=None):
    """Build a one-row CSV-safe summary from a solver or baseline result."""
    extra = extra or {}
    row = {
        "policy": result.get("policy", ""),
        "status": result.get("status", "ok"),
        "avg_cost": result.get("avg_cost"),
        "avg_quality": result.get("avg_quality"),
        "escalation_rate": result.get("escalation_rate"),
        "selected_models": "; ".join(result.get("selected_models", result.get("models_used", []))),
    }
    row.update(extra)
    return row
