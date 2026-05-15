import pandas as pd

from .metrics import assignment_metrics


ALPHA_GRID = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]


def solve_always_cheapest(data):
    """Choose the minimum-cost available model for each prompt."""
    assignment = {}
    for prompt in data["P"]:
        assignment[prompt] = min(
            data["M_p"][prompt],
            key=lambda model: (data["c"][(prompt, model)], -data["q"][(prompt, model)], model),
        )
    result = assignment_metrics(data, assignment, "Always cheapest")
    result["status"] = "ok"
    return result


def solve_always_best_quality(data):
    """Choose the highest-quality available model, tie-breaking by lower cost."""
    assignment = {}
    for prompt in data["P"]:
        assignment[prompt] = max(
            data["M_p"][prompt],
            key=lambda model: (data["q"][(prompt, model)], -data["c"][(prompt, model)], model),
        )
    result = assignment_metrics(data, assignment, "Always best quality")
    result["status"] = "ok"
    return result


def solve_weighted_baseline(data, alpha):
    """A0: choose argmin_m cost[p,m] - alpha * quality[p,m] over M_p."""
    assignment = {}
    for prompt in data["P"]:
        assignment[prompt] = min(
            data["M_p"][prompt],
            key=lambda model: data["c"][(prompt, model)] - alpha * data["q"][(prompt, model)],
        )
    result = assignment_metrics(data, assignment, f"A0 alpha={alpha:g}")
    result["alpha"] = alpha
    result["status"] = "ok"
    return result


def run_weighted_baselines(data, alphas=None):
    """Return A0 summary rows and full assignment results for all alphas."""
    results = [solve_weighted_baseline(data, alpha) for alpha in (alphas or ALPHA_GRID)]
    rows = []
    for result in results:
        rows.append(
            {
                "policy": result["policy"],
                "alpha": result["alpha"],
                "avg_cost": result["avg_cost"],
                "avg_quality": result["avg_quality"],
                "models_used": len(result["models_used"]),
                "selected_models": "; ".join(result["models_used"]),
            }
        )
    return pd.DataFrame(rows), results
