from pathlib import Path

import pandas as pd

from .baselines import ALPHA_GRID, run_weighted_baselines, solve_always_best_quality, solve_always_cheapest
from .load_data import load_dataset
from .metrics import domain_quality_rows, records_from_result, scenario_quality, usage_rows
from .plots import make_all_plots
from .pyomo_cascade import generate_cascades, solve_a2
from .pyomo_robust_cascade import build_scenarios, compute_domain_floors, solve_a3
from .pyomo_single_shot import solve_a1
from .solver_utils import write_json


def ensure_output_dirs(output_dir):
    root = Path(output_dir)
    for child in ["tables", "figures", "solutions"]:
        (root / child).mkdir(parents=True, exist_ok=True)
    return root


def compute_budget_grid(data, output_dir):
    cheapest = solve_always_cheapest(data)
    best = solve_always_best_quality(data)
    low_end = cheapest["avg_cost"]
    high_end = best["avg_cost"]
    if high_end < low_end:
        high_end = low_end
    span = high_end - low_end
    budgets = {
        "B_low": low_end + 0.25 * span,
        "B_mid": low_end + 0.50 * span,
        "B_high": low_end + 0.75 * span,
    }
    rows = [
        {"name": "always_cheapest_cost", "value": low_end, "source": "mean prompt min cost"},
        {"name": "always_best_quality_cost", "value": best["avg_cost"], "source": "quality oracle tie-break min cost"},
    ]
    rows.extend({"name": name, "value": value, "source": "interpolated"} for name, value in budgets.items())
    pd.DataFrame(rows).to_csv(Path(output_dir) / "tables" / "budget_grid.csv", index=False)
    return budgets, cheapest, best


def _summary_row(result, **extra):
    return records_from_result(result, extra)


def _best_result(results):
    feasible = [r for r in results if r.get("status") in {"ok", "optimal", "feasible"}]
    if not feasible:
        return results[0] if results else None
    return sorted(feasible, key=lambda r: (r.get("avg_quality", -1), -r.get("avg_cost", 1e9)), reverse=True)[0]


def _record_solution_tables(policy_results, data, scenarios, R=None, C=None, output_rows=None):
    domain_rows = []
    usage = []
    scenario_rows = []
    for result in policy_results:
        if result is None or result.get("status") not in {"ok", "optimal", "feasible"}:
            continue
        policy = result["policy"]
        domain_rows.extend(domain_quality_rows(policy, result.get("domain_quality", {})))
        if "assignment" in result:
            usage.extend(usage_rows(policy, result["model_usage"], len(data["P"]), "single"))
            for name, scenario in scenarios.items():
                weights = scenario["prompt_weights"]
                scenario_rows.append(
                    {
                        "policy": policy,
                        "scenario": name,
                        "avg_quality": scenario_quality(weights, result["assignment"], data["q"]),
                        "avg_cost": scenario_quality(weights, result["assignment"], data["c"]),
                    }
                )
        if "cascade_assignment" in result and R is not None and C is not None:
            usage.extend(usage_rows(policy, result.get("stage1_usage", {}), len(data["P"]), "stage1"))
            usage.extend(usage_rows(policy, result.get("expected_stage2_usage", {}), len(data["P"]), "expected_stage2"))
            for name, scenario in scenarios.items():
                weights = scenario["prompt_weights"]
                scenario_rows.append(
                    {
                        "policy": policy,
                        "scenario": name,
                        "avg_quality": scenario_quality(weights, result["cascade_assignment"], R),
                        "avg_cost": scenario_quality(weights, result["cascade_assignment"], C),
                    }
                )
    output_rows["domain"].extend(domain_rows)
    output_rows["usage"].extend(usage)
    output_rows["scenario"].extend(scenario_rows)


def run_experiments(data_path="data/routerbench.csv", output_dir="outputs", skip_a1=False, skip_a2=False, skip_a3=False, time_limit=60, max_cascades=250):
    root = ensure_output_dirs(output_dir)
    data = load_dataset(data_path, output_dir=root)
    budgets, cheapest, best = compute_budget_grid(data, root)
    scenarios = build_scenarios(data)
    table_rows = {"domain": [], "usage": [], "scenario": []}

    baseline_rows = [
        _summary_row(cheapest, family="baseline"),
        _summary_row(best, family="baseline"),
    ]
    a0_df, a0_results = run_weighted_baselines(data, ALPHA_GRID)
    a0_df.to_csv(root / "tables" / "a0_results.csv", index=False)
    pd.DataFrame(baseline_rows).to_csv(root / "tables" / "baseline_extremes.csv", index=False)
    _record_solution_tables([cheapest, best] + a0_results, data, scenarios, output_rows=table_rows)
    write_json(root / "solutions" / "baseline_assignments.json", {r["policy"]: r.get("assignment", {}) for r in [cheapest, best] + a0_results})

    a1_results = []
    if not skip_a1:
        for K in [1, 2, 3, 5, 8]:
            for budget_name, B in budgets.items():
                result = solve_a1(data, K=K, B=B, time_limit=time_limit)
                a1_results.append(result)
        pd.DataFrame([_summary_row(r, family="A1", K=r.get("K"), B=r.get("B")) for r in a1_results]).to_csv(root / "tables" / "a1_results.csv", index=False)
        write_json(root / "solutions" / "a1_solutions.json", {r["policy"]: r for r in a1_results})
        _record_solution_tables(a1_results, data, scenarios, output_rows=table_rows)

    cascades = pd.DataFrame()
    params = {"A_p": {}, "R": {}, "C": {}, "Esc": {}}
    a2_results = []
    if not skip_a2 or not skip_a3:
        cascades, params = generate_cascades(data, rho=0.75, max_cascades=max_cascades)
        cascades.to_csv(root / "tables" / "cascade_candidates.csv", index=False)

    if not skip_a2:
        for K in [2, 3, 5]:
            for budget_name, B in budgets.items():
                for Emax in [1.0, 0.75, 0.50]:
                    result = solve_a2(data, cascades, params["R"], params["C"], params["Esc"], params["A_p"], K=K, B=B, Emax=Emax, time_limit=time_limit)
                    result["budget_name"] = budget_name
                    a2_results.append(result)
        pd.DataFrame([_summary_row(r, family="A2", K=r.get("K"), B=r.get("B"), Emax=r.get("Emax"), budget_name=r.get("budget_name")) for r in a2_results]).to_csv(root / "tables" / "a2_results.csv", index=False)
        write_json(root / "solutions" / "a2_solutions.json", {r["policy"]: r for r in a2_results})
        _record_solution_tables(a2_results, data, scenarios, params["R"], params["C"], output_rows=table_rows)

    a3_results = []
    if not skip_a3:
        floors = compute_domain_floors(data)
        attempts = [(3, budgets["B_high"], 0.75), (3, budgets["B_high"], 1.0), (5, budgets["B_high"], 0.75), (5, budgets["B_high"], 1.0)]
        for K, B, Emax in attempts:
            result = solve_a3(data, cascades, params["R"], params["C"], params["Esc"], params["A_p"], scenarios, floors, K=K, B=B, Emax=Emax, lambda_slack=0.10, time_limit=time_limit)
            a3_results.append(result)
            if result.get("status") in {"optimal", "feasible"}:
                break
        a3_rows = []
        slack_rows = []
        scenario_metric_rows = []
        for result in a3_results:
            a3_rows.append(_summary_row(result, family="A3", K=result.get("K"), B=result.get("B"), Emax=result.get("Emax"), eta=result.get("eta")))
            for domain, value in result.get("domain_slacks", {}).items():
                slack_rows.append({"policy": result["policy"], "domain": domain, "slack": value})
            for scenario, values in result.get("scenario_metrics", {}).items():
                scenario_metric_rows.append({"policy": result["policy"], "scenario": scenario, **values})
        pd.DataFrame(a3_rows).to_csv(root / "tables" / "a3_results.csv", index=False)
        pd.DataFrame(slack_rows).to_csv(root / "tables" / "a3_domain_slacks.csv", index=False)
        pd.DataFrame(scenario_metric_rows).to_csv(root / "tables" / "a3_scenario_metrics.csv", index=False)
        write_json(root / "solutions" / "a3_solutions.json", {r["policy"]: r for r in a3_results})
        _record_solution_tables(a3_results, data, scenarios, params["R"], params["C"], output_rows=table_rows)

    representative = [
        cheapest,
        best,
        _best_result(a0_results),
        _best_result(a1_results),
        _best_result(a2_results),
        _best_result(a3_results),
    ]
    summary = [_summary_row(r, family=r["policy"].split()[0]) for r in representative if r is not None]
    pd.DataFrame(summary).to_csv(root / "tables" / "summary_comparison.csv", index=False)
    pd.DataFrame(table_rows["domain"]).to_csv(root / "tables" / "domain_quality.csv", index=False)
    pd.DataFrame(table_rows["usage"]).to_csv(root / "tables" / "selected_model_usage.csv", index=False)
    pd.DataFrame(table_rows["scenario"]).to_csv(root / "tables" / "scenario_quality.csv", index=False)

    make_all_plots(root)
    return {
        "output_dir": str(root),
        "budgets": budgets,
        "summary": summary,
        "a1_count": len(a1_results),
        "a2_count": len(a2_results),
        "a3_count": len(a3_results),
    }
