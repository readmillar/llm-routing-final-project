from pathlib import Path

import pandas as pd

from .audit import audit_cascade_result, audit_single_shot_result
from .baselines import (
    ALPHA_GRID,
    run_weighted_baselines,
    solve_always_best_quality,
    solve_always_cheapest,
)
from .complementarity import estimate_pair_recovery, recovery_lookup_from_frame
from .load_data import load_dataset
from .metrics import (
    domain_quality_rows,
    records_from_result,
    scenario_quality,
    usage_concentration_rows,
    usage_rows,
)
from .model_metadata import load_or_create_metadata, summarize_provider_pool
from .pareto import pareto_frontier
from .plots import make_all_plots
from .pyomo_cascade import generate_cascades, solve_a2
from .pyomo_robust_cascade import (
    build_scenarios,
    compute_domain_floors,
    solve_a3,
    solve_a3_lexicographic,
)
from .pyomo_single_shot import solve_a1
from .report_artifacts import write_report_numbers, write_report_tables
from .solver_utils import write_json
from .stress_testing import evaluate_policy_under_scenarios, sample_dirichlet_scenarios

STATUS_RANK = {
    "optimal": 4,
    "feasible": 3,
    "feasible_time_limited": 2,
    "ok": 1,
}

SUCCESS_STATUSES = {"ok", "optimal", "feasible", "feasible_time_limited"}

REPORT_COMPARISON_COLUMNS = [
    "policy",
    "family",
    "K",
    "budget_name",
    "Emax",
    "avg_cost",
    "avg_quality",
    "worst_scenario_quality",
    "p05_stress_quality",
    "worst_domain_quality",
    "total_slack",
    "escalation_rate",
    "num_models_selected",
    "provider_count",
    "storage_gb",
    "status",
    "mip_gap",
    "wall_time_sec",
]


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
        {
            "name": "always_best_quality_cost",
            "value": best["avg_cost"],
            "source": "quality oracle tie-break min cost",
        },
    ]
    rows.extend(
        {"name": name, "value": value, "source": "interpolated"} for name, value in budgets.items()
    )
    pd.DataFrame(rows).to_csv(Path(output_dir) / "tables" / "budget_grid.csv", index=False)
    return budgets, cheapest, best


def _summary_row(result, **extra):
    return records_from_result(result, extra)


def _successful(df):
    """Return solved rows with numeric cost and quality for comparisons."""
    if df.empty:
        return df.copy()
    status = df.get("status", pd.Series(index=df.index, dtype=object)).fillna("ok")
    avg_cost = df.get("avg_cost", pd.Series(index=df.index, dtype=object))
    avg_quality = df.get("avg_quality", pd.Series(index=df.index, dtype=object))
    mask = status.isin(SUCCESS_STATUSES)
    mask &= pd.to_numeric(avg_cost, errors="coerce").notna()
    mask &= pd.to_numeric(avg_quality, errors="coerce").notna()
    return df.loc[mask].copy()


def _is_successful_result(result):
    """Return True for reportable results with solved status and metrics."""
    return (
        isinstance(result, dict)
        and result.get("status", "ok") in SUCCESS_STATUSES
        and pd.notna(pd.to_numeric(result.get("avg_cost"), errors="coerce"))
        and pd.notna(pd.to_numeric(result.get("avg_quality"), errors="coerce"))
    )


def _total_slack(result):
    return float(sum((result.get("domain_slacks") or {}).values()))


def _safe_number(value, default):
    """Return a finite-sortable numeric value for nullable report fields."""
    if value is None or pd.isna(value):
        return default
    return float(value)


def _format_grid_value(value):
    return f"{float(value):g}"


def make_a3_grid_id(K, budget_name, Emax, floor_multiplier, lambda_slack, rho):
    """Build a stable unique identifier for an A3 grid point."""
    return (
        f"K={K}|budget={budget_name}|Emax={_format_grid_value(Emax)}|"
        f"floor={_format_grid_value(floor_multiplier)}|"
        f"lambda={_format_grid_value(lambda_slack)}|rho={_format_grid_value(rho)}"
    )


def select_report_a3_policy(results):
    """Select the report A3 policy using a documented lexicographic rule."""
    feasible = [r for r in results if r.get("status") in STATUS_RANK]
    if not feasible:
        return None
    return sorted(
        feasible,
        key=lambda r: (
            STATUS_RANK.get(r.get("status"), 0),
            _safe_number(r.get("eta"), 0.0),
            -_safe_number(r.get("total_slack", _total_slack(r)), 1e18),
            _safe_number(r.get("avg_quality"), 0.0),
            -_safe_number(r.get("avg_cost"), 1e18),
            -_safe_number(r.get("escalation_rate"), 1e18),
        ),
        reverse=True,
    )[0]


def _float_equals(left, right):
    """Compare nullable numeric settings without treating missing values as equal to numbers."""
    if pd.isna(left) or pd.isna(right):
        return pd.isna(left) and pd.isna(right)
    return float(left) == float(right)


def build_report_main_comparison(rows, K=5, budget_name="B_mid", Emax=0.75):
    """Build apples-to-apples policy rows for the report comparison table."""
    df = _successful(pd.DataFrame(rows))
    if df.empty:
        return pd.DataFrame(columns=REPORT_COMPARISON_COLUMNS)
    keep = []
    for row in df.to_dict("records"):
        family = row.get("family")
        if family in {"baseline", "A0"}:
            keep.append(row)
        elif family == "A1" and row.get("K") == K and row.get("budget_name") == budget_name:
            keep.append(row)
        elif (
            family in {"A2", "A3", "A4"}
            and row.get("K") == K
            and row.get("budget_name") == budget_name
            and _float_equals(row.get("Emax"), Emax)
        ):
            keep.append(row)
    out = pd.DataFrame(keep)
    for column in REPORT_COMPARISON_COLUMNS:
        if column not in out.columns:
            out[column] = None
    return out[REPORT_COMPARISON_COLUMNS]


def _best_result(results):
    feasible = [r for r in results if r.get("status") in {"ok", "optimal", "feasible"}]
    if not feasible:
        return results[0] if results else None
    return sorted(
        feasible, key=lambda r: (r.get("avg_quality", -1), -r.get("avg_cost", 1e9)), reverse=True
    )[0]


def _best_successful_result(results):
    feasible = [r for r in results if _is_successful_result(r)]
    if not feasible:
        return None
    return sorted(
        feasible, key=lambda r: (r.get("avg_quality", -1), -r.get("avg_cost", 1e9)), reverse=True
    )[0]


def select_report_policy(a3_results, a2_results, a1_results):
    """Choose a report headline policy, returning None when no result is reportable."""
    a3_successful = [result for result in a3_results if _is_successful_result(result)]
    return (
        select_report_a3_policy(a3_successful)
        or _best_successful_result(a2_results)
        or _best_successful_result(a1_results)
    )


def _nonempty_models(models):
    """Return sorted non-empty model names."""
    return sorted({model for model in models if isinstance(model, str) and model})


def _models_for_provider_summary(result):
    """Prefer actually assigned or used models over loose selected pool variables."""
    if result.get("assignment"):
        return _nonempty_models(result["assignment"].values())
    if result.get("cascade_assignment"):
        models = []
        for key in ["stage1_usage", "expected_stage2_usage", "expected_stage3_usage"]:
            models.extend(
                model
                for model, count in (result.get(key) or {}).items()
                if float(count or 0.0) > 0.0
            )
        if models:
            return _nonempty_models(models)
    return _nonempty_models(result.get("selected_models", result.get("models_used", [])))


def _has_assignment_result(result, key):
    """Return True when a solver result contains a complete assignment payload."""
    return (
        isinstance(result, dict)
        and result.get("status") in {"ok", "optimal", "feasible"}
        and key in result
    )


def _record_solution_tables(policy_results, data, scenarios, R=None, C=None, output_rows=None):
    domain_rows = []
    usage = []
    usage_concentration = []
    scenario_rows = []
    for result in policy_results:
        if result is None or result.get("status") not in {"ok", "optimal", "feasible"}:
            continue
        policy_label = result["policy"]
        grid_id = result.get("grid_id")
        family = policy_label.split()[0] if policy_label else ""
        policy = f"{family}|{grid_id}" if grid_id and family else grid_id or policy_label
        for row in domain_quality_rows(policy, result.get("domain_quality", {})):
            row["policy_label"] = policy_label
            domain_rows.append(row)
        if "assignment" in result:
            for row in usage_rows(policy, result["model_usage"], len(data["P"]), "single"):
                row["policy_label"] = policy_label
                usage.append(row)
            row = usage_concentration_rows(policy, result["model_usage"], "single")
            row["policy_label"] = policy_label
            usage_concentration.append(row)
            for name, scenario in scenarios.items():
                weights = scenario["prompt_weights"]
                scenario_rows.append(
                    {
                        "policy": policy,
                        "policy_label": policy_label,
                        "scenario": name,
                        "avg_quality": scenario_quality(weights, result["assignment"], data["q"]),
                        "avg_cost": scenario_quality(weights, result["assignment"], data["c"]),
                    }
                )
        if "cascade_assignment" in result and R is not None and C is not None:
            for row in usage_rows(policy, result.get("stage1_usage", {}), len(data["P"]), "stage1"):
                row["policy_label"] = policy_label
                usage.append(row)
            row = usage_concentration_rows(policy, result.get("stage1_usage", {}), "stage1")
            row["policy_label"] = policy_label
            usage_concentration.append(row)
            for row in usage_rows(
                policy,
                result.get("expected_stage2_usage", {}),
                len(data["P"]),
                "expected_stage2",
            ):
                row["policy_label"] = policy_label
                usage.append(row)
            row = usage_concentration_rows(
                policy, result.get("expected_stage2_usage", {}), "expected_stage2"
            )
            row["policy_label"] = policy_label
            usage_concentration.append(row)
            for name, scenario in scenarios.items():
                weights = scenario["prompt_weights"]
                scenario_rows.append(
                    {
                        "policy": policy,
                        "policy_label": policy_label,
                        "scenario": name,
                        "avg_quality": scenario_quality(weights, result["cascade_assignment"], R),
                        "avg_cost": scenario_quality(weights, result["cascade_assignment"], C),
                    }
                )
    output_rows["domain"].extend(domain_rows)
    output_rows["usage"].extend(usage)
    if "usage_concentration" in output_rows:
        output_rows["usage_concentration"].extend(usage_concentration)
    output_rows["scenario"].extend(scenario_rows)


def run_experiments(
    data_path="data/routerbench.csv",
    output_dir="outputs",
    skip_a1=False,
    skip_a2=False,
    skip_a3=False,
    time_limit=60,
    max_cascades=250,
):
    root = ensure_output_dirs(output_dir)
    data = load_dataset(data_path, output_dir=root)
    metadata = load_or_create_metadata(data["M"], path="data/model_metadata.csv")
    metadata.to_csv(root / "tables" / "model_metadata.csv", index=False)
    recovery_df = estimate_pair_recovery(data, min_support=5, global_rho=0.75)
    recovery_df.to_csv(root / "tables" / "model_pair_recovery.csv", index=False)
    recovery_lookup = recovery_lookup_from_frame(recovery_df)
    budgets, cheapest, best = compute_budget_grid(data, root)
    scenarios = build_scenarios(data)
    table_rows = {"domain": [], "usage": [], "usage_concentration": [], "scenario": []}
    audit_rows = []

    baseline_rows = [
        _summary_row(cheapest, family="baseline"),
        _summary_row(best, family="baseline"),
    ]
    a0_df, a0_results = run_weighted_baselines(data, ALPHA_GRID)
    a0_df.to_csv(root / "tables" / "a0_results.csv", index=False)
    pd.DataFrame(baseline_rows).to_csv(root / "tables" / "baseline_extremes.csv", index=False)
    _record_solution_tables([cheapest, best] + a0_results, data, scenarios, output_rows=table_rows)
    for result in [cheapest, best] + a0_results:
        if _has_assignment_result(result, "assignment"):
            audit_rows.extend(audit_single_shot_result(data, result))
    write_json(
        root / "solutions" / "baseline_assignments.json",
        {r["policy"]: r.get("assignment", {}) for r in [cheapest, best] + a0_results},
    )

    a1_results = []
    if not skip_a1:
        for K in [1, 2, 3, 5, 8]:
            for budget_name, B in budgets.items():
                result = solve_a1(data, K=K, B=B, time_limit=time_limit)
                result["budget_name"] = budget_name
                a1_results.append(result)
                if _has_assignment_result(result, "assignment"):
                    audit_rows.extend(
                        audit_single_shot_result(data, result, K=result.get("K"), B=result.get("B"))
                    )
        pd.DataFrame(
            [
                _summary_row(
                    r,
                    family="A1",
                    K=r.get("K"),
                    B=r.get("B"),
                    budget_name=r.get("budget_name"),
                )
                for r in a1_results
            ]
        ).to_csv(root / "tables" / "a1_results.csv", index=False)
        write_json(root / "solutions" / "a1_solutions.json", {r["policy"]: r for r in a1_results})
        _record_solution_tables(a1_results, data, scenarios, output_rows=table_rows)

    cascades = pd.DataFrame()
    params = {"A_p": {}, "R": {}, "C": {}, "Esc": {}}
    a2_results = []
    if not skip_a2 or not skip_a3:
        cascades, params = generate_cascades(
            data, rho=0.75, max_cascades=max_cascades, recovery_lookup=recovery_lookup
        )
        cascades.to_csv(root / "tables" / "cascade_candidates.csv", index=False)

    if not skip_a2:
        for K in [2, 3, 5]:
            for budget_name, B in budgets.items():
                for Emax in [1.0, 0.75, 0.50]:
                    result = solve_a2(
                        data,
                        cascades,
                        params["R"],
                        params["C"],
                        params["Esc"],
                        params["A_p"],
                        K=K,
                        B=B,
                        Emax=Emax,
                        time_limit=time_limit,
                    )
                    result["budget_name"] = budget_name
                    a2_results.append(result)
                    if _has_assignment_result(result, "cascade_assignment"):
                        audit_rows.extend(
                            audit_cascade_result(
                                data,
                                cascades,
                                params,
                                result,
                                K=result.get("K"),
                                B=result.get("B"),
                                Emax=result.get("Emax"),
                            )
                        )
        pd.DataFrame(
            [
                _summary_row(
                    r,
                    family="A2",
                    K=r.get("K"),
                    B=r.get("B"),
                    Emax=r.get("Emax"),
                    budget_name=r.get("budget_name"),
                )
                for r in a2_results
            ]
        ).to_csv(root / "tables" / "a2_results.csv", index=False)
        write_json(root / "solutions" / "a2_solutions.json", {r["policy"]: r for r in a2_results})
        _record_solution_tables(
            a2_results, data, scenarios, params["R"], params["C"], output_rows=table_rows
        )

    a3_results = []
    lex_columns = [
        "pass",
        "objective",
        "status",
        "solver",
        "eta",
        "total_slack",
        "grid_id",
        "K",
        "B",
        "Emax",
        "budget_name",
        "rho",
        "floor_multiplier",
        "message",
        "mip_gap",
        "termination_condition",
        "wall_time_sec",
        "best_bound",
        "objective_value",
    ]
    lex_passes = []
    if not skip_a3:
        a3_grid = []
        rho_cascades = {}
        for K in [3, 5, 8]:
            for budget_name, B in budgets.items():
                for Emax in [0.50, 0.75, 1.00]:
                    for floor_multiplier in [0.75, 0.80, 0.85, 0.90]:
                        for lambda_slack in [0.01, 0.05, 0.10, 0.25, 0.50]:
                            for rho in [0.50, 0.75, 1.00]:
                                a3_grid.append(
                                    (K, budget_name, B, Emax, floor_multiplier, lambda_slack, rho)
                                )
        for K, budget_name, B, Emax, floor_multiplier, lambda_slack, rho in a3_grid:
            if rho not in rho_cascades:
                rho_cascades[rho] = generate_cascades(
                    data,
                    rho=rho,
                    max_cascades=max_cascades,
                    recovery_lookup=recovery_lookup,
                )
            cascades_rho, params_rho = rho_cascades[rho]
            floors = compute_domain_floors(data, multiplier=floor_multiplier)
            grid_id = make_a3_grid_id(
                K=K,
                budget_name=budget_name,
                Emax=Emax,
                floor_multiplier=floor_multiplier,
                lambda_slack=lambda_slack,
                rho=rho,
            )
            result = solve_a3(
                data,
                cascades_rho,
                params_rho["R"],
                params_rho["C"],
                params_rho["Esc"],
                params_rho["A_p"],
                scenarios,
                floors,
                K=K,
                B=B,
                Emax=Emax,
                lambda_slack=lambda_slack,
                time_limit=time_limit,
            )
            result.update(
                {
                    "K": K,
                    "B": B,
                    "Emax": Emax,
                    "budget_name": budget_name,
                    "floor_multiplier": floor_multiplier,
                    "lambda_slack": lambda_slack,
                    "rho": rho,
                    "total_slack": _total_slack(result),
                    "grid_id": grid_id,
                }
            )
            a3_results.append(result)
            if _has_assignment_result(result, "cascade_assignment"):
                audit_rows.extend(
                    audit_cascade_result(
                        data,
                        cascades_rho,
                        params_rho,
                        result,
                        K=result.get("K"),
                        B=result.get("B"),
                        Emax=result.get("Emax"),
                    )
                )
            result_rows = {"domain": [], "usage": [], "usage_concentration": [], "scenario": []}
            _record_solution_tables(
                [result],
                data,
                scenarios,
                params_rho["R"],
                params_rho["C"],
                output_rows=result_rows,
            )
            for key in result_rows:
                for row in result_rows[key]:
                    row["grid_id"] = grid_id
                table_rows[key].extend(result_rows[key])
        a3_grid_rows = [
            _summary_row(
                r,
                family="A3",
                grid_id=r.get("grid_id"),
                K=r.get("K"),
                B=r.get("B"),
                Emax=r.get("Emax"),
                budget_name=r.get("budget_name"),
                eta=r.get("eta"),
                total_slack=r.get("total_slack"),
                floor_multiplier=r.get("floor_multiplier"),
                lambda_slack=r.get("lambda_slack"),
                rho=r.get("rho"),
            )
            for r in a3_results
        ]
        pd.DataFrame(a3_grid_rows).to_csv(root / "tables" / "a3_grid_results.csv", index=False)
        best_report = select_report_a3_policy(a3_results)
        best_report_extra = {
            "family": "A3",
            "grid_id": None,
            "K": None,
            "B": None,
            "Emax": None,
            "budget_name": None,
            "eta": None,
            "total_slack": None,
        }
        if best_report is not None:
            pd.DataFrame(
                [
                    _summary_row(
                        best_report,
                        family="A3",
                        grid_id=best_report.get("grid_id"),
                        K=best_report.get("K"),
                        B=best_report.get("B"),
                        Emax=best_report.get("Emax"),
                        budget_name=best_report.get("budget_name"),
                        eta=best_report.get("eta"),
                        total_slack=best_report.get("total_slack"),
                    )
                ]
            ).to_csv(root / "tables" / "a3_best_report_policy.csv", index=False)
        else:
            pd.DataFrame(columns=list(_summary_row({}, **best_report_extra).keys())).to_csv(
                root / "tables" / "a3_best_report_policy.csv", index=False
            )
        if best_report is not None:
            rho = best_report.get("rho", 0.75)
            if rho not in rho_cascades:
                rho_cascades[rho] = generate_cascades(
                    data,
                    rho=rho,
                    max_cascades=max_cascades,
                    recovery_lookup=recovery_lookup,
                )
            cascades_rho, params_rho = rho_cascades[rho]
            lex_result = solve_a3_lexicographic(
                data,
                cascades_rho,
                params_rho["R"],
                params_rho["C"],
                params_rho["Esc"],
                params_rho["A_p"],
                scenarios,
                compute_domain_floors(data, multiplier=best_report.get("floor_multiplier", 0.75)),
                K=best_report["K"],
                B=best_report["B"],
                Emax=best_report["Emax"],
                time_limit=time_limit,
            )
            lex_passes = lex_result.get("lexicographic_passes", [])
            for row in lex_passes:
                row.update(
                    {
                        "grid_id": best_report.get("grid_id"),
                        "K": best_report.get("K"),
                        "B": best_report.get("B"),
                        "Emax": best_report.get("Emax"),
                        "budget_name": best_report.get("budget_name"),
                        "rho": best_report.get("rho"),
                        "floor_multiplier": best_report.get("floor_multiplier"),
                    }
                )
        a3_rows = []
        slack_rows = []
        scenario_metric_rows = []
        for result in a3_results:
            a3_rows.append(
                _summary_row(
                    result,
                    family="A3",
                    grid_id=result.get("grid_id"),
                    K=result.get("K"),
                    B=result.get("B"),
                    Emax=result.get("Emax"),
                    budget_name=result.get("budget_name"),
                    eta=result.get("eta"),
                    total_slack=result.get("total_slack"),
                    floor_multiplier=result.get("floor_multiplier"),
                    lambda_slack=result.get("lambda_slack"),
                    rho=result.get("rho"),
                )
            )
            for domain, value in (result.get("domain_slacks") or {}).items():
                slack_rows.append(
                    {
                        "grid_id": result["grid_id"],
                        "policy": result["policy"],
                        "domain": domain,
                        "slack": value,
                    }
                )
            for scenario, values in (result.get("scenario_metrics") or {}).items():
                scenario_metric_rows.append(
                    {
                        "grid_id": result["grid_id"],
                        "policy": result["policy"],
                        "scenario": scenario,
                        **values,
                    }
                )
        pd.DataFrame(a3_rows).to_csv(root / "tables" / "a3_results.csv", index=False)
        pd.DataFrame(slack_rows).to_csv(root / "tables" / "a3_domain_slacks.csv", index=False)
        pd.DataFrame(scenario_metric_rows).to_csv(
            root / "tables" / "a3_scenario_metrics.csv", index=False
        )
        write_json(root / "solutions" / "a3_solutions.json", {r["grid_id"]: r for r in a3_results})

    pd.DataFrame(lex_passes, columns=lex_columns).to_csv(
        root / "tables" / "a3_lexicographic_passes.csv", index=False
    )

    representative = [
        cheapest,
        best,
        _best_result(a0_results),
        _best_result(a1_results),
        _best_result(a2_results),
        select_report_a3_policy(a3_results),
    ]
    provider_rows = []
    for result in representative:
        if result is None:
            continue
        selected = _models_for_provider_summary(result)
        row = {"policy": result["policy"], **summarize_provider_pool(selected, metadata)}
        provider_rows.append(row)
    provider_df = pd.DataFrame(provider_rows)
    provider_df.to_csv(root / "tables" / "provider_usage.csv", index=False)
    provider_df[["policy", "storage_gb"]].to_csv(root / "tables" / "storage_usage.csv", index=False)
    stress_scenarios = sample_dirichlet_scenarios(data, n=500, concentration=40.0, seed=164)
    stress_rows = []
    stress_cascade_cache = {}
    for result in representative:
        if result is None or result.get("status") not in {
            "ok",
            "optimal",
            "feasible",
            "feasible_time_limited",
        }:
            continue
        if "cascade_assignment" in result:
            rho = result.get("rho", 0.75)
            if rho not in stress_cascade_cache:
                stress_cascade_cache[rho] = generate_cascades(
                    data,
                    rho=rho,
                    max_cascades=max_cascades,
                    recovery_lookup=recovery_lookup,
                )[1]
            params_rho = stress_cascade_cache[rho]
            stress_rows.extend(
                evaluate_policy_under_scenarios(
                    result, stress_scenarios, params_rho["R"], params_rho["C"]
                ).to_dict("records")
            )
        elif "assignment" in result:
            stress_rows.extend(
                evaluate_policy_under_scenarios(
                    result, stress_scenarios, data["q"], data["c"]
                ).to_dict("records")
            )
    pd.DataFrame(stress_rows).to_csv(root / "tables" / "stress_test_results.csv", index=False)
    summary_rows = [
        _summary_row(r, family="baseline") for r in [cheapest, best]
    ] + [_summary_row(r, family="A0", alpha=r.get("alpha")) for r in a0_results]
    summary_rows.extend(
        _summary_row(
            r,
            family="A1",
            K=r.get("K"),
            B=r.get("B"),
            budget_name=r.get("budget_name"),
        )
        for r in a1_results
    )
    summary_rows.extend(
        _summary_row(
            r,
            family="A2",
            K=r.get("K"),
            B=r.get("B"),
            Emax=r.get("Emax"),
            budget_name=r.get("budget_name"),
        )
        for r in a2_results
    )
    summary_rows.extend(
        _summary_row(
            r,
            family="A3",
            grid_id=r.get("grid_id"),
            K=r.get("K"),
            B=r.get("B"),
            Emax=r.get("Emax"),
            budget_name=r.get("budget_name"),
            eta=r.get("eta"),
            total_slack=r.get("total_slack"),
            floor_multiplier=r.get("floor_multiplier"),
            lambda_slack=r.get("lambda_slack"),
            rho=r.get("rho"),
        )
        for r in a3_results
    )
    all_summary = pd.DataFrame(summary_rows)
    pareto_frontier(_successful(all_summary)).to_csv(
        root / "tables" / "pareto_frontier.csv", index=False
    )
    report_main = build_report_main_comparison(summary_rows, K=5, budget_name="B_mid", Emax=0.75)
    report_main.to_csv(root / "tables" / "report_main_comparison.csv", index=False)
    chosen = select_report_policy(a3_results, a2_results, a1_results)
    if chosen is not None:
        write_report_numbers(root, chosen)
    write_report_tables(root, report_main, pd.DataFrame(table_rows["domain"]))
    summary = [
        _summary_row(r, family=r["policy"].split()[0]) for r in representative if r is not None
    ]
    pd.DataFrame(summary).to_csv(root / "tables" / "summary_comparison.csv", index=False)
    pd.DataFrame(table_rows["domain"]).to_csv(root / "tables" / "domain_quality.csv", index=False)
    pd.DataFrame(table_rows["usage"]).to_csv(
        root / "tables" / "selected_model_usage.csv", index=False
    )
    pd.DataFrame(table_rows["usage_concentration"]).to_csv(
        root / "tables" / "usage_concentration.csv", index=False
    )
    pd.DataFrame(table_rows["scenario"]).to_csv(
        root / "tables" / "scenario_quality.csv", index=False
    )
    pd.DataFrame(audit_rows).to_csv(root / "tables" / "solution_audit.csv", index=False)

    make_all_plots(root)
    return {
        "output_dir": str(root),
        "budgets": budgets,
        "summary": summary,
        "a1_count": len(a1_results),
        "a2_count": len(a2_results),
        "a3_count": len(a3_results),
    }
