def test_a3_selection_prefers_status_eta_slack_quality_cost_escalation():
    from src.experiments import select_report_a3_policy

    rows = [
        {
            "policy": "feasible_high_eta",
            "status": "feasible",
            "eta": 0.91,
            "total_slack": 0.02,
            "avg_quality": 0.90,
            "avg_cost": 0.50,
            "escalation_rate": 0.50,
        },
        {
            "policy": "optimal_lower_eta",
            "status": "optimal",
            "eta": 0.89,
            "total_slack": 0.00,
            "avg_quality": 0.88,
            "avg_cost": 0.40,
            "escalation_rate": 0.30,
        },
        {
            "policy": "optimal_best",
            "status": "optimal",
            "eta": 0.91,
            "total_slack": 0.01,
            "avg_quality": 0.89,
            "avg_cost": 0.60,
            "escalation_rate": 0.20,
        },
    ]

    chosen = select_report_a3_policy(rows)

    assert chosen["policy"] == "optimal_best"


def test_a3_selection_prefers_lower_slack_when_status_eta_tie():
    from src.experiments import select_report_a3_policy

    rows = [
        {
            "policy": "higher_slack",
            "status": "optimal",
            "eta": 0.91,
            "total_slack": 0.02,
            "avg_quality": 0.90,
            "avg_cost": 0.50,
            "escalation_rate": 0.20,
        },
        {
            "policy": "lower_slack",
            "status": "optimal",
            "eta": 0.91,
            "total_slack": 0.01,
            "avg_quality": 0.80,
            "avg_cost": 0.90,
            "escalation_rate": 0.90,
        },
    ]

    chosen = select_report_a3_policy(rows)

    assert chosen["policy"] == "lower_slack"


def test_a3_selection_prefers_higher_quality_after_slack_tie():
    from src.experiments import select_report_a3_policy

    rows = [
        {
            "policy": "lower_quality",
            "status": "optimal",
            "eta": 0.91,
            "total_slack": 0.01,
            "avg_quality": 0.88,
            "avg_cost": 0.20,
            "escalation_rate": 0.10,
        },
        {
            "policy": "higher_quality",
            "status": "optimal",
            "eta": 0.91,
            "total_slack": 0.01,
            "avg_quality": 0.89,
            "avg_cost": 0.90,
            "escalation_rate": 0.90,
        },
    ]

    chosen = select_report_a3_policy(rows)

    assert chosen["policy"] == "higher_quality"


def test_a3_selection_prefers_lower_cost_after_quality_tie():
    from src.experiments import select_report_a3_policy

    rows = [
        {
            "policy": "higher_cost",
            "status": "optimal",
            "eta": 0.91,
            "total_slack": 0.01,
            "avg_quality": 0.89,
            "avg_cost": 0.60,
            "escalation_rate": 0.10,
        },
        {
            "policy": "lower_cost",
            "status": "optimal",
            "eta": 0.91,
            "total_slack": 0.01,
            "avg_quality": 0.89,
            "avg_cost": 0.50,
            "escalation_rate": 0.90,
        },
    ]

    chosen = select_report_a3_policy(rows)

    assert chosen["policy"] == "lower_cost"


def test_a3_selection_prefers_lower_escalation_after_cost_tie():
    from src.experiments import select_report_a3_policy

    rows = [
        {
            "policy": "higher_escalation",
            "status": "optimal",
            "eta": 0.91,
            "total_slack": 0.01,
            "avg_quality": 0.89,
            "avg_cost": 0.50,
            "escalation_rate": 0.30,
        },
        {
            "policy": "lower_escalation",
            "status": "optimal",
            "eta": 0.91,
            "total_slack": 0.01,
            "avg_quality": 0.89,
            "avg_cost": 0.50,
            "escalation_rate": 0.20,
        },
    ]

    chosen = select_report_a3_policy(rows)

    assert chosen["policy"] == "lower_escalation"


def test_a3_selection_handles_nullable_numeric_fields():
    from src.experiments import select_report_a3_policy

    rows = [
        {
            "policy": "nullable_values",
            "status": "optimal",
            "eta": None,
            "total_slack": None,
            "avg_quality": None,
            "avg_cost": None,
            "escalation_rate": None,
        },
        {
            "policy": "numeric_values",
            "status": "optimal",
            "eta": 0.10,
            "total_slack": 0.0,
            "avg_quality": 0.10,
            "avg_cost": 0.10,
            "escalation_rate": 0.10,
        },
    ]

    chosen = select_report_a3_policy(rows)

    assert chosen["policy"] == "numeric_values"


def test_a3_grid_id_identifies_full_grid_point():
    from src.experiments import make_a3_grid_id

    first = make_a3_grid_id(
        K=5,
        budget_name="B_mid",
        Emax=0.75,
        floor_multiplier=0.90,
        lambda_slack=0.10,
        rho=0.75,
    )
    second = make_a3_grid_id(
        K=5,
        budget_name="B_mid",
        Emax=0.75,
        floor_multiplier=0.90,
        lambda_slack=0.10,
        rho=1.00,
    )

    assert first == "K=5|budget=B_mid|Emax=0.75|floor=0.9|lambda=0.1|rho=0.75"
    assert second != first


def test_empty_matched_report_table_has_expected_schema():
    from src.experiments import build_report_main_comparison

    table = build_report_main_comparison([])

    assert list(table.columns) == [
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
    assert table.empty


def test_record_solution_tables_uses_grid_id_as_policy_and_preserves_label():
    from src.experiments import _record_solution_tables

    data = {"P": ["p1"], "q": {}, "c": {}}
    scenarios = {"empirical": {"prompt_weights": {"p1": 1.0}}}
    rows = {"domain": [], "usage": [], "scenario": []}
    results = [
        {
            "policy": "A3 K=5 B=0.1 Emax=0.75",
            "grid_id": "grid-1",
            "status": "optimal",
            "domain_quality": {"D": 0.8},
            "cascade_assignment": {"p1": "cascade-a"},
            "stage1_usage": {"m1": 1.0},
            "expected_stage2_usage": {"m2": 0.2},
        },
        {
            "policy": "A3 K=5 B=0.1 Emax=0.75",
            "grid_id": "grid-2",
            "status": "optimal",
            "domain_quality": {"D": 0.9},
            "cascade_assignment": {"p1": "cascade-b"},
            "stage1_usage": {"m3": 1.0},
            "expected_stage2_usage": {"m4": 0.4},
        },
    ]

    _record_solution_tables(
        results,
        data,
        scenarios,
        R={("p1", "cascade-a"): 0.8, ("p1", "cascade-b"): 0.9},
        C={("p1", "cascade-a"): 0.1, ("p1", "cascade-b"): 0.2},
        output_rows=rows,
    )

    for key in ["domain", "usage", "scenario"]:
        emitted_policies = {row["policy"] for row in rows[key]}
        emitted_labels = {row["policy_label"] for row in rows[key]}
        assert len(emitted_policies) == 2
        assert all(policy.startswith("A3") for policy in emitted_policies)
        assert emitted_labels == {"A3 K=5 B=0.1 Emax=0.75"}


def test_matched_report_table_contains_same_k_budget_and_emax():
    from src.experiments import build_report_main_comparison

    rows = [
        {
            "policy": "A1",
            "family": "A1",
            "K": 5,
            "budget_name": "B_mid",
            "Emax": None,
            "status": "optimal",
            "avg_quality": 0.8,
            "avg_cost": 0.2,
        },
        {
            "policy": "A2",
            "family": "A2",
            "K": 5,
            "budget_name": "B_mid",
            "Emax": 0.75,
            "status": "optimal",
            "avg_quality": 0.9,
            "avg_cost": 0.3,
        },
        {
            "policy": "A3",
            "family": "A3",
            "K": 5,
            "budget_name": "B_mid",
            "Emax": 0.75,
            "status": "optimal",
            "avg_quality": 0.85,
            "avg_cost": 0.25,
        },
        {
            "policy": "wrong",
            "family": "A2",
            "K": 3,
            "budget_name": "B_low",
            "Emax": 0.5,
            "status": "optimal",
            "avg_quality": 0.99,
            "avg_cost": 0.9,
        },
    ]

    table = build_report_main_comparison(rows, K=5, budget_name="B_mid", Emax=0.75)

    assert set(table["policy"]) == {"A1", "A2", "A3"}
    assert set(table["K"].dropna()) == {5}
    assert set(table["budget_name"].dropna()) == {"B_mid"}


def test_matched_report_table_excludes_infeasible_metricless_rows():
    from src.experiments import build_report_main_comparison

    rows = [
        {
            "policy": "A2 feasible",
            "family": "A2",
            "K": 5,
            "budget_name": "B_mid",
            "Emax": 0.75,
            "status": "optimal",
            "avg_quality": 0.9,
            "avg_cost": 0.3,
        },
        {
            "policy": "A2 no solution",
            "family": "A2",
            "K": 5,
            "budget_name": "B_mid",
            "Emax": 0.75,
            "status": "no_solution",
            "avg_quality": None,
            "avg_cost": None,
        },
    ]

    table = build_report_main_comparison(rows, K=5, budget_name="B_mid", Emax=0.75)

    assert set(table["policy"]) == {"A2 feasible"}


def test_report_policy_selection_returns_none_without_successful_fallback():
    from src.experiments import select_report_policy

    chosen = select_report_policy(
        a3_results=[],
        a2_results=[{"policy": "A2 no solution", "status": "no_solution"}],
        a1_results=[{"policy": "A1 infeasible", "status": "infeasible"}],
    )

    assert chosen is None


def test_pareto_filter_removes_dominated_points():
    import pandas as pd

    from src.pareto import pareto_frontier

    df = pd.DataFrame(
        [
            {"policy": "cheap_good", "avg_cost": 1.0, "avg_quality": 0.8},
            {"policy": "expensive_bad", "avg_cost": 2.0, "avg_quality": 0.7},
            {"policy": "expensive_best", "avg_cost": 3.0, "avg_quality": 0.9},
        ]
    )

    frontier = pareto_frontier(df)

    assert set(frontier["policy"]) == {"cheap_good", "expensive_best"}


def test_report_numbers_markdown_contains_chosen_policy(tmp_path):
    from src.report_artifacts import write_report_numbers

    path = write_report_numbers(
        tmp_path,
        {
            "policy": "A3 robust cascade",
            "avg_quality": 0.9,
            "avg_cost": 0.2,
            "eta": 0.85,
            "escalation_rate": 0.4,
            "selected_models": ["m1", "m2"],
        },
    )

    text = path.read_text()
    assert "Chosen policy: A3 robust cascade" in text
    assert "Average quality: 0.9000" in text
