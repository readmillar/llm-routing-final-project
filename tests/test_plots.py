import pandas as pd


def test_feasibility_pivot_preserves_family_rows():
    from src.plots import _build_feasibility_pivot

    df = pd.DataFrame(
        [
            {"family": "A1", "K": 2, "budget_name": "B_mid", "status": "optimal"},
            {"family": "A2", "K": 2, "budget_name": "B_mid", "status": "infeasible"},
        ]
    )

    pivot = _build_feasibility_pivot(df)

    assert list(pivot.index) == ["A1 K=2", "A2 K=2"]
    assert pivot.loc["A1 K=2", "B_mid"] == 3
    assert pivot.loc["A2 K=2", "B_mid"] == 1


def test_select_cascade_flow_result_uses_quality_cost_escalation_order():
    from src.plots import _select_cascade_flow_result

    payload = {
        "first_inserted": {
            "cascade_assignment": {"p1": "c1"},
            "avg_quality": 0.70,
            "avg_cost": 0.10,
            "escalation_rate": 0.10,
        },
        "best_quality": {
            "cascade_assignment": {"p1": "c2"},
            "avg_quality": 0.80,
            "avg_cost": 0.30,
            "escalation_rate": 0.20,
        },
        "same_quality_lower_cost": {
            "cascade_assignment": {"p1": "c3"},
            "avg_quality": 0.80,
            "avg_cost": 0.20,
            "escalation_rate": 0.40,
        },
        "same_quality_cost_lower_escalation": {
            "cascade_assignment": {"p1": "c4"},
            "avg_quality": 0.80,
            "avg_cost": 0.20,
            "escalation_rate": 0.30,
        },
    }

    result = _select_cascade_flow_result(payload)

    assert result["cascade_assignment"] == {"p1": "c4"}


def test_selected_report_policy_falls_back_to_a3_a2_a1_and_skips_baseline_a0():
    from src.plots import _select_report_policy_marker

    chosen = pd.DataFrame()
    report = pd.DataFrame(
        [
            {
                "policy": "Always best quality",
                "family": "baseline",
                "avg_quality": 1.0,
                "avg_cost": 0.5,
            },
            {"policy": "A0 alpha=1", "family": "A0", "avg_quality": 0.95, "avg_cost": 0.4},
            {"policy": "A1 K=5", "family": "A1", "avg_quality": 0.9, "avg_cost": 0.2},
            {"policy": "A2 K=5", "family": "A2", "avg_quality": 0.8, "avg_cost": 0.1},
            {"policy": "A3 K=5", "family": "A3", "avg_quality": 0.99, "avg_cost": float("nan")},
        ]
    )

    marker = _select_report_policy_marker(chosen, report)

    assert marker.iloc[0]["policy"] == "A2 K=5"


def test_usage_concentration_frame_keeps_most_concentrated_rows():
    from src.plots import _usage_concentration_plot_frame

    usage = pd.DataFrame(
        {
            "policy": [f"P{i}" for i in range(10)],
            "stage": ["single"] * 10,
            "top_1_model_share": [i / 10 for i in range(10)],
        }
    )

    plot_df = _usage_concentration_plot_frame(usage)

    assert set(plot_df["top_1_model_share"]) == {0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9}
