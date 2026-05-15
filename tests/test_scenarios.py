import pytest


def test_l1_shift_scenarios_sum_to_one(synthetic_data):
    from src.stress_testing import build_l1_shift_scenarios

    scenarios = build_l1_shift_scenarios(synthetic_data, radius=0.4)

    assert scenarios
    for scenario in scenarios.values():
        assert abs(sum(scenario["domain_weights"].values()) - 1.0) <= 1e-8
        assert abs(sum(scenario["prompt_weights"].values()) - 1.0) <= 1e-8


def test_dirichlet_stress_scenarios_are_reproducible(synthetic_data):
    from src.stress_testing import sample_dirichlet_scenarios

    first = sample_dirichlet_scenarios(synthetic_data, n=5, concentration=10.0, seed=164)
    second = sample_dirichlet_scenarios(synthetic_data, n=5, concentration=10.0, seed=164)

    assert list(first) == list(second)
    assert first["stress_000"]["domain_weights"] == second["stress_000"]["domain_weights"]


def test_evaluate_policy_under_scenarios_returns_expected_single_shot_values():
    from src.stress_testing import evaluate_policy_under_scenarios

    scenarios = {
        "mix": {
            "prompt_weights": {
                "p1": 0.25,
                "p2": 0.75,
            }
        }
    }
    result = {
        "policy": "A1 test",
        "assignment": {
            "p1": "m1",
            "p2": "m2",
        },
    }
    quality = {
        ("p1", "m1"): 1.0,
        ("p2", "m2"): 0.4,
    }
    cost = {
        ("p1", "m1"): 0.2,
        ("p2", "m2"): 0.6,
    }

    rows = evaluate_policy_under_scenarios(result, scenarios, quality, cost)

    row = rows.to_dict("records")[0]
    assert row["policy"] == "A1 test"
    assert row["scenario"] == "mix"
    assert row["avg_quality"] == pytest.approx(0.55)
    assert row["avg_cost"] == pytest.approx(0.5)


def test_evaluate_policy_under_scenarios_preserves_cascade_metadata():
    from src.stress_testing import evaluate_policy_under_scenarios

    scenarios = {"mix": {"prompt_weights": {"p1": 1.0}}}
    result = {
        "policy": "A3 test",
        "grid_id": "grid-1",
        "rho": 0.5,
        "cascade_assignment": {"p1": "c1"},
    }
    quality = {("p1", "c1"): 0.8}
    cost = {("p1", "c1"): 0.3}

    rows = evaluate_policy_under_scenarios(result, scenarios, quality, cost)

    assert rows.loc[0, "policy"] == "A3 test"
    assert rows.loc[0, "grid_id"] == "grid-1"
    assert rows.loc[0, "rho"] == 0.5


def test_read_csv_returns_empty_dataframe_for_no_column_csv(tmp_path):
    from src.plots import _read_csv

    path = tmp_path / "empty.csv"
    path.write_text("\n")

    assert _read_csv(path).empty
