import pandas as pd
import pytest


@pytest.fixture
def synthetic_csv(tmp_path):
    rows = [
        ("AIME", "p1", "free-small", 0.0, 0.0),
        ("AIME", "p1", "cheap-solid", 1.0, 0.1),
        ("AIME", "p1", "balanced", 1.0, 0.5),
        ("AIME", "p1", "strong", 1.0, 2.0),
        ("AIME", "p2", "free-small", 0.0, 0.0),
        ("AIME", "p2", "cheap-solid", 1.0, 0.1),
        ("AIME", "p2", "balanced", 0.0, 0.5),
        ("AIME", "p2", "strong", 1.0, 2.0),
        ("LCB", "p3", "free-small", 0.0, 0.0),
        ("LCB", "p3", "cheap-solid", 0.0, 0.1),
        ("LCB", "p3", "balanced", 1.0, 0.5),
        ("LCB", "p3", "strong", 1.0, 2.0),
        ("GPQA", "p4", "free-small", 1.0, 0.0),
        ("GPQA", "p4", "cheap-solid", 0.0, 0.1),
        ("GPQA", "p4", "balanced", 1.0, 0.5),
        ("GPQA", "p4", "strong", 1.0, 2.0),
    ]
    df = pd.DataFrame(rows, columns=["dataset", "prompt_id", "model", "score", "cost"])
    path = tmp_path / "synthetic_routerbench.csv"
    df.to_csv(path, index=False)
    return path


@pytest.fixture
def synthetic_data(synthetic_csv, tmp_path):
    from src.load_data import load_dataset

    return load_dataset(str(synthetic_csv), output_dir=str(tmp_path / "outputs"))


def test_preprocessing_normalizes_quality_and_preserves_zero_cost(synthetic_csv, tmp_path):
    """Covers score-to-quality normalization, r=q, and zero-cost row preservation."""
    from src.load_data import load_dataset

    data = load_dataset(str(synthetic_csv), output_dir=str(tmp_path / "outputs"))

    assert min(data["q"].values()) >= 0.0
    assert max(data["q"].values()) <= 1.0
    assert data["q"] == data["r"]
    assert any(value == 0.0 for value in data["c"].values())


def test_loader_rejects_missing_required_columns(tmp_path):
    """Covers clear validation failure when required semantic columns are absent."""
    from src.load_data import load_dataset

    path = tmp_path / "bad.csv"
    pd.DataFrame({"a": [1], "b": [2]}).to_csv(path, index=False)

    with pytest.raises(ValueError, match="Available columns"):
        load_dataset(str(path), output_dir=str(tmp_path / "outputs"))


def test_locked_routerbench_missing_pairs_are_reported(tmp_path):
    """Covers the locked CSV shape and known missing MMLU-Pro model block."""
    from src.load_data import load_dataset

    data = load_dataset("data/routerbench.csv", output_dir=str(tmp_path / "outputs"))
    missing = data["missing_pairs"]

    assert len(data["df"]) == 7860
    assert len(data["P"]) == 240
    assert len(data["M"]) == 33
    assert len(data["D"]) == 4
    assert len(missing) == 60
    assert set(missing["domain"]) == {"MMLU-Pro"}
    assert set(missing["model"]) == {"deepseek-v3.1-terminus"}


def test_baselines_assign_every_prompt_once(synthetic_data):
    """Covers assignment completeness for cheapest, best-quality, and A0 policies."""
    from src.baselines import (
        solve_always_best_quality,
        solve_always_cheapest,
        solve_weighted_baseline,
    )

    results = [
        solve_always_cheapest(synthetic_data),
        solve_always_best_quality(synthetic_data),
        solve_weighted_baseline(synthetic_data, alpha=1.0),
    ]

    for result in results:
        assert set(result["assignment"]) == set(synthetic_data["P"])
        assert result["avg_cost"] >= 0.0
        assert 0.0 <= result["avg_quality"] <= 1.0


def test_scenario_weights_sum_to_one(synthetic_data):
    """Covers conversion from domain-level scenario weights to prompt weights."""
    from src.metrics import scenario_weights

    weights = scenario_weights(
        synthetic_data["P"],
        synthetic_data["prompt_domain"],
        {"AIME": 0.5, "LCB": 0.25, "GPQA": 0.25},
    )

    assert abs(sum(weights.values()) - 1.0) <= 1e-8
    assert weights["p1"] == pytest.approx(0.25)
    assert weights["p2"] == pytest.approx(0.25)


def test_a1_solution_assigns_each_prompt_and_respects_selected_pool(synthetic_data):
    """Covers A1 assignment and x[p,m] <= y[m] pool linkage."""
    from src.pyomo_single_shot import solve_a1

    result = solve_a1(synthetic_data, K=2, B=10.0, time_limit=20)
    if result["status"] == "no_solver":
        pytest.skip(result["message"])

    assert result["status"] in {"optimal", "feasible"}
    assert set(result["assignment"]) == set(synthetic_data["P"])
    assert set(result["assignment"].values()).issubset(set(result["selected_models"]))
    assert len(result["selected_models"]) <= 2


def test_cascades_have_expected_parameters(synthetic_data):
    """Covers two-stage cascade R, C, and escalation parameter generation."""
    from src.pyomo_cascade import generate_cascades

    cascades, params = generate_cascades(synthetic_data, rho=0.75, max_cascades=20)
    prompt = synthetic_data["P"][0]
    cascade_id = params["A_p"][prompt][0]

    assert len(cascades) > 0
    assert 0.0 <= params["R"][(prompt, cascade_id)] <= 1.0
    assert params["C"][(prompt, cascade_id)] >= 0.0
    assert 0.0 <= params["Esc"][(prompt, cascade_id)] <= 1.0


def test_a2_solution_uses_only_selected_models(synthetic_data):
    """Covers A2 selected cascade linkage to the selected model pool."""
    from src.pyomo_cascade import generate_cascades, solve_a2

    cascades, params = generate_cascades(synthetic_data, rho=0.75, max_cascades=20)
    result = solve_a2(
        synthetic_data,
        cascades,
        params["R"],
        params["C"],
        params["Esc"],
        params["A_p"],
        K=3,
        B=10.0,
        Emax=1.0,
        time_limit=20,
    )
    if result["status"] == "no_solver":
        pytest.skip(result["message"])

    selected = set(result["selected_models"])
    cascade_lookup = cascades.set_index("cascade_id")
    for cascade_id in result["cascade_assignment"].values():
        row = cascade_lookup.loc[cascade_id]
        assert row["m1"] in selected
        assert row["m2"] in selected


def test_a3_returns_robust_metrics(synthetic_data):
    """Covers A3 robust objective outputs, scenario metrics, and domain slacks."""
    from src.pyomo_cascade import generate_cascades
    from src.pyomo_robust_cascade import build_scenarios, compute_domain_floors, solve_a3

    cascades, params = generate_cascades(synthetic_data, rho=0.75, max_cascades=20)
    scenarios = build_scenarios(synthetic_data)
    floors = compute_domain_floors(synthetic_data, multiplier=0.75)
    result = solve_a3(
        synthetic_data,
        cascades,
        params["R"],
        params["C"],
        params["Esc"],
        params["A_p"],
        scenarios,
        floors,
        K=3,
        B=10.0,
        Emax=1.0,
        lambda_slack=0.1,
        time_limit=20,
    )
    if result["status"] == "no_solver":
        pytest.skip(result["message"])

    assert result["status"] in {"optimal", "feasible"}
    assert 0.0 <= result["eta"] <= 1.0
    assert set(result["scenario_metrics"])
    assert set(result["domain_slacks"]) == set(synthetic_data["D"])
