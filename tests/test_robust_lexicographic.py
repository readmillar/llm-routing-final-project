from types import SimpleNamespace


def _set_fake_a3_solution(model):
    for prompt in model.P:
        selected = False
        for p, cascade_id in model.PA:
            if p == prompt and not selected:
                model.z[p, cascade_id].set_value(1)
                selected = True
            elif p == prompt:
                model.z[p, cascade_id].set_value(0)
    for model_name in model.M:
        model.y[model_name].set_value(1)
    model.eta.set_value(0.5)
    for domain in model.D:
        model.floor_slack[domain].set_value(0.0)


def test_a3_lexicographic_returns_three_monotone_passes(synthetic_data):
    import pytest

    from src.cascade_generation import generate_cascades
    from src.pyomo_robust_cascade import (
        build_scenarios,
        compute_domain_floors,
        solve_a3_lexicographic,
    )

    cascades, params = generate_cascades(synthetic_data, rho=0.75, max_cascades=40)
    result = solve_a3_lexicographic(
        synthetic_data,
        cascades,
        params["R"],
        params["C"],
        params["Esc"],
        params["A_p"],
        build_scenarios(synthetic_data),
        compute_domain_floors(synthetic_data, multiplier=0.75),
        K=3,
        B=10.0,
        Emax=1.0,
        time_limit=20,
    )
    if result["status"] == "no_solver":
        pytest.skip(result["message"])

    passes = result["lexicographic_passes"]
    assert [row["pass"] for row in passes] == [1, 2, 3]
    assert passes[1]["eta"] + 1e-6 >= passes[0]["eta"] - 1e-6
    assert passes[2]["eta"] + 1e-6 >= passes[0]["eta"] - 1e-6
    assert passes[2]["total_slack"] <= passes[1]["total_slack"] + 1e-6


def test_run_experiments_writes_empty_lexicographic_table_when_a3_skipped(
    synthetic_csv, tmp_path, monkeypatch
):
    import pandas as pd

    from src import experiments
    from src.model_metadata import load_or_create_metadata

    metadata_path = tmp_path / "model_metadata.csv"
    monkeypatch.setattr(
        experiments,
        "load_or_create_metadata",
        lambda models, path: load_or_create_metadata(models, path=metadata_path),
    )
    monkeypatch.setattr(experiments, "make_all_plots", lambda root: None)

    output_dir = tmp_path / "outputs"
    experiments.run_experiments(
        data_path=synthetic_csv,
        output_dir=output_dir,
        skip_a1=True,
        skip_a2=True,
        skip_a3=True,
        time_limit=1,
        max_cascades=5,
    )

    table_path = output_dir / "tables" / "a3_lexicographic_passes.csv"
    assert table_path.exists()
    table = pd.read_csv(table_path)
    assert {
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
    }.issubset(table.columns)
    assert table.empty


def test_a3_lexicographic_stops_when_pass1_is_not_optimal(synthetic_data, monkeypatch):
    from src import pyomo_robust_cascade as robust
    from src.cascade_generation import generate_cascades

    calls = []

    def fake_solve_model(model, time_limit, policy):
        calls.append(policy)
        _set_fake_a3_solution(model)
        diagnostics = {"policy": policy, "mip_gap": 0.25, "termination_condition": "maxTimeLimit"}
        results = SimpleNamespace(solver=SimpleNamespace(termination_condition="maxTimeLimit"))
        return "fake", results, diagnostics

    monkeypatch.setattr(robust, "solve_model", fake_solve_model)
    monkeypatch.setattr(robust, "result_status", lambda results: "feasible_time_limited")

    cascades, params = generate_cascades(synthetic_data, rho=0.75, max_cascades=40)
    result = robust.solve_a3_lexicographic(
        synthetic_data,
        cascades,
        params["R"],
        params["C"],
        params["Esc"],
        params["A_p"],
        robust.build_scenarios(synthetic_data),
        robust.compute_domain_floors(synthetic_data, multiplier=0.75),
        K=3,
        B=10.0,
        Emax=1.0,
        time_limit=20,
    )

    assert len(calls) == 1
    assert result["status"] == "lexicographic_incomplete"
    assert result["failed_pass"] == 1
    assert [row["pass"] for row in result["lexicographic_passes"]] == [1]
    assert result["lexicographic_passes"][0]["status"] == "feasible_time_limited"


def test_a3_lexicographic_stops_when_pass2_is_not_optimal(synthetic_data, monkeypatch):
    from src import pyomo_robust_cascade as robust
    from src.cascade_generation import generate_cascades

    statuses = iter(["optimal", "feasible_time_limited"])
    calls = []

    def fake_solve_model(model, time_limit, policy):
        calls.append(policy)
        _set_fake_a3_solution(model)
        diagnostics = {"policy": policy, "mip_gap": 0.50, "termination_condition": "maxTimeLimit"}
        results = SimpleNamespace(solver=SimpleNamespace(termination_condition="maxTimeLimit"))
        return "fake", results, diagnostics

    monkeypatch.setattr(robust, "solve_model", fake_solve_model)
    monkeypatch.setattr(robust, "result_status", lambda results: next(statuses))

    cascades, params = generate_cascades(synthetic_data, rho=0.75, max_cascades=40)
    result = robust.solve_a3_lexicographic(
        synthetic_data,
        cascades,
        params["R"],
        params["C"],
        params["Esc"],
        params["A_p"],
        robust.build_scenarios(synthetic_data),
        robust.compute_domain_floors(synthetic_data, multiplier=0.75),
        K=3,
        B=10.0,
        Emax=1.0,
        time_limit=20,
    )

    assert len(calls) == 2
    assert result["status"] == "lexicographic_incomplete"
    assert result["failed_pass"] == 2
    assert [row["pass"] for row in result["lexicographic_passes"]] == [1, 2]
    assert result["lexicographic_passes"][0]["status"] == "optimal"
    assert result["lexicographic_passes"][1]["status"] == "feasible_time_limited"
