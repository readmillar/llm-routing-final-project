def test_a4_cvar_returns_tail_risk_metrics(synthetic_data):
    import pytest

    from src.cascade_generation import generate_cascades
    from src.pyomo_tail_risk import solve_a4_cvar_cascade

    cascades, params = generate_cascades(synthetic_data, rho=0.75, max_cascades=40)
    floors = {domain: 0.5 for domain in synthetic_data["D"]}
    result = solve_a4_cvar_cascade(
        synthetic_data,
        cascades,
        params["R"],
        params["C"],
        params["Esc"],
        params["A_p"],
        floors,
        K=3,
        B=10.0,
        Emax=1.0,
        beta=0.9,
        lambda_cvar=0.1,
        time_limit=20,
    )
    if result["status"] == "no_solver":
        pytest.skip(result["message"])

    assert result["status"] in {"optimal", "feasible", "feasible_time_limited"}
    assert result["cvar_shortfall"] >= 0.0


def test_a4_experiment_path_is_not_gated_by_skip_a3():
    from src.experiments import needs_cascade_candidates, should_run_a4

    assert should_run_a4(skip_a3=True)
    assert needs_cascade_candidates(skip_a2=True, skip_a3=True)


def test_a4_rejects_invalid_beta_without_raising(synthetic_data):
    from src.pyomo_tail_risk import solve_a4_cvar_cascade

    result = solve_a4_cvar_cascade(
        synthetic_data,
        cascades=None,
        R={},
        C={},
        Esc={},
        A_p={},
        floors={},
        K=3,
        B=10.0,
        Emax=1.0,
        beta="bad",
        lambda_cvar=0.1,
    )

    assert result["status"] == "invalid"
    assert "beta" in result["message"]
    assert result["diagnostics"]["status"] == "invalid"


def test_a4_rejects_invalid_lambda_without_raising(synthetic_data):
    from src.cascade_generation import generate_cascades
    from src.pyomo_tail_risk import solve_a4_cvar_cascade

    cascades, params = generate_cascades(synthetic_data, rho=0.75, max_cascades=40)
    floors = {domain: 0.5 for domain in synthetic_data["D"]}

    for lambda_cvar in ["bad", 0.0, -0.1]:
        result = solve_a4_cvar_cascade(
            synthetic_data,
            cascades,
            params["R"],
            params["C"],
            params["Esc"],
            params["A_p"],
            floors,
            K=3,
            B=10.0,
            Emax=1.0,
            beta=0.9,
            lambda_cvar=lambda_cvar,
        )

        assert result["status"] == "invalid"
        assert "lambda_cvar" in result["message"]
        assert result["diagnostics"]["status"] == "invalid"


def test_a4_rejects_three_stage_cascades_until_esc3_is_supported(synthetic_data):
    from src.cascade_generation import generate_cascades
    from src.pyomo_tail_risk import solve_a4_cvar_cascade

    cascades, params = generate_cascades(
        synthetic_data, rho=0.75, max_cascades=60, include_three_stage=True
    )
    assert (cascades["depth"] == 3).any()
    floors = {domain: 0.5 for domain in synthetic_data["D"]}

    result = solve_a4_cvar_cascade(
        synthetic_data,
        cascades,
        params["R"],
        params["C"],
        params["Esc"],
        params["A_p"],
        floors,
        K=3,
        B=10.0,
        Emax=1.0,
        beta=0.9,
        lambda_cvar=0.1,
        time_limit=20,
    )

    assert result["status"] == "invalid"
    assert "depth-3" in result["message"]
    assert result["diagnostics"]["status"] == "invalid"
