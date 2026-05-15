import pytest


def test_degenerate_cascades_cover_every_observed_prompt_model_pair(synthetic_data):
    from src.cascade_generation import generate_single_stage_cascades, precompute_cascade_parameters

    cascades = generate_single_stage_cascades(synthetic_data)
    params = precompute_cascade_parameters(synthetic_data, cascades, rho=0.75)

    single_pairs = {(row.m1, row.depth) for row in cascades.itertuples(index=False)}
    assert all((model, 1) in single_pairs for model in synthetic_data["M"])
    for prompt in synthetic_data["P"]:
        assert set(params["A_p"][prompt])
        available_single_models = {
            cascades.set_index("cascade_id").loc[cascade_id, "m1"]
            for cascade_id in params["A_p"][prompt]
            if cascades.set_index("cascade_id").loc[cascade_id, "depth"] == 1
        }
        assert available_single_models == set(synthetic_data["M_p"][prompt])


def test_single_stage_parameters_match_original_pair_values(synthetic_data):
    from src.cascade_generation import generate_single_stage_cascades, precompute_cascade_parameters

    cascades = generate_single_stage_cascades(synthetic_data)
    params = precompute_cascade_parameters(synthetic_data, cascades, rho=0.75)
    lookup = cascades.set_index("cascade_id")

    for prompt in synthetic_data["P"]:
        for cascade_id in params["A_p"][prompt]:
            row = lookup.loc[cascade_id]
            if row["depth"] == 1:
                model = row["m1"]
                assert params["R"][(prompt, cascade_id)] == synthetic_data["q"][(prompt, model)]
                assert params["C"][(prompt, cascade_id)] == synthetic_data["c"][(prompt, model)]
                assert params["Esc"][(prompt, cascade_id)] == 0.0


def test_two_stage_parameters_match_manual_formula(synthetic_data):
    from src.cascade_generation import generate_two_stage_cascades, precompute_cascade_parameters

    cascades = generate_two_stage_cascades(synthetic_data, rho=0.75, max_two_stage=20)
    params = precompute_cascade_parameters(synthetic_data, cascades, rho=0.75)
    row = cascades[cascades["depth"] == 2].iloc[0]
    cascade_id = row["cascade_id"]
    if (synthetic_data["P"][0], cascade_id) not in params["R"]:
        pytest.skip("First generated cascade is not feasible for the first synthetic prompt.")

    p = synthetic_data["P"][0]
    r1 = synthetic_data["r"][(p, row["m1"])]
    r2 = synthetic_data["r"][(p, row["m2"])]
    assert params["R"][(p, cascade_id)] == pytest.approx(r1 + (1 - r1) * 0.75 * r2)
    assert params["C"][(p, cascade_id)] == pytest.approx(
        synthetic_data["c"][(p, row["m1"])] + (1 - r1) * synthetic_data["c"][(p, row["m2"])]
    )
    assert params["Esc"][(p, cascade_id)] == pytest.approx(1 - r1)


def test_a2_can_use_single_stage_cascades_when_escalation_cap_is_zero(synthetic_data):
    from src.pyomo_cascade import generate_cascades, solve_a2

    cascades, params = generate_cascades(synthetic_data, rho=0.75, max_cascades=20)
    result = solve_a2(
        synthetic_data,
        cascades,
        params["R"],
        params["C"],
        params["Esc"],
        params["A_p"],
        K=2,
        B=10.0,
        Emax=0.0,
        time_limit=20,
    )
    if result["status"] == "no_solver":
        pytest.skip(result["message"])

    assert result["status"] in {"optimal", "feasible"}
    cascade_lookup = cascades.set_index("cascade_id")
    assert all(
        cascade_lookup.loc[cascade_id, "depth"] == 1
        for cascade_id in result["cascade_assignment"].values()
    )


def test_a2_generalizes_a1_on_synthetic_data(synthetic_data):
    from src.cascade_generation import generate_cascades
    from src.pyomo_cascade import solve_a2
    from src.pyomo_single_shot import solve_a1

    a1 = solve_a1(synthetic_data, K=2, B=10.0, time_limit=20)
    if a1["status"] == "no_solver":
        pytest.skip(a1["message"])

    cascades, params = generate_cascades(synthetic_data, rho=0.75, max_cascades=40)
    a2 = solve_a2(
        synthetic_data,
        cascades,
        params["R"],
        params["C"],
        params["Esc"],
        params["A_p"],
        K=2,
        B=10.0,
        Emax=1.0,
        time_limit=20,
    )

    assert a2["status"] in {"optimal", "feasible", "feasible_time_limited"}
    assert a2["avg_quality"] + 1e-8 >= a1["avg_quality"]


def test_three_stage_parameters_are_bounded_when_enabled(synthetic_data):
    from src.cascade_generation import generate_cascades

    cascades, params = generate_cascades(
        synthetic_data, rho=0.75, max_cascades=60, include_three_stage=True
    )
    three_stage_ids = set(cascades.loc[cascades["depth"] == 3, "cascade_id"])
    if not three_stage_ids:
        pytest.skip(
            "Synthetic data does not produce a three-stage candidate under the configured filter."
        )

    for key, value in params["R"].items():
        if key[1] in three_stage_ids:
            assert 0.0 <= value <= 1.0
            assert params["C"][key] >= 0.0
            assert params["Esc"][key] >= 0.0


def test_three_stage_opt_in_does_not_reduce_two_stage_count(synthetic_data):
    from src.cascade_generation import generate_cascades

    default_cascades, _ = generate_cascades(synthetic_data, rho=0.75, max_cascades=6)
    opt_in_cascades, _ = generate_cascades(
        synthetic_data, rho=0.75, max_cascades=6, include_three_stage=True
    )

    default_two_stage = int((default_cascades["depth"] == 2).sum())
    opt_in_two_stage = int((opt_in_cascades["depth"] == 2).sum())
    assert opt_in_two_stage >= default_two_stage


def test_three_stage_candidate_summaries_are_computed(synthetic_data):
    from src.cascade_generation import generate_three_stage_cascades

    cascades = generate_three_stage_cascades(synthetic_data, rho=0.75, max_three_stage=20)
    if cascades.empty:
        pytest.skip(
            "Synthetic data does not produce a three-stage candidate under the configured filter."
        )

    assert (cascades["avg_R"] >= 0.0).all()
    assert (cascades["avg_C"] >= 0.0).all()
    assert (cascades[["avg_R", "avg_C", "avg_Esc"]].sum(axis=1) > 0.0).any()


def test_cascade_metrics_use_esc3_for_expected_third_stage_usage(synthetic_data):
    from src.cascade_generation import generate_cascades
    from src.metrics import cascade_assignment_metrics

    cascades, params = generate_cascades(
        synthetic_data, rho=0.75, max_cascades=60, include_three_stage=True
    )
    three_stage = cascades[cascades["depth"] == 3]
    if three_stage.empty:
        pytest.skip(
            "Synthetic data does not produce a three-stage candidate under the configured filter."
        )

    cascade_id = three_stage.iloc[0]["cascade_id"]
    assignment = {prompt: cascade_id for prompt in synthetic_data["P"]}
    result = cascade_assignment_metrics(
        synthetic_data,
        cascades,
        assignment,
        params["R"],
        params["C"],
        params["Esc"],
        "three-stage-test",
        esc3_param=params["Esc3"],
    )

    m3 = three_stage.iloc[0]["m3"]
    expected_usage = sum(params["Esc3"][(prompt, cascade_id)] for prompt in synthetic_data["P"])
    assert result["expected_stage3_usage"][m3] == pytest.approx(expected_usage)
