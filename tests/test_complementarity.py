def test_pair_recovery_uses_domain_level_when_support_sufficient(synthetic_data):
    from src.complementarity import estimate_pair_recovery

    recovery = estimate_pair_recovery(synthetic_data, min_support=1, global_rho=0.75)
    row = recovery[
        (recovery["m1"] == "free-small")
        & (recovery["m2"] == "cheap-solid")
        & (recovery["domain"] == "AIME")
    ].iloc[0]

    assert row["support"] == 2
    assert row["fallback_level"] == "domain_pair"
    assert row["recovery_rate"] == 1.0


def test_pair_recovery_falls_back_to_global_rho_when_support_low(synthetic_data):
    from src.complementarity import estimate_pair_recovery

    recovery = estimate_pair_recovery(synthetic_data, min_support=999, global_rho=0.75)
    assert set(recovery["fallback_level"]) == {"global_rho"}
    assert set(recovery["recovery_rate"]) == {0.75}


def test_empirical_recovery_lookup_is_used_directly_for_cascade_parameters(synthetic_data):
    import pandas as pd
    import pytest

    from src.cascade_generation import precompute_cascade_parameters
    from src.complementarity import recovery_lookup_from_frame

    cascades = pd.DataFrame(
        [
            {
                "cascade_id": "s2::free-small::cheap-solid",
                "depth": 2,
                "m1": "free-small",
                "m2": "cheap-solid",
                "m3": "",
            }
        ]
    )
    recovery = pd.DataFrame(
        [
            {
                "m1": "free-small",
                "m2": "cheap-solid",
                "domain": "AIME",
                "support": 2,
                "recovery_rate": 0.25,
                "fallback_level": "domain_pair",
            }
        ]
    )

    params = precompute_cascade_parameters(
        synthetic_data,
        cascades,
        rho=0.75,
        recovery_lookup=recovery_lookup_from_frame(recovery),
    )
    prompt = "p1"
    cascade_id = "s2::free-small::cheap-solid"
    r1 = synthetic_data["r"][(prompt, "free-small")]

    assert params["R"][(prompt, cascade_id)] == pytest.approx(r1 + (1 - r1) * 0.25)


def test_two_stage_candidate_summary_uses_empirical_recovery_lookup(synthetic_data):
    import pandas as pd
    import pytest

    from src.cascade_generation import generate_two_stage_cascades
    from src.complementarity import recovery_lookup_from_frame

    recovery = pd.DataFrame(
        [
            {
                "m1": "free-small",
                "m2": "balanced",
                "domain": "AIME",
                "support": 2,
                "recovery_rate": 0.25,
                "fallback_level": "domain_pair",
            }
        ]
    )

    cascades = generate_two_stage_cascades(
        synthetic_data,
        rho=0.75,
        max_two_stage=20,
        recovery_lookup=recovery_lookup_from_frame(recovery),
    )
    row = cascades[
        (cascades["m1"] == "free-small") & (cascades["m2"] == "balanced")
    ].iloc[0]

    assert row["avg_R"] == pytest.approx((0.25 + 0.25 + 0.75 + 1.0) / 4)
