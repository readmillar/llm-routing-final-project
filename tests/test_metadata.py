import pandas as pd
import pytest


def test_metadata_inference_covers_all_models(synthetic_data):
    from src.model_metadata import build_metadata_for_models

    metadata = build_metadata_for_models(synthetic_data["M"])

    assert set(metadata["model"]) == set(synthetic_data["M"])
    assert metadata["provider_family"].notna().all()
    assert (metadata["estimated_storage_gb"] >= 0).all()


def test_provider_usage_counts_selected_models(synthetic_data):
    from src.model_metadata import build_metadata_for_models, summarize_provider_pool

    metadata = build_metadata_for_models(synthetic_data["M"])
    summary = summarize_provider_pool(["free-small", "strong"], metadata)

    assert summary["num_models_selected"] == 2
    assert summary["provider_count"] >= 1
    assert summary["storage_gb"] >= 0.0


def test_checked_in_metadata_covers_all_real_models(tmp_path):
    from src.load_data import load_dataset

    data = load_dataset("data/routerbench.csv", output_dir=tmp_path / "outputs")
    metadata = pd.read_csv("data/model_metadata.csv")

    assert not metadata["model"].duplicated().any()
    assert set(metadata["model"]) == set(data["M"])
    assert metadata["provider_family"].notna().all()
    assert metadata["estimated_storage_gb"].notna().all()
    assert pd.api.types.is_numeric_dtype(metadata["estimated_storage_gb"])
    assert (metadata["estimated_storage_gb"] >= 0).all()


def test_a2_rejects_incomplete_metadata(synthetic_data):
    from src.model_metadata import build_metadata_for_models
    from src.pyomo_cascade import generate_cascades, solve_a2

    cascades, params = generate_cascades(synthetic_data, rho=0.75, max_cascades=20)
    metadata = build_metadata_for_models(synthetic_data["M"])
    metadata = metadata[metadata["model"] != synthetic_data["M"][0]]

    with pytest.raises(ValueError, match=synthetic_data["M"][0]):
        solve_a2(
            synthetic_data,
            cascades,
            params["R"],
            params["C"],
            params["Esc"],
            params["A_p"],
            K=2,
            B=10.0,
            Emax=1.0,
            metadata=metadata,
            time_limit=20,
        )


def test_a3_rejects_incomplete_metadata(synthetic_data):
    from src.model_metadata import build_metadata_for_models
    from src.pyomo_cascade import generate_cascades
    from src.pyomo_robust_cascade import build_scenarios, compute_domain_floors, solve_a3

    cascades, params = generate_cascades(synthetic_data, rho=0.75, max_cascades=20)
    scenarios = build_scenarios(synthetic_data)
    floors = compute_domain_floors(synthetic_data, multiplier=0.75)
    metadata = build_metadata_for_models(synthetic_data["M"])
    metadata = metadata[metadata["model"] != synthetic_data["M"][0]]

    with pytest.raises(ValueError, match=synthetic_data["M"][0]):
        solve_a3(
            synthetic_data,
            cascades,
            params["R"],
            params["C"],
            params["Esc"],
            params["A_p"],
            scenarios,
            floors,
            K=2,
            B=10.0,
            Emax=1.0,
            metadata=metadata,
            time_limit=20,
        )


def test_a3_provider_traffic_caps_are_scenario_weighted(synthetic_data):
    from src.model_metadata import build_metadata_for_models
    from src.pyomo_cascade import generate_cascades
    from src.pyomo_robust_cascade import solve_a3

    for prompt in ["p1", "p2"]:
        for model in synthetic_data["M_p"][prompt]:
            value = 1.0 if model == "strong" else 0.0
            synthetic_data["q"][(prompt, model)] = value
            synthetic_data["r"][(prompt, model)] = value

    cascades, params = generate_cascades(synthetic_data, rho=0.75, max_cascades=20)
    metadata = build_metadata_for_models(synthetic_data["M"])
    metadata.loc[metadata["model"] == "strong", "provider_family"] = "Limited"
    scenarios = {"aime_heavy": {"prompt_weights": {"p1": 0.45, "p2": 0.45, "p3": 0.05, "p4": 0.05}}}
    floors = {"AIME": 1.0, "LCB": 0.0, "GPQA": 0.0}

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
        metadata=metadata,
        provider_traffic_caps={"Limited": 0.6},
        lambda_slack=100.0,
        time_limit=20,
    )

    cascade_lookup = cascades.set_index("cascade_id").to_dict("index")
    traffic = 0.0
    for prompt, cascade_id in result["cascade_assignment"].items():
        row = cascade_lookup[cascade_id]
        weight = scenarios["aime_heavy"]["prompt_weights"][prompt]
        if row["m1"] == "strong":
            traffic += weight
        if row.get("m2") == "strong":
            traffic += weight * params["Esc"][(prompt, cascade_id)]

    assert result["status"] == "optimal"
    assert traffic <= 0.6


def test_provider_summary_uses_assignment_before_selected_models():
    from src.experiments import _models_for_provider_summary
    from src.model_metadata import build_metadata_for_models, summarize_provider_pool

    result = {
        "assignment": {"p1": "free-small", "p2": "free-small"},
        "selected_models": ["free-small", "strong"],
    }
    metadata = build_metadata_for_models(["free-small", "strong"])
    metadata.loc[metadata["model"] == "strong", "estimated_storage_gb"] = 100.0

    models = _models_for_provider_summary(result)
    summary = summarize_provider_pool(models, metadata)

    assert models == ["free-small"]
    assert summary["num_models_selected"] == 1
    assert summary["storage_gb"] == 32.0
