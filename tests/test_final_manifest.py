import hashlib
import inspect
import sys

import pandas as pd
import pytest


FOUR_DOMAIN_TEST_CONFIG = """
random_seed: 164
profile: test_core
time_limit: 20
max_cascades: 30
base_rho: 0.75
features:
  empirical_recovery: false
  stress_tests: false
  a4_cvar: false
  three_stage: false
  lexicographic_a3: false
  provider_storage_constraints: false
a1:
  K: [1, 2]
  budget_names: ["B_mid"]
a2:
  K: [2]
  budget_names: ["B_mid"]
  Emax: [1.0]
a3:
  K: [2]
  budget_names: ["B_mid"]
  Emax: [1.0]
  floor_multiplier: [0.75]
  lambda_slack: [0.1]
  rho: [0.75]
matched_report:
  K: 2
  budget_name: "B_mid"
  Emax: 1.0
  floor_multiplier: 0.75
  lambda_slack: 0.1
  rho: 0.75
stress:
  dirichlet_samples: 0
  concentration: 40.0
production_constraints:
  storage_cap_gb: null
  provider_pool_caps: {}
  provider_traffic_caps: {}
"""


def write_four_domain_routerbench_fixture(path):
    """Write a tiny RouterBench-like CSV with all core domains and zero-cost rows."""
    rows = []
    row_id = 0
    domains = ["AIME", "GPQA", "LCB", "MMLU-Pro"]
    models = ["free", "cheap", "balanced", "strong"]
    costs = {"free": 0.0, "cheap": 0.0001, "balanced": 0.0002, "strong": 0.0004}
    for domain_idx, domain in enumerate(domains):
        for prompt_idx in range(2):
            prompt_id = f"{domain}-{prompt_idx}"
            for model_idx, model in enumerate(models):
                score = int(model == "strong" or (domain_idx + prompt_idx + model_idx) % 3 == 0)
                rows.append(
                    {
                        "row_id": row_id,
                        "dataset": domain,
                        "prompt_id": prompt_id,
                        "index": prompt_idx,
                        "model": model,
                        "score": score,
                        "cost": costs[model],
                        "prompt_tokens": 100 + prompt_idx,
                        "completion_tokens": 20 + model_idx,
                    }
                )
                row_id += 1
    pd.DataFrame(rows).to_csv(path, index=False)


def test_load_config_reads_yaml_and_empty_path_returns_empty(tmp_path):
    from src.experiments import load_config

    assert load_config(None) == {}

    config_path = tmp_path / "final.yaml"
    config_path.write_text("random_seed: 271\nnested:\n  enabled: true\n", encoding="utf-8")

    assert load_config(config_path) == {"random_seed": 271, "nested": {"enabled": True}}


def test_file_sha256_hashes_file_contents(tmp_path):
    from src.experiments import file_sha256

    path = tmp_path / "data.csv"
    path.write_text("row_id,score\n1,1\n", encoding="utf-8")

    assert file_sha256(path) == hashlib.sha256(path.read_bytes()).hexdigest()


def test_write_table_with_aliases_preserves_canonical_and_grid_aliases(tmp_path):
    import pandas as pd

    from src.experiments import write_table_with_aliases

    frame = pd.DataFrame([{"policy": "x"}])

    a1_path = tmp_path / "a1_results.csv"
    write_table_with_aliases(frame, a1_path)
    assert a1_path.exists()
    assert (tmp_path / "a1_grid_results.csv").exists()
    pd.testing.assert_frame_equal(
        pd.read_csv(a1_path), pd.read_csv(tmp_path / "a1_grid_results.csv")
    )

    a2_path = tmp_path / "a2_results.csv"
    write_table_with_aliases(frame, a2_path)
    assert a2_path.exists()
    assert (tmp_path / "a2_grid_results.csv").exists()
    pd.testing.assert_frame_equal(
        pd.read_csv(a2_path), pd.read_csv(tmp_path / "a2_grid_results.csv")
    )


def test_collect_solver_diagnostics_keeps_stable_metadata():
    from src.experiments import collect_solver_diagnostics

    rows = collect_solver_diagnostics(
        [
            (
                "A1",
                [
                    {
                        "policy": "A1 K=2",
                        "status": "optimal",
                        "K": 2,
                        "B": 0.5,
                        "budget_name": "B_mid",
                        "diagnostics": {
                            "policy": "A1 K=2",
                            "solver": "highs",
                            "termination_condition": "optimal",
                            "wall_time_sec": 1.25,
                        },
                    },
                    {"policy": "A1 skipped", "status": "invalid"},
                ],
            ),
            (
                "A3",
                [
                    {
                        "policy": "A3 robust",
                        "status": "feasible_time_limited",
                        "grid_id": "grid-1",
                        "K": 5,
                        "B": 0.8,
                        "Emax": 0.75,
                        "budget_name": "B_high",
                        "rho": 0.75,
                        "floor_multiplier": 0.9,
                        "lambda_slack": 0.1,
                        "diagnostics": {
                            "policy": "A3 robust",
                            "solver": "highs",
                            "termination_condition": "maxTimeLimit",
                            "mip_gap": 0.2,
                        },
                    }
                ],
            ),
        ]
    )

    assert rows[0]["family"] == "A1"
    assert rows[0]["status"] == "optimal"
    assert rows[0]["solver"] == "highs"
    assert rows[0]["K"] == 2
    assert rows[1]["family"] == "A3"
    assert rows[1]["status"] == "feasible_time_limited"
    assert rows[1]["grid_id"] == "grid-1"
    assert rows[1]["mip_gap"] == 0.2


def test_default_configs_disable_extensions():
    from src.experiments import DEFAULT_FEATURES, config_feature, merged_experiment_config

    for config_path in ["config/final.yaml", "config/smoke.yaml"]:
        config = merged_experiment_config(config_path)
        for feature in DEFAULT_FEATURES:
            assert config_feature(config, feature) is False


def test_final_config_a3_grid_is_core_sized():
    from src.experiments import build_a3_grid, merged_experiment_config

    budgets = {"B_low": 0.01, "B_mid": 0.02, "B_high": 0.03}

    assert build_a3_grid(merged_experiment_config("config/final.yaml"), budgets) == [
        (5, "B_mid", 0.02, 0.75, 0.85, 0.1, 0.75)
    ]


def test_smoke_config_a3_grid_is_tiny():
    from src.experiments import build_a3_grid, merged_experiment_config

    budgets = {"B_low": 0.01, "B_mid": 0.02, "B_high": 0.03}

    assert build_a3_grid(merged_experiment_config("config/smoke.yaml"), budgets) == [
        (2, "B_mid", 0.02, 1.0, 0.75, 0.1, 0.75)
    ]


def test_grid_builders_require_explicit_config_values():
    from src.experiments import build_a1_grid

    with pytest.raises(ValueError, match="a1.K"):
        build_a1_grid({"a1": {"budget_names": ["B_mid"]}}, {"B_mid": 0.02})


def test_a4_and_cascade_helpers_are_config_driven():
    from src.experiments import needs_cascade_candidates, should_run_a4

    assert should_run_a4({"features": {"a4_cvar": False}}) is False
    assert should_run_a4({"features": {"a4_cvar": True}}) is True

    assert needs_cascade_candidates(skip_a2=True, skip_a3=True) is False
    assert needs_cascade_candidates(skip_a2=True, skip_a3=True, run_a4=False) is False
    assert needs_cascade_candidates(skip_a2=True, skip_a3=True, run_a4=True) is True


def test_configured_solver_limits_prefer_cli_then_config_then_defaults():
    from src.experiments import configured_max_cascades, configured_time_limit

    config = {"time_limit": 20, "max_cascades": 40}

    assert configured_time_limit(config, 5) == 5.0
    assert configured_time_limit(config, None) == 20.0
    assert configured_time_limit({}, None) == 60.0

    assert configured_max_cascades(config, 10) == 10
    assert configured_max_cascades(config, None) == 40
    assert configured_max_cascades({}, None) == 250


def test_run_experiments_solver_limit_defaults_defer_to_config():
    from src.experiments import run_experiments

    signature = inspect.signature(run_experiments)

    assert signature.parameters["time_limit"].default is None
    assert signature.parameters["max_cascades"].default is None


def test_cli_solver_limit_defaults_defer_to_config(monkeypatch):
    from run_experiments import parse_args

    monkeypatch.setattr(sys, "argv", ["run_experiments.py"])

    args = parse_args()

    assert args.time_limit is None
    assert args.max_cascades is None


def test_configured_matched_report_prefers_config_then_defaults():
    from src.experiments import configured_matched_report

    assert configured_matched_report({"matched_report": {"K": 2, "budget_name": "B_low"}}) == {
        "K": 2,
        "budget_name": "B_low",
        "Emax": 0.75,
    }
    assert configured_matched_report({}) == {"K": 5, "budget_name": "B_mid", "Emax": 0.75}


def test_select_report_a3_policy_ignores_incomplete_time_limited_rows():
    from src.experiments import select_report_a3_policy

    incomplete = {
        "policy": "A3 incomplete",
        "status": "feasible_time_limited",
        "grid_id": "bad",
        "eta": None,
        "avg_quality": None,
        "avg_cost": None,
    }
    complete = {
        "policy": "A3 complete",
        "status": "feasible",
        "grid_id": "good",
        "eta": 0.7,
        "avg_quality": 0.8,
        "avg_cost": 0.1,
    }

    assert select_report_a3_policy([incomplete, complete]) == complete
    assert select_report_a3_policy([incomplete]) is None


def test_report_main_comparison_includes_robust_summary_metrics():
    from src.experiments import build_report_main_comparison

    rows = [
        {
            "policy": "A3 core",
            "family": "A3",
            "K": 5,
            "budget_name": "B_mid",
            "Emax": 0.75,
            "status": "feasible",
            "avg_cost": 0.1,
            "avg_quality": 0.8,
            "eta": 0.7,
            "total_slack": 0.0,
            "domain_quality": {"AIME": 0.9, "GPQA": 0.75},
            "scenario_metrics": {
                "empirical": {"avg_quality": 0.8},
                "math_heavy": {"avg_quality": 0.7},
            },
        }
    ]

    report = build_report_main_comparison(rows, K=5, budget_name="B_mid", Emax=0.75)

    assert "eta" in report.columns
    row = report.iloc[0]
    assert row["eta"] == 0.7
    assert row["total_slack"] == 0.0
    assert row["worst_domain_quality"] == 0.75
    assert row["worst_scenario_quality"] == 0.7


def test_run_experiments_core_config_writes_four_domain_manifest_artifacts(
    tmp_path, monkeypatch
):
    from src import experiments
    from src.model_metadata import load_or_create_metadata

    data_path = tmp_path / "routerbench_tiny.csv"
    config_path = tmp_path / "core.yaml"
    output_dir = tmp_path / "outputs"
    metadata_path = tmp_path / "model_metadata.csv"

    write_four_domain_routerbench_fixture(data_path)
    config_path.write_text(FOUR_DOMAIN_TEST_CONFIG, encoding="utf-8")
    monkeypatch.setattr(
        experiments,
        "load_or_create_metadata",
        lambda models, path: load_or_create_metadata(models, path=metadata_path),
    )
    monkeypatch.setattr(experiments, "make_all_plots", lambda root: None)

    result = experiments.run_experiments(
        data_path=data_path,
        output_dir=output_dir,
        config_path=config_path,
    )

    assert result["a3_count"] == 1
    assert (output_dir / "manifest.json").exists()
    assert (output_dir / "RUN_LOG.md").exists()

    required_tables = [
        "a0_results.csv",
        "a1_results.csv",
        "a1_grid_results.csv",
        "a2_results.csv",
        "a2_grid_results.csv",
        "a3_results.csv",
        "a3_grid_results.csv",
        "a3_domain_slacks.csv",
        "a3_scenario_metrics.csv",
        "model_pair_recovery.csv",
        "report_main_comparison.csv",
        "solver_diagnostics.csv",
        "stress_test_results.csv",
        "a4_cvar_results.csv",
    ]
    for table_name in required_tables:
        assert (output_dir / "tables" / table_name).exists()

    a3_results = pd.read_csv(output_dir / "tables" / "a3_results.csv")
    assert len(a3_results) == 1
    a3_row = a3_results.iloc[0]
    assert a3_row["status"] in {"ok", "optimal", "feasible", "feasible_time_limited"}
    assert pd.notna(a3_row["eta"])
    assert pd.notna(a3_row["avg_quality"])
    assert pd.notna(a3_row["grid_id"])

    for disabled_table in [
        "model_pair_recovery.csv",
        "stress_test_results.csv",
        "a4_cvar_results.csv",
    ]:
        assert pd.read_csv(output_dir / "tables" / disabled_table).empty

    domain_slacks = pd.read_csv(output_dir / "tables" / "a3_domain_slacks.csv")
    assert set(domain_slacks["domain"]) == {"AIME", "GPQA", "LCB", "MMLU-Pro"}
    assert len(domain_slacks["domain"]) == 4

    scenario_metrics = pd.read_csv(output_dir / "tables" / "a3_scenario_metrics.csv")
    assert not scenario_metrics.empty
    expected_scenarios = {
        "empirical",
        "balanced",
        "coding_heavy",
        "math_heavy",
        "knowledge_heavy",
        "l1_shift_to_AIME",
        "l1_shift_to_GPQA",
        "l1_shift_to_LCB",
        "l1_shift_to_MMLU-Pro",
    }
    assert set(scenario_metrics["scenario"]) >= expected_scenarios
    eta = pd.to_numeric(pd.Series([a3_row["eta"]]), errors="raise").iloc[0]
    scenario_quality = pd.to_numeric(scenario_metrics["avg_quality"], errors="raise")
    assert eta <= scenario_quality.min() + 1e-9
