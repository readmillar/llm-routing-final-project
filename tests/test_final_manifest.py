import hashlib


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
