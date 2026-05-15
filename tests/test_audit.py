def test_audit_assignment_budget_and_observed_pairs_for_single_shot(synthetic_data):
    from src.audit import audit_single_shot_result
    from src.baselines import solve_always_cheapest

    result = solve_always_cheapest(synthetic_data)
    rows = audit_single_shot_result(synthetic_data, result, K=4, B=10.0)

    assert rows
    assert all(row["passed"] for row in rows)
    assert {row["check_name"] for row in rows} >= {
        "assignment_completeness",
        "observed_pairs_only",
        "budget",
        "pool_size",
    }


def test_audit_detects_budget_violation(synthetic_data):
    from src.audit import audit_single_shot_result
    from src.baselines import solve_always_best_quality

    result = solve_always_best_quality(synthetic_data)
    rows = audit_single_shot_result(synthetic_data, result, K=4, B=0.0)
    budget = [row for row in rows if row["check_name"] == "budget"][0]

    assert not budget["passed"]
    assert budget["violation"] > 0.0


def test_cli_audits_saved_cascade_solution(synthetic_csv, synthetic_data, tmp_path):
    from src.audit import main
    from src.cascade_generation import generate_cascades
    from src.solver_utils import write_json

    root = tmp_path / "outputs"
    (root / "solutions").mkdir(parents=True, exist_ok=True)
    (root / "tables").mkdir(parents=True, exist_ok=True)
    cascades, params = generate_cascades(synthetic_data, rho=0.75, max_cascades=20)
    assignment = {prompt: params["A_p"][prompt][0] for prompt in synthetic_data["P"]}
    selected_models = sorted(
        {
            cascades.set_index("cascade_id").loc[cascade_id, "m1"]
            for cascade_id in assignment.values()
        }
    )
    cascades.to_csv(root / "tables" / "cascade_candidates.csv", index=False)
    write_json(
        root / "solutions" / "a2_solutions.json",
        {
            "synthetic-a2": {
                "policy": "synthetic-a2",
                "status": "optimal",
                "cascade_assignment": assignment,
                "selected_models": selected_models,
                "K": len(selected_models),
                "B": 10.0,
                "Emax": 1.0,
            }
        },
    )

    main(["--data", str(synthetic_csv), "--output-dir", str(root)])

    audit = __import__("pandas").read_csv(root / "tables" / "solution_audit.csv")
    assert not audit.empty
    assert set(audit["check_name"]) >= {
        "assignment_completeness",
        "cascade_ids_in_saved_candidates",
        "observed_pairs_only",
        "selected_model_linking",
        "budget",
        "escalation",
    }
    assert audit["passed"].all()


def test_cli_audits_saved_a4_cascade_solution(synthetic_csv, synthetic_data, tmp_path):
    from src.audit import main
    from src.cascade_generation import generate_cascades
    from src.solver_utils import write_json

    root = tmp_path / "outputs"
    (root / "solutions").mkdir(parents=True, exist_ok=True)
    (root / "tables").mkdir(parents=True, exist_ok=True)
    cascades, params = generate_cascades(synthetic_data, rho=0.75, max_cascades=20)
    assignment = {prompt: params["A_p"][prompt][0] for prompt in synthetic_data["P"]}
    selected_models = sorted(
        {
            cascades.set_index("cascade_id").loc[cascade_id, "m1"]
            for cascade_id in assignment.values()
        }
    )
    cascades.to_csv(root / "tables" / "cascade_candidates.csv", index=False)
    write_json(
        root / "solutions" / "a4_solutions.json",
        {
            "synthetic-a4": {
                "policy": "synthetic-a4",
                "status": "optimal",
                "cascade_assignment": assignment,
                "selected_models": selected_models,
                "K": len(selected_models),
                "B": 10.0,
                "Emax": 1.0,
            }
        },
    )

    main(["--data", str(synthetic_csv), "--output-dir", str(root)])

    audit = __import__("pandas").read_csv(root / "tables" / "solution_audit.csv")
    assert set(audit["policy"]) == {"synthetic-a4"}
    assert set(audit["check_name"]) >= {
        "assignment_completeness",
        "cascade_ids_in_saved_candidates",
        "observed_pairs_only",
        "selected_model_linking",
        "budget",
        "escalation",
    }
    assert audit["passed"].all()


def test_cli_reports_missing_saved_cascade_id(synthetic_csv, synthetic_data, tmp_path):
    import pytest

    from src.audit import main
    from src.cascade_generation import generate_cascades
    from src.solver_utils import write_json

    root = tmp_path / "outputs"
    (root / "solutions").mkdir(parents=True, exist_ok=True)
    (root / "tables").mkdir(parents=True, exist_ok=True)
    cascades, _ = generate_cascades(synthetic_data, rho=0.75, max_cascades=20)
    cascades.to_csv(root / "tables" / "cascade_candidates.csv", index=False)
    write_json(
        root / "solutions" / "a3_solutions.json",
        {
            "synthetic-a3": {
                "policy": "synthetic-a3",
                "status": "optimal",
                "cascade_assignment": {
                    prompt: "missing::cascade" for prompt in synthetic_data["P"]
                },
                "selected_models": [],
                "K": 1,
                "B": 10.0,
                "Emax": 1.0,
                "rho": 0.5,
            }
        },
    )

    with pytest.raises(SystemExit):
        main(["--data", str(synthetic_csv), "--output-dir", str(root)])

    audit = __import__("pandas").read_csv(root / "tables" / "solution_audit.csv")
    cascade_id_row = audit[audit["check_name"] == "cascade_ids_in_saved_candidates"].iloc[0]
    assert not cascade_id_row["passed"]
    assert cascade_id_row["violation"] == len(synthetic_data["P"])


def test_cli_skips_modern_no_solution_single_shot_and_keeps_legacy_assignment(
    synthetic_csv, synthetic_data, tmp_path
):
    from src.audit import main
    from src.baselines import solve_always_cheapest
    from src.solver_utils import write_json

    root = tmp_path / "outputs"
    (root / "solutions").mkdir(parents=True, exist_ok=True)
    legacy = solve_always_cheapest(synthetic_data)
    write_json(
        root / "solutions" / "baseline_assignments.json",
        {"legacy-cheapest": legacy["assignment"]},
    )
    write_json(
        root / "solutions" / "a1_solutions.json",
        {
            "no-solution-a1": {
                "message": "Budget below cheapest assignment",
                "policy": "A1 K=1 B=0",
                "solver": "appsi_highs",
                "status": "no_solution",
            }
        },
    )

    main(["--data", str(synthetic_csv), "--output-dir", str(root)])

    audit = __import__("pandas").read_csv(root / "tables" / "solution_audit.csv")
    assert set(audit["policy"]) == {"legacy-cheapest"}
    assert audit["passed"].all()


def test_cascade_audit_fails_linking_when_selected_models_missing_or_empty(synthetic_data):
    import pytest

    from src.audit import audit_cascade_result
    from src.cascade_generation import generate_cascades

    cascades, params = generate_cascades(synthetic_data, rho=0.75, max_cascades=20)
    assignment = {prompt: params["A_p"][prompt][0] for prompt in synthetic_data["P"]}

    for result in [
        {"policy": "missing-selected", "cascade_assignment": assignment},
        {"policy": "empty-selected", "cascade_assignment": assignment, "selected_models": []},
    ]:
        rows = audit_cascade_result(synthetic_data, cascades, params, result)
        linking = [row for row in rows if row["check_name"] == "selected_model_linking"][0]
        assert not linking["passed"]
        assert linking["violation"] == pytest.approx(len(synthetic_data["P"]))


def test_single_shot_audit_reports_bad_pairs_without_budget_crash(synthetic_data):
    from src.audit import audit_single_shot_result

    assignment = {prompt: synthetic_data["M_p"][prompt][0] for prompt in synthetic_data["P"]}
    assignment[synthetic_data["P"][0]] = "not-an-observed-model"
    result = {"policy": "bad-single", "assignment": assignment}

    rows = audit_single_shot_result(synthetic_data, result, B=10.0)

    observed_pairs = [row for row in rows if row["check_name"] == "observed_pairs_only"][0]
    assert not observed_pairs["passed"]
    assert observed_pairs["violation"] == 1.0


def test_cascade_audit_reports_bad_cascade_id_without_crash(synthetic_data):
    from src.audit import audit_cascade_result
    from src.cascade_generation import generate_cascades

    cascades, params = generate_cascades(synthetic_data, rho=0.75, max_cascades=20)
    assignment = {prompt: params["A_p"][prompt][0] for prompt in synthetic_data["P"]}
    assignment[synthetic_data["P"][0]] = "not-a-saved-cascade"
    result = {
        "policy": "bad-cascade",
        "cascade_assignment": assignment,
        "selected_models": list(synthetic_data["M"]),
    }

    rows = audit_cascade_result(synthetic_data, cascades, params, result, B=10.0, Emax=1.0)

    validity = [row for row in rows if row["check_name"] == "cascade_ids_in_candidates"][0]
    assert not validity["passed"]
    assert validity["violation"] == 1.0
