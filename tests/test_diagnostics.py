import pyomo.environ as pyo


class _FakeSolverInfo:
    def __init__(self, termination_condition):
        self.termination_condition = termination_condition
        self.status = "warning"


class _FakeResults:
    def __init__(self, termination_condition, solutions=None):
        self.solver = _FakeSolverInfo(termination_condition)
        self.solution = [] if solutions is None else solutions


def test_collect_model_stats_counts_binary_variables_and_constraints():
    from src.solver_utils import collect_model_stats

    model = pyo.ConcreteModel()
    model.I = pyo.Set(initialize=[1, 2])
    model.x = pyo.Var(model.I, within=pyo.Binary)
    model.y = pyo.Var(within=pyo.NonNegativeReals)
    model.c = pyo.Constraint(expr=sum(model.x[i] for i in model.I) + model.y <= 2)
    model.obj = pyo.Objective(expr=model.y)

    stats = collect_model_stats(model)

    assert stats["num_variables"] == 3
    assert stats["num_binary_variables"] == 2
    assert stats["num_constraints"] == 1


def test_normalize_status_distinguishes_time_limited_feasible():
    from src.solver_utils import normalize_status

    assert normalize_status("optimal", has_incumbent=True) == "optimal"
    assert normalize_status("maxTimeLimit", has_incumbent=True) == "feasible_time_limited"
    assert normalize_status("maxTimeLimit", has_incumbent=False) == "no_solution"
    assert normalize_status("infeasible", has_incumbent=False) == "infeasible"


def test_result_status_time_limit_without_incumbent_is_no_solution():
    from src.solver_utils import result_status

    assert result_status(_FakeResults("maxTimeLimit", solutions=[])) == "no_solution"


def test_result_status_time_limit_with_incumbent_is_feasible_time_limited():
    from src.solver_utils import result_status

    assert (
        result_status(_FakeResults("maxTimeLimit", solutions=[object()]))
        == "feasible_time_limited"
    )


def test_solve_model_catches_runtime_error_from_timelimit_fallback(monkeypatch):
    from src import solver_utils

    class FallbackRuntimeErrorSolver:
        def solve(self, model, **kwargs):
            if "timelimit" in kwargs:
                raise TypeError("timelimit not supported")
            raise RuntimeError("fallback failed\nwith detail")

    model = pyo.ConcreteModel()
    model.x = pyo.Var(within=pyo.Binary)

    monkeypatch.setattr(
        solver_utils, "get_solver", lambda time_limit=300: ("fake", FallbackRuntimeErrorSolver())
    )

    solver_name, results, diagnostics = solver_utils.solve_model(
        model, time_limit=1, policy="fake policy"
    )

    assert solver_name == "fake"
    assert results.solver.status == "error"
    assert results.solver.termination_condition == "fallback failed"
    assert diagnostics["policy"] == "fake policy"
    assert diagnostics["solver"] == "fake"
    assert diagnostics["termination_condition"] == "fallback failed"
    assert diagnostics["num_binary_variables"] == 1


def test_pre_solve_infeasible_results_include_diagnostics(synthetic_data):
    from src.pyomo_cascade import generate_cascades, solve_a2
    from src.pyomo_robust_cascade import build_scenarios, compute_domain_floors, solve_a3
    from src.pyomo_single_shot import solve_a1

    cascades, params = generate_cascades(synthetic_data, rho=0.75, max_cascades=20)
    scenarios = build_scenarios(synthetic_data)
    floors = compute_domain_floors(synthetic_data, multiplier=0.75)

    results = [
        solve_a1(synthetic_data, K=0, B=10.0, time_limit=20),
        solve_a1(synthetic_data, K=2, B=-1.0, time_limit=20),
        solve_a2(
            synthetic_data,
            cascades,
            params["R"],
            params["C"],
            params["Esc"],
            params["A_p"],
            K=0,
            B=10.0,
            Emax=1.0,
            time_limit=20,
        ),
        solve_a3(
            synthetic_data,
            cascades,
            params["R"],
            params["C"],
            params["Esc"],
            params["A_p"],
            scenarios,
            floors,
            K=1,
            B=10.0,
            Emax=1.0,
            time_limit=20,
        ),
    ]

    for result in results:
        assert result["status"] == "infeasible"
        assert result["diagnostics"]["policy"] == result["policy"]
        assert result["diagnostics"]["status"] == "infeasible"
        assert result["diagnostics"]["termination_condition"] == result["message"]
