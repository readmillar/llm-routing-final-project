import json
import time
from pathlib import Path

import pyomo.environ as pyo

SOLVER_ORDER = ["appsi_highs", "highs", "cbc", "glpk"]
SUCCESS_STATUSES = {"ok", "optimal", "feasible", "feasible_time_limited"}


class SolverFailure:
    def __init__(self, message):
        self.solver = type(
            "SolverStatus",
            (),
            {
                "status": "error",
                "termination_condition": message,
            },
        )()


def get_solver(time_limit=300):
    """Return the first available MILP solver and its Pyomo name."""
    for name in SOLVER_ORDER:
        try:
            solver = pyo.SolverFactory(name)
            if solver is None or not solver.available(False):
                continue
            if time_limit:
                try:
                    solver.options["time_limit"] = float(time_limit)
                except Exception:
                    pass
                if name == "cbc":
                    solver.options["seconds"] = float(time_limit)
                if name == "glpk":
                    solver.options["tmlim"] = int(time_limit)
            return name, solver
        except Exception:
            continue
    return None, None


def collect_model_stats(model):
    """Count active Pyomo model size metrics for diagnostics tables."""
    variables = list(model.component_data_objects(pyo.Var, active=True))
    constraints = list(model.component_data_objects(pyo.Constraint, active=True))
    binaries = [var for var in variables if var.is_binary()]
    return {
        "num_variables": len(variables),
        "num_binary_variables": len(binaries),
        "num_constraints": len(constraints),
    }


def _safe_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def normalize_status(termination_condition, has_incumbent=False):
    """Map solver termination to project statuses that separate optimality from incumbents."""
    text = str(termination_condition).lower()
    if "optimal" in text:
        return "optimal"
    if "infeasible" in text:
        return "infeasible"
    if "feasible solution was not found" in text or "no solution" in text:
        return "no_solution"
    if "max" in text or "time" in text:
        return "feasible_time_limited" if has_incumbent else "no_solution"
    if "feasible" in text:
        return "feasible"
    return text.replace(" ", "_")


def _positive_count(value):
    try:
        return int(value) > 0
    except (TypeError, ValueError):
        return False


def result_has_incumbent(results):
    """Conservatively detect whether solver results include a loaded incumbent solution."""
    solution = getattr(results, "solution", None)
    if solution is not None:
        try:
            return len(solution) > 0
        except TypeError:
            pass
        for attr in ("number_of_solutions", "num_solutions", "solution_count"):
            if _positive_count(getattr(solution, attr, None)):
                return True

    problem = getattr(results, "problem", None)
    for attr in ("number_of_solutions", "num_solutions", "solution_count"):
        if _positive_count(getattr(problem, attr, None)):
            return True
    return False


def extract_solver_diagnostics(policy, solver_name, results, model, wall_time_sec):
    """Build a CSV-safe diagnostics row for one solve attempt."""
    stats = collect_model_stats(model)
    solver = getattr(results, "solver", None)
    termination = getattr(solver, "termination_condition", "")
    status = getattr(solver, "status", "")
    problem = getattr(results, "problem", None)
    upper = _safe_float(getattr(problem, "upper_bound", None))
    lower = _safe_float(getattr(problem, "lower_bound", None))
    mip_gap = None
    if upper is not None and lower is not None and abs(upper) > 1e-12:
        mip_gap = abs(upper - lower) / abs(upper)
    row = {
        "policy": policy,
        "solver": solver_name,
        "solver_status": str(status),
        "termination_condition": str(termination),
        "wall_time_sec": wall_time_sec,
        "best_bound": lower,
        "objective_value": upper,
        "mip_gap": mip_gap,
    }
    row.update(stats)
    return row


def pre_solve_diagnostics(policy, status, message):
    """Build diagnostics for validation outcomes before a Pyomo model is solved."""
    return {
        "policy": policy,
        "status": status,
        "termination_condition": message,
        "message": message,
    }


def solve_model(model, time_limit=300, policy=""):
    """Solve a Pyomo model and return solver results with diagnostics."""
    solver_name, solver = get_solver(time_limit=time_limit)
    if solver is None:
        return None, None, {"policy": policy, "status": "no_solver"}
    start = time.perf_counter()
    try:
        try:
            results = solver.solve(model, tee=False, timelimit=time_limit)
        except TypeError:
            results = solver.solve(model, tee=False)
    except RuntimeError as exc:
        elapsed = time.perf_counter() - start
        failure = SolverFailure(str(exc).splitlines()[0])
        return solver_name, failure, extract_solver_diagnostics(
            policy, solver_name, failure, model, elapsed
        )
    elapsed = time.perf_counter() - start
    return solver_name, results, extract_solver_diagnostics(
        policy, solver_name, results, model, elapsed
    )


def result_status(results):
    """Map Pyomo termination output to compact project statuses."""
    if results is None:
        return "no_solver"
    termination = str(results.solver.termination_condition)
    return normalize_status(termination, has_incumbent=result_has_incumbent(results))


def has_solution(status):
    """Return True when it is safe to extract variable values."""
    return status in SUCCESS_STATUSES


def no_solver_result(policy, diagnostics=None):
    result = {
        "policy": policy,
        "status": "no_solver",
        "message": "No MILP solver is available. Install highspy, CBC, or GLPK.",
    }
    if diagnostics is not None:
        result["diagnostics"] = diagnostics
    return result


def write_json(path, payload):
    """Write JSON with stable indentation, creating parent directories."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=str)
