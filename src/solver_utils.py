import json
from pathlib import Path

import pyomo.environ as pyo

SOLVER_ORDER = ["appsi_highs", "highs", "cbc", "glpk"]


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


def solve_model(model, time_limit=300):
    """Solve a Pyomo model with fallback handling for missing solvers."""
    solver_name, solver = get_solver(time_limit=time_limit)
    if solver is None:
        return None, None
    try:
        results = solver.solve(model, tee=False, timelimit=time_limit)
    except TypeError:
        results = solver.solve(model, tee=False)
    except RuntimeError as exc:
        return solver_name, SolverFailure(str(exc).splitlines()[0])
    return solver_name, results


def result_status(results):
    """Map Pyomo termination output to compact project statuses."""
    if results is None:
        return "no_solver"
    termination = str(results.solver.termination_condition).lower()
    if "feasible solution was not found" in termination or "no solution" in termination:
        return "no_solution"
    if "optimal" in termination:
        return "optimal"
    if "feasible" in termination:
        return "feasible"
    if "max" in termination or "time" in termination:
        return "time_limited"
    if "infeasible" in termination:
        return "infeasible"
    return termination.replace(" ", "_")


def has_solution(status):
    """Return True when it is safe to extract variable values."""
    return status in {"optimal", "feasible", "time_limited"}


def no_solver_result(policy):
    return {
        "policy": policy,
        "status": "no_solver",
        "message": "No MILP solver is available. Install highspy, CBC, or GLPK.",
    }


def write_json(path, payload):
    """Write JSON with stable indentation, creating parent directories."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=str)
