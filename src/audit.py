from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .cascade_generation import precompute_cascade_parameters
from .load_data import load_dataset

TOL = 1e-7
MODERN_RESULT_KEYS = {"status", "policy", "message", "solver", "diagnostics", "K", "B"}


def audit_row(policy, check_name, lhs, rhs, sense="<=", tolerance=TOL):
    """Build one audit row with a signed violation."""
    if sense == "<=":
        violation = max(0.0, float(lhs) - float(rhs))
    elif sense == ">=":
        violation = max(0.0, float(rhs) - float(lhs))
    elif sense == "==":
        violation = abs(float(lhs) - float(rhs))
    else:
        raise ValueError(f"Unknown audit sense: {sense}")
    return {
        "policy": policy,
        "check_name": check_name,
        "passed": violation <= tolerance,
        "lhs": float(lhs),
        "sense": sense,
        "rhs": float(rhs),
        "violation": violation,
        "tolerance": tolerance,
    }


def audit_single_shot_result(data, result, K=None, B=None):
    """Audit an A0/A1-style prompt-to-model assignment."""
    policy = result.get("policy", "")
    assignment = result.get("assignment", {})
    observed_pairs = set(data["PM"])
    rows = []
    rows.append(
        audit_row(policy, "assignment_completeness", len(set(assignment)), len(data["P"]), "==")
    )
    missing_pairs = sum(1 for p, m in assignment.items() if (p, m) not in observed_pairs)
    rows.append(audit_row(policy, "observed_pairs_only", missing_pairs, 0, "=="))
    if K is not None:
        rows.append(audit_row(policy, "pool_size", len(set(assignment.values())), K, "<="))
    if B is not None and assignment and missing_pairs == 0:
        avg_cost = sum(data["c"][(p, m)] for p, m in assignment.items()) / len(data["P"])
        rows.append(audit_row(policy, "budget", avg_cost, B, "<="))
    return rows


def audit_cascade_result(data, cascades, params, result, K=None, B=None, Emax=None):
    """Audit an A2/A3-style prompt-to-cascade assignment."""
    policy = result.get("policy", "")
    assignment = result.get("cascade_assignment", {})
    rows = [
        audit_row(policy, "assignment_completeness", len(set(assignment)), len(data["P"]), "==")
    ]
    lookup = (
        cascades.set_index("cascade_id")
        if "cascade_id" in cascades.columns
        else pd.DataFrame(index=pd.Index([], name="cascade_id"))
    )
    candidate_ids = set(lookup.index)
    invalid_cascade_ids = sum(
        1 for cascade_id in assignment.values() if cascade_id not in candidate_ids
    )
    rows.append(audit_row(policy, "cascade_ids_in_candidates", invalid_cascade_ids, 0, "=="))
    observed_pairs = set(data["PM"])
    selected = set(result.get("selected_models") or [])
    unavailable = 0
    unlinked = 0
    for prompt, cascade_id in assignment.items():
        if cascade_id not in candidate_ids:
            continue
        row = lookup.loc[cascade_id]
        models = [
            m for m in [row["m1"], row.get("m2", ""), row.get("m3", "")] if isinstance(m, str) and m
        ]
        unavailable += sum(1 for model in models if (prompt, model) not in observed_pairs)
        if selected:
            unlinked += sum(1 for model in models if model not in selected)
        else:
            unlinked += len(models)
    rows.append(audit_row(policy, "observed_pairs_only", unavailable, 0, "=="))
    rows.append(audit_row(policy, "selected_model_linking", unlinked, 0, "=="))
    if K is not None:
        rows.append(audit_row(policy, "pool_size", len(selected), K, "<="))
    if B is not None and assignment:
        cost_params = params.get("C", {})
        missing_cost_params = sum(1 for p, a in assignment.items() if (p, a) not in cost_params)
        if missing_cost_params:
            rows.append(
                audit_row(policy, "cascade_cost_params_available", missing_cost_params, 0, "==")
            )
        else:
            avg_cost = sum(cost_params[(p, a)] for p, a in assignment.items()) / len(data["P"])
            rows.append(audit_row(policy, "budget", avg_cost, B, "<="))
    if Emax is not None and assignment:
        esc_params = params.get("Esc", {})
        missing_esc_params = sum(1 for p, a in assignment.items() if (p, a) not in esc_params)
        if missing_esc_params:
            rows.append(
                audit_row(
                    policy, "cascade_escalation_params_available", missing_esc_params, 0, "=="
                )
            )
        else:
            avg_esc = sum(esc_params[(p, a)] for p, a in assignment.items()) / len(data["P"])
            rows.append(audit_row(policy, "escalation", avg_esc, Emax, "<="))
    return rows


def _load_json(path):
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _single_shot_results_from_payload(payload):
    """Yield full result dictionaries from current or legacy solution JSON."""
    for policy, result in payload.items():
        if not isinstance(result, dict):
            continue
        if "assignment" in result:
            yield result
        elif MODERN_RESULT_KEYS.intersection(result):
            continue
        elif all(isinstance(model, str) for model in result.values()):
            yield {"policy": policy, "assignment": result}


def _cascade_results_from_payload(payload):
    """Yield saved cascade result dictionaries."""
    for result in payload.values():
        if isinstance(result, dict) and "cascade_assignment" in result:
            yield result


def _load_saved_cascades(root):
    """Load saved cascade candidates if the experiment wrote them."""
    path = root / "tables" / "cascade_candidates.csv"
    if not path.exists():
        return None
    return pd.read_csv(path, keep_default_na=False)


def _audit_saved_cascade_result(data, cascades, result):
    """Audit a saved cascade result without crashing on stale cascade IDs."""
    policy = result.get("policy", "")
    assignment = result.get("cascade_assignment", {})
    rows = [
        audit_row(policy, "assignment_completeness", len(set(assignment)), len(data["P"]), "==")
    ]
    if cascades is None or cascades.empty or "cascade_id" not in cascades.columns:
        rows.append(audit_row(policy, "cascade_candidates_available", 0, 1, "=="))
        return rows

    saved_ids = set(cascades["cascade_id"])
    missing_ids = sum(1 for cascade_id in assignment.values() if cascade_id not in saved_ids)
    rows.append(audit_row(policy, "cascade_ids_in_saved_candidates", missing_ids, 0, "=="))
    if missing_ids:
        return rows

    rho = result.get("rho", 0.75)
    params = precompute_cascade_parameters(data, cascades, rho=rho)
    rows.extend(
        audit_cascade_result(
            data,
            cascades,
            params,
            result,
            K=result.get("K"),
            B=result.get("B"),
            Emax=result.get("Emax"),
        )[1:]
    )
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description="Audit saved LLM routing solutions.")
    parser.add_argument("--data", default="data/routerbench.csv")
    parser.add_argument("--output-dir", default="outputs_final")
    args = parser.parse_args(argv)
    root = Path(args.output_dir)
    data = load_dataset(args.data, output_dir=root)
    rows = []
    for file_name in ["baseline_assignments.json", "a1_solutions.json"]:
        payload = _load_json(root / "solutions" / file_name)
        for result in _single_shot_results_from_payload(payload):
            rows.extend(
                audit_single_shot_result(data, result, K=result.get("K"), B=result.get("B"))
            )
    cascades = _load_saved_cascades(root)
    for file_name in ["a2_solutions.json", "a3_solutions.json", "a4_solutions.json"]:
        payload = _load_json(root / "solutions" / file_name)
        for result in _cascade_results_from_payload(payload):
            rows.extend(_audit_saved_cascade_result(data, cascades, result))
    out = pd.DataFrame(rows)
    (root / "tables").mkdir(parents=True, exist_ok=True)
    out.to_csv(root / "tables" / "solution_audit.csv", index=False)
    if not out.empty and not out["passed"].all():
        raise SystemExit("One or more audit checks failed. See solution_audit.csv.")


if __name__ == "__main__":
    main()
