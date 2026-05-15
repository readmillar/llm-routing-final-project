from __future__ import annotations

import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import pyomo


def write_report_numbers(output_dir, chosen_policy):
    """Write a compact Markdown file with headline report numbers."""
    root = Path(output_dir)
    report_dir = root / "report_artifacts"
    report_dir.mkdir(parents=True, exist_ok=True)
    selected_models = chosen_policy.get("selected_models", [])
    if isinstance(selected_models, str):
        selected_models_text = selected_models
    else:
        selected_models_text = ", ".join(selected_models)
    text = "\n".join(
        [
            "# Report Numbers",
            "",
            f"Chosen policy: {chosen_policy.get('policy', '')}",
            f"Average quality: {float(chosen_policy.get('avg_quality') or 0.0):.4f}",
            f"Average cost: {float(chosen_policy.get('avg_cost') or 0.0):.6f}",
            f"Worst-scenario quality eta: {float(chosen_policy.get('eta') or 0.0):.4f}",
            f"Escalation rate: {float(chosen_policy.get('escalation_rate') or 0.0):.4f}",
            f"Selected models: {selected_models_text}",
            "",
        ]
    )
    path = report_dir / "report_numbers.md"
    path.write_text(text, encoding="utf-8")
    return path


def write_manifest(output_dir, data_sha256, command, random_seed=164, git_commit="unknown"):
    """Write a reproducibility manifest for generated report artifacts."""
    root = Path(output_dir)
    manifest = {
        "git_commit": git_commit,
        "data_sha256": data_sha256,
        "python_version": platform.python_version(),
        "pyomo_version": pyomo.version.version,
        "command": command,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "random_seed": random_seed,
    }
    path = root / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return path


def write_report_tables(output_dir, main_comparison, domain_table):
    """Write report-ready comparison and domain tables."""
    root = Path(output_dir)
    artifact_dir = root / "report_artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    main_comparison.to_csv(artifact_dir / "report_main_comparison.csv", index=False)
    domain_table.to_csv(artifact_dir / "report_domain_table.csv", index=False)
