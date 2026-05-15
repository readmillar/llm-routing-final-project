from pathlib import Path

import pandas as pd

from .preprocessing import (
    build_sets_and_matrices,
    normalize_quality,
    validate_data,
    write_data_summaries,
)

COLUMN_CANDIDATES = {
    "prompt_id": ["prompt_id", "prompt", "question_id", "qid", "id"],
    "domain": ["dataset", "domain", "benchmark", "task"],
    "model": ["model", "model_name", "llm", "system"],
    "quality": ["score", "quality", "performance", "accuracy", "correct", "reward"],
    "cost": ["cost", "avg_cost", "price", "dollar_cost", "api_cost"],
}


def load_raw_data(path):
    """Load the CSV input without changing availability or imputing rows."""
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Data file not found: {csv_path}")
    return pd.read_csv(csv_path)


def detect_columns(df):
    """Detect semantic columns, preferring the locked routerbench names."""
    lower_to_actual = {str(col).lower(): col for col in df.columns}
    detected = {}
    for canonical, candidates in COLUMN_CANDIDATES.items():
        for candidate in candidates:
            if candidate.lower() in lower_to_actual:
                detected[canonical] = lower_to_actual[candidate.lower()]
                break
    missing = sorted(set(COLUMN_CANDIDATES) - set(detected))
    if missing:
        raise ValueError(
            "Could not detect required columns " f"{missing}. Available columns: {list(df.columns)}"
        )
    return detected


def standardize_long_format(df, columns):
    """Rename detected columns to the canonical long-format schema."""
    out = pd.DataFrame(
        {
            "prompt_id": df[columns["prompt_id"]].astype(str),
            "domain": df[columns["domain"]].astype(str),
            "model": df[columns["model"]].astype(str),
            "quality": pd.to_numeric(df[columns["quality"]], errors="raise"),
            "cost": pd.to_numeric(df[columns["cost"]], errors="raise"),
        }
    )
    optional = [
        c for c in ["row_id", "index", "prompt_tokens", "completion_tokens"] if c in df.columns
    ]
    for column in optional:
        out[column] = df[column]
    return out


def load_dataset(path, output_dir="outputs"):
    """Load, validate, summarize, and return dict-backed optimization data."""
    raw = load_raw_data(path)
    columns = detect_columns(raw)
    df = standardize_long_format(raw, columns)
    df = normalize_quality(df)
    validate_data(df)
    data = build_sets_and_matrices(df)
    write_data_summaries(data, output_dir)
    return data
