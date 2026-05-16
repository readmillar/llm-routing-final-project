from pathlib import Path

import pandas as pd


def normalize_quality(df):
    """Return a copy with quality and success probability in [0, 1]."""
    out = df.copy()
    out["quality"] = pd.to_numeric(out["quality"], errors="raise").astype(float)
    out["cost"] = pd.to_numeric(out["cost"], errors="raise").astype(float)
    if out["quality"].min() < 0 or out["quality"].max() > 1:
        q_min = out["quality"].min()
        q_max = out["quality"].max()
        if q_max == q_min:
            out["q_norm"] = 0.0
        else:
            out["q_norm"] = (out["quality"] - q_min) / (q_max - q_min)
    else:
        out["q_norm"] = out["quality"]
    out["r"] = out["q_norm"]
    return out


def validate_data(df):
    """Validate the locked long-format data contract without imputing rows."""
    required = {"prompt_id", "domain", "model", "quality", "cost", "q_norm", "r"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing canonical columns: {sorted(missing)}")
    if df.duplicated(["prompt_id", "model"]).any():
        dupes = int(df.duplicated(["prompt_id", "model"]).sum())
        raise ValueError(f"Found {dupes} duplicate prompt-model rows")
    if (df["cost"] < 0).any():
        raise ValueError("Cost must be nonnegative")
    if df["q_norm"].min() < 0 or df["q_norm"].max() > 1:
        raise ValueError("Normalized quality must be in [0, 1]")
    if df["r"].min() < 0 or df["r"].max() > 1:
        raise ValueError("Success probability must be in [0, 1]")


def find_missing_pairs(df):
    prompts = sorted(df["prompt_id"].unique())
    models = sorted(df["model"].unique())
    pairs = set(zip(df["prompt_id"], df["model"], strict=False))
    prompt_domain = df.drop_duplicates("prompt_id").set_index("prompt_id")["domain"].to_dict()
    rows = []
    for prompt in prompts:
        for model in models:
            if (prompt, model) not in pairs:
                rows.append(
                    {
                        "prompt_id": prompt,
                        "domain": prompt_domain[prompt],
                        "model": model,
                    }
                )
    return pd.DataFrame(rows, columns=["prompt_id", "domain", "model"])


def build_sets_and_matrices(df):
    """Build prompt-specific availability sets and parameter dictionaries."""
    prompts = sorted(df["prompt_id"].unique())
    models = sorted(df["model"].unique())
    domains = sorted(df["domain"].unique())
    pair_rows = df[["prompt_id", "model"]].drop_duplicates()
    pm = sorted((row.prompt_id, row.model) for row in pair_rows.itertuples(index=False))
    m_p = {
        prompt: sorted(df.loc[df["prompt_id"] == prompt, "model"].unique()) for prompt in prompts
    }
    p_d = {
        domain: sorted(df.loc[df["domain"] == domain, "prompt_id"].unique()) for domain in domains
    }
    prompt_domain = (
        df[["prompt_id", "domain"]].drop_duplicates().set_index("prompt_id")["domain"].to_dict()
    )
    q = {(r.prompt_id, r.model): float(r.q_norm) for r in df.itertuples(index=False)}
    c = {(r.prompt_id, r.model): float(r.cost) for r in df.itertuples(index=False)}
    success = {(r.prompt_id, r.model): float(r.r) for r in df.itertuples(index=False)}
    return {
        "df": df,
        "P": prompts,
        "M": models,
        "D": domains,
        "PM": pm,
        "M_p": m_p,
        "P_d": p_d,
        "prompt_domain": prompt_domain,
        "q": q,
        "r": success,
        "c": c,
        "missing_pairs": find_missing_pairs(df),
    }


def write_data_summaries(data, output_dir):
    """Write locked-data summary CSVs used by the report."""
    tables = Path(output_dir) / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    df = data["df"]
    missing = data["missing_pairs"]
    summary = pd.DataFrame(
        [
            ("rows", len(df)),
            ("unique_prompts", len(data["P"])),
            ("unique_models", len(data["M"])),
            ("unique_domains", len(data["D"])),
            ("available_pairs", len(data["PM"])),
            ("missing_pairs", len(missing)),
            ("quality_min", df["q_norm"].min()),
            ("quality_max", df["q_norm"].max()),
            ("cost_min", df["cost"].min()),
            ("cost_max", df["cost"].max()),
            ("zero_cost_rows", int((df["cost"] == 0).sum())),
            ("zero_cost_models", int(df.loc[df["cost"] == 0, "model"].nunique())),
        ],
        columns=["metric", "value"],
    )
    model_summary = (
        df.groupby("model", as_index=False)
        .agg(
            rows=("prompt_id", "size"),
            avg_quality=("q_norm", "mean"),
            avg_cost=("cost", "mean"),
            zero_cost_rows=("cost", lambda s: int((s == 0).sum())),
        )
        .sort_values(["avg_quality", "avg_cost"], ascending=[False, True])
    )
    domain_summary = (
        df[["prompt_id", "domain"]]
        .drop_duplicates()
        .groupby("domain", as_index=False)
        .agg(prompts=("prompt_id", "count"))
    )
    summary.to_csv(tables / "data_summary.csv", index=False)
    model_summary.to_csv(tables / "model_summary.csv", index=False)
    domain_summary.to_csv(tables / "domain_summary.csv", index=False)
    missing.to_csv(tables / "missing_pairs.csv", index=False)
