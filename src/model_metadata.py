from __future__ import annotations

from pathlib import Path

import pandas as pd

PROVIDER_PATTERNS = [
    ("gpt-", "OpenAI"),
    ("o1", "OpenAI"),
    ("o3", "OpenAI"),
    ("gemini", "Google"),
    ("qwen", "Qwen/Alibaba"),
    ("deepseek", "DeepSeek"),
    ("glm", "Zhipu"),
    ("llama", "Meta"),
    ("nvidia", "NVIDIA"),
    ("mistral", "Mistral"),
    ("claude", "Anthropic"),
]


def infer_provider_family(model):
    """Infer the provider family from a model name using transparent string rules."""
    lowered = model.lower()
    for pattern, provider in PROVIDER_PATTERNS:
        if pattern in lowered:
            return provider
    return "Other"


def infer_storage_gb(model):
    """Estimate local storage footprint; hosted API models are treated as zero-storage."""
    lowered = model.lower()
    if any(token in lowered for token in ["gpt", "gemini", "claude"]):
        return 0.0
    if "70b" in lowered or "72b" in lowered:
        return 140.0
    if "32b" in lowered:
        return 64.0
    if "14b" in lowered:
        return 28.0
    if "7b" in lowered or "8b" in lowered:
        return 16.0
    return 32.0


def build_metadata_for_models(models):
    """Build provider and deployment metadata for every supplied model name."""
    rows = []
    for model in sorted(models):
        provider = infer_provider_family(model)
        hosted = provider in {"OpenAI", "Google", "Anthropic"}
        rows.append(
            {
                "model": model,
                "provider_family": provider,
                "is_open_source": not hosted,
                "is_hosted_api": hosted,
                "estimated_params_b": 0.0,
                "estimated_storage_gb": infer_storage_gb(model),
                "contract_group": provider,
            }
        )
    return pd.DataFrame(rows)


def load_or_create_metadata(models, path="data/model_metadata.csv"):
    """Load metadata from disk, appending transparent inferred rows for new models."""
    metadata_path = Path(path)
    if metadata_path.exists():
        metadata = pd.read_csv(metadata_path)
    else:
        metadata = build_metadata_for_models(models)
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        metadata.to_csv(metadata_path, index=False)
    missing = set(models) - set(metadata["model"])
    if missing:
        metadata = pd.concat([metadata, build_metadata_for_models(missing)], ignore_index=True)
        metadata = metadata.drop_duplicates("model", keep="first")
        metadata.to_csv(metadata_path, index=False)
    return metadata


def validate_metadata_covers_models(metadata, models):
    """Raise if deployment metadata is missing any model used by the optimization data."""
    missing = sorted(set(models) - set(metadata["model"]))
    if missing:
        preview = ", ".join(missing[:10])
        if len(missing) > 10:
            preview = f"{preview}, ..."
        raise ValueError(f"Model metadata is missing {len(missing)} model(s): {preview}")
    return metadata


def summarize_provider_pool(selected_models, metadata):
    """Summarize provider diversity and storage footprint for a selected model pool."""
    selected = metadata[metadata["model"].isin(selected_models)]
    return {
        "num_models_selected": int(len(selected_models)),
        "provider_count": int(selected["provider_family"].nunique()),
        "storage_gb": float(selected["estimated_storage_gb"].sum()),
    }
