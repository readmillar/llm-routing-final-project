"""Estimate model-pair recovery rates from observed prompt outcomes."""

from __future__ import annotations

import pandas as pd


def _failure_prompts(data, domain, m1, fail_threshold):
    prompts = data["P_d"][domain]
    return [
        prompt
        for prompt in prompts
        if (prompt, m1) in data["q"] and data["q"][(prompt, m1)] <= fail_threshold
    ]


def estimate_pair_recovery(data, min_support=5, fail_threshold=0.0, global_rho=0.75):
    """Estimate P(model 2 succeeds | model 1 fails) by ordered pair and domain."""
    rows = []
    for domain in data["D"]:
        for m1 in data["M"]:
            failures = _failure_prompts(data, domain, m1, fail_threshold)
            for m2 in data["M"]:
                if m1 == m2:
                    continue
                domain_prompts = [p for p in failures if (p, m2) in data["q"]]
                if len(domain_prompts) >= min_support:
                    rate = sum(data["q"][(p, m2)] for p in domain_prompts) / len(domain_prompts)
                    level = "domain_pair"
                    support = len(domain_prompts)
                else:
                    pair_failures = [
                        p
                        for d in data["D"]
                        for p in _failure_prompts(data, d, m1, fail_threshold)
                        if (p, m2) in data["q"]
                    ]
                    if len(pair_failures) >= min_support:
                        rate = sum(data["q"][(p, m2)] for p in pair_failures) / len(pair_failures)
                        level = "pair"
                        support = len(pair_failures)
                    else:
                        rate = global_rho
                        level = "global_rho"
                        support = len(domain_prompts)
                rows.append(
                    {
                        "m1": m1,
                        "m2": m2,
                        "domain": domain,
                        "support": support,
                        "recovery_rate": float(rate),
                        "fallback_level": level,
                    }
                )
    return pd.DataFrame(rows)


def recovery_lookup_from_frame(frame):
    """Convert recovery table into a lookup consumed by cascade parameter generation."""
    return {
        (row.m1, row.m2, row.domain): {
            "support": int(row.support),
            "recovery_rate": float(row.recovery_rate),
            "fallback_level": row.fallback_level,
        }
        for row in frame.itertuples(index=False)
    }
