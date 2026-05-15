"""Cascade candidate generation and prompt-specific parameter precomputation."""

import pandas as pd


def summarize_models(data):
    """Summarize observed cost and quality by model."""
    df = data["df"]
    return (
        df.groupby("model", as_index=False)
        .agg(
            qbar=("q_norm", "mean"),
            cbar=("cost", "mean"),
            zero_cost_rows=("cost", lambda s: int((s == 0).sum())),
            rows=("prompt_id", "count"),
        )
        .sort_values(["cbar", "qbar"], ascending=[True, False])
    )


def generate_single_stage_cascades(data):
    """Create one degenerate depth-1 cascade for each observed model."""
    summary = summarize_models(data).set_index("model")
    pm = set(data["PM"])
    rows = []
    for model in data["M"]:
        feasible_prompts = [prompt for prompt in data["P"] if (prompt, model) in pm]
        avg_r = (
            sum(data["q"][(prompt, model)] for prompt in feasible_prompts) / len(feasible_prompts)
            if feasible_prompts
            else 0.0
        )
        avg_c = (
            sum(data["c"][(prompt, model)] for prompt in feasible_prompts) / len(feasible_prompts)
            if feasible_prompts
            else 0.0
        )
        rows.append(
            {
                "cascade_id": f"s1::{model}",
                "depth": 1,
                "m1": model,
                "m2": "",
                "m3": "",
                "qbar_m1": float(summary.loc[model, "qbar"]),
                "qbar_m2": 0.0,
                "qbar_m3": 0.0,
                "cbar_m1": float(summary.loc[model, "cbar"]),
                "cbar_m2": 0.0,
                "cbar_m3": 0.0,
                "avg_R": avg_r,
                "avg_C": avg_c,
                "avg_Esc": 0.0,
                "feasible_prompts": len(feasible_prompts),
            }
        )
    return pd.DataFrame(rows)


def _all_two_stage_cascades(data, rho, recovery_lookup=None):
    summary = summarize_models(data).set_index("model")
    cost_cutoff = summary["cbar"].quantile(0.30)
    quality_cutoff = summary["qbar"].quantile(0.50)
    cheap = summary[(summary["cbar"] <= cost_cutoff) | (summary["zero_cost_rows"] > 0)].index
    strong = summary[summary["qbar"] >= quality_cutoff].index
    rows = []
    pm = set(data["PM"])
    for m1 in cheap:
        for m2 in strong:
            if m1 == m2 or summary.loc[m2, "qbar"] < summary.loc[m1, "qbar"]:
                continue
            feasible_prompts = [p for p in data["P"] if (p, m1) in pm and (p, m2) in pm]
            if not feasible_prompts:
                continue
            avg_r = sum(
                data["r"][(p, m1)]
                + (1 - data["r"][(p, m1)])
                * _recovery_term(data, p, m1, m2, rho, recovery_lookup)
                for p in feasible_prompts
            ) / len(feasible_prompts)
            avg_c = sum(
                data["c"][(p, m1)] + (1 - data["r"][(p, m1)]) * data["c"][(p, m2)]
                for p in feasible_prompts
            ) / len(feasible_prompts)
            avg_esc = sum(1 - data["r"][(p, m1)] for p in feasible_prompts) / len(feasible_prompts)
            rows.append(
                {
                    "cascade_id": f"s2::{m1}::{m2}",
                    "depth": 2,
                    "m1": m1,
                    "m2": m2,
                    "m3": "",
                    "qbar_m1": float(summary.loc[m1, "qbar"]),
                    "qbar_m2": float(summary.loc[m2, "qbar"]),
                    "qbar_m3": 0.0,
                    "cbar_m1": float(summary.loc[m1, "cbar"]),
                    "cbar_m2": float(summary.loc[m2, "cbar"]),
                    "cbar_m3": 0.0,
                    "avg_R": avg_r,
                    "avg_C": avg_c,
                    "avg_Esc": avg_esc,
                    "feasible_prompts": len(feasible_prompts),
                }
            )
    return pd.DataFrame(rows)


def generate_two_stage_cascades(data, rho=0.75, max_two_stage=250, recovery_lookup=None):
    """Generate feasible cheap-then-strong depth-2 cascade candidates."""
    cascades = _all_two_stage_cascades(data, rho, recovery_lookup=recovery_lookup)
    if cascades.empty:
        raise ValueError("No feasible two-stage cascade candidates generated")
    if len(cascades) <= max_two_stage:
        return cascades.reset_index(drop=True)

    low_cost_n = max(1, max_two_stage // 2)
    high_quality_n = max_two_stage - low_cost_n
    low_cost = cascades.sort_values(["avg_C", "avg_R"], ascending=[True, False]).head(low_cost_n)
    high_quality = cascades.sort_values(
        ["avg_R", "avg_C", "feasible_prompts"],
        ascending=[False, True, False],
    ).head(high_quality_n)
    selected = pd.concat([low_cost, high_quality], ignore_index=True).drop_duplicates(["m1", "m2"])
    if len(selected) < max_two_stage:
        fill = cascades.sort_values(["avg_C", "avg_R"], ascending=[True, False])
        selected = (
            pd.concat([selected, fill], ignore_index=True)
            .drop_duplicates(["m1", "m2"])
            .head(max_two_stage)
        )
    return selected.reset_index(drop=True)


def generate_three_stage_cascades(data, rho=0.75, max_three_stage=50, recovery_lookup=None):
    """Generate a small high-value set of feasible depth-3 cascade candidates."""
    summary = summarize_models(data).set_index("model")
    cheap = summary.sort_values(["cbar", "qbar"], ascending=[True, False]).head(6).index
    middle = summary.sort_values(["qbar", "cbar"], ascending=[False, True]).head(10).index
    final = summary.sort_values(["qbar", "cbar"], ascending=[False, True]).head(6).index
    pm = set(data["PM"])
    rows = []
    for m1 in cheap:
        for m2 in middle:
            for m3 in final:
                if len({m1, m2, m3}) < 3:
                    continue
                feasible_prompts = [
                    p for p in data["P"] if (p, m1) in pm and (p, m2) in pm and (p, m3) in pm
                ]
                if not feasible_prompts:
                    continue
                values = [
                    _three_stage_summary_values(data, p, m1, m2, m3, rho, recovery_lookup)
                    for p in feasible_prompts
                ]
                rows.append(
                    {
                        "depth": 3,
                        "m1": m1,
                        "m2": m2,
                        "m3": m3,
                        "qbar_m1": summary.loc[m1, "qbar"],
                        "qbar_m2": summary.loc[m2, "qbar"],
                        "qbar_m3": summary.loc[m3, "qbar"],
                        "cbar_m1": summary.loc[m1, "cbar"],
                        "cbar_m2": summary.loc[m2, "cbar"],
                        "cbar_m3": summary.loc[m3, "cbar"],
                        "avg_R": sum(value["R"] for value in values) / len(values),
                        "avg_C": sum(value["C"] for value in values) / len(values),
                        "avg_Esc": sum(value["Esc"] for value in values) / len(values),
                        "feasible_prompts": len(feasible_prompts),
                    }
                )
    frame = pd.DataFrame(rows).head(max_three_stage).reset_index(drop=True)
    if not frame.empty:
        frame.insert(
            0,
            "cascade_id",
            [f"s3::{row.m1}::{row.m2}::{row.m3}" for row in frame.itertuples()],
        )
    return frame


def _three_stage_summary_values(data, prompt, m1, m2, m3, rho, recovery_lookup):
    """Return prompt-specific summary values for a three-stage cascade."""
    r1 = data["r"][(prompt, m1)]
    fail1 = 1 - r1
    recovery2 = _recovery_term(data, prompt, m1, m2, rho, recovery_lookup)
    fail2 = fail1 * (1 - recovery2)
    recovery3 = _recovery_term(data, prompt, m2, m3, rho, recovery_lookup)
    return {
        "R": r1 + fail1 * recovery2 + fail2 * recovery3,
        "C": data["c"][(prompt, m1)]
        + fail1 * data["c"][(prompt, m2)]
        + fail2 * data["c"][(prompt, m3)],
        "Esc": fail1,
    }


def _cascade_models(row):
    """Return non-empty stage model names for a cascade row."""
    models = []
    for field in ("m1", "m2", "m3"):
        value = row.get(field, "")
        if isinstance(value, str) and value:
            models.append(value)
    return models


def _recovery_term(data, prompt, m1, m2, rho, recovery_lookup):
    """Return the conditional recovery contribution for a later-stage model."""
    if recovery_lookup:
        domain = data["prompt_domain"][prompt]
        recovery = recovery_lookup.get((m1, m2, domain))
        if recovery and recovery.get("fallback_level") != "global_rho":
            return float(recovery["recovery_rate"])
    return rho * data["r"][(prompt, m2)]


def precompute_cascade_parameters(data, cascades, rho=0.75, recovery_lookup=None):
    """Precompute prompt-specific cascade availability, quality, cost, and escalation."""
    pm = set(data["PM"])
    a_p = {prompt: [] for prompt in data["P"]}
    r_param = {}
    c_param = {}
    esc_param = {}
    esc2_param = {}
    esc3_param = {}

    for row in cascades.to_dict("records"):
        cascade_id = row["cascade_id"]
        models = _cascade_models(row)
        for prompt in data["P"]:
            if not models or any((prompt, model) not in pm for model in models):
                continue

            m1 = models[0]
            r1 = data["r"][(prompt, m1)]
            c1 = data["c"][(prompt, m1)]
            a_p[prompt].append(cascade_id)

            if len(models) == 1:
                r_param[(prompt, cascade_id)] = data["q"][(prompt, m1)]
                c_param[(prompt, cascade_id)] = c1
                esc_param[(prompt, cascade_id)] = 0.0
                esc2_param[(prompt, cascade_id)] = 0.0
                esc3_param[(prompt, cascade_id)] = 0.0
                continue

            m2 = models[1]
            recovery2 = _recovery_term(data, prompt, m1, m2, rho, recovery_lookup)
            fail1 = 1 - r1
            r_value = r1 + fail1 * recovery2
            c_value = c1 + fail1 * data["c"][(prompt, m2)]
            esc3 = 0.0

            if len(models) >= 3:
                m3 = models[2]
                fail2 = fail1 * (1 - recovery2)
                recovery3 = _recovery_term(data, prompt, m2, m3, rho, recovery_lookup)
                r_value += fail2 * recovery3
                c_value += fail2 * data["c"][(prompt, m3)]
                esc3 = fail2

            r_param[(prompt, cascade_id)] = r_value
            c_param[(prompt, cascade_id)] = c_value
            esc_param[(prompt, cascade_id)] = fail1
            esc2_param[(prompt, cascade_id)] = fail1
            esc3_param[(prompt, cascade_id)] = esc3

    return {
        "A_p": a_p,
        "R": r_param,
        "C": c_param,
        "Esc": esc_param,
        "Esc2": esc2_param,
        "Esc3": esc3_param,
    }


def generate_cascades(
    data,
    rho=0.75,
    max_cascades=250,
    recovery_lookup=None,
    include_three_stage=False,
):
    """Generate available cascade candidates plus prompt-specific parameters."""
    singles = generate_single_stage_cascades(data)
    two_stage_budget = max(0, max_cascades - len(singles))
    frames = [singles]
    if two_stage_budget:
        try:
            frames.append(
                generate_two_stage_cascades(
                    data,
                    rho=rho,
                    max_two_stage=two_stage_budget,
                    recovery_lookup=recovery_lookup,
                )
            )
        except ValueError:
            pass
    used_capacity = sum(len(frame) for frame in frames)
    three_stage_budget = min(50, max(0, max_cascades - used_capacity))
    if include_three_stage and three_stage_budget:
        threes = generate_three_stage_cascades(
            data,
            rho=rho,
            max_three_stage=three_stage_budget,
            recovery_lookup=recovery_lookup,
        )
        if not threes.empty:
            frames.append(threes)
    cascades = pd.concat(frames, ignore_index=True)
    params = precompute_cascade_parameters(data, cascades, rho=rho, recovery_lookup=recovery_lookup)
    uncovered = [prompt for prompt, values in params["A_p"].items() if not values]
    if uncovered:
        raise ValueError(f"No feasible cascades for prompts: {uncovered[:5]}")
    return cascades, params
