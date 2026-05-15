from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

SUCCESS = {"ok", "optimal", "feasible", "feasible_time_limited"}


def _read_csv(path):
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _successful(df):
    if df.empty or "status" not in df.columns:
        return df
    return df[df["status"].fillna("ok").isin(SUCCESS)].copy()


def _numeric_complete_rows(df, columns):
    """Return rows with finite numeric values for the requested columns."""
    if df.empty or not set(columns).issubset(df.columns):
        return pd.DataFrame()
    out = df.copy()
    for column in columns:
        out[column] = pd.to_numeric(out[column], errors="coerce")
    return out.dropna(subset=columns)


def _select_report_policy_marker(chosen, all_results):
    """Return the report policy row to highlight on the Pareto plot."""
    chosen = _numeric_complete_rows(chosen, ["avg_cost", "avg_quality"])
    if not chosen.empty:
        return chosen.head(1)

    reportable = _numeric_complete_rows(all_results, ["avg_cost", "avg_quality"])
    if reportable.empty:
        return pd.DataFrame()
    if "status" in reportable.columns:
        reportable = reportable[reportable["status"].fillna("ok").isin(SUCCESS)].copy()
    if reportable.empty:
        return pd.DataFrame()
    if "family" not in reportable.columns:
        reportable["family"] = reportable.get("policy", "").astype(str).str.split().str[0]
    family_rank = {"A3": 0, "A2": 1, "A1": 2}
    reportable = reportable[reportable["family"].isin(family_rank)].copy()
    if reportable.empty:
        return pd.DataFrame()
    reportable["_family_rank"] = reportable["family"].map(family_rank)
    return reportable.sort_values(
        ["_family_rank", "avg_quality", "avg_cost"], ascending=[True, False, True]
    ).head(1)


def _format_plot_value(value):
    """Format numeric labels without noisy trailing decimals."""
    number = pd.to_numeric(value, errors="coerce")
    if pd.notna(number):
        return f"{float(number):g}"
    return str(value)


def _build_feasibility_pivot(df):
    """Build a solver-status grid with family preserved in each row."""
    required = {"family", "K", "budget_name", "status"}
    if df.empty or not required.issubset(df.columns):
        return pd.DataFrame()
    score = {
        "optimal": 3,
        "feasible": 2,
        "feasible_time_limited": 2,
        "infeasible": 1,
        "no_solution": 0,
        "no_solver": 0,
    }
    out = df.copy()
    out["status_score"] = out["status"].map(score).fillna(0)
    out["_K_sort"] = pd.to_numeric(out["K"], errors="coerce")
    out["_family_sort"] = out["family"].map({"A1": 1, "A2": 2, "A3": 3}).fillna(99)
    out["_row_label"] = out["family"].astype(str) + " K=" + out["K"].map(_format_plot_value)
    out = out.sort_values(["_family_sort", "_K_sort", "family", "_row_label"])
    pivot = out.pivot_table(
        index="_row_label",
        columns="budget_name",
        values="status_score",
        aggfunc="max",
        sort=False,
    )
    pivot.index.name = "family_K"
    return pivot


def _safe_result_number(result, key, default):
    value = pd.to_numeric(result.get(key), errors="coerce")
    if pd.isna(value):
        return default
    return float(value)


def _select_cascade_flow_result(payload):
    """Choose a deterministic representative A2 result for cascade-flow plots."""
    candidates = []
    for name, result in payload.items():
        if not isinstance(result, dict) or "cascade_assignment" not in result:
            continue
        if result.get("status", "ok") not in SUCCESS:
            continue
        candidates.append((name, result))
    if not candidates:
        return None
    return sorted(
        candidates,
        key=lambda item: (
            -_safe_result_number(item[1], "avg_quality", float("-inf")),
            _safe_result_number(item[1], "avg_cost", float("inf")),
            _safe_result_number(item[1], "escalation_rate", float("inf")),
            str(item[0]),
        ),
    )[0][1]


def _usage_concentration_plot_frame(usage):
    """Return the most concentrated usage rows for plotting."""
    required = {"policy", "stage", "top_1_model_share"}
    if usage.empty or not required.issubset(usage.columns):
        return pd.DataFrame()
    out = usage.copy()
    out["top_1_model_share"] = pd.to_numeric(out["top_1_model_share"], errors="coerce")
    out = out.dropna(subset=["top_1_model_share"])
    return out.sort_values("top_1_model_share", ascending=False).head(8)


def plot_cost_quality_frontier(root):
    tables = root / "tables"
    frames = [
        _read_csv(tables / "baseline_extremes.csv"),
        _read_csv(tables / "a0_results.csv").assign(status="ok", family="A0"),
        _read_csv(tables / "a1_results.csv"),
        _read_csv(tables / "a2_results.csv"),
        _read_csv(tables / "a3_results.csv"),
        _read_csv(tables / "a4_cvar_results.csv"),
    ]
    df = _successful(pd.concat([f for f in frames if not f.empty], ignore_index=True))
    plt.figure(figsize=(8, 5.5))
    for label, group in df.groupby(df.get("family", df["policy"]).fillna("policy")):
        plt.scatter(group["avg_cost"], group["avg_quality"], label=label, s=42, alpha=0.85)
    plt.xlabel("Average expected cost")
    plt.ylabel("Average expected quality")
    plt.title("Cost-quality frontier")
    plt.grid(alpha=0.25)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(root / "figures" / "cost_quality_frontier.png", dpi=200)
    plt.close()


def plot_quality_vs_pool_size(root):
    tables = root / "tables"
    frames = []
    for family, file_name in [
        ("A1", "a1_results.csv"),
        ("A2", "a2_results.csv"),
        ("A3", "a3_results.csv"),
    ]:
        df = _successful(_read_csv(tables / file_name))
        if not df.empty and "K" in df:
            frames.append(df.assign(family=family))
    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    plt.figure(figsize=(7.5, 5))
    if not df.empty:
        best = df.groupby(["family", "K"], as_index=False)["avg_quality"].max()
        for family, group in best.groupby("family"):
            plt.plot(group["K"], group["avg_quality"], marker="o", label=family)
    plt.xlabel("Model pool size K")
    plt.ylabel("Best average quality")
    plt.title("Quality versus model-pool size")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(root / "figures" / "quality_vs_pool_size.png", dpi=200)
    plt.close()


def plot_selected_model_usage(root):
    usage = _read_csv(root / "tables" / "selected_model_usage.csv")
    plt.figure(figsize=(9, 5.5))
    if not usage.empty:
        preferred = usage[usage["policy"].str.startswith("A3")]
        if preferred.empty:
            preferred = usage[usage["policy"].str.startswith("A2")]
        if preferred.empty:
            preferred = usage
        policy = preferred["policy"].iloc[-1]
        plot_df = preferred[preferred["policy"] == policy].copy()
        plot_df = (
            plot_df.groupby("model", as_index=False)["usage_count"]
            .sum()
            .nlargest(12, "usage_count")
        )
        plt.barh(plot_df["model"], plot_df["usage_count"], color="#2f6f73")
        plt.gca().invert_yaxis()
        plt.title(f"Selected model usage: {policy}")
    plt.xlabel("Expected prompt count")
    plt.tight_layout()
    plt.savefig(root / "figures" / "selected_model_usage.png", dpi=200)
    plt.close()


def plot_domain_performance(root):
    domain = _read_csv(root / "tables" / "domain_quality.csv")
    plt.figure(figsize=(9, 5.5))
    if not domain.empty:
        keep = domain[domain["policy"].str.startswith(("Always", "A1", "A2", "A3"))]
        pivot = keep.pivot_table(
            index="domain", columns="policy", values="avg_quality", aggfunc="mean"
        )
        pivot.iloc[:, -6:].plot(kind="bar", ax=plt.gca())
    plt.ylabel("Average quality")
    plt.title("Domain quality comparison")
    plt.ylim(0, 1.05)
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(root / "figures" / "domain_performance_comparison.png", dpi=200)
    plt.close()


def plot_robustness_heatmap(root):
    scenario = _read_csv(root / "tables" / "scenario_quality.csv")
    plt.figure(figsize=(8.5, 5.5))
    if not scenario.empty:
        keep = scenario[scenario["policy"].str.startswith(("Always", "A1", "A2", "A3"))]
        pivot = keep.pivot_table(
            index="policy", columns="scenario", values="avg_quality", aggfunc="mean"
        )
        pivot = pivot.tail(8)
        image = plt.imshow(pivot.values, aspect="auto", vmin=0, vmax=1, cmap="viridis")
        plt.xticks(range(len(pivot.columns)), pivot.columns, rotation=30, ha="right")
        plt.yticks(range(len(pivot.index)), pivot.index)
        for i in range(pivot.shape[0]):
            for j in range(pivot.shape[1]):
                plt.text(
                    j,
                    i,
                    f"{pivot.iloc[i, j]:.2f}",
                    ha="center",
                    va="center",
                    color="white",
                    fontsize=8,
                )
        plt.colorbar(image, label="Average quality")
    plt.title("Robustness under prompt-mix scenarios")
    plt.tight_layout()
    plt.savefig(root / "figures" / "robustness_heatmap.png", dpi=200)
    plt.close()


def plot_model_complementarity_heatmap(root):
    recovery = _read_csv(root / "tables" / "model_pair_recovery.csv")
    plt.figure(figsize=(10, 8))
    if not recovery.empty:
        if "fallback_level" in recovery.columns:
            recovery = recovery[recovery["fallback_level"] != "global_rho"].copy()
    if not recovery.empty:
        recovery["weighted_recovery"] = recovery["recovery_rate"] * recovery["support"]
        pair = (
            recovery.groupby(["m1", "m2"], as_index=False)
            .agg(weighted_recovery=("weighted_recovery", "sum"), support=("support", "sum"))
            .sort_values("support", ascending=False)
        )
        pair = pair[pair["support"] > 0].copy()
        pair["recovery_rate"] = pair["weighted_recovery"] / pair["support"]
        top_models = sorted(set(pair.head(20)["m1"]).union(pair.head(20)["m2"]))
        plot_df = pair[pair["m1"].isin(top_models) & pair["m2"].isin(top_models)]
        pivot = plot_df.pivot_table(
            index="m1", columns="m2", values="recovery_rate", aggfunc="mean"
        )
        if not pivot.empty:
            image = plt.imshow(
                pivot.fillna(0.0).values, aspect="auto", vmin=0, vmax=1, cmap="magma"
            )
            plt.xticks(
                range(len(pivot.columns)), pivot.columns, rotation=60, ha="right", fontsize=7
            )
            plt.yticks(range(len(pivot.index)), pivot.index, fontsize=7)
            plt.colorbar(image, label="Recovery rate")
    plt.title("Model complementarity: P(second succeeds | first fails)")
    plt.tight_layout()
    plt.savefig(root / "figures" / "model_complementarity_heatmap.png", dpi=220)
    plt.close()


def plot_pareto_frontier_report(root):
    """Plot the matched report comparison and its non-dominated frontier."""
    all_results = _read_csv(root / "tables" / "report_main_comparison.csv")
    frontier = _read_csv(root / "tables" / "pareto_frontier.csv")
    plt.figure(figsize=(8.5, 5.5))
    if not all_results.empty and {"family", "avg_cost", "avg_quality"}.issubset(
        all_results.columns
    ):
        for family, group in all_results.groupby("family"):
            plt.scatter(
                group["avg_cost"],
                group["avg_quality"],
                label=family,
                s=48,
                alpha=0.85,
            )
    if not frontier.empty and {"avg_cost", "avg_quality"}.issubset(frontier.columns):
        ordered = frontier.sort_values("avg_cost")
        plt.plot(
            ordered["avg_cost"],
            ordered["avg_quality"],
            color="black",
            linewidth=1.5,
            label="Non-dominated frontier",
        )
    chosen = _select_report_policy_marker(
        _read_csv(root / "tables" / "a3_best_report_policy.csv"), all_results
    )
    if not chosen.empty:
        plt.scatter(
            chosen["avg_cost"],
            chosen["avg_quality"],
            marker="*",
            s=240,
            color="#d62728",
            label="Chosen policy",
        )
    plt.xlabel("Average expected cost")
    plt.ylabel("Average expected quality")
    plt.title("Matched cost-quality frontier")
    plt.grid(alpha=0.25)
    handles, labels = plt.gca().get_legend_handles_labels()
    if handles:
        plt.legend(handles, labels, fontsize=8)
    plt.tight_layout()
    plt.savefig(root / "figures" / "pareto_frontier_report.png", dpi=240)
    plt.close()


def plot_stress_test_quality_distribution(root):
    stress = _read_csv(root / "tables" / "stress_test_results.csv")
    plt.figure(figsize=(9, 5.5))
    if not stress.empty:
        policies = list(
            stress.groupby("policy")["avg_quality"]
            .mean()
            .sort_values(ascending=False)
            .head(6)
            .index
        )
        data = [stress.loc[stress["policy"] == policy, "avg_quality"].values for policy in policies]
        plt.boxplot(data, tick_labels=policies, showfliers=False)
        plt.xticks(rotation=30, ha="right")
        plt.ylabel("Average quality under sampled traffic mix")
    plt.title("Stress-test quality distribution")
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(root / "figures" / "stress_test_quality_distribution.png", dpi=220)
    plt.close()


def plot_domain_reliability_report(root):
    """Plot per-domain quality for the report policy set."""
    domain = _read_csv(root / "tables" / "domain_quality.csv")
    plt.figure(figsize=(9, 5.5))
    required = {"policy", "domain", "avg_quality"}
    if not domain.empty and required.issubset(domain.columns):
        keep = domain[domain["policy"].fillna("").str.startswith(("A1", "A2", "A3", "A4"))]
        if not keep.empty:
            pivot = keep.pivot_table(
                index="domain", columns="policy", values="avg_quality", aggfunc="mean"
            )
            if not pivot.empty:
                pivot.iloc[:, -6:].plot(kind="bar", ax=plt.gca())
    plt.ylabel("Average quality")
    plt.ylim(0, 1.05)
    plt.title("Domain reliability by policy")
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(root / "figures" / "domain_reliability_report.png", dpi=240)
    plt.close()


def plot_feasibility_map(root):
    """Plot solver status across pool-size and budget grid points."""
    frames = []
    for family, file_name in [
        ("A1", "a1_results.csv"),
        ("A2", "a2_results.csv"),
        ("A3", "a3_grid_results.csv"),
    ]:
        frame = _read_csv(root / "tables" / file_name)
        if not frame.empty:
            frames.append(frame.assign(family=family))
    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    plt.figure(figsize=(8, 5.5))
    pivot = _build_feasibility_pivot(df)
    if not pivot.empty:
        image = plt.imshow(pivot.values, aspect="auto", vmin=0, vmax=3, cmap="viridis")
        plt.xticks(range(len(pivot.columns)), pivot.columns)
        plt.yticks(range(len(pivot.index)), pivot.index)
        plt.colorbar(image, label="0 no solution, 3 optimal")
    plt.xlabel("Budget")
    plt.ylabel("Policy family / pool size K")
    plt.title("Optimization feasibility map")
    plt.tight_layout()
    plt.savefig(root / "figures" / "feasibility_map.png", dpi=240)
    plt.close()


def plot_provider_traffic_share(root):
    provider = _read_csv(root / "tables" / "provider_usage.csv")
    plt.figure(figsize=(8, 5))
    if not provider.empty and "provider_count" in provider:
        provider.tail(8).plot(
            kind="bar", x="policy", y="provider_count", ax=plt.gca(), legend=False
        )
        plt.ylabel("Provider families selected")
        plt.xticks(rotation=30, ha="right")
    plt.title("Provider diversity by selected policy")
    plt.tight_layout()
    plt.savefig(root / "figures" / "provider_traffic_share.png", dpi=220)
    plt.close()


def plot_usage_concentration(root):
    """Plot concentration of policy traffic in the most-used model."""
    usage = _read_csv(root / "tables" / "usage_concentration.csv")
    plt.figure(figsize=(8.5, 5))
    plot_df = _usage_concentration_plot_frame(usage)
    if not plot_df.empty:
        plt.barh(
            plot_df["policy"] + " / " + plot_df["stage"],
            plot_df["top_1_model_share"],
            color="#4c78a8",
        )
        plt.xlabel("Top model usage share")
    plt.title("Model usage concentration")
    plt.tight_layout()
    plt.savefig(root / "figures" / "usage_concentration.png", dpi=240)
    plt.close()


def plot_cvar_tradeoff(root):
    cvar = _read_csv(root / "tables" / "a4_cvar_results.csv")
    plt.figure(figsize=(7.5, 5))
    if not cvar.empty:
        cvar = _numeric_complete_rows(cvar, ["cvar_shortfall", "avg_quality"])
        if not cvar.empty:
            plt.scatter(cvar["cvar_shortfall"], cvar["avg_quality"], s=60)
        plt.xlabel("CVaR shortfall")
        plt.ylabel("Average quality")
    plt.title("Tail-risk tradeoff")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(root / "figures" / "cvar_tradeoff.png", dpi=220)
    plt.close()


def plot_cascade_flow(root):
    """Plot the most common first-stage to second-stage cascade assignments."""
    solutions = root / "solutions" / "a2_solutions.json"
    candidates = _read_csv(root / "tables" / "cascade_candidates.csv")
    plt.figure(figsize=(9, 6))
    required = {"cascade_id", "m1", "m2"}
    if solutions.exists() and not candidates.empty and required.issubset(candidates.columns):
        import json

        payload = json.loads(solutions.read_text())
        result = _select_cascade_flow_result(payload)
        if result is not None:
            lookup = candidates.drop_duplicates("cascade_id").set_index("cascade_id")
            flows = {}
            for cascade_id in result["cascade_assignment"].values():
                if cascade_id not in lookup.index:
                    continue
                row = lookup.loc[cascade_id]
                if isinstance(row.get("m2", ""), str) and row["m2"]:
                    flows[(row["m1"], row["m2"])] = flows.get((row["m1"], row["m2"]), 0) + 1
            top = sorted(flows.items(), key=lambda item: item[1], reverse=True)[:12]
            labels = [f"{m1} -> {m2}" for (m1, m2), _ in top]
            values = [count for _, count in top]
            plt.barh(labels, values, color="#2f6f73")
            plt.gca().invert_yaxis()
    plt.xlabel("Assigned prompt count")
    plt.title("Cascade flow: first-stage to second-stage")
    plt.tight_layout()
    plt.savefig(root / "figures" / "cascade_flow.png", dpi=240)
    plt.close()


def make_all_plots(output_dir):
    root = Path(output_dir)
    (root / "figures").mkdir(parents=True, exist_ok=True)
    plot_cost_quality_frontier(root)
    plot_quality_vs_pool_size(root)
    plot_selected_model_usage(root)
    plot_domain_performance(root)
    plot_robustness_heatmap(root)
    plot_pareto_frontier_report(root)
    plot_cascade_flow(root)
    plot_model_complementarity_heatmap(root)
    plot_stress_test_quality_distribution(root)
    plot_domain_reliability_report(root)
    plot_feasibility_map(root)
    plot_provider_traffic_share(root)
    plot_usage_concentration(root)
    plot_cvar_tradeoff(root)
