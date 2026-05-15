from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


SUCCESS = {"ok", "optimal", "feasible"}


def _read_csv(path):
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    return pd.read_csv(path)


def _successful(df):
    if df.empty or "status" not in df.columns:
        return df
    return df[df["status"].fillna("ok").isin(SUCCESS)].copy()


def plot_cost_quality_frontier(root):
    tables = root / "tables"
    frames = [
        _read_csv(tables / "baseline_extremes.csv"),
        _read_csv(tables / "a0_results.csv").assign(status="ok", family="A0"),
        _read_csv(tables / "a1_results.csv"),
        _read_csv(tables / "a2_results.csv"),
        _read_csv(tables / "a3_results.csv"),
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
    for family, file_name in [("A1", "a1_results.csv"), ("A2", "a2_results.csv"), ("A3", "a3_results.csv")]:
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
        plot_df = plot_df.groupby("model", as_index=False)["usage_count"].sum().nlargest(12, "usage_count")
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
        pivot = keep.pivot_table(index="domain", columns="policy", values="avg_quality", aggfunc="mean")
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
        pivot = keep.pivot_table(index="policy", columns="scenario", values="avg_quality", aggfunc="mean")
        pivot = pivot.tail(8)
        image = plt.imshow(pivot.values, aspect="auto", vmin=0, vmax=1, cmap="viridis")
        plt.xticks(range(len(pivot.columns)), pivot.columns, rotation=30, ha="right")
        plt.yticks(range(len(pivot.index)), pivot.index)
        for i in range(pivot.shape[0]):
            for j in range(pivot.shape[1]):
                plt.text(j, i, f"{pivot.iloc[i, j]:.2f}", ha="center", va="center", color="white", fontsize=8)
        plt.colorbar(image, label="Average quality")
    plt.title("Robustness under prompt-mix scenarios")
    plt.tight_layout()
    plt.savefig(root / "figures" / "robustness_heatmap.png", dpi=200)
    plt.close()


def make_all_plots(output_dir):
    root = Path(output_dir)
    (root / "figures").mkdir(parents=True, exist_ok=True)
    plot_cost_quality_frontier(root)
    plot_quality_vs_pool_size(root)
    plot_selected_model_usage(root)
    plot_domain_performance(root)
    plot_robustness_heatmap(root)
