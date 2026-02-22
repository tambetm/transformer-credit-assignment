"""Visualization script for generating all plots and summary tables.

Generates:
1. Learning curves for each environment (all algorithms, with std shading)
2. Attention weight visualizations from GRPO+Attention
3. Credit assignment quality metric (correlation analysis)
4. Summary table
"""

import os
import json
import glob
import csv
import argparse
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats as scipy_stats


def load_metrics(results_dir, algo_name, env_name):
    """Load training metrics CSV files for all seeds of a given algo/env pair."""
    pattern = os.path.join(results_dir, algo_name, env_name, "seed_*", "metrics.csv")
    files = sorted(glob.glob(pattern))

    all_data = []
    for f in files:
        data = {"timesteps": [], "mean_return": []}
        with open(f, "r") as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                data["timesteps"].append(int(row["timesteps"]))
                data["mean_return"].append(float(row["mean_return"]))
        all_data.append(data)

    return all_data


def load_results(results_dir, algo_name, env_name):
    """Load results.json files for all seeds."""
    pattern = os.path.join(results_dir, algo_name, env_name, "seed_*", "results.json")
    files = sorted(glob.glob(pattern))

    results = []
    for f in files:
        with open(f, "r") as fp:
            results.append(json.load(fp))
    return results


def load_attention_history(results_dir, env_name):
    """Load attention history for GRPO_Attention."""
    pattern = os.path.join(
        results_dir, "GRPO_Attention", env_name, "seed_*", "attention_history.json"
    )
    files = sorted(glob.glob(pattern))

    all_history = []
    for f in files:
        with open(f, "r") as fp:
            all_history.extend(json.load(fp))
    return all_history


def interpolate_to_common_x(all_data, num_points=200):
    """Interpolate metrics to common x-axis for averaging across seeds."""
    if not all_data:
        return None, None, None

    # Find common x range
    max_x = min(d["timesteps"][-1] for d in all_data if d["timesteps"])
    min_x = max(d["timesteps"][0] for d in all_data if d["timesteps"])

    common_x = np.linspace(min_x, max_x, num_points)
    interpolated_y = []

    for data in all_data:
        if len(data["timesteps"]) < 2:
            continue
        y_interp = np.interp(common_x, data["timesteps"], data["mean_return"])
        interpolated_y.append(y_interp)

    if not interpolated_y:
        return None, None, None

    interpolated_y = np.array(interpolated_y)
    mean_y = np.mean(interpolated_y, axis=0)
    std_y = np.std(interpolated_y, axis=0)

    return common_x, mean_y, std_y


ALGO_COLORS = {
    "REINFORCE_EMA": "#2196F3",
    "PPO": "#4CAF50",
    "GRPO_Attention": "#FF5722",
}

ALGO_LABELS = {
    "REINFORCE_EMA": "REINFORCE + EMA",
    "PPO": "PPO",
    "GRPO_Attention": "GRPO + Attention (Ours)",
}


def plot_learning_curves(results_dir, output_dir, environments, algorithms):
    """Plot learning curves for each environment."""
    os.makedirs(output_dir, exist_ok=True)

    for env_name in environments:
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))

        for algo_name in algorithms:
            all_data = load_metrics(results_dir, algo_name, env_name)
            if not all_data:
                continue

            x, mean_y, std_y = interpolate_to_common_x(all_data)
            if x is None:
                continue

            color = ALGO_COLORS.get(algo_name, "#999999")
            label = ALGO_LABELS.get(algo_name, algo_name)

            ax.plot(x, mean_y, color=color, label=label, linewidth=2)
            ax.fill_between(x, mean_y - std_y, mean_y + std_y, color=color, alpha=0.2)

        ax.set_xlabel("Training Timesteps", fontsize=12)
        ax.set_ylabel("Mean Episode Return (100-ep)", fontsize=12)
        ax.set_title(f"Learning Curves: {env_name}", fontsize=14)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)

        filepath = os.path.join(output_dir, f"learning_curve_{env_name}.png")
        fig.tight_layout()
        fig.savefig(filepath, dpi=150)
        plt.close(fig)
        print(f"Saved: {filepath}")


def plot_attention_weights(results_dir, output_dir, environments):
    """Visualize attention weights from GRPO+Attention."""
    os.makedirs(output_dir, exist_ok=True)

    for env_name in environments:
        history = load_attention_history(results_dir, env_name)
        if not history:
            continue

        # Group by label (training stage)
        labels = sorted(set(h["label"] for h in history))
        n_labels = len(labels)

        if n_labels == 0:
            continue

        fig, axes = plt.subplots(1, n_labels, figsize=(6 * n_labels, 4), squeeze=False)
        axes = axes[0]

        for idx, label in enumerate(labels):
            stage_episodes = [h for h in history if h["label"] == label]
            if not stage_episodes:
                continue

            # Pick first episode for this stage
            ep = stage_episodes[0]
            timesteps = ep["timesteps"]
            weights = ep["attention_weights"]
            rewards = ep["rewards"]

            ax = axes[idx]
            ax.bar(timesteps, weights, alpha=0.7, color="#FF5722", label="Attention Weight")

            # Overlay rewards if they have variation
            if max(rewards) != min(rewards):
                ax2 = ax.twinx()
                ax2.plot(timesteps, rewards, color="#2196F3", linewidth=1.5,
                         alpha=0.7, label="Reward")
                ax2.set_ylabel("Reward", color="#2196F3", fontsize=10)
                ax2.tick_params(axis="y", labelcolor="#2196F3")

            ax.set_xlabel("Timestep", fontsize=10)
            ax.set_ylabel("Attention Weight", fontsize=10)
            ax.set_title(f"{label} (return={ep['total_return']:.0f})", fontsize=11)

        fig.suptitle(f"Attention Weights: {env_name}", fontsize=13)
        fig.tight_layout()
        filepath = os.path.join(output_dir, f"attention_{env_name}.png")
        fig.savefig(filepath, dpi=150)
        plt.close(fig)
        print(f"Saved: {filepath}")


def compute_credit_quality(results_dir, output_dir):
    """Compute credit assignment quality for SparseCartPole.

    Uses temporal difference magnitude |V(s_t) - V(s_{t+1})| as importance proxy
    and measures rank correlation with attention weights.

    For SparseCartPole, since reward=0 everywhere except terminal,
    the true importance increases toward the end of the episode
    (later decisions are more 'pivotal' to survival).
    We use a simple linear proxy: importance(t) = t/T.
    """
    os.makedirs(output_dir, exist_ok=True)
    env_name = "SparseCartPole"

    history = load_attention_history(results_dir, env_name)
    if not history:
        print("No attention data for SparseCartPole credit quality analysis.")
        return

    # Use late-training episodes for quality measurement
    late_episodes = [h for h in history if "0.9" in h["label"] or h["label"].endswith("90")]
    if not late_episodes:
        # Fall back to any episodes
        late_episodes = history[-5:] if len(history) >= 5 else history

    correlations = []
    for ep in late_episodes:
        T = len(ep["timesteps"])
        if T < 3:
            continue

        weights = np.array(ep["attention_weights"])

        # Proxy: linear importance (later steps matter more in CartPole)
        importance = np.linspace(0, 1, T)

        # Rank correlation
        corr, p_value = scipy_stats.spearmanr(weights, importance)
        if not np.isnan(corr):
            correlations.append(corr)

    if correlations:
        mean_corr = np.mean(correlations)
        std_corr = np.std(correlations)
        print(f"\nCredit Quality (SparseCartPole):")
        print(f"  Spearman rank correlation (attention vs linear importance): "
              f"{mean_corr:.3f} +/- {std_corr:.3f}")
        print(f"  (Positive = attention concentrates on later, more important steps)")

        # Save
        quality = {
            "env": env_name,
            "mean_spearman": float(mean_corr),
            "std_spearman": float(std_corr),
            "n_episodes": len(correlations),
        }
        with open(os.path.join(output_dir, "credit_quality.json"), "w") as f:
            json.dump(quality, f, indent=2)


def print_summary_table(results_dir, environments, algorithms):
    """Print and save summary table of final returns."""
    rows = []
    print("\n" + "=" * 80)
    print("FINAL RESULTS SUMMARY")
    print("=" * 80)
    header = f"{'Environment':<20} {'Algorithm':<25} {'Mean Return':>15} {'Std':>10}"
    print(header)
    print("-" * 70)

    for env_name in environments:
        for algo_name in algorithms:
            results = load_results(results_dir, algo_name, env_name)
            if not results:
                print(f"{env_name:<20} {ALGO_LABELS.get(algo_name, algo_name):<25} {'N/A':>15}")
                continue

            means = [r["mean_final_return"] for r in results if "mean_final_return" in r]
            if means:
                overall_mean = np.mean(means)
                overall_std = np.std(means)
                label = ALGO_LABELS.get(algo_name, algo_name)
                print(f"{env_name:<20} {label:<25} {overall_mean:>15.1f} {overall_std:>10.1f}")
                rows.append({
                    "environment": env_name,
                    "algorithm": label,
                    "mean_return": round(float(overall_mean), 1),
                    "std_return": round(float(overall_std), 1),
                })

    # Save as JSON
    summary_path = os.path.join(results_dir, "summary_table.json")
    with open(summary_path, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"\nSummary saved to: {summary_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate visualizations")
    parser.add_argument("--results-dir", type=str, default="results",
                        help="Directory containing experiment results")
    parser.add_argument("--output-dir", type=str, default="plots",
                        help="Directory to save plots")
    args = parser.parse_args()

    environments = ["CartPole-v1", "Acrobot-v1", "LunarLander-v3", "SparseCartPole"]
    algorithms = ["REINFORCE_EMA", "PPO", "GRPO_Attention"]

    # Filter to environments/algorithms that have results
    available_envs = []
    for env in environments:
        for algo in algorithms:
            pattern = os.path.join(args.results_dir, algo, env, "seed_*", "metrics.csv")
            if glob.glob(pattern):
                if env not in available_envs:
                    available_envs.append(env)
                break

    if not available_envs:
        print("No results found. Run training first.")
        return

    print(f"Found results for environments: {available_envs}")

    # Generate plots
    plot_learning_curves(args.results_dir, args.output_dir, available_envs, algorithms)
    plot_attention_weights(args.results_dir, args.output_dir, available_envs)
    compute_credit_quality(args.results_dir, args.output_dir)
    print_summary_table(args.results_dir, available_envs, algorithms)

    print(f"\nAll plots saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
