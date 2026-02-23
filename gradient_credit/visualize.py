"""Visualization script for gradient credit assignment experiments.

Generates:
1. Learning curves for each environment (7 algorithms)
2. Credit pattern comparison (value diffs vs gradient methods) for SparseCartPole
3. Credit evolution over training (heatmaps)
4. Predictor quality vs policy performance scatter
5. Summary table
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


ALGO_COLORS = {
    "REINFORCE_EMA": "#9E9E9E",      # Gray
    "PPO": "#2196F3",                  # Blue
    "RUDDER_ValueDiff": "#E91E63",     # Pink dashed
    "Gradient_Norm": "#F44336",        # Red
    "Gradient_Input": "#FF9800",       # Orange
    "Integrated_Gradients": "#4CAF50", # Green
    "Hybrid_Credit": "#9C27B0",        # Purple
}

ALGO_LABELS = {
    "REINFORCE_EMA": "REINFORCE + EMA",
    "PPO": "PPO",
    "RUDDER_ValueDiff": "RUDDER Value Diff",
    "Gradient_Norm": "Gradient Norm",
    "Gradient_Input": "Gradient x Input",
    "Integrated_Gradients": "Integrated Gradients",
    "Hybrid_Credit": "Hybrid (Grad + VD)",
}

ALGO_LINESTYLES = {
    "REINFORCE_EMA": "-",
    "PPO": "-",
    "RUDDER_ValueDiff": "--",
    "Gradient_Norm": "-",
    "Gradient_Input": "-",
    "Integrated_Gradients": "-",
    "Hybrid_Credit": "-",
}


def load_metrics(results_dir, algo_name, env_name):
    pattern = os.path.join(results_dir, algo_name, env_name, "seed_*", "metrics.csv")
    files = sorted(glob.glob(pattern))
    all_data = []
    for f in files:
        data = {"timesteps": [], "mean_return": []}
        extra_keys = []
        with open(f, "r") as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                data["timesteps"].append(int(row["timesteps"]))
                data["mean_return"].append(float(row["mean_return"]))
                for key in row:
                    if key not in ("timesteps", "mean_return"):
                        if key not in data:
                            data[key] = []
                            extra_keys.append(key)
                        data[key].append(float(row[key]))
        all_data.append(data)
    return all_data


def load_results(results_dir, algo_name, env_name):
    pattern = os.path.join(results_dir, algo_name, env_name, "seed_*", "results.json")
    files = sorted(glob.glob(pattern))
    results = []
    for f in files:
        with open(f, "r") as fp:
            results.append(json.load(fp))
    return results


def load_credit_history(results_dir, algo_name, env_name):
    pattern = os.path.join(results_dir, algo_name, env_name, "seed_*", "credit_history.json")
    files = sorted(glob.glob(pattern))
    all_history = []
    for f in files:
        with open(f, "r") as fp:
            all_history.extend(json.load(fp))
    return all_history


def interpolate_to_common_x(all_data, num_points=200):
    if not all_data:
        return None, None, None
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
    return common_x, np.mean(interpolated_y, axis=0), np.std(interpolated_y, axis=0)


def plot_learning_curves(results_dir, output_dir, environments, algorithms):
    """Plot learning curves for each environment with all 7 algorithms."""
    os.makedirs(output_dir, exist_ok=True)

    for env_name in environments:
        fig, ax = plt.subplots(1, 1, figsize=(12, 7))

        for algo_name in algorithms:
            all_data = load_metrics(results_dir, algo_name, env_name)
            if not all_data:
                continue
            x, mean_y, std_y = interpolate_to_common_x(all_data)
            if x is None:
                continue

            color = ALGO_COLORS.get(algo_name, "#999999")
            label = ALGO_LABELS.get(algo_name, algo_name)
            linestyle = ALGO_LINESTYLES.get(algo_name, "-")

            ax.plot(x, mean_y, color=color, label=label, linewidth=2, linestyle=linestyle)
            ax.fill_between(x, mean_y - std_y, mean_y + std_y, color=color, alpha=0.12)

        ax.set_xlabel("Training Timesteps", fontsize=12)
        ax.set_ylabel("Mean Episode Return (100-ep)", fontsize=12)
        ax.set_title(f"Learning Curves: {env_name}", fontsize=14)
        ax.legend(fontsize=9, loc="best")
        ax.grid(True, alpha=0.3)

        filepath = os.path.join(output_dir, f"learning_curve_{env_name}.png")
        fig.tight_layout()
        fig.savefig(filepath, dpi=150)
        plt.close(fig)
        print(f"Saved: {filepath}")


def plot_credit_patterns(results_dir, output_dir, environments, gradient_algos):
    """Side-by-side credit pattern comparison for each environment.

    Shows value diffs, gradient norm, gradient x input, and IG credit
    for the same environment at late training.
    """
    os.makedirs(output_dir, exist_ok=True)

    credit_algos = ["RUDDER_ValueDiff"] + gradient_algos

    for env_name in environments:
        histories = {}
        for algo_name in credit_algos:
            h = load_credit_history(results_dir, algo_name, env_name)
            if h:
                histories[algo_name] = h

        if len(histories) < 2:
            continue

        n_algos = len(histories)
        fig, axes = plt.subplots(1, n_algos, figsize=(5 * n_algos, 4), squeeze=False)
        axes = axes[0]

        for idx, (algo_name, history) in enumerate(histories.items()):
            # Use late-training episodes
            late_eps = [h for h in history if "0.9" in h["label"]]
            if not late_eps:
                late_eps = history[-3:] if len(history) >= 3 else history
            if not late_eps:
                continue

            ep = late_eps[0]
            timesteps = ep["timesteps"]
            credit = np.array(ep["credit_values"])

            ax = axes[idx]
            colors_bar = ["#4CAF50" if c >= 0 else "#F44336" for c in credit]
            ax.bar(timesteps, credit, color=colors_bar, alpha=0.7)
            ax.axhline(y=0, color="black", linewidth=0.5)
            ax.set_xlabel("Timestep", fontsize=10)
            ax.set_ylabel("Credit", fontsize=10)
            label = ALGO_LABELS.get(algo_name, algo_name)
            ax.set_title(f"{label}\n(return={ep['total_return']:.0f})", fontsize=10)
            ax.grid(True, alpha=0.3)

        fig.suptitle(f"Credit Pattern Comparison: {env_name}", fontsize=13)
        fig.tight_layout()
        filepath = os.path.join(output_dir, f"credit_patterns_{env_name}.png")
        fig.savefig(filepath, dpi=150)
        plt.close(fig)
        print(f"Saved: {filepath}")


def plot_credit_evolution(results_dir, output_dir, environments, gradient_algos):
    """Heatmap: how credit distribution changes during training.

    X-axis: relative position in episode, Y-axis: training stage, Color: credit magnitude.
    """
    os.makedirs(output_dir, exist_ok=True)

    for env_name in environments:
        for algo_name in gradient_algos:
            history = load_credit_history(results_dir, algo_name, env_name)
            if not history:
                continue

            labels = sorted(set(h["label"] for h in history))
            if len(labels) < 2:
                continue

            num_bins = 20
            heatmap = np.zeros((len(labels), num_bins))

            for row, label in enumerate(labels):
                stage_eps = [h for h in history if h["label"] == label]
                bin_vals = [[] for _ in range(num_bins)]

                for ep in stage_eps:
                    credit = np.abs(np.array(ep["credit_values"]))
                    T = len(credit)
                    if T < 2:
                        continue
                    for t in range(T):
                        bin_idx = min(int(t / T * num_bins), num_bins - 1)
                        bin_vals[bin_idx].append(credit[t])

                for b in range(num_bins):
                    if bin_vals[b]:
                        heatmap[row, b] = np.mean(bin_vals[b])

            fig, ax = plt.subplots(1, 1, figsize=(10, 4))
            im = ax.imshow(heatmap, aspect='auto', cmap='hot', interpolation='nearest')
            ax.set_xlabel("Relative Episode Position", fontsize=11)
            ax.set_ylabel("Training Stage", fontsize=11)
            ax.set_xticks(np.arange(0, num_bins, 4))
            ax.set_xticklabels([f"{i/num_bins:.1f}" for i in range(0, num_bins, 4)])
            ax.set_yticks(range(len(labels)))
            ax.set_yticklabels([l.replace("step_", "") for l in labels], fontsize=9)
            fig.colorbar(im, ax=ax, label="|Credit|")

            algo_label = ALGO_LABELS.get(algo_name, algo_name)
            ax.set_title(f"Credit Evolution: {algo_label} on {env_name}", fontsize=12)
            fig.tight_layout()
            filepath = os.path.join(output_dir, f"credit_evolution_{algo_name}_{env_name}.png")
            fig.savefig(filepath, dpi=150)
            plt.close(fig)
            print(f"Saved: {filepath}")


def plot_predictor_vs_performance(results_dir, output_dir, environments, predictor_algos):
    """Scatter: predictor MSE vs policy performance.

    Tests hypothesis that gradient credit works even when predictions are inaccurate.
    """
    os.makedirs(output_dir, exist_ok=True)

    for env_name in environments:
        fig, ax = plt.subplots(1, 1, figsize=(10, 7))
        has_data = False

        for algo_name in predictor_algos:
            all_data = load_metrics(results_dir, algo_name, env_name)
            if not all_data:
                continue

            for seed_data in all_data:
                if "predictor_mse" not in seed_data or "mean_return" not in seed_data:
                    continue
                mse_vals = seed_data["predictor_mse"]
                return_vals = seed_data["mean_return"]
                n = min(len(mse_vals), len(return_vals))
                if n < 5:
                    continue

                color = ALGO_COLORS.get(algo_name, "#999999")
                label = ALGO_LABELS.get(algo_name, algo_name)
                ax.scatter(mse_vals[:n], return_vals[:n], c=color, alpha=0.3,
                           s=15, label=label)
                has_data = True

        if has_data:
            # Deduplicate legend
            handles, labels = ax.get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            ax.legend(by_label.values(), by_label.keys(), fontsize=9)

            ax.set_xlabel("Predictor MSE (normalized)", fontsize=12)
            ax.set_ylabel("Mean Episode Return", fontsize=12)
            ax.set_title(f"Predictor Quality vs Policy Performance: {env_name}", fontsize=13)
            ax.grid(True, alpha=0.3)

            filepath = os.path.join(output_dir, f"predictor_vs_performance_{env_name}.png")
            fig.tight_layout()
            fig.savefig(filepath, dpi=150)
            print(f"Saved: {filepath}")
        plt.close(fig)


def print_summary_table(results_dir, environments, algorithms):
    """Print and save summary table."""
    rows = []
    print("\n" + "=" * 100)
    print("FINAL RESULTS SUMMARY")
    print("=" * 100)
    header = f"{'Environment':<20} {'Algorithm':<25} {'Mean Return':>15} {'Std':>10}"
    print(header)
    print("-" * 70)

    for env_name in environments:
        for algo_name in algorithms:
            results = load_results(results_dir, algo_name, env_name)
            if not results:
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
                    "algo_key": algo_name,
                    "mean_return": round(float(overall_mean), 1),
                    "std_return": round(float(overall_std), 1),
                })

    summary_path = os.path.join(results_dir, "summary_table.json")
    with open(summary_path, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"\nSummary saved to: {summary_path}")


def print_timing_table(results_dir, algorithms):
    """Print timing comparison."""
    print("\n" + "=" * 60)
    print("TIMING: Mean Update Time (seconds)")
    print("=" * 60)

    for algo_name in algorithms:
        pattern = os.path.join(results_dir, algo_name, "*", "seed_*", "results.json")
        files = glob.glob(pattern)
        times = []
        for f in files:
            with open(f) as fp:
                r = json.load(fp)
            if "mean_update_time" in r:
                times.append(r["mean_update_time"])
        if times:
            label = ALGO_LABELS.get(algo_name, algo_name)
            mean_t = np.mean(times)
            print(f"  {label:<25} {mean_t:.4f}s")


def main():
    parser = argparse.ArgumentParser(description="Generate visualizations")
    parser.add_argument("--results-dir", type=str, default="results_gradient")
    parser.add_argument("--output-dir", type=str, default="plots_gradient")
    args = parser.parse_args()

    environments = ["CartPole-v1", "Acrobot-v1", "LunarLander-v3", "SparseCartPole"]
    algorithms = list(ALGO_COLORS.keys())

    gradient_algos = ["Gradient_Norm", "Gradient_Input", "Integrated_Gradients", "Hybrid_Credit"]
    predictor_algos = ["RUDDER_ValueDiff"] + gradient_algos

    # Filter to available envs
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

    plot_learning_curves(args.results_dir, args.output_dir, available_envs, algorithms)
    plot_credit_patterns(args.results_dir, args.output_dir, available_envs, gradient_algos)
    plot_credit_evolution(args.results_dir, args.output_dir, available_envs, gradient_algos)
    plot_predictor_vs_performance(args.results_dir, args.output_dir, available_envs, predictor_algos)
    print_summary_table(args.results_dir, available_envs, algorithms)
    print_timing_table(args.results_dir, algorithms)

    print(f"\nAll plots saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
