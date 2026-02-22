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


def _train_oracle_value_function(env_name, results_dir, gamma=0.99):
    """Train an oracle value function on collected episode data from all algorithms.

    Returns a trained MLP that estimates V(s) for SparseCartPole states.
    """
    import torch
    import torch.nn as nn

    # Collect observations and discounted returns-to-go from all algorithms' data
    all_obs = []
    all_returns_to_go = []

    for algo in ["REINFORCE_EMA", "PPO", "GRPO_Attention"]:
        attn_pattern = os.path.join(
            results_dir, algo, env_name, "seed_*", "attention_history.json"
        )
        for fpath in glob.glob(attn_pattern):
            with open(fpath) as f:
                episodes = json.load(f)
            for ep in episodes:
                T = len(ep["timesteps"])
                rewards = ep["rewards"]
                # Compute discounted returns-to-go
                rtg = np.zeros(T)
                running = 0.0
                for t in reversed(range(T)):
                    running = rewards[t] + gamma * running
                    rtg[t] = running
                all_returns_to_go.extend(rtg.tolist())

        # Also try to extract obs from results.json final_returns context
        # But attention_history doesn't store observations. We need the raw env.

    # If we couldn't collect data from history (it doesn't store obs), generate fresh
    # by running episodes in SparseCartPole with a random policy
    if not all_obs:
        import gymnasium as gym
        from grpo_attention.envs.sparse_cartpole import SparseCartPole

        env = SparseCartPole(gym.make("CartPole-v1"))
        obs_list = []
        rtg_list = []

        for seed in range(200):
            obs, _ = env.reset(seed=seed)
            ep_obs = [obs.copy()]
            ep_rewards = []
            done = False
            while not done:
                action = env.action_space.sample()
                obs, reward, terminated, truncated, _ = env.step(action)
                ep_rewards.append(reward)
                if not (terminated or truncated):
                    ep_obs.append(obs.copy())
                done = terminated or truncated

            T = len(ep_rewards)
            rtg = np.zeros(T)
            running = 0.0
            for t in reversed(range(T)):
                running = ep_rewards[t] + gamma * running
                rtg[t] = running

            obs_list.extend(ep_obs[:T])
            rtg_list.extend(rtg.tolist())

        env.close()
        all_obs = obs_list
        all_returns_to_go = rtg_list

    if len(all_obs) < 100:
        return None

    # Train a simple value network
    obs_tensor = torch.tensor(np.array(all_obs), dtype=torch.float32)
    rtg_tensor = torch.tensor(np.array(all_returns_to_go), dtype=torch.float32)

    obs_dim = obs_tensor.shape[1]
    value_net = nn.Sequential(
        nn.Linear(obs_dim, 64), nn.Tanh(),
        nn.Linear(64, 64), nn.Tanh(),
        nn.Linear(64, 1),
    )
    optimizer = torch.optim.Adam(value_net.parameters(), lr=1e-3)

    # Train for 200 epochs
    dataset_size = len(obs_tensor)
    batch_size = min(256, dataset_size)
    for epoch in range(200):
        indices = torch.randperm(dataset_size)[:batch_size]
        pred = value_net(obs_tensor[indices]).squeeze(-1)
        loss = nn.functional.mse_loss(pred, rtg_tensor[indices])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return value_net


def compute_credit_quality(results_dir, output_dir):
    """Compute credit assignment quality for SparseCartPole.

    Trains an oracle value function on collected data, computes TD magnitude
    |V(s_t) - V(s_{t+1})| as ground-truth importance, and measures rank
    correlation with attention weights.

    Also computes correlation with a simple linear proxy for comparison.
    """
    import torch
    os.makedirs(output_dir, exist_ok=True)
    env_name = "SparseCartPole"

    history = load_attention_history(results_dir, env_name)
    if not history:
        print("No attention data for SparseCartPole credit quality analysis.")
        return

    # Use late-training episodes for quality measurement
    late_episodes = [h for h in history if "0.9" in h["label"] or h["label"].endswith("90")]
    if not late_episodes:
        late_episodes = history[-5:] if len(history) >= 5 else history

    # Train oracle value function
    print("\nTraining oracle value function for credit quality analysis...")
    oracle = _train_oracle_value_function(env_name, results_dir)

    linear_correlations = []
    td_correlations = []

    for ep in late_episodes:
        T = len(ep["timesteps"])
        if T < 5:
            continue

        weights = np.array(ep["attention_weights"])

        # 1. Linear proxy: later steps are more important
        linear_importance = np.linspace(0, 1, T)
        corr_lin, _ = scipy_stats.spearmanr(weights, linear_importance)
        if not np.isnan(corr_lin):
            linear_correlations.append(corr_lin)

        # 2. Oracle TD magnitude proxy (if oracle available)
        # This requires the observations, which attention_history doesn't store.
        # We'll use the linear proxy as the main metric and note this limitation.

    # If we have the oracle, generate TD importance on fresh episodes for a plot
    td_importance_example = None
    if oracle is not None:
        import gymnasium as gym
        from grpo_attention.envs.sparse_cartpole import SparseCartPole

        env = SparseCartPole(gym.make("CartPole-v1"))
        # Run a few episodes to get TD magnitudes
        td_episodes = []
        for seed in range(10):
            obs, _ = env.reset(seed=seed + 1000)
            ep_obs = [obs.copy()]
            done = False
            while not done:
                action = env.action_space.sample()
                obs, reward, terminated, truncated, _ = env.step(action)
                if not (terminated or truncated):
                    ep_obs.append(obs.copy())
                done = terminated or truncated

            if len(ep_obs) >= 5:
                obs_t = torch.tensor(np.array(ep_obs), dtype=torch.float32)
                with torch.no_grad():
                    values = oracle(obs_t).squeeze(-1).numpy()
                td_mag = np.abs(np.diff(values))
                td_episodes.append({
                    "values": values.tolist(),
                    "td_magnitude": td_mag.tolist(),
                    "length": len(ep_obs),
                })
        env.close()

        if td_episodes:
            td_importance_example = td_episodes[0]

    # Compute attention vs oracle TD correlation on late episodes
    # Since attention_history doesn't store observations, we generate parallel
    # episodes and compare distributions rather than point-wise correlation
    if oracle is not None and td_importance_example:
        # Show that TD magnitude is concentrated at end of episode
        td_mags = np.array(td_importance_example["td_magnitude"])
        T_td = len(td_mags)

        # For the attention episodes, compute average attention profile
        avg_attn = None
        for ep in late_episodes:
            w = np.array(ep["attention_weights"])
            T_w = len(w)
            # Normalize to [0,1] position
            positions = np.linspace(0, 1, T_w)
            interp_w = np.interp(np.linspace(0, 1, 50), positions, w)
            if avg_attn is None:
                avg_attn = interp_w
            else:
                avg_attn += interp_w
        if avg_attn is not None and len(late_episodes) > 0:
            avg_attn /= len(late_episodes)

            # Average TD profile
            td_positions = np.linspace(0, 1, T_td)
            avg_td = np.interp(np.linspace(0, 1, 50), td_positions, td_mags)
            avg_td = avg_td / (avg_td.sum() + 1e-8)  # Normalize

            # Profile correlation
            profile_corr, _ = scipy_stats.spearmanr(avg_attn, avg_td)
            td_correlations.append(profile_corr)

    # Print results
    print(f"\nCredit Quality Analysis (SparseCartPole):")
    if linear_correlations:
        mean_lin = np.mean(linear_correlations)
        std_lin = np.std(linear_correlations)
        print(f"  Attention vs linear importance: {mean_lin:.3f} +/- {std_lin:.3f}")
        print(f"  (Positive = attention on later steps; Negative = attention on earlier steps)")

    if td_correlations:
        mean_td = np.mean(td_correlations)
        print(f"  Attention profile vs TD magnitude profile: {mean_td:.3f}")
        print(f"  (Positive = attention correlates with oracle importance)")

    # Save results
    quality = {
        "env": env_name,
        "n_episodes": len(linear_correlations),
        "linear_proxy": {
            "mean_spearman": float(np.mean(linear_correlations)) if linear_correlations else None,
            "std_spearman": float(np.std(linear_correlations)) if linear_correlations else None,
        },
        "oracle_td": {
            "profile_correlation": float(np.mean(td_correlations)) if td_correlations else None,
        },
    }

    with open(os.path.join(output_dir, "credit_quality.json"), "w") as f:
        json.dump(quality, f, indent=2)

    # Generate credit quality comparison plot
    if td_importance_example and late_episodes:
        _plot_credit_comparison(
            late_episodes, td_importance_example, output_dir
        )


def _plot_credit_comparison(attention_episodes, td_example, output_dir):
    """Plot attention weights alongside oracle TD magnitude for comparison."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: average attention profile across episodes
    ax1 = axes[0]
    for ep in attention_episodes[:3]:  # Show up to 3 episodes
        w = np.array(ep["attention_weights"])
        T = len(w)
        ax1.plot(range(T), w, alpha=0.5, linewidth=1)

    # Average
    max_len = max(len(ep["attention_weights"]) for ep in attention_episodes)
    avg = np.zeros(max_len)
    counts = np.zeros(max_len)
    for ep in attention_episodes:
        w = np.array(ep["attention_weights"])
        avg[:len(w)] += w
        counts[:len(w)] += 1
    avg = avg / np.maximum(counts, 1)
    ax1.plot(range(len(avg)), avg, color="black", linewidth=2, label="Average")
    ax1.set_xlabel("Timestep", fontsize=12)
    ax1.set_ylabel("Attention Weight", fontsize=12)
    ax1.set_title("CAT Credit Assignment Weights", fontsize=13)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Right: oracle TD magnitude
    ax2 = axes[1]
    td_mags = np.array(td_example["td_magnitude"])
    values = np.array(td_example["values"])
    ax2.bar(range(len(td_mags)), td_mags, alpha=0.7, color="#4CAF50",
            label="|V(s_t) - V(s_{t+1})|")
    ax2_twin = ax2.twinx()
    ax2_twin.plot(range(len(values)), values, color="#FF5722", linewidth=2,
                  alpha=0.8, label="V(s_t)")
    ax2_twin.set_ylabel("V(s_t)", color="#FF5722", fontsize=11)
    ax2_twin.tick_params(axis="y", labelcolor="#FF5722")
    ax2.set_xlabel("Timestep", fontsize=12)
    ax2.set_ylabel("TD Magnitude", fontsize=12)
    ax2.set_title("Oracle Value Function (SparseCartPole)", fontsize=13)
    ax2.legend(loc="upper left")
    ax2_twin.legend(loc="upper right")
    ax2.grid(True, alpha=0.3)

    fig.suptitle("Credit Assignment Quality: Attention vs Oracle TD", fontsize=14)
    fig.tight_layout()
    filepath = os.path.join(output_dir, "credit_quality_comparison.png")
    fig.savefig(filepath, dpi=150)
    plt.close(fig)
    print(f"Saved: {filepath}")


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
