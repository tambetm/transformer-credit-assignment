"""Main training script that runs all experiments.

Runs REINFORCE+EMA, PPO, and GRPO+Attention on all environments
with multiple seeds, saving results for visualization.
"""

import os
import sys
import json
import argparse
import numpy as np
import gymnasium as gym
import torch

from .envs.sparse_cartpole import SparseCartPole
from .algorithms.reinforce_ema import ReinforceEMA
from .algorithms.ppo import PPO
from .algorithms.grpo_attention import GRPOAttention
from .algorithms.rudder_transformer import RUDDERTransformerAlgo


# Default configuration
DEFAULT_CONFIG = {
    "num_seeds": 5,
    "total_timesteps": 200_000,
    "lr_policy": 3e-4,
    "lr_cat": 1e-3,
    "lr_value": 1e-3,
    "gamma": 0.99,
    "ema_alpha": 0.01,
    "ppo_clip": 0.2,
    "ppo_epochs": 4,
    "ppo_gae_lambda": 0.95,
    "batch_size": 2048,
    "episodes_per_update": 10,
    "device": "cpu",
}

# Environment definitions
ENVIRONMENTS = {
    "CartPole-v1": lambda: gym.make("CartPole-v1"),
    "Acrobot-v1": lambda: gym.make("Acrobot-v1"),
    "LunarLander-v3": lambda: gym.make("LunarLander-v3"),
    "SparseCartPole": lambda: SparseCartPole(gym.make("CartPole-v1")),
}

# Algorithm definitions
ALGORITHMS = {
    "REINFORCE_EMA": ReinforceEMA,
    "PPO": PPO,
    "GRPO_Attention": GRPOAttention,
    "RUDDER_Transformer": RUDDERTransformerAlgo,
}


def run_experiment(algo_name, env_name, env_fn, config, seed, output_dir):
    """Run a single experiment (one algorithm, one env, one seed)."""
    log_dir = os.path.join(output_dir, algo_name, env_name, f"seed_{seed}")
    os.makedirs(log_dir, exist_ok=True)

    algo_cls = ALGORITHMS[algo_name]
    agent = algo_cls(env_fn=env_fn, config=config, seed=seed, log_dir=log_dir)

    print(f"\n{'='*70}")
    print(f"Running {algo_name} on {env_name} (seed={seed})")
    print(f"{'='*70}")

    returns = agent.train()

    # Save results
    results = {
        "algorithm": algo_name,
        "environment": env_name,
        "seed": seed,
        "final_returns": returns[-100:] if len(returns) >= 100 else returns,
        "mean_final_return": float(np.mean(returns[-100:])) if returns else 0.0,
        "std_final_return": float(np.std(returns[-100:])) if returns else 0.0,
    }

    # Save attention history for GRPO_Attention
    if algo_name == "GRPO_Attention" and hasattr(agent, "attention_history"):
        attn_path = os.path.join(log_dir, "attention_history.json")
        with open(attn_path, "w") as f:
            json.dump(agent.attention_history, f)

    # Save value prediction history for RUDDER_Transformer
    if algo_name == "RUDDER_Transformer" and hasattr(agent, "value_history"):
        val_path = os.path.join(log_dir, "value_history.json")
        with open(val_path, "w") as f:
            json.dump(agent.value_history, f)

    results_path = os.path.join(log_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    return results


def main():
    parser = argparse.ArgumentParser(description="GRPO + Attention Credit Assignment Experiments")
    parser.add_argument("--output-dir", type=str, default="results",
                        help="Output directory for results")
    parser.add_argument("--envs", nargs="+", default=None,
                        help="Environments to run (default: all)")
    parser.add_argument("--algos", nargs="+", default=None,
                        help="Algorithms to run (default: all)")
    parser.add_argument("--seeds", type=int, default=None,
                        help="Number of seeds (default: from config)")
    parser.add_argument("--timesteps", type=int, default=None,
                        help="Total timesteps per experiment")
    parser.add_argument("--device", type=str, default="cpu",
                        choices=["cpu", "cuda"],
                        help="Device to use")
    args = parser.parse_args()

    config = DEFAULT_CONFIG.copy()
    if args.seeds is not None:
        config["num_seeds"] = args.seeds
    if args.timesteps is not None:
        config["total_timesteps"] = args.timesteps
    config["device"] = args.device

    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        config["device"] = "cpu"

    env_names = args.envs if args.envs else list(ENVIRONMENTS.keys())
    algo_names = args.algos if args.algos else list(ALGORITHMS.keys())

    # Validate
    for e in env_names:
        if e not in ENVIRONMENTS:
            print(f"Unknown environment: {e}. Available: {list(ENVIRONMENTS.keys())}")
            sys.exit(1)
    for a in algo_names:
        if a not in ALGORITHMS:
            print(f"Unknown algorithm: {a}. Available: {list(ALGORITHMS.keys())}")
            sys.exit(1)

    # Save config
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    # Run all experiments
    all_results = []
    total_runs = len(algo_names) * len(env_names) * config["num_seeds"]
    run_idx = 0

    for env_name in env_names:
        env_fn = ENVIRONMENTS[env_name]
        for algo_name in algo_names:
            for seed in range(config["num_seeds"]):
                run_idx += 1
                print(f"\n>>> Run {run_idx}/{total_runs}")
                try:
                    result = run_experiment(
                        algo_name, env_name, env_fn, config, seed, args.output_dir
                    )
                    all_results.append(result)
                except Exception as e:
                    print(f"ERROR in {algo_name}/{env_name}/seed_{seed}: {e}")
                    import traceback
                    traceback.print_exc()
                    all_results.append({
                        "algorithm": algo_name,
                        "environment": env_name,
                        "seed": seed,
                        "error": str(e),
                    })

    # Save summary
    summary_path = os.path.join(args.output_dir, "all_results.json")
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)

    # Print summary table
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"{'Environment':<20} {'Algorithm':<20} {'Mean Return':>15} {'Std':>10}")
    print("-" * 65)

    for env_name in env_names:
        for algo_name in algo_names:
            env_algo_results = [
                r for r in all_results
                if r.get("environment") == env_name
                and r.get("algorithm") == algo_name
                and "error" not in r
            ]
            if env_algo_results:
                means = [r["mean_final_return"] for r in env_algo_results]
                overall_mean = np.mean(means)
                overall_std = np.std(means)
                print(f"{env_name:<20} {algo_name:<20} {overall_mean:>15.1f} {overall_std:>10.1f}")
            else:
                print(f"{env_name:<20} {algo_name:<20} {'FAILED':>15}")

    print("\nResults saved to:", args.output_dir)


if __name__ == "__main__":
    main()
