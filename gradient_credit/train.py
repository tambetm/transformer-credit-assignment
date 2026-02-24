"""Main training script for gradient-based credit assignment experiments.

Runs 9 algorithms on 2 environments with multiple seeds.
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
from .algorithms.rudder_value_diff import RUDDERValueDiffAlgo
from .algorithms.gradient_norm_credit import GradientNormCreditAlgo
from .algorithms.gradient_input_credit import GradientInputCreditAlgo
from .algorithms.integrated_gradients_credit import IntegratedGradientsCreditAlgo
from .algorithms.hybrid_credit import HybridCreditAlgo
from .algorithms.gsvd_credit import GSVDCreditAlgo
from .algorithms.attention_rollout_credit import AttentionRolloutCreditAlgo


DEFAULT_CONFIG = {
    "num_seeds": 5,
    "total_timesteps": 200_000,
    "lr_policy": 3e-4,
    "lr_predictor": 5e-4,
    "lr_value": 1e-3,
    "gamma": 0.99,
    "ema_alpha": 0.01,
    "ppo_clip": 0.2,
    "ppo_epochs": 4,
    "ppo_gae_lambda": 0.95,
    "batch_size": 2048,
    "episodes_per_update": 10,
    "predictor_d_model": 32,
    "predictor_nhead": 2,
    "predictor_layers": 2,
    "predictor_d_ff": 64,
    "predictor_dropout": 0.1,
    "replay_buffer_size": 300,
    "predictor_train_steps": 3,
    "integrated_grad_steps": 10,
    "device": "cpu",
}

ENVIRONMENTS = {
    "CartPole-v1": lambda: gym.make("CartPole-v1"),
    "SparseCartPole": lambda: SparseCartPole(gym.make("CartPole-v1")),
}

ALGORITHMS = {
    "REINFORCE_EMA": ReinforceEMA,
    "PPO": PPO,
    "RUDDER_ValueDiff": RUDDERValueDiffAlgo,
    "Gradient_Norm": GradientNormCreditAlgo,
    "Gradient_Input": GradientInputCreditAlgo,
    "Integrated_Gradients": IntegratedGradientsCreditAlgo,
    "Hybrid_Credit": HybridCreditAlgo,
    "GSVD": GSVDCreditAlgo,
    "Attention_Rollout": AttentionRolloutCreditAlgo,
}


def run_experiment(algo_name, env_name, env_fn, config, seed, output_dir):
    """Run a single experiment."""
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

    # Save credit history for gradient-based methods
    if hasattr(agent, "credit_history") and agent.credit_history:
        credit_path = os.path.join(log_dir, "credit_history.json")
        with open(credit_path, "w") as f:
            json.dump(agent.credit_history, f)

    # Save timing info
    if hasattr(agent, "update_times") and agent.update_times:
        results["mean_update_time"] = float(np.mean(agent.update_times))
        results["total_update_time"] = float(np.sum(agent.update_times))

    results_path = os.path.join(log_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    return results


def main():
    parser = argparse.ArgumentParser(description="Gradient Credit Assignment Experiments")
    parser.add_argument("--output-dir", type=str, default="results_gradient",
                        help="Output directory for results")
    parser.add_argument("--envs", nargs="+", default=None)
    parser.add_argument("--algos", nargs="+", default=None)
    parser.add_argument("--seeds", type=int, default=None)
    parser.add_argument("--timesteps", type=int, default=None)
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
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

    for e in env_names:
        if e not in ENVIRONMENTS:
            print(f"Unknown environment: {e}. Available: {list(ENVIRONMENTS.keys())}")
            sys.exit(1)
    for a in algo_names:
        if a not in ALGORITHMS:
            print(f"Unknown algorithm: {a}. Available: {list(ALGORITHMS.keys())}")
            sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

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
                        algo_name, env_name, env_fn, config, seed, args.output_dir)
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

    summary_path = os.path.join(args.output_dir, "all_results.json")
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)

    # Print summary table
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"{'Environment':<20} {'Algorithm':<25} {'Mean Return':>15} {'Std':>10}")
    print("-" * 70)

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
                print(f"{env_name:<20} {algo_name:<25} {overall_mean:>15.1f} {overall_std:>10.1f}")
            else:
                print(f"{env_name:<20} {algo_name:<25} {'FAILED':>15}")

    # Print timing summary
    print("\n" + "=" * 80)
    print("TIMING (mean update time in seconds)")
    print("=" * 80)
    for algo_name in algo_names:
        algo_results = [r for r in all_results if r.get("algorithm") == algo_name and "mean_update_time" in r]
        if algo_results:
            mean_time = np.mean([r["mean_update_time"] for r in algo_results])
            print(f"  {algo_name:<25} {mean_time:.4f}s per update")

    print(f"\nResults saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
