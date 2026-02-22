# GRPO + Attention-Based Credit Assignment

Validating a novel RL algorithm that combines **GRPO-style critic-free advantage estimation** with **attention-based credit assignment** using a lightweight "Credit Assignment Transformer" (CAT).

## Core Idea

Standard GRPO computes episode-level advantages:
```
A_episode = (G - μ_EMA) / σ_EMA
```

Applying a single scalar to all timesteps provides poor credit assignment. We train a lightweight transformer that predicts episode returns from state-action sequences and use its cross-attention weights to redistribute the advantage:
```
w_t = credit_attention([RETURN] → (s_t, a_t))     # from the CAT
w_blended_t = (1-λ) * (1/T) + λ * w_t             # mix with uniform (warmup)
Â_t = w_blended_t * T * A_episode                  # per-timestep advantage
```

## Algorithms Compared

1. **REINFORCE + EMA** — Uniform episode advantage, no per-step credit assignment
2. **PPO** — Learned value function with GAE (strong baseline)
3. **GRPO + Attention** — Our method: attention-based credit assignment without a critic

## Environments

1. **CartPole-v1** — Sanity check (dense reward)
2. **Acrobot-v1** — Sparse reward, moderate difficulty
3. **LunarLander-v3** — Shaped reward, coordinated behavior
4. **SparseCartPole** — CartPole with terminal-only reward (key test)

## Usage

```bash
pip install -r requirements.txt

# Run all experiments
python -m grpo_attention.train --output-dir results

# Run specific experiments
python -m grpo_attention.train --envs CartPole-v1 SparseCartPole --algos GRPO_Attention PPO --seeds 3

# Generate plots
python -m grpo_attention.visualize --results-dir results --output-dir plots
```

## Results

All experiments run with 5 seeds, 200k timesteps each. Mean ± std of final 100-episode returns:

| Environment    | REINFORCE + EMA   | PPO              | GRPO + Attention (Ours) |
|----------------|-------------------|------------------|-------------------------|
| CartPole-v1    | 223.5 ± 5.7      | **490.2 ± 6.0**  | 171.2 ± 48.9           |
| Acrobot-v1     | -279.7 ± 180.2   | **-88.5 ± 3.9**  | -351.9 ± 162.3         |
| LunarLander-v3 | -111.0 ± 8.5     | **33.2 ± 40.5**  | -124.7 ± 18.7          |
| SparseCartPole | **103.8 ± 6.7**  | 42.5 ± 4.2      | 91.5 ± 17.8            |

### Key Findings

- **SparseCartPole** (terminal-only reward) — the key test: Both critic-free methods (REINFORCE+EMA and GRPO+Attention) significantly outperform PPO (~100 vs ~42). PPO's value function struggles to learn meaningful value estimates with purely sparse rewards. GRPO+Attention reaches ~91, comparable to REINFORCE's ~104.
- **CartPole-v1**: PPO solves the environment (~490); both critic-free methods learn but converge more slowly without a value function critic. GRPO+Attention shows higher seed variance than REINFORCE.
- **Acrobot-v1**: PPO converges reliably while both critic-free methods show high variance across seeds, with some seeds not converging within 200k steps.
- **LunarLander-v3**: PPO learns a positive-return policy; both critic-free methods plateau around -110 to -125.
- The attention weights from the CAT show non-uniform patterns: higher weights on early timesteps and near episode end, with characteristic peaks around critical decision points.

### Architecture Evolution

The CAT was iteratively improved:
1. **v1**: State-only input, self-attention weights, 1 layer — matched REINFORCE but no improvement
2. **v2**: Added action embeddings, dedicated cross-attention credit head, learnable temperature, 2 layers, normalized return targets — initially collapsed (attention too peaked), fixed via uniform-weight mixing with warmup schedule

### Limitations & Discussion

The core challenge: the CAT's attention weights reflect *what predicts returns* (correlation), not *what caused returns* (causation). In CartPole, early states with small pole angles are predictive of long episodes, so the CAT attends to them — but the causally important actions are near the end where a wrong move causes failure. Bridging this prediction-causation gap is a fundamental challenge for attention-based credit assignment and a direction for future work.

## Project Structure

```
grpo_attention/
├── envs/sparse_cartpole.py         # Sparse reward wrapper
├── models/
│   ├── policy.py                   # Shared policy network (MLP)
│   ├── value.py                    # Value network (PPO only)
│   └── credit_transformer.py       # Credit Assignment Transformer
├── algorithms/
│   ├── reinforce_ema.py            # REINFORCE + EMA baseline
│   ├── ppo.py                      # PPO implementation
│   └── grpo_attention.py           # GRPO + Attention (ours)
├── utils/
│   ├── buffer.py                   # Trajectory storage
│   ├── ema.py                      # EMA statistics tracker
│   └── logger.py                   # CSV logging
├── train.py                        # Main training script
└── visualize.py                    # Plot generation
```
