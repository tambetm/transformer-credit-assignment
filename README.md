# GRPO + Attention-Based Credit Assignment

Validating a novel RL algorithm that combines **GRPO-style critic-free advantage estimation** with **attention-based credit assignment** using a lightweight "Credit Assignment Transformer" (CAT).

## Core Idea

Standard GRPO computes episode-level advantages:
```
A_episode = (G - μ_EMA) / σ_EMA
```

Applying a single scalar to all timesteps provides poor credit assignment. We train a lightweight transformer that predicts episode returns and use its attention weights to redistribute the advantage:
```
w_t = attention_weight([RETURN] → s_t)    # from the CAT
Â_t = w_t * T * A_episode                  # per-timestep advantage
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

| Environment    | REINFORCE + EMA | PPO          | GRPO + Attention (Ours) |
|----------------|-----------------|--------------|-------------------------|
| CartPole-v1    | 223.5 ± 5.7    | **490.2 ± 6.0** | 196.8 ± 19.2        |
| Acrobot-v1     | -279.7 ± 180.2 | **-88.5 ± 3.9** | -333.3 ± 175.1     |
| LunarLander-v3 | -111.0 ± 8.5   | **33.2 ± 40.5** | -118.9 ± 8.7       |
| SparseCartPole | 103.8 ± 6.7    | 42.5 ± 4.2  | **105.5 ± 6.6**        |

### Key Findings

- **CartPole-v1**: PPO solves the environment; REINFORCE and GRPO+Attention both learn but converge more slowly without a value function critic.
- **SparseCartPole** (terminal-only reward): Both critic-free methods (REINFORCE+EMA and GRPO+Attention) significantly outperform PPO. PPO's value function struggles to learn meaningful value estimates with purely sparse rewards. GRPO+Attention performs on par with REINFORCE+EMA here.
- **Acrobot-v1**: PPO converges reliably while both critic-free methods show high variance across seeds, with some seeds not converging within 200k steps.
- **LunarLander-v3**: PPO learns a positive-return policy; both critic-free methods plateau around -110.
- The attention weights from the CAT concentrate on early timesteps in CartPole, suggesting the transformer identifies initial balance decisions as most predictive of episode return.

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
