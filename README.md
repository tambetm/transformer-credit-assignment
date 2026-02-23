# Transformer-Based Credit Assignment for RL

Comparing approaches to **per-timestep credit assignment** in episodic reinforcement learning using lightweight transformers, without a critic network.

## Approaches

### 1. GRPO + Attention (CAT)

Train a bidirectional transformer to predict episode returns. Use cross-attention weights from a dedicated `[RETURN]` token to redistribute the episode-level advantage:
```
w_t = credit_attention([RETURN] → (s_t, a_t))
Â_t = w_blended_t * T * A_episode
```

### 2. RUDDER + Transformer (new)

Train a **causal** transformer to predict episode returns at each timestep. Each position only attends to past positions (like RUDDER's LSTM, but with a transformer). Per-timestep advantages come from **value differences**:
```
V̂(t) = TransformerCausal(s_0, a_0, ..., s_t, a_t)   # predict G
Â_t = V̂(t) - V̂(t-1)                                  # value difference
```

These telescope: `Σ Â_t = V̂(T-1) - V̂(-1) ≈ G - E[G]`, automatically decomposing the episode advantage.

Key advantages over attention-based credit:
- **Signed**: value diffs can be negative (bad action reduces predicted return)
- **No warmup**: naturally well-behaved from the start
- **T× training signal**: MSE loss at every position, not just one `[RETURN]` token
- **Principled**: measures marginal information contribution, not correlation

## Algorithms Compared

1. **REINFORCE + EMA** — Uniform episode advantage, no per-step credit assignment
2. **PPO** — Learned value function with GAE (strong baseline)
3. **GRPO + Attention** — Cross-attention credit weights from bidirectional transformer
4. **RUDDER + Transformer** — Value differences from causal transformer

## Environments

1. **CartPole-v1** — Dense reward (sanity check)
2. **Acrobot-v1** — Sparse reward, moderate difficulty
3. **LunarLander-v3** — Shaped reward, coordinated behavior
4. **SparseCartPole** — Terminal-only reward (key credit assignment test)

## Usage

```bash
pip install -r requirements.txt

# Run all experiments
python -m grpo_attention.train --output-dir results

# Run specific experiments
python -m grpo_attention.train --envs CartPole-v1 SparseCartPole --algos RUDDER_Transformer PPO --seeds 3

# Generate plots
python -m grpo_attention.visualize --results-dir results --output-dir plots
```

## Results

All experiments: 5 seeds, 200k timesteps each. Mean ± std of final 100-episode returns:

| Algorithm          | CartPole-v1       | Acrobot-v1         | LunarLander-v3    | SparseCartPole     |
|--------------------|-------------------|--------------------|-------------------|--------------------|
| REINFORCE + EMA    | 223.5 ± 5.7      | -279.7 ± 180.2     | -111.0 ± 8.5     | 103.8 ± 6.7       |
| PPO                | **490.2 ± 6.0**  | **-88.5 ± 3.9**   | **33.2 ± 40.5**  | 42.5 ± 4.2        |
| GRPO + Attention   | 171.2 ± 48.9     | -351.9 ± 162.3     | -124.7 ± 18.7    | 91.5 ± 17.8       |
| RUDDER + Transf.   | 126.1 ± 33.1     | -365.0 ± 135.1     | -121.7 ± 12.4    | **130.6 ± 29.4**  |

### Key Findings

- **SparseCartPole** (terminal-only reward) — the key credit assignment test: **RUDDER+Transformer achieves the best performance (130.6)**, outperforming REINFORCE+EMA (103.8), GRPO+Attention (91.5), and PPO (42.5). The causal value-difference approach provides better credit assignment than both attention weights and uniform advantages in sparse-reward settings.
- **PPO dominates dense/shaped rewards** (CartPole, Acrobot, LunarLander) — a learned value function with GAE is hard to beat when reward signal is available at every step.
- **Critic-free methods outperform PPO on SparseCartPole** — PPO's state-value function struggles with terminal-only reward. All three critic-free methods (REINFORCE, GRPO+Attn, RUDDER+TF) substantially outperform PPO here.
- **RUDDER+TF vs GRPO+Attention**: On the sparse reward task that specifically tests credit assignment, RUDDER's value differences (+130.6) significantly outperform attention weights (+91.5). The signed, telescoping value differences provide more useful learning signal than non-negative attention weights.
- **Dense reward environments**: RUDDER+TF and GRPO+Attention perform comparably on LunarLander (~-122 vs ~-125). On CartPole and Acrobot, both underperform REINFORCE, suggesting the transformer overhead doesn't help when uniform credit is sufficient.

### Architecture Comparison

| Property                    | GRPO + Attention (CAT)           | RUDDER + Transformer          |
|-----------------------------|----------------------------------|-------------------------------|
| Attention masking           | Bidirectional (full)             | Causal (autoregressive)       |
| Credit mechanism            | Cross-attention weights          | Value differences V̂(t)-V̂(t-1)|
| Credit sign                 | Non-negative only                | Positive or negative          |
| Training signal per episode | 1 (return from [RETURN] token)   | T (return at every position)  |
| Warmup needed               | Yes (uniform → CAT over 20%)    | No                            |
| Telescoping guarantee       | No (ad-hoc w*T scaling)          | Yes: Σ Â_t ≈ G - E[G]       |
| Special tokens              | [RETURN] token                   | None                          |

### Limitations

- All transformer-based methods add overhead vs plain REINFORCE. On dense-reward tasks where uniform credit suffices, this overhead hurts.
- The causal transformer's early-timestep predictions are noisy (limited info), though value *differences* can still be informative even when absolute predictions are poor.
- Higher seed variance than REINFORCE on some environments, likely due to the additional optimization landscape of the transformer.

## Project Structure

```
grpo_attention/
├── envs/sparse_cartpole.py         # Sparse reward wrapper
├── models/
│   ├── policy.py                   # Shared policy network (MLP)
│   ├── value.py                    # Value network (PPO only)
│   ├── credit_transformer.py       # Bidirectional CAT (GRPO+Attention)
│   └── rudder_transformer.py       # Causal transformer (RUDDER+Transformer)
├── algorithms/
│   ├── reinforce_ema.py            # REINFORCE + EMA baseline
│   ├── ppo.py                      # PPO implementation
│   ├── grpo_attention.py           # GRPO + Attention
│   └── rudder_transformer.py       # RUDDER + Transformer
├── utils/
│   ├── buffer.py                   # Trajectory storage
│   ├── ema.py                      # EMA statistics tracker
│   └── logger.py                   # CSV logging
├── train.py                        # Main training script
└── visualize.py                    # Plot generation
```
