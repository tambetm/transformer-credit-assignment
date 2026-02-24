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

### 3. Gradient-Based Credit Assignment (new)

Use **input gradients** of the causal return predictor as credit weights instead of value differences. Three methods:
```
grad_norm:    w_t = ||∂ĝ/∂emb_t||₂            # unsigned saliency
grad_x_input: w_t = ∂ĝ/∂emb_t · emb_t         # signed contribution
integrated:   w_t = Σ_k (∂f(α_k·emb)/∂emb · emb) / K  # principled attribution
```

Plus a **hybrid** that adaptively blends gradient credit with value differences:
```
Â_t = β * gradient_advantages + (1-β) * value_diff_advantages
```
where β adapts based on predictor quality.

## Algorithms Compared

1. **REINFORCE + EMA** — Uniform episode advantage, no per-step credit assignment
2. **PPO** — Learned value function with GAE (strong baseline)
3. **GRPO + Attention** — Cross-attention credit weights from bidirectional transformer
4. **RUDDER + Transformer** — Value differences from causal transformer
5. **RUDDER + LSTM** — Value differences from LSTM (ablation: LSTM vs Transformer for RUDDER)
6. **Gradient Norm** — Gradient L2-norm credit from causal transformer
7. **Gradient x Input** — Gradient-input dot product credit (signed)
8. **Integrated Gradients** — Principled attribution via path integrals (signed)
9. **Hybrid (Grad + VD)** — Adaptive blend of gradient credit and value differences

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

### Experiment 1 & 2: Attention vs Value Differences (4 environments, 200k timesteps, 5 seeds)

| Algorithm          | CartPole-v1       | Acrobot-v1         | LunarLander-v3    | SparseCartPole     |
|--------------------|-------------------|--------------------|-------------------|--------------------|
| REINFORCE + EMA    | 223.5 ± 5.7      | -279.7 ± 180.2     | -111.0 ± 8.5     | 103.8 ± 6.7       |
| PPO                | **490.2 ± 6.0**  | **-88.5 ± 3.9**   | **33.2 ± 40.5**  | 42.5 ± 4.2        |
| GRPO + Attention   | 171.2 ± 48.9     | -351.9 ± 162.3     | -124.7 ± 18.7    | 91.5 ± 17.8       |
| RUDDER + Transf.   | 126.1 ± 33.1     | -365.0 ± 135.1     | -121.7 ± 12.4    | **130.6 ± 29.4**  |
| RUDDER + LSTM      | 201.2 ± 37.8     | -350.6 ± 158.4     | -104.5 ± 6.6     | 110.5 ± 7.6       |

### Experiment 3: Gradient-Based Credit (CartPole + SparseCartPole, 200k timesteps, 5 seeds)

Same model architecture (d_model=32, nhead=2, d_ff=64) for comparable results.

| Algorithm             | CartPole-v1       | SparseCartPole     |
|-----------------------|-------------------|--------------------|
| REINFORCE + EMA       | 223.5 ± 5.7      | 103.8 ± 6.7       |
| PPO                   | **490.2 ± 6.0**  | 42.5 ± 4.2        |
| RUDDER Value Diff     | 106.6 ± 8.8      | **120.1 ± 32.3**  |
| Gradient Norm         | 46.5 ± 8.9       | 61.6 ± 17.3       |
| Gradient x Input      | 26.0 ± 12.4      | 27.0 ± 9.6        |
| Integrated Gradients  | 25.3 ± 9.8       | 34.9 ± 12.7       |
| Hybrid (Grad + VD)    | 65.5 ± 10.7      | 72.5 ± 5.9        |

### Key Findings

**Experiments 1 & 2:**
- **SparseCartPole** (terminal-only reward) — the key credit assignment test: **RUDDER+Transformer achieves the best performance (130.6)**, outperforming RUDDER+LSTM (110.5), REINFORCE+EMA (103.8), GRPO+Attention (91.5), and PPO (42.5).
- **PPO dominates dense/shaped rewards** (CartPole, Acrobot, LunarLander) — a learned value function with GAE is hard to beat when reward signal is available at every step.
- **Critic-free methods outperform PPO on SparseCartPole** — PPO's state-value function struggles with terminal-only reward.
- **Transformer vs LSTM for RUDDER**: RUDDER+Transformer (130.6) outperforms RUDDER+LSTM (110.5) on SparseCartPole. However, RUDDER+LSTM is competitive on dense-reward tasks (201.2 on CartPole vs 126.1 for Transformer).

**Experiment 3 — Gradient credit methods:**
- **Value differences still win**: RUDDER Value Diff (120.1) remains the best credit assignment method on SparseCartPole, outperforming all gradient-based alternatives.
- **Gradient methods underperform value differences**: All three gradient credit methods (Gradient Norm: 61.6, Integrated Gradients: 34.9, Gradient x Input: 27.0) score well below value differences (120.1) on SparseCartPole.
- **Hybrid helps over pure gradient**: The hybrid blend (72.5) outperforms all pure gradient methods on SparseCartPole, but still falls short of pure value differences — suggesting the gradient component adds noise rather than useful signal.
- **Gradient Norm > signed methods**: The unsigned Gradient Norm (61.6) outperforms the signed methods (Gradient x Input: 27.0, Integrated Gradients: 34.9) on SparseCartPole, suggesting the sign information from gradients is unreliable for credit assignment.
- **CartPole confirms the pattern**: On dense-reward CartPole, all gradient methods (25–66) significantly underperform even REINFORCE+EMA (223.5), while value differences (106.6) are closer to the baseline.

### Architecture Comparison

| Property                    | GRPO + Attention (CAT)           | RUDDER Value Diff               | Gradient Credit                 |
|-----------------------------|----------------------------------|----------------------------------|---------------------------------|
| Sequence model              | Bidirectional Transformer        | Causal Transformer               | Causal Transformer              |
| Credit mechanism            | Cross-attention weights          | Value differences V̂(t)-V̂(t-1)  | Input gradients ∂ĝ/∂emb        |
| Credit sign                 | Non-negative only                | Positive or negative             | Norm: unsigned; GxI/IG: signed  |
| Telescoping guarantee       | No (ad-hoc w*T scaling)          | Yes: Σ Â_t ≈ G - E[G]          | No                              |
| Computational cost          | 1 forward pass                   | 1 forward pass                   | 1+ backward passes (IG: K)     |
| Special tokens              | [RETURN] token                   | None                             | None                            |

### Limitations

- **Gradient credit is computationally expensive**: Integrated Gradients requires K backward passes per credit computation. Even Gradient Norm/GxI need a backward pass, adding ~1-2s per update vs value differences' pure forward pass.
- **Value differences remain the best credit assignment method**: Despite gradient methods being more theoretically principled (especially Integrated Gradients), they underperform the simpler value-difference approach in practice.
- All transformer-based methods add overhead vs plain REINFORCE. On dense-reward tasks where uniform credit suffices, this overhead hurts.
- Higher seed variance than REINFORCE on some environments, likely due to the additional optimization landscape of the transformer.

## Project Structure

```
grpo_attention/                      # Experiments 1 & 2
├── envs/sparse_cartpole.py         # Sparse reward wrapper
├── models/
│   ├── policy.py                   # Shared policy network (MLP)
│   ├── value.py                    # Value network (PPO only)
│   ├── credit_transformer.py       # Bidirectional CAT (GRPO+Attention)
│   ├── rudder_transformer.py       # Causal transformer (RUDDER+Transformer)
│   └── lstm_predictor.py           # LSTM return predictor (RUDDER+LSTM)
├── algorithms/
│   ├── reinforce_ema.py            # REINFORCE + EMA baseline
│   ├── ppo.py                      # PPO implementation
│   ├── grpo_attention.py           # GRPO + Attention
│   ├── rudder_transformer.py       # RUDDER + Transformer
│   └── lstm_rudder.py              # RUDDER + LSTM (ablation)
├── utils/                          # Shared utilities
├── train.py                        # Main training script
└── visualize.py                    # Plot generation

gradient_credit/                     # Experiment 3
├── models/
│   └── causal_predictor.py         # Causal transformer with gradient credit methods
├── algorithms/
│   ├── base_gradient.py            # Shared base class for gradient credit algos
│   ├── reinforce_ema.py            # REINFORCE + EMA baseline
│   ├── ppo.py                      # PPO baseline
│   ├── rudder_value_diff.py        # RUDDER value differences (comparison)
│   ├── gradient_norm_credit.py     # Gradient L2-norm credit
│   ├── gradient_input_credit.py    # Gradient x Input credit
│   ├── integrated_gradients_credit.py  # Integrated Gradients credit
│   └── hybrid_credit.py           # Adaptive gradient + value diff blend
├── utils/
│   └── gradient_utils.py           # Credit-to-advantage conversion utilities
├── train.py                        # Training script (7 algos x 2 envs)
└── visualize.py                    # Visualization suite
```
