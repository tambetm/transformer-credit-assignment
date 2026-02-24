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

### 3. Gradient-Based Credit Assignment

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

### 4. Gradient-Sharpened Value Differences — GSVD (new)

Key insight: value differences are the best credit signal but can be **diffuse** — spreading credit uniformly when the predictor hasn't fully converged. Gradient saliency from the same model identifies which timesteps the predictor is most **sensitive** to. GSVD uses gradients to **sharpen** value differences rather than replace them:

```
saliency_t = ||∂ĝ/∂emb_t||₂                 # gradient magnitude (sensitivity)
weights_t  = softmax(saliency_t / τ) × T     # sharpening weights (mean = 1)
Â_t        = (V̂(t) - V̂(t-1)) × weights_t    # sharpened value differences
```

Temperature τ is **adaptive**: starts high (τ=10, ≈ pure value diffs when predictor is bad) and decreases (τ=0.5, gradient-sharpened) as the predictor's MSE improves. This makes GSVD a strict superset of value differences — when τ→∞ it exactly recovers them.

### 5. Attention Rollout Credit — ARC (new)

Instead of computing costly backward passes, extract credit directly from the transformer's **attention weights** (already computed during the forward pass). Attention rollout (Abnar & Zuidema, 2020) traces multi-layer information flow:

```
R = ∏_l (0.5 × A^l + 0.5 × I)         # compose across layers w/ residual
flow_t = R[T-1, t]                       # total information flow: input t → output
Â_t = (V̂(t) - V̂(t-1)) × (flow_t / mean(flow))   # flow-weighted value diffs
```

Zero extra computation — uses the same causal predictor as RUDDER, just reads out the attention matrices that were already computed. When flow is uniform, reduces to pure value differences.

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
10. **GSVD** *(new)* — Gradient-sharpened value differences with adaptive temperature
11. **Attention Rollout** *(new)* — Multi-layer attention flow weighting of value differences

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

| Algorithm              | CartPole-v1       | SparseCartPole     |
|------------------------|-------------------|--------------------|
| REINFORCE + EMA        | 223.5 ± 5.7      | 103.8 ± 6.7       |
| PPO                    | **489.4 ± 4.8**  | 42.5 ± 4.2        |
| RUDDER Value Diff      | 106.6 ± 8.8      | **120.8 ± 33.1**  |
| Gradient Norm          | 46.5 ± 8.9       | 61.6 ± 17.3       |
| Gradient x Input       | 26.0 ± 12.4      | 27.0 ± 9.6        |
| Integrated Gradients   | 25.3 ± 9.8       | 34.9 ± 12.7       |
| Hybrid (Grad + VD)     | 65.5 ± 10.7      | 72.5 ± 5.9        |
| **GSVD** *(new)*       | 107.1 ± 12.7     | **112.7 ± 10.3**  |
| **Attn Rollout** *(new)* | **126.4 ± 25.5** | 108.2 ± 18.9   |

### Key Findings

**Experiments 1 & 2:**
- **SparseCartPole** (terminal-only reward) — the key credit assignment test: **RUDDER+Transformer achieves the best performance (130.6)**, outperforming RUDDER+LSTM (110.5), REINFORCE+EMA (103.8), GRPO+Attention (91.5), and PPO (42.5).
- **PPO dominates dense/shaped rewards** (CartPole, Acrobot, LunarLander) — a learned value function with GAE is hard to beat when reward signal is available at every step.
- **Critic-free methods outperform PPO on SparseCartPole** — PPO's state-value function struggles with terminal-only reward.
- **Transformer vs LSTM for RUDDER**: RUDDER+Transformer (130.6) outperforms RUDDER+LSTM (110.5) on SparseCartPole. However, RUDDER+LSTM is competitive on dense-reward tasks (201.2 on CartPole vs 126.1 for Transformer).

**Experiment 3 — Gradient credit methods:**
- **Value differences still win on SparseCartPole**: RUDDER Value Diff (120.8) achieves the highest mean on SparseCartPole, but with high variance (±33.1).
- **GSVD closes the gap with 3× lower variance**: GSVD (112.7 ± 10.3) nearly matches value diffs (120.8 ± 33.1) on SparseCartPole, while achieving **3× lower standard deviation** — making it the most reliable transformer-based credit method. This confirms the design goal: gradient sharpening stabilizes value differences without destroying them.
- **GSVD dominates all prior gradient methods**: On SparseCartPole, GSVD (112.7) outperforms Hybrid (72.5), Gradient Norm (61.6), Integrated Gradients (34.9), and Gradient x Input (27.0) by a wide margin — a **55% improvement** over the best prior gradient method.
- **Attention Rollout is best on dense-reward CartPole**: Among all transformer-based methods, Attention Rollout (126.4) outperforms RUDDER Value Diff (106.6) and GSVD (107.1) on CartPole-v1, showing that attention flow captures useful credit structure even with dense rewards.
- **Attention Rollout is competitive on SparseCartPole**: Attention Rollout (108.2) outperforms REINFORCE+EMA (103.8), all pure gradient methods, and the hybrid — despite requiring zero backward passes.
- **Multiplicative > additive combination**: Both GSVD and Attention Rollout use gradient/attention information **multiplicatively** with value differences, preserving the value-diff sign structure. This dramatically outperforms the Hybrid's additive blend (72.5), confirming that gradient information should *sharpen* value diffs, not replace them.
- **Gradient Norm > signed methods**: The unsigned Gradient Norm (61.6) outperforms the signed methods (Gradient x Input: 27.0, Integrated Gradients: 34.9) on SparseCartPole, suggesting the sign information from gradients is unreliable for credit assignment.

### Architecture Comparison

| Property                    | RUDDER Value Diff               | Gradient Credit                 | GSVD *(new)*                    | Attention Rollout *(new)*       |
|-----------------------------|----------------------------------|---------------------------------|---------------------------------|---------------------------------|
| Sequence model              | Causal Transformer               | Causal Transformer              | Causal Transformer              | Causal Transformer              |
| Credit mechanism            | Value differences V̂(t)-V̂(t-1)  | Input gradients ∂ĝ/∂emb        | VD × softmax(grad_norm/τ)      | VD × attention rollout          |
| Credit sign                 | Positive or negative             | Norm: unsigned; GxI/IG: signed  | From value diffs (signed)       | From value diffs (signed)       |
| Telescoping guarantee       | Yes: Σ Â_t ≈ G - E[G]          | No                              | Approximate (weighted VD)       | Approximate (weighted VD)       |
| Backward passes             | 0                                | 1 (IG: K)                       | 1                               | 0                               |
| Adapts over training        | No                               | No                              | Yes (temperature annealing)     | Yes (attention learns)          |
| SparseCartPole score        | 120.8 ± 33.1                     | 27–72                           | **112.7 ± 10.3**               | 108.2 ± 18.9                    |

### Limitations

- **Value differences still edge out on mean performance**: While GSVD and Attention Rollout nearly match RUDDER Value Diff on SparseCartPole (112.7 and 108.2 vs 120.8), the mean is still slightly lower. The tradeoff is dramatically lower variance.
- **Gradient credit is computationally expensive**: Integrated Gradients requires K backward passes per credit computation. GSVD needs one backward pass (same cost as value diffs + grad_norm). Attention Rollout requires zero backward passes.
- All transformer-based methods add overhead vs plain REINFORCE. On dense-reward tasks where uniform credit suffices, this overhead hurts.
- **Multiplicative weighting sacrifices exact telescoping**: GSVD and Attention Rollout multiply value diffs by non-uniform weights, so Σ Â_t ≠ Σ vd_t exactly. In practice the deviation is small.

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
│   └── causal_predictor.py         # Causal transformer with gradient + attention credit
├── algorithms/
│   ├── base_gradient.py            # Shared base class for gradient credit algos
│   ├── reinforce_ema.py            # REINFORCE + EMA baseline
│   ├── ppo.py                      # PPO baseline
│   ├── rudder_value_diff.py        # RUDDER value differences (comparison)
│   ├── gradient_norm_credit.py     # Gradient L2-norm credit
│   ├── gradient_input_credit.py    # Gradient x Input credit
│   ├── integrated_gradients_credit.py  # Integrated Gradients credit
│   ├── hybrid_credit.py           # Adaptive gradient + value diff blend
│   ├── gsvd_credit.py             # Gradient-Sharpened Value Differences (new)
│   └── attention_rollout_credit.py # Attention Rollout Credit (new)
├── utils/
│   └── gradient_utils.py           # Credit-to-advantage conversion utilities
├── train.py                        # Training script (9 algos x 2 envs)
└── visualize.py                    # Visualization suite
```
