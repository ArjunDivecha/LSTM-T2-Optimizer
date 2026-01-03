# Model Configurations

This directory contains YAML configuration files for different training scenarios.

## Available Configurations

### 1. `model_config_mac_test.yaml` - Mac M4 Max Testing

**Purpose:** Quick sanity check on local Mac before full training

**Key Settings:**
- 2 layers (vs 6 in production)
- 256 hidden dim (vs 1024)
- 60-day lookback (vs 250)
- Batch size 32 (vs 128)
- 5 epochs only
- MPS (Metal) backend

**Use Case:** Verify code works, catch bugs early, fast iteration

**Expected Runtime:** ~10-15 minutes per epoch

**Command:**
```bash
python train.py --config configs/model_config_mac_test.yaml
```

---

### 2. `model_config_h100_full.yaml` - H100 Full Production

**Purpose:** Complete training per PRD specifications

**Key Settings:**
- 6 layers (full depth)
- 1024 hidden dim
- 250-day lookback
- Batch size 128
- 300 max epochs (early stopping at ~200)
- CUDA + BF16 + torch.compile

**Use Case:** Final production training

**Expected Runtime:** 2-5 minutes per epoch, ~10-15 hours total

**Command:**
```bash
python train.py --config configs/model_config_h100_full.yaml
```

---

## Configuration Parameters

### Model Architecture

```yaml
model:
  encoder:
    num_layers: 6  # Depth of xLSTM
    hidden_dim: 1024  # Hidden state size
    cell_types: ['sLSTM', 'sLSTM', 'mLSTM', 'mLSTM', 'mLSTM', 'mLSTM']
    dropout: [0.1, 0.1, 0.1, 0.15, 0.15, 0.2]
```

### Data

```yaml
data:
  lookback_days: 250  # Input window
  prediction_horizon: 5  # Forecast horizon
  num_factors: 104
  num_features: 9  # Per factor
```

### Training

```yaml
training:
  batch_size: 128
  max_epochs: 300
  learning_rate: 1e-4
  device: "cuda"  # or "mps" for Mac
  precision: "bfloat16"  # or "float32"
```

### Loss Weights

```yaml
training:
  alpha_return: 1.0  # Return prediction
  beta_cov: 0.5  # Covariance prediction
  gamma_sharpe: 2.0  # Portfolio performance (most important!)
  lambda_tc: 0.0  # Transaction costs (disabled)
```

---

## Development Workflow

1. **Mac Testing (10 min):**
   ```bash
   python train.py --config configs/model_config_mac_test.yaml
   ```
   Verify: Code runs, no crashes, reasonable loss values

2. **H100 Full Training (15 hours):**
   ```bash
   # On Lambda Labs H100 instance
   python train.py --config configs/model_config_h100_full.yaml
   ```
   Get: Production-ready model

3. **Ensemble Training (3 days):**
   ```bash
   # Train 5 models with different seeds
   for seed in 42 123 456 789 1024; do
     python train.py \
       --config configs/model_config_h100_full.yaml \
       --seed $seed \
       --output models/ensemble/model_$seed
   done
   ```

---

## Philosophy: NO FALLBACKS, FAIL IS FAIL

- If training fails, debug and fix the root cause
- Don't reduce model size to "make it work"
- Don't skip epochs or reduce batch size to avoid errors
- Proper failures reveal real problems!

---

## Monitoring

**Key metrics to watch:**
- `val_sharpe_ratio` - Primary success metric
- `loss_sharpe` - Portfolio-level training loss
- `gradient_norm` - Should be stable, not exploding
- `val_turnover` - Should be <100% per rebalance

**Good training:**
- Validation Sharpe increases over time
- Training doesn't overfit (val loss tracks train loss)
- Gradients remain bounded (<10.0)

**Bad training (needs debugging):**
- Validation Sharpe stuck or decreasing
- NaN losses (gradient explosion, bad initialization)
- Optimization solver fails (bad predictions)
