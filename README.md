# xLSTM-TS Factor Return Forecasting & Portfolio Optimization

A deep learning system using Extended Long Short-Term Memory (xLSTM) architecture to forecast returns and covariance for financial factors, then optimize portfolio weights to maximize Sharpe ratio.

## Overview

This system:
- Forecasts 5-day ahead returns for 100+ financial factors
- Predicts covariance matrices using a factor model decomposition
- Optimizes portfolio weights via differentiable convex optimization
- Operates on a 5-day rebalancing cycle with 250-day lookback windows

## Project Structure

```
LSTM/
├── configs/                    # Model configuration files
│   ├── model_config_mac_test.yaml   # Quick testing on Mac
│   └── model_config_h100_full.yaml  # Full production training
├── data/
│   ├── raw/                    # Raw data (T2_Optimizer.xlsx)
│   └── processed/              # Preprocessed yearly data
├── src/
│   ├── data/                   # Data loading and preprocessing
│   │   ├── loader.py
│   │   ├── market_data.py
│   │   └── preprocessing.py
│   ├── models/                 # Model implementations
│   │   ├── xlstm_factor.py     # xLSTM encoder
│   │   ├── portfolio_layer.py  # Differentiable optimization
│   │   └── portfolio_opt.py    # Portfolio optimization utils
│   └── utils/                  # Utilities
│       └── save_results.py
├── outputs/                    # Training outputs and results
├── run_walkforward_final.py    # Main walk-forward training script
├── evaluate_sharpe.py          # Evaluation utilities
└── train_sharpe.py             # Core training logic
```

## Requirements

- Python 3.9+
- PyTorch 2.0+
- cvxpy / cvxpylayers (differentiable optimization)
- pandas, numpy, scipy
- matplotlib (visualization)

## Quick Start

### 1. Data Preparation

Place `T2_Optimizer.xlsx` in the project root or `data/raw/` directory.

### 2. Run Walk-Forward Training

```bash
python run_walkforward_final.py
```

This runs expanding-window walk-forward optimization:
- Training starts from 2010
- Expands training window each year
- Generates alpha predictions for each out-of-sample year

### 3. Evaluate Results

```bash
python evaluate_sharpe.py
```

## Model Architecture

**Encoder:** 6-layer xLSTM (sLSTM + mLSTM hybrid)
- Layers 1-2: sLSTM for local pattern detection
- Layers 3-6: mLSTM for long-range dependencies

**Dual Prediction Heads:**
1. **Return Head:** Predicts expected 5-day returns (μ)
2. **Covariance Head:** Factor model decomposition (Σ = BFB' + D)

**Portfolio Layer:** Differentiable convex optimization maximizing Sharpe ratio subject to:
- Long-only constraint (w ≥ 0)
- Fully invested (Σw = 1)
- Minimum volatility floor (10% annualized)

## Configuration

See `configs/README.md` for detailed configuration options.

**Mac Testing:**
```bash
python train.py --config configs/model_config_mac_test.yaml
```

**H100 Production:**
```bash
python train.py --config configs/model_config_h100_full.yaml
```

## Results

Walk-forward backtest results are saved to `outputs/`:
- `alpha_YYYY.xlsx` - Yearly alpha predictions
- `cumulative_returns.pdf` - Performance visualization
- `training_log.txt` - Training metrics

## Philosophy

**NO FALLBACKS, FAIL IS FAIL**

- No synthetic data generation
- No fallback strategies (equal-weight, etc.)
- If the model fails, debug and fix the root cause
- Failures are valuable signals, not problems to hide

## License

Proprietary - All Rights Reserved

## References

- [xLSTM Paper](https://arxiv.org/abs/2405.04517) - Beck et al., 2024
- See `PRD_xLSTM_Factor_Forecasting.md` for full specification
