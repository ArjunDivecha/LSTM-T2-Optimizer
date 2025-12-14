# Product Requirements Document (PRD)
# xLSTM-TS Factor Return Forecasting & Portfolio Optimization System

**Version:** 1.0
**Date:** December 14, 2025
**Status:** Approved - Ready for Implementation
**Project Code:** LSTM-T2-OPT

---

## Executive Summary

Build a deep learning system using Extended Long Short-Term Memory (xLSTM) architecture to forecast returns and covariance for 100 financial factors, then optimize portfolio weights to maximize Sharpe ratio subject to a 10% minimum volatility constraint. The system will operate on a 5-day rebalancing cycle with 250-day lookback windows.

**Key Objectives:**
- Forecast 5-day ahead returns for 100 factors
- Forecast 5-day ahead covariance matrix (100×100)
- Optimize portfolio weights to maximize Sharpe ratio
- Maintain minimum 10% annualized volatility
- Deploy on H100 GPUs with production-ready pipeline

---

## 1. Project Scope

### 1.1 In Scope
- xLSTM-TS model for multi-output forecasting (returns + covariance)
- Differentiable portfolio optimization layer
- End-to-end training with multi-objective loss
- Walk-forward backtesting framework
- 5-model ensemble system
- Production inference pipeline
- Monitoring and alerting dashboard
- Model versioning and rollback capability

### 1.2 Out of Scope
- Live trading execution (order routing, broker integration)
- Real-time tick data processing
- Options/derivatives strategies
- Short selling (long-only constraint)
- Leverage (100% invested, no margin)
- Manual overrides/discretionary trading
- Multi-asset class expansion (stocks, bonds, commodities)

---

## 2. Business Requirements

### 2.1 Primary Objective
Maximize risk-adjusted returns (Sharpe ratio) on a portfolio of 100 financial factors with weekly rebalancing.

### 2.2 Success Criteria

**Minimum Viable Performance:**
- Out-of-sample Sharpe ratio ≥ 1.0 (annualized)
- Maximum drawdown < 20%
- Win rate ≥ 55% (positive 5-day periods)
- Information Coefficient (IC) ≥ 0.05
- Portfolio turnover < 100% per rebalance

**Target Performance:**
- Sharpe ratio ≥ 1.5
- Maximum drawdown < 15%
- Win rate ≥ 60%
- IC ≥ 0.08
- Portfolio turnover < 50%

### 2.3 Performance Benchmarks
System must outperform:
1. Equal-weight portfolio (1/100 per factor)
2. Minimum variance portfolio (sample covariance)
3. Risk parity allocation
4. Mean-variance optimization (sample statistics)
5. Factor momentum strategy

---

## 3. Data Requirements

### 3.1 Primary Data

**Factor Returns:**
- Source: T2_Optimizer.xlsx
- Factors: 100 financial factors
- Frequency: Daily
- History: January 1, 2000 - Present (~6,300+ observations)
- Format: Date-indexed returns (decimal, e.g., 0.01 = 1%)

**Market Data:**
- SPY/SPX daily returns
- VIX daily levels
- Risk-free rate (3-month T-bill)

**Macro Data:**
- 10-Year Treasury yield
- 2-Year Treasury yield
- Yield curve spread (10Y - 2Y)

### 3.2 Data Quality Requirements

**Missing Data Handling:**
- Strategy: Drop that specific factor for that sample/month only
- Keep remaining 99 factors for that period
- No forward fill, no imputation
- Log all missingness events

**Outlier Treatment:**
- Method: Winsorization (not capping)
- Threshold: 1st and 99th percentile (per factor, rolling 252-day)
- Apply separately to each factor
- Preserve sign (don't convert to absolute values)

**Data Validation:**
- No duplicate timestamps
- All dates align to trading calendar
- Returns within plausible range (|-50%| to |+50%| daily)
- Minimum 80% data availability per factor (drop factor entirely if <80%)

### 3.3 Feature Engineering

**Per-Factor Features (7 features × 100 factors):**
1. Raw daily return
2. 20-day moving average return
3. 20-day rolling volatility (standard deviation)
4. 60-day moving average return
5. 60-day rolling volatility
6. Cross-sectional rank (percentile among 100 factors daily)
7. Z-score (rolling 60-day: (x - μ_60) / σ_60)

**Market/Macro Features (broadcast to all):**
1. Market return (SPY)
2. Market volatility (VIX level)
3. Volatility regime indicator (VIX > 20 binary flag)
4. Yield curve spread (10Y - 2Y)
5. Risk-free rate level
6. Market correlation regime (average pairwise correlation of 100 factors)

**Normalization:**
- Method: Rolling Z-score standardization
- Window: 60 trading days
- Formula: (x - rolling_mean_60) / rolling_std_60
- Handle edge cases: First 60 days use expanding window, add ε=1e-8 for zero variance

---

## 4. Model Architecture

### 4.1 Overall Design

**Architecture Type:** Dual-Head xLSTM with Differentiable Portfolio Optimization

**Input:**
- Shape: (batch, sequence=250, features=706)
- 250 days lookback
- 100 factors × 7 features per factor = 700
- Plus 6 macro features = 706 total

**Output:**
- μ: Expected 5-day returns (100 × 1)
- Σ: Covariance matrix (100 × 100, positive semi-definite)
- w*: Optimal portfolio weights (100 × 1, sum to 1, ≥ 0)

### 4.2 Encoder Architecture

**Shared xLSTM Encoder (6 layers):**

```
Layer 1: sLSTM
  - Hidden dimension: 1024
  - LayerNorm + Dropout(0.1)
  - Purpose: Capture local/short-term patterns

Layer 2: sLSTM
  - Hidden dimension: 1024
  - LayerNorm + Dropout(0.1)
  - Purpose: Refine local patterns

Layer 3: mLSTM (matrix memory)
  - Hidden dimension: 1024
  - LayerNorm + Dropout(0.1)
  - Purpose: Begin capturing long-range dependencies

Layer 4: mLSTM
  - Hidden dimension: 1024
  - LayerNorm + Dropout(0.15)
  - Purpose: Deep long-range pattern recognition

Layer 5: mLSTM
  - Hidden dimension: 1024
  - LayerNorm + Dropout(0.15)
  - Purpose: Complex temporal interactions

Layer 6: mLSTM
  - Hidden dimension: 1024
  - LayerNorm + Dropout(0.2)
  - Purpose: Final abstract representation

Output: (batch, 1024) - Final hidden state
```

**Rationale:**
- sLSTM early layers: Efficient local pattern detection
- mLSTM later layers: Long-term dependencies across 250 days
- 1024 hidden dim: Sufficient capacity for 100 factors
- Progressive dropout: More regularization in deeper layers

**Total Encoder Parameters:** ~40M

### 4.3 Head 1: Return Prediction (μ)

```
Input: Encoder output (1024)
  ↓
Dense(1024 → 512) + ReLU + Dropout(0.2)
  ↓
Dense(512 → 256) + ReLU + Dropout(0.2)
  ↓
Dense(256 → 100) + Linear (no activation)
  ↓
Output: μ (batch, 100)
```

**Purpose:** Predict expected 5-day return for each factor

**Parameters:** ~1M

### 4.4 Head 2: Covariance Prediction (Σ)

**Method:** Factor Model Decomposition

**Formula:** Σ = B F B^T + D

Where:
- B: Factor loadings (100 × 20)
- F: Latent factor covariance (20 × 20, PSD)
- D: Idiosyncratic variance (100, diagonal, positive)

**Architecture:**

```
Input: Encoder output (1024)
  ↓
Dense(1024 → 512) + ReLU + Dropout(0.2)
  ↓
Dense(512 → 256) + ReLU + Dropout(0.2)
  ↓
Split into 3 branches:

Branch B (Factor Loadings):
  Dense(256 → 2000) → Reshape(100, 20)
  Output: B matrix (100 × 20)

Branch F (Factor Covariance):
  Dense(256 → 210)
  Reshape to lower triangular L (20 × 20)
  Apply softplus to diagonal: L_diag = softplus(L_diag)
  Compute: F = L @ L^T (Cholesky, guaranteed PSD)
  Output: F matrix (20 × 20, PSD)

Branch D (Idiosyncratic Variance):
  Dense(256 → 100) + Softplus (ensure positive)
  Output: D vector (100)

Combine:
  Σ = B @ F @ B^T + diag(D)
  Output: Σ matrix (100 × 100, PSD)
```

**Advantages:**
- Guaranteed positive semi-definite (PSD) covariance
- Only ~1,100 parameters (vs 5,050 for full Cholesky)
- Interpretable latent factor structure
- Numerically stable

**Parameters:** ~3M

**Total Model Parameters:** ~44M

### 4.5 Portfolio Optimization Layer

**Problem Formulation:**

```
Given: μ (predicted returns), Σ (predicted covariance)

Maximize: Sharpe Ratio = (μ^T w) / sqrt(w^T Σ w)

Subject to:
  1. sum(w) = 1                          (fully invested)
  2. w ≥ 0                                (long-only)
  3. sqrt(w^T Σ w) ≥ 0.10 * sqrt(5/252)  (min vol constraint)
```

**Implementation:** Differentiable convex optimization (cvxpylayers)

**Reformulation (for solver):**

Since Sharpe maximization is non-convex, use equivalent formulation:

```
For a grid of target returns R_target:
  Minimize: w^T Σ w
  Subject to:
    - μ^T w ≥ R_target
    - sum(w) = 1
    - w ≥ 0
    - sqrt(w^T Σ w) ≥ vol_min

Select R_target that maximizes Sharpe ratio
```

**Gradient Flow:**
- Optimization is differentiable via implicit function theorem
- Gradients backpropagate through w* to μ and Σ
- Enables end-to-end training

**Output:** w* (optimal portfolio weights, 100 × 1)

---

## 5. Training Specification

### 5.1 Loss Function

**Multi-Objective Loss:**

```
L_total = α * L_return + β * L_cov + γ * L_portfolio + λ_tc * L_turnover
```

**Component 1: Return Prediction Loss**
```
L_return = Huber(μ_pred, μ_actual, δ=1.0)

where:
  μ_actual = mean(actual_returns[t+1:t+6], axis=0)
  Huber loss is robust to outliers
```

**Component 2: Covariance Prediction Loss**
```
L_cov = Frobenius_norm(Σ_pred - Σ_actual)
      = sqrt(sum((Σ_pred - Σ_actual)^2))

where:
  Σ_actual = cov(actual_returns[t+1:t+6])
```

**Component 3: Portfolio Performance Loss**
```
L_portfolio = -Sharpe(w*, actual_returns)
            = -(w* @ μ_actual) / sqrt(w* @ Σ_actual @ w*)

where:
  w* = optimal weights from optimization layer
  μ_actual, Σ_actual = realized statistics
```

**Component 4: Turnover Penalty (optional, set to 0 initially)**
```
L_turnover = sum(|w*_t - w*_{t-1}|)

Transaction cost: λ_tc = 0.0007 (7 bps)
Initially: λ_tc = 0 (build capability, don't use)
```

**Loss Weights:**
- α = 1.0 (return prediction baseline)
- β = 0.5 (covariance secondary importance)
- γ = 2.0 (portfolio performance most important)
- λ_tc = 0.0 (transaction costs disabled for now)

### 5.2 Optimizer Configuration

**Optimizer:** AdamW (Adam with decoupled weight decay)

**Parameters:**
```
learning_rate: 1e-4
betas: (0.9, 0.999)
weight_decay: 1e-5
eps: 1e-8
amsgrad: False
```

**Learning Rate Schedule:**
```
Phase 1: Warmup (epochs 1-10)
  Linear increase: 0 → 1e-4

Phase 2: Cosine Annealing (epochs 11-300)
  lr(t) = lr_min + 0.5 * (lr_max - lr_min) * (1 + cos(π * t / T))
  lr_max = 1e-4
  lr_min = 1e-6
  T = 290 epochs
```

### 5.3 Training Hyperparameters

```
batch_size: 128
max_epochs: 300
early_stopping_patience: 30 (monitor validation Sharpe)
gradient_clip_norm: 1.0
mixed_precision: bfloat16 (H100 optimized)
compile: True (torch.compile for 2x speedup)
num_workers: 8 (data loading)
pin_memory: True
```

**Regularization:**
```
dropout_rates: [0.1, 0.1, 0.1, 0.15, 0.15, 0.2] (per layer)
weight_decay: 1e-5
gradient_clipping: 1.0
```

### 5.4 Data Split Strategy

**Primary Strategy: Expanding Window**

```
Training Set:   2000-01-01 to 2015-12-31 (15 years, ~3,750 samples)
Validation Set: 2016-01-01 to 2020-12-31 (5 years, ~1,250 samples)
Test Set:       2021-01-01 to 2025-12-31 (5 years, ~1,250 samples)
```

**Retraining Schedule:**
- Frequency: Annual
- Process: Add 1 year of data, retrain from scratch
- Validation: Previous year becomes part of training, new year for validation

**Ensemble Strategy:**
- Train 5 models with different random seeds
- Same architecture, same data, different initialization
- Average predictions: μ_final = mean(μ_1, ..., μ_5)
- Average covariances: Σ_final = mean(Σ_1, ..., Σ_5)

### 5.5 Hardware Configuration

**Platform:** Lambda Labs H100 GPU Instances

**Specifications:**
```
GPU: 1x H100 (80GB HBM3)
vCPUs: 24-48 cores
RAM: 200GB+
Storage: 1TB NVMe SSD
```

**Expected Performance:**
```
Training time per epoch: 2-5 minutes
Total training time: ~10-15 hours (with early stopping)
Inference latency: <100ms per prediction
Memory usage: ~30-40GB GPU memory
```

**Optimizations:**
- torch.compile() for 2x training speedup
- bfloat16 mixed precision for 1.5-2x speedup
- Gradient checkpointing if memory constrained
- DataLoader with prefetching

---

## 6. Backtesting Framework

### 6.1 Backtest Configuration

**Period:** 2021-01-01 to 2025-12-31 (out-of-sample)

**Simulation Parameters:**
```
Initial capital: $1,000,000
Rebalancing frequency: Every 5 trading days (weekly)
Transaction costs: 0 bps (capability built, disabled)
Slippage: 0 bps (capability built, disabled)
Position limits: None (long-only, no individual caps)
Leverage: 1.0x (100% invested, sum(w) = 1)
```

### 6.2 Backtest Procedure

**For each rebalancing date t (every 5 days):**

```
1. Generate Forecast
   - Input: factor_returns[t-250:t] (250-day lookback)
   - Model: Ensemble average of 5 models
   - Output: μ_t (100×1), Σ_t (100×100)

2. Optimize Portfolio
   - Solve: max Sharpe s.t. vol ≥ 10%, sum(w)=1, w≥0
   - Output: w*_t (target weights)

3. Calculate Trades
   - trades = w*_t - w_{t-1}
   - turnover = sum(|trades|)

4. Apply Costs (when enabled)
   - cost = turnover * transaction_cost_bps
   - portfolio_value -= cost

5. Execute Trades
   - Update positions to w*_t
   - Mark-to-market daily for next 5 days

6. Record Metrics
   - Daily returns, cumulative P&L
   - Rolling Sharpe, drawdown
   - Turnover, position concentration
```

### 6.3 Performance Metrics

**Portfolio Returns:**
```
- Cumulative return
- Annualized return (CAGR)
- Annualized volatility
- Sharpe ratio (annualized)
- Sortino ratio (downside deviation)
- Calmar ratio (return / max drawdown)
- Maximum drawdown (peak-to-trough)
- Win rate (% of positive 5-day periods)
- Average turnover per rebalance
```

**Prediction Quality:**
```
- Information Coefficient (IC): corr(μ_pred, μ_actual)
- Rank IC: Spearman(μ_pred, μ_actual)
- Hit rate: sign(μ_pred) == sign(μ_actual)
- Covariance RMSE: sqrt(mean((Σ_pred - Σ_actual)^2))
- Realized vs predicted volatility error
```

**Risk Metrics:**
```
- Value at Risk (VaR 95%, 99%)
- Conditional VaR (Expected Shortfall)
- Beta to market (SPY)
- Maximum position size (max w_i)
- Position concentration (HHI)
- Longest drawdown duration
```

### 6.4 Benchmark Comparisons

**Benchmarks:**
1. Equal-weight (1/100 per factor)
2. Minimum variance (sample covariance, 250-day)
3. Risk parity (equal risk contribution)
4. Mean-variance (sample mean & cov, 250-day)
5. Factor momentum (rank by 60-day return)

**Statistical Tests:**
- Sharpe ratio difference (Jobson-Korkie test)
- Return distribution (t-test)
- Out-of-sample R²

---

## 7. Validation Strategy

### 7.1 Walk-Forward Validation

**Method:** Expanding window with annual retraining

```
Year 2021:
  Train: 2000-2020 → Validate: 2020 → Test: 2021

Year 2022:
  Train: 2000-2021 → Validate: 2021 → Test: 2022

Year 2023:
  Train: 2000-2022 → Validate: 2022 → Test: 2023

Year 2024:
  Train: 2000-2023 → Validate: 2023 → Test: 2024

Year 2025:
  Train: 2000-2024 → Validate: 2024 → Test: 2025
```

**Result:** 5 separate out-of-sample test years, no look-ahead bias

### 7.2 Robustness Checks

**Stress Testing:**
```
Crisis Periods:
  - 2008 Financial Crisis (if in sample)
  - 2020 COVID Crash
  - 2022 Rate Hike Selloff

Metrics:
  - Maximum drawdown during crisis
  - Recovery time (days to new high)
  - VaR exceedances
```

**Regime Analysis:**
```
Low Volatility (VIX < 15):
  - Model performance
  - Position sizing behavior

High Volatility (VIX > 25):
  - Model performance
  - Risk management

Correlation Breakdowns:
  - Factor correlation > 0.7 (crisis)
  - Model adaptability
```

**Sensitivity Analysis:**
```
Parameter Perturbations:
  - Lookback: 200, 250, 300 days
  - Rebalance: 3, 5, 10 days
  - Min vol: 8%, 10%, 12%

Data Perturbations:
  - Add noise (5%, 10% σ)
  - Remove random factors (10%, 20%)
  - Missing data scenarios
```

---

## 8. Hyperparameter Tuning

### 8.1 Tuning Priorities

**Tier 1 (Highest Impact):**
```
1. Lookback period: [60, 120, 180, 250] days
2. Hidden dimension: [512, 1024, 2048]
3. Number of layers: [4, 6, 8]
4. Loss weights: α ∈ [0.5, 2.0], β ∈ [0.1, 1.0], γ ∈ [1.0, 5.0]
5. Learning rate: [1e-5, 3e-5, 1e-4, 3e-4]
```

**Tier 2 (Moderate Impact):**
```
6. Covariance latent factors: [10, 20, 30]
7. Dropout rates: [0.1, 0.2, 0.3]
8. Batch size: [64, 128, 256]
9. Feature ablation (which features to include)
10. Normalization window: [30, 60, 120] days
```

**Tier 3 (Fine-Tuning):**
```
11. Weight decay: [1e-6, 1e-5, 1e-4]
12. Gradient clip: [0.5, 1.0, 2.0]
13. Cell type ratios (sLSTM vs mLSTM)
14. Activation functions (ReLU, GELU, SiLU)
15. Optimizer (AdamW, Lion, Sophia)
```

### 8.2 Search Strategy

**Phase 1: Grid Search (Tier 1)**
```
Configurations: 4 × 3 × 3 = 36 combinations (lookback × hidden × layers)
Epochs: 50 (early stopping)
Time: ~3 days on 1 H100
Metric: Validation Sharpe ratio
```

**Phase 2: Random Search (Tier 2)**
```
Configurations: 100 random samples
Epochs: 20
Time: ~4 days on 1 H100
Metric: Validation Sharpe ratio
```

**Phase 3: Bayesian Optimization (Tier 3)**
```
Tool: Optuna or Ray Tune
Trials: 50
Epochs: Full training
Time: ~5 days on 1 H100
Metric: Validation Sharpe ratio
```

**Total Tuning Time:** ~2 weeks (can parallelize with multiple H100s)

### 8.3 Ablation Studies

**Feature Importance:**
```
Baseline: All 7 features per factor + 6 macro features

Ablate:
  - Remove macro features → Measure Sharpe drop
  - Remove rolling statistics → Measure Sharpe drop
  - Remove cross-sectional rank → Measure Sharpe drop
  - Use only raw returns → Compare to baseline
```

**Architecture Components:**
```
Ablate:
  - sLSTM only (no mLSTM) → Compare
  - mLSTM only (no sLSTM) → Compare
  - Single head (returns only, use sample cov) → Compare
  - No optimization layer (predict weights directly) → Compare
```

**Loss Function:**
```
Ablate:
  - α=1, β=0, γ=0 (returns only)
  - α=0, β=1, γ=0 (covariance only)
  - α=0, β=0, γ=1 (portfolio only)
  - Compare to multi-objective (α=1, β=0.5, γ=2)
```

---

## 9. Production Deployment

### 9.1 Inference Pipeline

**Daily Workflow (T+0, after market close):**

```
1. Data Ingestion (T+0, 4:30 PM ET)
   - Fetch factor returns for day T
   - Validate data quality
   - Update rolling features

2. Feature Engineering (T+0, 4:35 PM ET)
   - Compute 20-day, 60-day statistics
   - Calculate cross-sectional ranks
   - Normalize with rolling z-score

3. Model Inference (T+0, 4:40 PM ET)
   - Load last 250 days of features
   - Run ensemble (5 models)
   - Average predictions: μ_t, Σ_t

4. Portfolio Optimization (T+0, 4:42 PM ET)
   - Solve constrained optimization
   - Output: target weights w*_t

5. Trade Generation (T+0, 4:45 PM ET)
   - Calculate trades: Δw = w*_t - w_{t-1}
   - Validate constraints
   - Generate order list

6. Pre-Market Validation (T+1, 9:00 AM ET)
   - Final checks before execution
   - Risk limits validation
```

**Latency Requirements:**
- Data ingestion: <5 minutes
- Feature engineering: <2 minutes
- Model inference: <1 minute (<100ms per model)
- Portfolio optimization: <1 minute
- Total pipeline: <10 minutes

### 9.2 Model Retraining

**Schedule:** Annual (every January)

**Process:**
```
1. Data Update
   - Add previous year's data
   - Validate data quality

2. Model Retraining
   - Train 5 new models (different seeds)
   - Full 300 epochs (with early stopping)
   - Time: ~15 hours per model (~3 days total sequential)

3. Validation
   - Backtest on previous year (out-of-sample)
   - Compare to existing production model
   - Statistical significance test

4. A/B Testing (optional)
   - Paper trade new model for 1 month
   - Compare to production model

5. Deployment
   - If new model outperforms: Deploy
   - If not: Keep existing model, investigate

6. Archival
   - Save old model to models/archive/
   - Version control: models/v{year}/
```

### 9.3 Monitoring & Alerts

**Real-Time Dashboard:**

```
Portfolio Metrics:
  - Current positions (bar chart)
  - Daily P&L
  - Rolling 30-day Sharpe
  - Current drawdown from peak
  - Realized vs target volatility

Model Predictions:
  - Top 10 predicted returns
  - Predicted covariance eigenvalues
  - Optimization convergence status
  - Ensemble disagreement (variance)

Risk Monitoring:
  - VaR utilization (95%, 99%)
  - Maximum position size
  - Number of active positions
  - Turnover rate

Data Quality:
  - Missing data count
  - Outliers detected (winsorized)
  - Feature distribution shifts
```

**Alerting Levels:**

```
CRITICAL (immediate action):
  - Model inference failure
  - Optimization solver infeasible
  - Data pipeline failure
  - Predicted returns >3σ outliers

WARNING (review within 1 day):
  - Data quality issues (>5% missing)
  - High ensemble disagreement (std > threshold)
  - Turnover >50% (unusual rebalancing)
  - Sharpe <0.5 over rolling 30 days

INFO (daily report):
  - Rebalancing summary
  - Performance summary
  - Model health check
```

### 9.4 Model Versioning

**Directory Structure:**

```
models/
├── production/
│   ├── model_1.pt (ensemble member 1)
│   ├── model_2.pt
│   ├── model_3.pt
│   ├── model_4.pt
│   ├── model_5.pt
│   ├── config.yaml
│   └── metadata.json
│
├── v2025/
│   ├── trained_2025-01-15/
│   │   ├── model_1.pt
│   │   ├── ...
│   │   ├── training_log.csv
│   │   └── backtest_results.pkl
│
├── v2024/
│   └── trained_2024-01-10/
│       └── ...
│
└── archive/
    └── (old models)
```

**Version Control:**
- Git LFS for model weights
- Track config.yaml and metadata.json in Git
- Tag releases: v2025.1, v2025.2, etc.

**Rollback Criteria:**

```
If new model (deployed <60 days):
  - Sharpe <0.5 for 30 consecutive days
  - Drawdown >20%
  - 3 consecutive weeks of losses
  - Critical bugs/errors

Then:
  - Rollback to previous version
  - Investigate root cause
  - Retrain if necessary
```

---

## 10. Risk Management

### 10.1 Portfolio Constraints

**Hard Constraints (enforced in optimization):**
```
1. Long-only: w_i ≥ 0 for all i
2. Fully invested: sum(w) = 1
3. Minimum volatility: σ_portfolio ≥ 10% annualized
   (scaled to 5-day: σ_5day ≥ 0.10 * sqrt(5/252))
```

**Soft Constraints (monitoring only, no hard limits per specification):**
```
1. No position size limits (can theoretically go to 100%)
2. No sector/group exposure limits
3. No turnover limits
4. No drawdown limits (monitoring only)
5. No stop-loss rules
```

### 10.2 Risk Monitoring

**Daily Risk Checks:**
```
1. Portfolio volatility (realized vs predicted)
2. Maximum position size (track, no limit)
3. Factor concentration (HHI index)
4. Drawdown from peak
5. VaR exceedances (95%, 99%)
```

**Weekly Risk Review:**
```
1. Rolling Sharpe ratio (4-week)
2. Correlation structure changes
3. Regime indicators (VIX, correlations)
4. Prediction accuracy metrics (IC, hit rate)
```

**Monthly Risk Report:**
```
1. Attribution analysis (which factors contributed)
2. Drawdown analysis
3. Comparison to benchmarks
4. Model health metrics
```

### 10.3 Fail-Safe Mechanisms

**Model Failure:**
```
If model inference fails:
  1. Use last valid prediction (up to 5 days old)
  2. If >5 days: Revert to equal-weight portfolio
  3. Alert: CRITICAL
```

**Optimization Failure:**
```
If solver doesn't converge:
  1. Relax constraints slightly (vol 9.5% instead of 10%)
  2. If still fails: Use minimum variance solution
  3. If still fails: Use equal-weight
  4. Alert: CRITICAL
```

**Data Quality Failure:**
```
If >10% factors have missing data:
  1. Continue with available factors (per specification)
  2. If >50% missing: Halt trading
  3. Alert: CRITICAL
```

---

## 11. Success Metrics & KPIs

### 11.1 Model Performance KPIs

**Primary KPI:**
- Out-of-sample Sharpe ratio ≥ 1.0 (minimum), ≥ 1.5 (target)

**Secondary KPIs:**
```
1. Maximum drawdown < 20% (minimum), < 15% (target)
2. Win rate ≥ 55% (minimum), ≥ 60% (target)
3. Information Coefficient ≥ 0.05 (minimum), ≥ 0.08 (target)
4. Calmar ratio ≥ 0.5 (return/max_dd)
5. Sortino ratio ≥ 1.2
```

### 11.2 Prediction Quality KPIs

```
1. Return IC (Information Coefficient): ≥ 0.05
2. Rank IC (Spearman): ≥ 0.08
3. Hit rate (sign prediction): ≥ 55%
4. Covariance RMSE: <20% of average variance
5. Volatility forecast error: <15%
```

### 11.3 Operational KPIs

```
1. Model uptime: ≥ 99.5%
2. Inference latency: <100ms
3. Pipeline completion time: <10 minutes
4. Retraining success rate: 100%
5. Data quality: <5% missing per day
```

### 11.4 Benchmark Outperformance

**Must beat all benchmarks on Sharpe ratio:**
```
1. Equal-weight
2. Minimum variance
3. Risk parity
4. Sample mean-variance
5. Factor momentum
```

**Statistical significance:** p < 0.05 (Jobson-Korkie test)

---

## 12. Implementation Timeline

### 12.1 Phase Breakdown (12 weeks total)

**Week 1-2: Data Preparation**
```
□ Load T2_Optimizer.xlsx (Excel → Parquet)
□ Data quality validation (missing, outliers, duplicates)
□ Feature engineering pipeline
□ Train/val/test split (expanding window)
□ Normalization and scaling (rolling z-score)
□ Save processed datasets

Deliverable: Cleaned data ready for training
```

**Week 3-4: Model Implementation**
```
□ Implement xLSTM encoder (6-layer, sLSTM + mLSTM)
□ Implement return prediction head
□ Implement covariance prediction head (factor model)
□ Implement differentiable portfolio optimization (cvxpylayers)
□ Multi-objective loss function
□ Training loop with logging
□ Checkpoint and early stopping logic

Deliverable: Working xLSTM model, untrained
```

**Week 5-6: Initial Training & Validation**
```
□ Train baseline model (default hyperparameters)
□ Debug training issues
□ Implement validation metrics (Sharpe, IC, etc.)
□ Walk-forward backtesting framework
□ Benchmark implementations (equal-weight, min-var, etc.)
□ Initial performance comparison

Deliverable: Baseline model with backtest results
```

**Week 7-8: Hyperparameter Tuning**
```
□ Grid search (Tier 1 parameters)
□ Random search (Tier 2 parameters)
□ Bayesian optimization (Tier 3 parameters)
□ Ablation studies (features, architecture, loss)
□ Select best configuration

Deliverable: Optimized model configuration
```

**Week 9-10: Ensemble & Robustness**
```
□ Train 5-model ensemble (different seeds)
□ Ensemble averaging logic
□ Stress testing (crisis periods, regime shifts)
□ Sensitivity analysis (parameter perturbations)
□ Robustness validation
□ Final model selection

Deliverable: Production-ready ensemble model
```

**Week 11-12: Production Deployment**
```
□ Production inference pipeline
□ Model versioning system
□ Monitoring dashboard (Grafana/Plotly Dash)
□ Alerting system
□ Documentation (user guide, API docs)
□ Final backtest report and presentation

Deliverable: Deployed system, ready for live operation
```

### 12.2 Critical Path

**Blocking Dependencies:**
```
Week 1-2 (Data) → Must complete before Week 3
Week 3-4 (Model) → Must complete before Week 5
Week 5-6 (Training) → Must complete before Week 7
Week 7-8 (Tuning) → Must complete before Week 9
Week 9-10 (Ensemble) → Must complete before Week 11
```

**Parallelizable Tasks:**
```
- Benchmark implementations (can run during model dev)
- Dashboard development (can run during tuning)
- Documentation (ongoing throughout)
```

### 12.3 Milestones

```
✅ Milestone 1 (Week 2): Clean data pipeline operational
✅ Milestone 2 (Week 4): First model training run completes
✅ Milestone 3 (Week 6): Baseline backtest Sharpe > 0.8
✅ Milestone 4 (Week 8): Tuned model Sharpe > 1.2
✅ Milestone 5 (Week 10): Ensemble Sharpe > 1.5 (target)
✅ Milestone 6 (Week 12): Production system deployed
```

---

## 13. Risks & Mitigation

### 13.1 Technical Risks

**Risk 1: Model Overfitting**
- Probability: Medium
- Impact: High (poor out-of-sample performance)
- Mitigation:
  - Expanding window validation (no look-ahead)
  - Dropout, weight decay, early stopping
  - Walk-forward backtesting
  - Ensemble averaging

**Risk 2: Covariance Prediction Fails**
- Probability: Medium
- Impact: Medium (suboptimal portfolios)
- Mitigation:
  - Hybrid approach: 0.7 * predicted + 0.3 * sample cov
  - Factor model reduces parameters (20 factors vs 5,050)
  - Fallback to sample covariance if predictions unstable

**Risk 3: Optimization Layer Issues**
- Probability: Low
- Impact: High (no gradients, can't train end-to-end)
- Mitigation:
  - Test cvxpylayers thoroughly
  - Fallback: Two-stage (predict returns, then optimize)
  - Alternative: Direct weight prediction (no optimization layer)

**Risk 4: Insufficient Data**
- Probability: Low
- Impact: Medium
- Mitigation:
  - 25 years of data is substantial
  - Regularization prevents overfitting
  - Ensemble reduces variance

**Risk 5: Infrastructure Failures**
- Probability: Low
- Impact: Medium
- Mitigation:
  - Lambda Labs has high uptime
  - Checkpoint frequently
  - Version control all code/models
  - Cloud storage backups

### 13.2 Financial Risks

**Risk 1: Regime Change**
- Probability: High (inevitable)
- Impact: High (performance degradation)
- Mitigation:
  - Annual retraining (adapts to new data)
  - Ensemble diversity
  - Monitor regime indicators (VIX, correlations)
  - No hard stop-loss, but monitoring

**Risk 2: Factor Correlation Breakdown**
- Probability: Medium (during crises)
- Impact: High (predicted cov matrix wrong)
- Mitigation:
  - Stress test on 2008, 2020 data
  - Monitor ensemble disagreement
  - Hybrid predicted + sample cov

**Risk 3: Transaction Costs (future)**
- Probability: High (when enabled)
- Impact: Medium (erodes returns)
- Mitigation:
  - Turnover penalty in loss function (when enabled)
  - Target <50% turnover per rebalance
  - 5-day rebalancing (not daily) reduces turnover

### 13.3 Operational Risks

**Risk 1: Data Quality Issues**
- Probability: Medium
- Impact: High (garbage in, garbage out)
- Mitigation:
  - Automated data validation (range checks, missing data)
  - Alerts for anomalies
  - Handle missing data gracefully (drop factor for that period)

**Risk 2: Model Drift**
- Probability: High (over time)
- Impact: Medium
- Mitigation:
  - Annual retraining
  - Monitor IC, Sharpe, hit rate over time
  - Automatic rollback if performance degrades

**Risk 3: Human Error**
- Probability: Low
- Impact: Medium
- Mitigation:
  - Extensive testing before deployment
  - Version control for all code/models
  - Automated pipelines (minimize manual intervention)

---

## 14. Alternative Approaches & Contingencies

### 14.1 Plan B: Simplified Models

**If xLSTM proves too complex:**

```
Option 1: Vanilla LSTM
  - Replace xLSTM with standard LSTM
  - Fewer parameters, faster training
  - May sacrifice some accuracy

Option 2: Transformer
  - Temporal Transformer with positional encoding
  - Attention mechanism for dependencies
  - Parallel training (faster)

Option 3: Gradient Boosting (XGBoost/LightGBM)
  - Ensemble of trees
  - Strong baseline for tabular data
  - No covariance prediction (use sample cov)
```

### 14.2 Plan C: Dimensionality Reduction

**If 100 factors too noisy:**

```
Option 1: PCA
  - Reduce 100 → 20-30 principal components
  - Explain 80-90% variance
  - Easier to forecast

Option 2: Factor Clustering
  - Cluster into 10 groups (momentum, value, quality, etc.)
  - Predict group returns
  - Within-group optimization

Option 3: Factor Selection
  - Pre-filter to top 30-50 factors by Sharpe, predictability
  - Smaller model, less overfitting
```

### 14.3 Plan D: Alternative Objectives

**If Sharpe maximization too volatile:**

```
Option 1: Minimum Variance
  - Objective: min w'Σw
  - Ignore return predictions (focus on covariance)
  - More conservative

Option 2: Risk Parity
  - Objective: Equal risk contribution
  - More stable allocations
  - Less sensitive to return forecasts

Option 3: Maximum Diversification
  - Objective: max (w'σ) / sqrt(w'Σw)
  - Diversification ratio
  - Balances risk and concentration
```

---

## 15. Documentation Requirements

### 15.1 Technical Documentation

```
1. Architecture Document
   - Model design, layer specifications
   - Data flow diagrams
   - Loss function derivation

2. API Documentation
   - Inference API (input/output schemas)
   - Model loading/saving
   - Utilities (data processing, metrics)

3. Training Guide
   - Hyperparameter descriptions
   - Tuning procedures
   - Debugging tips

4. Deployment Guide
   - Installation instructions
   - Configuration files
   - Monitoring setup
```

### 15.2 Research Documentation

```
1. Experiment Log
   - All hyperparameter trials
   - Validation results
   - Ablation study findings

2. Backtest Report
   - Performance metrics (tables, charts)
   - Comparison to benchmarks
   - Statistical significance tests

3. Model Card
   - Model description
   - Intended use
   - Limitations and risks
   - Performance characteristics
```

### 15.3 User Documentation

```
1. User Guide
   - How to run inference
   - How to interpret predictions
   - How to monitor performance

2. FAQ
   - Common issues and solutions
   - When to retrain
   - How to interpret alerts

3. Runbook
   - Daily operations checklist
   - Incident response procedures
   - Rollback procedures
```

---

## 16. Acceptance Criteria

### 16.1 Functional Requirements

```
✅ Model successfully trains on 100 factors
✅ Predicts 5-day returns (μ) for all factors
✅ Predicts 5-day covariance matrix (Σ, PSD)
✅ Optimizes portfolio weights (long-only, sum=1)
✅ Enforces minimum 10% volatility constraint
✅ Ensemble of 5 models averages predictions
✅ Walk-forward backtest runs on 2021-2025
✅ Inference completes in <100ms
✅ Pipeline runs end-to-end without errors
```

### 16.2 Performance Requirements

```
✅ Out-of-sample Sharpe ≥ 1.0 (minimum acceptable)
✅ Maximum drawdown < 20%
✅ Win rate ≥ 55%
✅ Information Coefficient ≥ 0.05
✅ Beats all 5 benchmarks on Sharpe ratio
✅ Statistical significance p < 0.05
```

### 16.3 Operational Requirements

```
✅ Model checkpoints saved every 10 epochs
✅ Training logs capture all metrics
✅ Monitoring dashboard displays real-time metrics
✅ Alerts trigger on critical errors
✅ Version control for models and code
✅ Documentation complete and accurate
```

---

## 17. Appendices

### Appendix A: Mathematical Notation

```
μ: Expected returns (100 × 1 vector)
Σ: Covariance matrix (100 × 100, PSD)
w: Portfolio weights (100 × 1 vector)
R: Portfolio return = w^T μ
σ: Portfolio volatility = sqrt(w^T Σ w)
S: Sharpe ratio = μ / σ
B: Factor loadings (100 × 20 matrix)
F: Latent factor covariance (20 × 20, PSD)
D: Idiosyncratic variance (100 × 1 vector, positive)
L: Cholesky factor (lower triangular)
```

### Appendix B: File Structure

```
LSTM-T2-Optimizer/
├── data/
│   ├── raw/
│   │   └── T2_Optimizer.xlsx
│   ├── processed/
│   │   ├── train.parquet
│   │   ├── val.parquet
│   │   └── test.parquet
│   └── statistics/
│       ├── normalization_params.pkl
│       └── data_quality_report.html
│
├── src/
│   ├── data/
│   │   ├── loader.py
│   │   ├── preprocessing.py
│   │   └── features.py
│   ├── models/
│   │   ├── xlstm.py
│   │   ├── heads.py
│   │   └── optimizer_layer.py
│   ├── training/
│   │   ├── trainer.py
│   │   ├── loss.py
│   │   └── metrics.py
│   ├── backtesting/
│   │   ├── backtest.py
│   │   ├── benchmarks.py
│   │   └── analysis.py
│   └── utils/
│       ├── config.py
│       └── logging.py
│
├── models/
│   ├── production/
│   ├── v2025/
│   └── archive/
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_model_development.ipynb
│   └── 03_backtest_analysis.ipynb
│
├── configs/
│   ├── model_config.yaml
│   ├── training_config.yaml
│   └── deployment_config.yaml
│
├── tests/
│   ├── test_data.py
│   ├── test_model.py
│   └── test_backtest.py
│
├── docs/
│   ├── architecture.md
│   ├── user_guide.md
│   └── api_reference.md
│
├── requirements.txt
├── setup.py
├── README.md
└── PRD_xLSTM_Factor_Forecasting.md (this document)
```

### Appendix C: Key Libraries

```
Core:
  - torch >= 2.0.0 (PyTorch)
  - numpy >= 1.24.0
  - pandas >= 2.0.0

Optimization:
  - cvxpy >= 1.4.0
  - cvxpylayers >= 0.1.6

ML Tools:
  - optuna >= 3.0.0 (hyperparameter tuning)
  - scikit-learn >= 1.3.0

Data:
  - pyarrow >= 12.0.0 (Parquet)
  - openpyxl >= 3.1.0 (Excel)

Visualization:
  - matplotlib >= 3.7.0
  - seaborn >= 0.12.0
  - plotly >= 5.15.0

Development:
  - pytest >= 7.4.0
  - black >= 23.0.0 (code formatting)
  - mypy >= 1.4.0 (type checking)
```

### Appendix D: References

```
1. xLSTM: Extended Long Short-Term Memory (Beck et al., 2024)
2. xLSTM-TS: Time Series Forecasting with xLSTM (2024)
3. Portfolio Optimization with Deep Learning (multiple papers)
4. Differentiable Convex Optimization Layers (Agrawal et al., 2019)
5. Financial Time Series Forecasting (literature review)
```

---

## 18. Sign-Off

**Project Sponsor:** [Your Name]
**Technical Lead:** Claude (AI Assistant)
**Status:** ✅ **APPROVED - Ready for Implementation**
**Date:** December 14, 2025

---

**Next Action:** Begin Week 1 - Data Preparation

**Contact for Questions:** [Specify communication channel]

---

*End of PRD*
