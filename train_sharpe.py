"""
Training script with Sharpe ratio loss.

Philosophy: NO FALLBACKS, FAIL IS FAIL
- Predict returns only
- Use historical covariance (fixed)
- Optimize for Sharpe ratio
- Full end-to-end differentiability
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pickle
from pathlib import Path
import sys
import time

sys.path.append('src')
from models.xlstm_returns import xLSTMReturnsModel, count_parameters
from models.portfolio_opt import (
    MeanVarianceLayer,
    compute_portfolio_return,
    compute_portfolio_volatility,
    compute_sharpe_ratio,
)


class FactorDataset(Dataset):
    """Dataset for factor returns."""

    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def load_year_data(year: int, data_dir: str = "data/processed"):
    """Load training data for a specific year."""
    year_dir = Path(data_dir) / str(year)

    with open(year_dir / "train_X.pkl", "rb") as f:
        X_train = pickle.load(f)
    with open(year_dir / "train_y.pkl", "rb") as f:
        y_train = pickle.load(f)

    return X_train, y_train


def compute_historical_covariance(y_train: np.ndarray, lookback: int = 60) -> torch.Tensor:
    """
    Compute historical covariance from training returns.

    Args:
        y_train: Training returns (num_samples, horizon, num_factors)
        lookback: Number of samples to use for covariance

    Returns:
        Covariance matrix (num_factors, num_factors)
    """
    # Use the last `lookback` samples
    # Average across the 5-day horizon to get daily-equivalent returns
    recent_returns = y_train[-lookback:].mean(axis=1)  # (lookback, num_factors)

    # Compute covariance
    cov = np.cov(recent_returns.T)  # (num_factors, num_factors)

    return torch.FloatTensor(cov)


def train_one_epoch(
    model: nn.Module,
    portfolio_layer: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    covariance: torch.Tensor,
    device: torch.device,
    alpha_sharpe: float = 1.0,
    alpha_mse: float = 0.1,
) -> dict:
    """
    Train for one epoch.

    Loss = -alpha_sharpe * Sharpe + alpha_mse * MSE

    Args:
        model: xLSTM returns model
        portfolio_layer: Portfolio optimization layer
        dataloader: Training data loader
        optimizer: Optimizer
        covariance: Fixed covariance matrix
        device: Device to use
        alpha_sharpe: Weight for Sharpe ratio loss
        alpha_mse: Weight for MSE loss (regularization)

    Returns:
        Dictionary with metrics
    """
    model.train()

    total_loss = 0.0
    total_sharpe = 0.0
    total_mse = 0.0
    total_return = 0.0
    total_vol = 0.0
    num_batches = 0

    covariance = covariance.to(device)

    for X_batch, y_batch in dataloader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        # Forward pass
        # Model predicts 5-day returns: (batch, 5, 104)
        pred_returns = model(X_batch)

        # Average to get expected return per factor: (batch, 104)
        pred_mean_returns = pred_returns.mean(dim=1)

        # Get portfolio weights from predicted returns
        weights = portfolio_layer(pred_mean_returns)

        # Actual returns (average over 5-day horizon): (batch, 104)
        actual_mean_returns = y_batch.mean(dim=1)

        # Compute portfolio metrics using ACTUAL returns (not predicted)
        # This is the realized performance
        port_return = compute_portfolio_return(weights, actual_mean_returns)
        port_vol = compute_portfolio_volatility(weights, covariance)
        sharpe = port_return / (port_vol + 1e-8) * np.sqrt(252)

        # MSE loss on return predictions (regularization)
        mse_loss = nn.functional.mse_loss(pred_returns, y_batch)

        # Combined loss: maximize Sharpe (minimize -Sharpe) + MSE regularization
        loss = -alpha_sharpe * sharpe.mean() + alpha_mse * mse_loss

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        # Track metrics
        total_loss += loss.item()
        total_sharpe += sharpe.mean().item()
        total_mse += mse_loss.item()
        total_return += port_return.mean().item()
        total_vol += port_vol.mean().item()
        num_batches += 1

    return {
        "loss": total_loss / num_batches,
        "sharpe": total_sharpe / num_batches,
        "mse": total_mse / num_batches,
        "return": total_return / num_batches,
        "volatility": total_vol / num_batches,
    }


def main():
    """Train on 2010 data with Sharpe ratio loss."""
    print("=" * 80)
    print("TRAINING WITH SHARPE RATIO LOSS")
    print("=" * 80)

    # Configuration
    year = 2010
    batch_size = 32
    num_epochs = 10
    learning_rate = 1e-3
    hidden_dim = 256
    num_layers = 4
    risk_aversion = 0.5  # Lower = more risk = higher vol

    # Set device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("\n✓ Using MPS (Metal Performance Shaders)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("\n✓ Using CUDA")
    else:
        device = torch.device("cpu")
        print("\n⚠️  Using CPU")

    # Load data
    print(f"\nLoading {year} data...")
    X_train, y_train = load_year_data(year)
    print(f"  X_train: {X_train.shape}")
    print(f"  y_train: {y_train.shape}")

    # Compute historical covariance
    print("\nComputing historical covariance...")
    covariance = compute_historical_covariance(y_train, lookback=252)
    print(f"  Covariance: {covariance.shape}")

    # Create dataset and dataloader
    train_dataset = FactorDataset(X_train, y_train)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )
    print(f"\nDataset:")
    print(f"  Samples: {len(train_dataset)}")
    print(f"  Batches: {len(train_loader)}")

    # Create model
    print("\nCreating model...")
    model = xLSTMReturnsModel(
        num_factors=104,
        num_features=9,
        lookback_days=250,
        prediction_horizon=5,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=0.1,
    )
    model = model.to(device)
    print(f"  Parameters: {count_parameters(model):,}")

    # Create portfolio layer
    portfolio_layer = MeanVarianceLayer(
        num_assets=104,
        risk_aversion=risk_aversion,
    )
    portfolio_layer.set_covariance(covariance)
    portfolio_layer = portfolio_layer.to(device)

    # Create optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    # Training loop
    print(f"\nTraining for {num_epochs} epochs...")
    print("-" * 80)
    print(f"{'Epoch':>5} | {'Loss':>10} | {'Sharpe':>10} | {'Return':>10} | {'Vol':>10} | {'MSE':>10} | {'Time':>8}")
    print("-" * 80)

    total_time = 0.0

    for epoch in range(num_epochs):
        start_time = time.time()

        metrics = train_one_epoch(
            model=model,
            portfolio_layer=portfolio_layer,
            dataloader=train_loader,
            optimizer=optimizer,
            covariance=covariance,
            device=device,
            alpha_sharpe=1.0,
            alpha_mse=0.1,
        )

        epoch_time = time.time() - start_time
        total_time += epoch_time

        scheduler.step()

        print(
            f"{epoch+1:>5} | "
            f"{metrics['loss']:>10.4f} | "
            f"{metrics['sharpe']:>10.4f} | "
            f"{metrics['return']*100:>9.4f}% | "
            f"{metrics['volatility']*100:>9.4f}% | "
            f"{metrics['mse']:>10.6f} | "
            f"{epoch_time:>7.2f}s"
        )

    print("-" * 80)
    print(f"\n✓ Training complete!")
    print(f"  Total time: {total_time:.1f}s")
    print(f"  Avg time/epoch: {total_time/num_epochs:.2f}s")
    print(f"  Final Sharpe: {metrics['sharpe']:.4f}")
    print(f"  Final Volatility: {metrics['volatility']*100:.2f}%")

    # Estimate full training time
    print(f"\n--- Time Estimates ---")
    print(f"  Mac M4 Max (100 epochs): {total_time/num_epochs * 100 / 60:.1f} min")
    print(f"  Mac M4 Max (15 years): {total_time/num_epochs * 100 * 15 / 3600:.1f} hours")
    print(f"  H100 estimate (10x faster): {total_time/num_epochs * 100 * 15 / 3600 / 10:.1f} hours")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ FAIL: Training failed!")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
