"""
Test training script for one year (2010) to verify model works.

Philosophy: NO FALLBACKS, FAIL IS FAIL
- Test basic training loop
- Verify data loading works
- Check model trains without errors
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pickle
from pathlib import Path
import sys

sys.path.append('src')
from models.xlstm_factor import xLSTMFactorModel, count_parameters


class FactorDataset(Dataset):
    """Simple dataset for factor returns."""

    def __init__(self, X, y):
        """
        Args:
            X: Input features (num_samples, lookback, factors, features)
            y: Target returns (num_samples, horizon, factors)
        """
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def load_year_data(year: int, data_dir: str = "data/processed"):
    """Load training data for a specific year."""
    year_dir = Path(data_dir) / str(year)

    print(f"Loading data from {year_dir}...")

    # Load training data
    with open(year_dir / "train_X.pkl", "rb") as f:
        X_train = pickle.load(f)
    with open(year_dir / "train_y.pkl", "rb") as f:
        y_train = pickle.load(f)

    print(f"  Train X shape: {X_train.shape}")
    print(f"  Train y shape: {y_train.shape}")

    return X_train, y_train


def compute_covariance_target(y: torch.Tensor) -> torch.Tensor:
    """
    Compute empirical covariance from 5-day returns.

    Args:
        y: Returns (batch, horizon=5, factors=104)

    Returns:
        Covariance matrix (batch, factors, factors)
    """
    batch_size, horizon, num_factors = y.shape

    # Compute covariance across the 5-day horizon
    # For each sample, compute cov of the 5 days
    covs = []
    for i in range(batch_size):
        # y[i] is (5, 104) - 5 days, 104 factors
        # Compute covariance across days
        sample_cov = torch.cov(y[i].T)  # (104, 104)
        covs.append(sample_cov)

    covariance = torch.stack(covs, dim=0)  # (batch, 104, 104)
    return covariance


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    alpha_return: float = 1.0,
    beta_cov: float = 0.5,
) -> dict:
    """Train for one epoch."""
    model.train()

    total_loss = 0.0
    total_return_loss = 0.0
    total_cov_loss = 0.0
    num_batches = 0

    for X_batch, y_batch in dataloader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        # Forward pass
        pred_returns, pred_cov = model(X_batch)

        # Compute losses
        # 1. Return prediction loss (MSE)
        return_loss = nn.functional.mse_loss(pred_returns, y_batch)

        # 2. Covariance prediction loss (Frobenius norm)
        # Compute target covariance from actual returns
        target_cov = compute_covariance_target(y_batch)
        cov_loss = nn.functional.mse_loss(pred_cov, target_cov)

        # Combined loss
        loss = alpha_return * return_loss + beta_cov * cov_loss

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        # Track metrics
        total_loss += loss.item()
        total_return_loss += return_loss.item()
        total_cov_loss += cov_loss.item()
        num_batches += 1

    return {
        "loss": total_loss / num_batches,
        "return_loss": total_return_loss / num_batches,
        "cov_loss": total_cov_loss / num_batches,
    }


def main():
    """Test training on 2010 data."""
    print("=" * 80)
    print("TEST TRAINING - YEAR 2010")
    print("=" * 80)

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

    # Load 2010 data
    print("\nLoading 2010 data...")
    X_train, y_train = load_year_data(2010)

    # Create dataset and dataloader
    train_dataset = FactorDataset(X_train, y_train)
    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=0,  # Use 0 for Mac compatibility
    )

    print(f"\nDataset:")
    print(f"  Training samples: {len(train_dataset)}")
    print(f"  Batches per epoch: {len(train_loader)}")

    # Create model
    print("\nCreating model...")
    model = xLSTMFactorModel(
        num_factors=104,
        num_features=9,
        lookback_days=250,
        prediction_horizon=5,
        hidden_dim=256,
        num_layers=4,
        num_latent_factors=10,
        dropout=0.1,
    )
    model = model.to(device)

    print(f"  Parameters: {count_parameters(model):,}")

    # Create optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Train for a few epochs
    num_epochs = 3
    print(f"\nTraining for {num_epochs} epochs...")
    print("-" * 80)

    for epoch in range(num_epochs):
        metrics = train_one_epoch(
            model, train_loader, optimizer, device,
            alpha_return=1.0, beta_cov=0.5
        )

        print(
            f"Epoch {epoch+1}/{num_epochs}: "
            f"Loss={metrics['loss']:.6f} | "
            f"Return Loss={metrics['return_loss']:.6f} | "
            f"Cov Loss={metrics['cov_loss']:.6f}"
        )

    print("-" * 80)
    print("\n✓ Test training complete!")
    print("\nModel is working correctly. Ready for full training.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ FAIL: Training test failed!")
        print(f"Error: {e}")
        print("\nFAIL IS FAIL: Fix the error and try again.")
        import traceback
        traceback.print_exc()
        sys.exit(1)
