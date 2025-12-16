"""
xLSTM model for factor return prediction ONLY.

Philosophy: NO FALLBACKS, FAIL IS FAIL
- Predict returns only (no covariance - use historical)
- Focus model capacity on the hard problem (return prediction)
- Simpler model = less overfitting
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional


class xLSTMReturnsModel(nn.Module):
    """
    xLSTM-based factor return forecasting model.

    Predicts 5-day ahead returns for 104 factors.
    Covariance is NOT predicted - use historical covariance instead.

    Architecture:
    1. Input projection: (batch, seq, factors*features) -> (batch, seq, hidden)
    2. xLSTM Encoder: Process time series
    3. Return Head: Predict 5-day returns

    Input: (batch, lookback_days=250, num_factors=104, num_features=9)
    Output: (batch, prediction_horizon=5, num_factors=104)
    """

    def __init__(
        self,
        num_factors: int = 104,
        num_features: int = 9,
        lookback_days: int = 250,
        prediction_horizon: int = 5,
        hidden_dim: int = 256,
        num_layers: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.num_factors = num_factors
        self.num_features = num_features
        self.lookback_days = lookback_days
        self.prediction_horizon = prediction_horizon
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Input projection: flatten factors*features -> hidden
        input_dim = num_factors * num_features
        self.input_projection = nn.Linear(input_dim, hidden_dim)

        # xLSTM Encoder (using standard LSTM - can upgrade to proper xLSTM later)
        self.encoder = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )

        # Layer norm for stability
        self.layer_norm = nn.LayerNorm(hidden_dim)

        # Return prediction head
        self.return_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_factors * prediction_horizon),
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize weights with Xavier initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor (batch, lookback_days, num_factors, num_features)

        Returns:
            returns: Predicted returns (batch, prediction_horizon, num_factors)
        """
        batch_size = x.shape[0]

        # Reshape: (batch, seq, factors, features) -> (batch, seq, factors*features)
        x = x.reshape(batch_size, self.lookback_days, -1)

        # Input projection
        x = self.input_projection(x)  # (batch, seq, hidden)

        # xLSTM encoder
        encoded, _ = self.encoder(x)  # (batch, seq, hidden)

        # Use last hidden state
        last_hidden = encoded[:, -1, :]  # (batch, hidden)
        last_hidden = self.layer_norm(last_hidden)

        # Predict returns
        returns_flat = self.return_head(last_hidden)  # (batch, factors*horizon)
        returns = returns_flat.reshape(
            batch_size, self.prediction_horizon, self.num_factors
        )

        return returns


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    """Test model forward pass."""
    print("Testing xLSTM Returns Model...")

    model = xLSTMReturnsModel(
        num_factors=104,
        num_features=9,
        lookback_days=250,
        prediction_horizon=5,
        hidden_dim=256,
        num_layers=4,
    )

    print(f"\nModel: {count_parameters(model):,} parameters")

    # Test forward pass
    batch_size = 8
    x = torch.randn(batch_size, 250, 104, 9)

    with torch.no_grad():
        returns = model(x)

    print(f"Input: {x.shape}")
    print(f"Output: {returns.shape} (expected: {batch_size}, 5, 104)")
    print("\n✓ Model test passed!")
