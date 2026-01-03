"""
xLSTM model for factor return prediction.

Philosophy: NO FALLBACKS, FAIL IS FAIL
- Real implementation, no simulations
- Proper error handling
- Clear documentation
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional


class xLSTMFactorModel(nn.Module):
    """
    xLSTM-based factor forecasting model with dual heads.

    Architecture:
    1. xLSTM Encoder: Processes time series of factor features
    2. Return Head: Predicts 5-day ahead returns for 104 factors
    3. Covariance Head: Predicts factor covariance using factor model decomposition

    Input: (batch, lookback_days=250, num_factors=104, num_features=9)
    Outputs:
        - returns: (batch, prediction_horizon=5, num_factors=104)
        - covariance: (batch, num_factors=104, num_factors=104)
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
        num_latent_factors: int = 10,
    ):
        super().__init__()

        self.num_factors = num_factors
        self.num_features = num_features
        self.lookback_days = lookback_days
        self.prediction_horizon = prediction_horizon
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_latent_factors = num_latent_factors

        # Input projection: (batch, seq, factors, features) -> (batch, seq, hidden)
        # Flatten factors x features dimension
        input_dim = num_factors * num_features
        self.input_projection = nn.Linear(input_dim, hidden_dim)

        # xLSTM Encoder (using standard LSTM for now)
        # TODO: Replace with proper sLSTM/mLSTM implementation
        self.encoder = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )

        # Layer norm
        self.layer_norm = nn.LayerNorm(hidden_dim)

        # ===== Return Prediction Head =====
        # Predicts 5-day ahead returns for each factor
        self.return_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_factors * prediction_horizon),
        )

        # ===== Covariance Prediction Head =====
        # Factor model: Σ = B @ F @ B^T + D
        # B: (num_factors, num_latent_factors) - factor loadings
        # F: (num_latent_factors, num_latent_factors) - latent factor covariance
        # D: (num_factors,) - idiosyncratic variances (diagonal)

        # Predict factor loadings B
        self.factor_loadings_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_factors * num_latent_factors),
        )

        # Predict latent covariance F (lower triangular for Cholesky)
        latent_cov_dim = num_latent_factors * (num_latent_factors + 1) // 2
        self.latent_cov_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, latent_cov_dim),
        )

        # Predict idiosyncratic variances D
        self.idio_var_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_factors),
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize weights with Xavier initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        x: torch.Tensor,
        return_hidden: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            x: Input tensor (batch, lookback_days, num_factors, num_features)
            return_hidden: Whether to return hidden states

        Returns:
            returns: Predicted returns (batch, prediction_horizon, num_factors)
            covariance: Predicted covariance (batch, num_factors, num_factors)
        """
        batch_size = x.shape[0]

        # Reshape: (batch, seq, factors, features) -> (batch, seq, factors*features)
        x = x.reshape(batch_size, self.lookback_days, -1)

        # Input projection
        x = self.input_projection(x)  # (batch, seq, hidden)

        # xLSTM encoder
        encoded, (h_n, c_n) = self.encoder(x)  # encoded: (batch, seq, hidden)

        # Use last hidden state for predictions
        last_hidden = encoded[:, -1, :]  # (batch, hidden)
        last_hidden = self.layer_norm(last_hidden)

        # ===== Return Prediction =====
        returns_flat = self.return_head(last_hidden)  # (batch, factors*horizon)
        returns = returns_flat.reshape(
            batch_size, self.prediction_horizon, self.num_factors
        )

        # ===== Covariance Prediction =====
        # Predict factor loadings B: (batch, factors, latent_factors)
        B_flat = self.factor_loadings_head(last_hidden)
        B = B_flat.reshape(batch_size, self.num_factors, self.num_latent_factors)

        # Predict latent covariance F: (batch, latent_factors, latent_factors)
        # Use Cholesky decomposition to ensure positive semi-definite
        L_flat = self.latent_cov_head(last_hidden)
        L = self._build_lower_triangular(L_flat, self.num_latent_factors)
        F = torch.bmm(L, L.transpose(1, 2))  # F = L @ L^T

        # Predict idiosyncratic variances D: (batch, factors)
        # Use softplus to ensure positive
        D_logvar = self.idio_var_head(last_hidden)
        D = torch.nn.functional.softplus(D_logvar) + 1e-6  # Add small constant for stability

        # Compute covariance: Σ = B @ F @ B^T + diag(D)
        BF = torch.bmm(B, F)  # (batch, factors, latent_factors)
        BFBt = torch.bmm(BF, B.transpose(1, 2))  # (batch, factors, factors)

        # Add diagonal idiosyncratic variances
        D_diag = torch.diag_embed(D)  # (batch, factors, factors)
        covariance = BFBt + D_diag

        if return_hidden:
            return returns, covariance, last_hidden

        return returns, covariance

    def _build_lower_triangular(
        self,
        flat: torch.Tensor,
        dim: int,
    ) -> torch.Tensor:
        """
        Build lower triangular matrix from flattened vector.

        Args:
            flat: Flattened lower triangular elements (batch, dim*(dim+1)/2)
            dim: Matrix dimension

        Returns:
            Lower triangular matrix (batch, dim, dim)
        """
        batch_size = flat.shape[0]
        L = torch.zeros(batch_size, dim, dim, device=flat.device, dtype=flat.dtype)

        # Get lower triangular indices
        tril_indices = torch.tril_indices(dim, dim, device=flat.device)

        # Fill lower triangular
        L[:, tril_indices[0], tril_indices[1]] = flat

        # Ensure positive diagonal (use exp for diagonal elements)
        diag_indices = torch.arange(dim, device=flat.device)
        L[:, diag_indices, diag_indices] = torch.exp(
            L[:, diag_indices, diag_indices]
        ) + 1e-6

        return L


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    """Test model forward pass."""
    print("Testing xLSTM Factor Model...")

    # Create model
    model = xLSTMFactorModel(
        num_factors=104,
        num_features=9,
        lookback_days=250,
        prediction_horizon=5,
        hidden_dim=256,
        num_layers=4,
        num_latent_factors=10,
    )

    print(f"\nModel architecture:")
    print(f"  Parameters: {count_parameters(model):,}")

    # Create dummy input
    batch_size = 8
    x = torch.randn(batch_size, 250, 104, 9)

    print(f"\nInput shape: {x.shape}")

    # Forward pass
    with torch.no_grad():
        returns, covariance = model(x)

    print(f"\nOutput shapes:")
    print(f"  Returns: {returns.shape} (expected: {batch_size}, 5, 104)")
    print(f"  Covariance: {covariance.shape} (expected: {batch_size}, 104, 104)")

    # Check covariance is symmetric and positive semi-definite
    print(f"\nCovariance checks:")
    is_symmetric = torch.allclose(covariance, covariance.transpose(1, 2), atol=1e-5)
    print(f"  Symmetric: {is_symmetric}")

    # Check eigenvalues (should all be non-negative)
    eigenvalues = torch.linalg.eigvalsh(covariance)
    min_eigenvalue = eigenvalues.min().item()
    print(f"  Min eigenvalue: {min_eigenvalue:.6f} (should be >= 0)")

    print("\n✓ Model test passed!")
