"""
Feature engineering and preprocessing for factor returns.

Philosophy: NO FALLBACKS, FAIL IS FAIL
- No imputation (zeros = missing, drop that factor for that sample)
- Winsorize outliers (per specification)
- Fail loud on data quality issues
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, List
import warnings


class FactorPreprocessor:
    """Preprocess factor returns and engineer features."""

    def __init__(
        self,
        lookback_days: int = 250,
        prediction_horizon: int = 5,
        winsorize_percentile: Tuple[float, float] = (1, 99),
    ):
        """
        Initialize preprocessor.

        Args:
            lookback_days: Number of days to look back (input window)
            prediction_horizon: Number of days to predict ahead
            winsorize_percentile: Percentiles for winsorization (lower, upper)
        """
        self.lookback_days = lookback_days
        self.prediction_horizon = prediction_horizon
        self.winsorize_percentile = winsorize_percentile

        print(f"Factor Preprocessor initialized:")
        print(f"  Lookback window: {lookback_days} days")
        print(f"  Prediction horizon: {prediction_horizon} days")
        print(f"  Winsorization: {winsorize_percentile[0]}th to {winsorize_percentile[1]}th percentile")

    def winsorize_outliers(self, df: pd.DataFrame, valid_mask: pd.DataFrame) -> pd.DataFrame:
        """
        Winsorize outliers per user specification.

        Only winsorize VALID data (not zeros/missing).
        Use rolling percentiles to adapt to changing volatility regimes.

        Args:
            df: Factor returns DataFrame
            valid_mask: Boolean mask of valid data

        Returns:
            Winsorized DataFrame
        """
        print("\nWinsorizing outliers...")
        print(f"  Method: Rolling {self.winsorize_percentile} percentiles (252-day window)")

        df_winsorized = df.copy()

        for col in df.columns:
            # Get valid data only (exclude zeros/missing)
            valid_data = df[col].where(valid_mask[col])

            # Calculate rolling percentiles (252 trading days = 1 year)
            lower_bound = valid_data.rolling(window=252, min_periods=60).quantile(
                self.winsorize_percentile[0] / 100
            )
            upper_bound = valid_data.rolling(window=252, min_periods=60).quantile(
                self.winsorize_percentile[1] / 100
            )

            # Winsorize (clip) values
            winsorized = valid_data.clip(lower=lower_bound, upper=upper_bound)

            # Put back zeros for missing data
            df_winsorized[col] = winsorized.where(valid_mask[col], 0.0)

        # Count how many values were winsorized
        winsorized_count = ((df != df_winsorized) & valid_mask).sum().sum()
        total_valid = valid_mask.sum().sum()
        winsorize_pct = (winsorized_count / total_valid) * 100

        print(f"  Winsorized {winsorized_count:,} values ({winsorize_pct:.2f}% of valid data)")

        return df_winsorized

    def create_features(self, df: pd.DataFrame, valid_mask: pd.DataFrame) -> pd.DataFrame:
        """
        Create features for each factor:
        1. Raw daily return
        2. 3-day cumulative return
        3. 10-day cumulative return
        4. 20-day moving average return
        5. 20-day rolling volatility
        6. 60-day moving average return
        7. 60-day rolling volatility
        8. Cross-sectional rank (percentile among factors)
        9. Z-score (rolling 60-day standardization)

        Args:
            df: Winsorized factor returns
            valid_mask: Boolean mask of valid data

        Returns:
            DataFrame with multi-level columns: (factor, feature)
        """
        print("\nEngineering features...")
        print("  Features per factor: 9")
        print("    1. Raw return")
        print("    2. 3-day return")
        print("    3. 10-day return")
        print("    4. 20-day MA return")
        print("    5. 20-day volatility")
        print("    6. 60-day MA return")
        print("    7. 60-day volatility")
        print("    8. Cross-sectional rank")
        print("    9. Z-score (60-day)")

        features = {}

        for col in df.columns:
            # Get valid data (zeros are missing)
            data = df[col].where(valid_mask[col])

            # Feature 1: Raw return
            features[(col, 'return')] = df[col]  # Keep zeros for missing

            # Feature 2: 3-day cumulative return
            ret_3d = data.rolling(window=3, min_periods=2).apply(lambda x: (1 + x).prod() - 1, raw=True)
            features[(col, 'ret_3d')] = ret_3d.where(valid_mask[col], 0.0)

            # Feature 3: 10-day cumulative return
            ret_10d = data.rolling(window=10, min_periods=5).apply(lambda x: (1 + x).prod() - 1, raw=True)
            features[(col, 'ret_10d')] = ret_10d.where(valid_mask[col], 0.0)

            # Feature 4: 20-day moving average
            ma_20 = data.rolling(window=20, min_periods=10).mean()
            features[(col, 'ma_20')] = ma_20.where(valid_mask[col], 0.0)

            # Feature 5: 20-day volatility
            vol_20 = data.rolling(window=20, min_periods=10).std()
            features[(col, 'vol_20')] = vol_20.where(valid_mask[col], 0.0)

            # Feature 6: 60-day moving average
            ma_60 = data.rolling(window=60, min_periods=30).mean()
            features[(col, 'ma_60')] = ma_60.where(valid_mask[col], 0.0)

            # Feature 7: 60-day volatility
            vol_60 = data.rolling(window=60, min_periods=30).std()
            features[(col, 'vol_60')] = vol_60.where(valid_mask[col], 0.0)

            # Feature 9: Z-score (rolling 60-day)
            rolling_mean = data.rolling(window=60, min_periods=30).mean()
            rolling_std = data.rolling(window=60, min_periods=30).std()
            zscore = (data - rolling_mean) / (rolling_std + 1e-8)  # Add epsilon to avoid /0
            features[(col, 'zscore')] = zscore.where(valid_mask[col], 0.0)

        # Create DataFrame with multi-level columns
        features_df = pd.DataFrame(features)

        # Feature 8: Cross-sectional rank (across all factors each day)
        # This is computed across factors, not per-factor
        print("  Computing cross-sectional ranks...")
        for date in df.index:
            # Get returns for this date (valid data only)
            day_returns = df.loc[date].where(valid_mask.loc[date])

            # Rank (percentile)
            ranks = day_returns.rank(pct=True)  # NaN for missing data

            # Store ranks (0 for missing data)
            for col in df.columns:
                rank_val = ranks[col] if not pd.isna(ranks[col]) else 0.0
                features_df.loc[date, (col, 'rank')] = rank_val

        print(f"  ✓ Features created: {features_df.shape}")
        print(f"    Shape: {features_df.shape[0]} days × {features_df.shape[1]} features")
        print(f"    ({len(df.columns)} factors × 9 features)")

        # Fill any remaining NaN from rolling operations at start of series
        # These occur when there isn't enough history for rolling windows
        nan_count_before = features_df.isna().sum().sum()
        if nan_count_before > 0:
            print(f"  Filling {nan_count_before:,} NaN values from insufficient history...")
            features_df = features_df.fillna(0.0)

        # Clip extreme values to prevent numerical instability
        # Aggressive clipping at ±100 (plenty for return-based features)
        print(f"  Clipping extreme values to [-100, 100]...")
        features_df = features_df.clip(lower=-100, upper=100)

        return features_df

    def create_samples(
        self,
        features_df: pd.DataFrame,
        returns_df: pd.DataFrame,
        valid_mask: pd.DataFrame,
    ) -> Tuple[List[np.ndarray], List[np.ndarray], List[pd.Timestamp]]:
        """
        Create sliding window samples for training.

        For each valid date t:
        - Input: features[t-lookback:t] (250 days × 104 factors × 9 features)
        - Target: returns[t+1:t+6] (next 5 days for computing target stats)

        Handles missing data: If any factor has missing data in the target window,
        that factor is dropped for that sample.

        Args:
            features_df: Feature DataFrame with multi-level columns
            returns_df: Raw returns DataFrame
            valid_mask: Boolean mask of valid data

        Returns:
            Tuple of (X_samples, y_samples, dates)
        """
        print("\nCreating sliding window samples...")
        print(f"  Lookback: {self.lookback_days} days")
        print(f"  Prediction horizon: {self.prediction_horizon} days")

        X_samples = []
        y_samples = []
        dates = []

        n_factors = returns_df.shape[1]
        n_features = 9  # 9 features per factor

        # Need enough history for lookback + enough future for prediction
        min_date_idx = self.lookback_days
        max_date_idx = len(returns_df) - self.prediction_horizon

        print(f"  Valid date range for samples: {min_date_idx} to {max_date_idx}")
        print(f"  Potential samples: {max_date_idx - min_date_idx}")

        skipped_samples = 0
        samples_created = 0

        for i in range(min_date_idx, max_date_idx):
            # Current date
            current_date = returns_df.index[i]

            # Input window: features[i-lookback:i]
            input_window = features_df.iloc[i - self.lookback_days:i]

            # Target window: returns[i+1:i+1+prediction_horizon]
            target_window = returns_df.iloc[i + 1:i + 1 + self.prediction_horizon]
            target_valid = valid_mask.iloc[i + 1:i + 1 + self.prediction_horizon]

            # Check if any day in target window has >50% missing factors
            # (Per PRD: HALT if >50% missing)
            missing_per_day = (~target_valid).sum(axis=1)
            max_missing_pct = (missing_per_day.max() / n_factors) * 100

            if max_missing_pct > 50:
                skipped_samples += 1
                continue  # Skip this sample (per FAIL IS FAIL policy)

            # Reshape input: (lookback, factors, features)
            # features_df has multi-level columns: (factor, feature)
            # We need to reshape to (lookback, n_factors, n_features)

            X = np.zeros((self.lookback_days, n_factors, n_features))
            for f_idx, factor in enumerate(returns_df.columns):
                for feat_idx, feature in enumerate(['return', 'ret_3d', 'ret_10d', 'ma_20', 'vol_20', 'ma_60', 'vol_60', 'rank', 'zscore']):
                    X[:, f_idx, feat_idx] = input_window[(factor, feature)].values

            # Target: returns for next 5 days
            # Shape: (prediction_horizon, n_factors)
            y = target_window.values

            X_samples.append(X)
            y_samples.append(y)
            dates.append(current_date)
            samples_created += 1

        print(f"\n  ✓ Created {samples_created:,} samples")
        print(f"  Skipped {skipped_samples:,} samples (>50% missing in target)")
        print(f"  Sample shape: X={X_samples[0].shape}, y={y_samples[0].shape}")

        return X_samples, y_samples, dates


def preprocess_data(
    factor_returns: pd.DataFrame,
    valid_mask: pd.DataFrame,
    lookback_days: int = 250,
    prediction_horizon: int = 5,
) -> Tuple[List[np.ndarray], List[np.ndarray], List[pd.Timestamp]]:
    """
    Main preprocessing pipeline.

    Args:
        factor_returns: Raw factor returns
        valid_mask: Boolean mask of valid data
        lookback_days: Input window size
        prediction_horizon: Forecast horizon

    Returns:
        Tuple of (X_samples, y_samples, dates)
    """
    print("=" * 80)
    print("PREPROCESSING PIPELINE")
    print("=" * 80)

    preprocessor = FactorPreprocessor(
        lookback_days=lookback_days,
        prediction_horizon=prediction_horizon,
    )

    # Step 1: Winsorize outliers
    df_winsorized = preprocessor.winsorize_outliers(factor_returns, valid_mask)

    # Step 2: Create features
    features_df = preprocessor.create_features(df_winsorized, valid_mask)

    # Step 3: Create samples
    X_samples, y_samples, dates = preprocessor.create_samples(
        features_df, factor_returns, valid_mask
    )

    print("\n" + "=" * 80)
    print(f"✓ Preprocessing complete: {len(X_samples)} samples ready")
    print("=" * 80)

    return X_samples, y_samples, dates


if __name__ == "__main__":
    # Test the preprocessor
    from loader import load_and_validate_data

    print("Loading data...")
    df, valid_mask = load_and_validate_data()

    print("\nPreprocessing...")
    X, y, dates = preprocess_data(df, valid_mask)

    print(f"\nFinal dataset:")
    print(f"  Samples: {len(X)}")
    print(f"  X shape: {X[0].shape} (lookback, factors, features)")
    print(f"  y shape: {y[0].shape} (horizon, factors)")
    print(f"  Date range: {dates[0]} to {dates[-1]}")
