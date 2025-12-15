"""
Data loader for factor returns and market data.

Philosophy: NO FALLBACKS, FAIL IS FAIL
- No synthetic data
- No imputation (zeros = missing, drop that factor for that day)
- Fail loud on data quality issues
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Dict
import warnings

# Suppress openpyxl warnings
warnings.filterwarnings('ignore', category=UserWarning, module='openpyxl')


class FactorDataLoader:
    """Load and validate factor returns data."""

    def __init__(self, data_path: str = "data/raw/T2_Optimizer.xlsx"):
        self.data_path = Path(data_path)
        if not self.data_path.exists():
            raise FileNotFoundError(
                f"Data file not found: {self.data_path}\n"
                "FAIL IS FAIL: Cannot proceed without data."
            )

    def load_factor_returns(self) -> pd.DataFrame:
        """
        Load factor returns from Excel file.

        Returns:
            DataFrame with DatetimeIndex and factor columns

        Raises:
            ValueError: If data validation fails
        """
        print(f"Loading factor returns from {self.data_path}...")

        # Load Excel file
        df = pd.read_excel(self.data_path, sheet_name='Monthly_Net_Returns')

        # Validate structure
        if 'Date' not in df.columns:
            raise ValueError(
                "FAIL: 'Date' column not found in data.\n"
                "FAIL IS FAIL: Cannot proceed without proper date column."
            )

        # Set date as index
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.set_index('Date')
        df = df.sort_index()

        # Validate data
        n_rows, n_factors = df.shape
        print(f"  Shape: {n_rows} days × {n_factors} factors")
        print(f"  Date range: {df.index.min()} to {df.index.max()}")
        print(f"  Trading days span: {(df.index.max() - df.index.min()).days} days")

        # Check for non-numeric columns
        non_numeric = df.select_dtypes(exclude=[np.number]).columns
        if len(non_numeric) > 0:
            raise ValueError(
                f"FAIL: Non-numeric columns found: {list(non_numeric)}\n"
                "FAIL IS FAIL: All factor columns must be numeric."
            )

        # Validate returns are in reasonable range (|-100%| to |+1000%|)
        # This catches obvious data errors
        extreme_values = (df.abs() > 1000).sum().sum()
        if extreme_values > 0:
            raise ValueError(
                f"FAIL: {extreme_values} values exceed ±1000%.\n"
                "FAIL IS FAIL: Data contains unrealistic returns."
            )

        print(f"  ✓ Factor returns loaded successfully")
        return df

    def identify_missing_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Identify missing data (zeros indicate missing per user specification).

        Args:
            df: Factor returns DataFrame

        Returns:
            Boolean DataFrame (True = valid data, False = missing)
        """
        print("\nIdentifying missing data...")
        print("  (Zeros indicate missing data per specification)")

        # Zeros are missing data
        valid_mask = df != 0.0

        # Calculate missing data statistics
        total_values = df.shape[0] * df.shape[1]
        missing_values = (~valid_mask).sum().sum()
        missing_pct = (missing_values / total_values) * 100

        print(f"  Total data points: {total_values:,}")
        print(f"  Missing (zeros): {missing_values:,} ({missing_pct:.2f}%)")

        # Per-factor missing data
        factor_missing = (~valid_mask).sum()
        factors_with_missing = (factor_missing > 0).sum()
        print(f"  Factors with missing data: {factors_with_missing}/{df.shape[1]}")

        # Per-day missing data
        day_missing = (~valid_mask).sum(axis=1)
        max_missing_day = day_missing.max()
        max_missing_pct = (max_missing_day / df.shape[1]) * 100
        print(f"  Worst day: {max_missing_day}/{df.shape[1]} factors missing ({max_missing_pct:.1f}%)")

        # Check if any day has >50% missing (would trigger HALT per PRD)
        days_over_50pct = (day_missing / df.shape[1] > 0.5).sum()
        if days_over_50pct > 0:
            print(f"  ⚠️  WARNING: {days_over_50pct} days have >50% factors missing")
            print("  These days will cause issues per FAIL IS FAIL policy")

        return valid_mask

    def get_data_quality_report(self, df: pd.DataFrame, valid_mask: pd.DataFrame) -> Dict:
        """
        Generate comprehensive data quality report.

        Args:
            df: Factor returns DataFrame
            valid_mask: Boolean mask of valid data

        Returns:
            Dictionary with data quality metrics
        """
        report = {
            'total_days': len(df),
            'total_factors': len(df.columns),
            'date_range': (df.index.min(), df.index.max()),
            'total_values': df.shape[0] * df.shape[1],
            'missing_values': (~valid_mask).sum().sum(),
            'missing_pct': ((~valid_mask).sum().sum() / (df.shape[0] * df.shape[1])) * 100,
            'factors_with_missing': (valid_mask.sum() < len(df)).sum(),
            'days_with_missing': (valid_mask.sum(axis=1) < len(df.columns)).sum(),
            'max_missing_per_day': (~valid_mask).sum(axis=1).max(),
            'max_missing_per_factor': (~valid_mask).sum().max(),
        }

        return report


def load_and_validate_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Main function to load and validate factor data.

    Returns:
        Tuple of (factor_returns, valid_mask)
    """
    loader = FactorDataLoader()
    df = loader.load_factor_returns()
    valid_mask = loader.identify_missing_data(df)
    report = loader.get_data_quality_report(df, valid_mask)

    print("\n" + "=" * 80)
    print("DATA QUALITY REPORT")
    print("=" * 80)
    for key, value in report.items():
        print(f"{key:.<30} {value}")
    print("=" * 80)

    return df, valid_mask


if __name__ == "__main__":
    # Test the loader
    df, valid_mask = load_and_validate_data()
    print("\n✓ Data loading successful!")
    print(f"\nFactor returns shape: {df.shape}")
    print(f"Valid mask shape: {valid_mask.shape}")
