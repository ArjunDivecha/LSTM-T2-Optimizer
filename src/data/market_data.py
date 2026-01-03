"""
Fetch market and macro data from public sources.

Philosophy: NO FALLBACKS, FAIL IS FAIL
- No synthetic data
- Fail loud if data sources are unavailable
"""

import pandas as pd
import pandas_datareader as pdr
from datetime import datetime
from pathlib import Path
import warnings


class MarketDataFetcher:
    """Fetch market and macro data for the model."""

    def __init__(self, start_date: str = "2000-01-01", end_date: str = None):
        """
        Initialize market data fetcher.

        Args:
            start_date: Start date for data fetch (YYYY-MM-DD)
            end_date: End date for data fetch (YYYY-MM-DD), defaults to today
        """
        self.start_date = pd.to_datetime(start_date)
        self.end_date = pd.to_datetime(end_date) if end_date else pd.Timestamp.now()

        print(f"Market data fetcher initialized")
        print(f"  Date range: {self.start_date.date()} to {self.end_date.date()}")

    def fetch_spy_returns(self) -> pd.Series:
        """
        Fetch SPY (S&P 500 ETF) daily returns.

        Returns:
            Series of daily returns

        Raises:
            RuntimeError: If data fetch fails
        """
        print("\nFetching SPY (S&P 500) data...")
        try:
            # Fetch from Yahoo Finance via pandas_datareader
            spy = pdr.get_data_yahoo('SPY', start=self.start_date, end=self.end_date)
            spy_returns = spy['Adj Close'].pct_change()
            spy_returns = spy_returns.dropna()

            print(f"  ✓ Fetched {len(spy_returns)} days of SPY returns")
            print(f"    Range: {spy_returns.index.min().date()} to {spy_returns.index.max().date()}")

            return spy_returns

        except Exception as e:
            raise RuntimeError(
                f"FAIL: Could not fetch SPY data.\n"
                f"Error: {e}\n"
                "FAIL IS FAIL: Cannot proceed without market data."
            )

    def fetch_vix(self) -> pd.Series:
        """
        Fetch VIX (CBOE Volatility Index) daily levels.

        Returns:
            Series of VIX levels

        Raises:
            RuntimeError: If data fetch fails
        """
        print("\nFetching VIX (Volatility Index) data...")
        try:
            # Fetch from Yahoo Finance
            vix = pdr.get_data_yahoo('^VIX', start=self.start_date, end=self.end_date)
            vix_levels = vix['Adj Close']
            vix_levels = vix_levels.dropna()

            print(f"  ✓ Fetched {len(vix_levels)} days of VIX levels")
            print(f"    Range: {vix_levels.index.min().date()} to {vix_levels.index.max().date()}")
            print(f"    Mean VIX: {vix_levels.mean():.2f}")

            return vix_levels

        except Exception as e:
            raise RuntimeError(
                f"FAIL: Could not fetch VIX data.\n"
                f"Error: {e}\n"
                "FAIL IS FAIL: Cannot proceed without VIX data."
            )

    def fetch_treasury_yields(self) -> pd.DataFrame:
        """
        Fetch Treasury yields (10Y, 2Y) from FRED.

        Returns:
            DataFrame with 10Y and 2Y yield columns

        Raises:
            RuntimeError: If data fetch fails
        """
        print("\nFetching Treasury yield data from FRED...")
        try:
            # 10-Year Treasury Constant Maturity Rate
            print("  Fetching 10Y Treasury yields...")
            yield_10y = pdr.get_data_fred('DGS10', start=self.start_date, end=self.end_date)

            # 2-Year Treasury Constant Maturity Rate
            print("  Fetching 2Y Treasury yields...")
            yield_2y = pdr.get_data_fred('DGS2', start=self.start_date, end=self.end_date)

            # Combine
            yields = pd.DataFrame({
                'yield_10y': yield_10y['DGS10'],
                'yield_2y': yield_2y['DGS2']
            })

            # Drop NaNs (weekends/holidays)
            yields = yields.dropna()

            # Calculate spread
            yields['yield_spread'] = yields['yield_10y'] - yields['yield_2y']

            print(f"  ✓ Fetched {len(yields)} days of Treasury yields")
            print(f"    Range: {yields.index.min().date()} to {yields.index.max().date()}")
            print(f"    Mean 10Y: {yields['yield_10y'].mean():.2f}%")
            print(f"    Mean 2Y: {yields['yield_2y'].mean():.2f}%")
            print(f"    Mean spread: {yields['yield_spread'].mean():.2f}%")

            return yields

        except Exception as e:
            raise RuntimeError(
                f"FAIL: Could not fetch Treasury yield data.\n"
                f"Error: {e}\n"
                "FAIL IS FAIL: Cannot proceed without macro data."
            )

    def fetch_risk_free_rate(self) -> pd.Series:
        """
        Fetch 3-month T-bill rate as risk-free rate.

        Returns:
            Series of risk-free rates

        Raises:
            RuntimeError: If data fetch fails
        """
        print("\nFetching 3M T-Bill rate (risk-free rate)...")
        try:
            # 3-Month Treasury Bill: Secondary Market Rate
            rf = pdr.get_data_fred('DTB3', start=self.start_date, end=self.end_date)
            rf_rate = rf['DTB3'].dropna()

            print(f"  ✓ Fetched {len(rf_rate)} days of risk-free rates")
            print(f"    Range: {rf_rate.index.min().date()} to {rf_rate.index.max().date()}")
            print(f"    Mean RF rate: {rf_rate.mean():.2f}%")

            return rf_rate

        except Exception as e:
            raise RuntimeError(
                f"FAIL: Could not fetch risk-free rate data.\n"
                f"Error: {e}\n"
                "FAIL IS FAIL: Cannot proceed without risk-free rate."
            )

    def fetch_all_market_data(self) -> pd.DataFrame:
        """
        Fetch all market and macro data and combine into single DataFrame.

        Returns:
            DataFrame with all market data aligned by date

        Raises:
            RuntimeError: If any data fetch fails
        """
        print("\n" + "=" * 80)
        print("FETCHING ALL MARKET DATA")
        print("=" * 80)

        # Fetch individual series
        spy_returns = self.fetch_spy_returns()
        vix_levels = self.fetch_vix()
        yields = self.fetch_treasury_yields()
        rf_rate = self.fetch_risk_free_rate()

        # Combine into single DataFrame
        print("\nCombining market data...")
        market_data = pd.DataFrame({
            'spy_return': spy_returns,
            'vix': vix_levels,
            'yield_10y': yields['yield_10y'],
            'yield_2y': yields['yield_2y'],
            'yield_spread': yields['yield_spread'],
            'rf_rate': rf_rate
        })

        # Forward fill to handle weekends/holidays
        # (Market data has gaps for non-trading days)
        print("  Forward filling for non-trading days...")
        market_data = market_data.fillna(method='ffill')

        print(f"\n  ✓ Combined market data shape: {market_data.shape}")
        print(f"    Date range: {market_data.index.min().date()} to {market_data.index.max().date()}")
        print(f"    Missing values: {market_data.isna().sum().sum()}")

        if market_data.isna().sum().sum() > 0:
            print("\n  ⚠️  WARNING: Some missing values remain after forward fill")
            print(market_data.isna().sum())

        print("\n" + "=" * 80)
        return market_data

    def save_market_data(self, market_data: pd.DataFrame, output_path: str = "data/raw/market_data.parquet"):
        """
        Save market data to parquet file.

        Args:
            market_data: Market data DataFrame
            output_path: Path to save parquet file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        market_data.to_parquet(output_path)
        print(f"\n✓ Market data saved to {output_path}")


def fetch_and_save_market_data(start_date: str = "2000-01-01", end_date: str = None):
    """
    Main function to fetch and save all market data.

    Args:
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD), defaults to today
    """
    fetcher = MarketDataFetcher(start_date=start_date, end_date=end_date)
    market_data = fetcher.fetch_all_market_data()
    fetcher.save_market_data(market_data)
    return market_data


if __name__ == "__main__":
    # Test the fetcher
    market_data = fetch_and_save_market_data()
    print("\n✓ Market data fetch complete!")
    print(f"\nMarket data preview:")
    print(market_data.head())
    print(f"\nMarket data summary:")
    print(market_data.describe())
