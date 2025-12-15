"""
Process factor data and save train/val/test splits.

Philosophy: NO FALLBACKS, FAIL IS FAIL
- Process all data with proper validation
- Create expanding window splits per PRD
- Save to parquet for efficient loading
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from typing import Dict, Tuple
import sys

from loader import load_and_validate_data
from preprocessing import preprocess_data


def create_train_val_test_splits(
    X_samples: list,
    y_samples: list,
    dates: list,
    train_end_year: int = 2015,
    val_end_year: int = 2020,
) -> Dict:
    """
    Create train/val/test splits using expanding window.

    Per PRD:
    - Training: 2000-2015 (15 years)
    - Validation: 2016-2020 (5 years)
    - Test: 2021-2025 (5 years)

    Args:
        X_samples: List of input arrays
        y_samples: List of target arrays
        dates: List of sample dates
        train_end_year: Last year of training data
        val_end_year: Last year of validation data

    Returns:
        Dictionary with train/val/test splits
    """
    print("\n" + "=" * 80)
    print("CREATING TRAIN/VAL/TEST SPLITS (Expanding Window)")
    print("=" * 80)

    # Convert dates to years
    years = np.array([d.year for d in dates])

    # Create split indices
    train_mask = years <= train_end_year
    val_mask = (years > train_end_year) & (years <= val_end_year)
    test_mask = years > val_end_year

    # Split data
    X_train = [X_samples[i] for i in range(len(X_samples)) if train_mask[i]]
    y_train = [y_samples[i] for i in range(len(y_samples)) if train_mask[i]]
    dates_train = [dates[i] for i in range(len(dates)) if train_mask[i]]

    X_val = [X_samples[i] for i in range(len(X_samples)) if val_mask[i]]
    y_val = [y_samples[i] for i in range(len(y_samples)) if val_mask[i]]
    dates_val = [dates[i] for i in range(len(dates)) if val_mask[i]]

    X_test = [X_samples[i] for i in range(len(X_samples)) if test_mask[i]]
    y_test = [y_samples[i] for i in range(len(y_samples)) if test_mask[i]]
    dates_test = [dates[i] for i in range(len(dates)) if test_mask[i]]

    # Report split sizes
    print(f"\nSplit Sizes:")
    print(f"  Training:   {len(X_train):,} samples ({dates_train[0]} to {dates_train[-1]})")
    print(f"  Validation: {len(X_val):,} samples ({dates_val[0]} to {dates_val[-1]})")
    print(f"  Test:       {len(X_test):,} samples ({dates_test[0]} to {dates_test[-1]})")
    print(f"  Total:      {len(X_samples):,} samples")

    # Validate splits
    if len(X_train) == 0:
        raise ValueError("FAIL: Training set is empty! Check date ranges.")
    if len(X_val) == 0:
        raise ValueError("FAIL: Validation set is empty! Check date ranges.")
    if len(X_test) == 0:
        print("  ⚠️  WARNING: Test set is empty (may be expected if data doesn't extend to test period)")

    splits = {
        'X_train': X_train,
        'y_train': y_train,
        'dates_train': dates_train,
        'X_val': X_val,
        'y_val': y_val,
        'dates_val': dates_val,
        'X_test': X_test,
        'y_test': y_test,
        'dates_test': dates_test,
    }

    print("\n✓ Splits created successfully")
    return splits


def convert_to_arrays(samples_dict: Dict) -> Dict:
    """
    Convert lists of arrays to single numpy arrays for efficient storage.

    Args:
        samples_dict: Dictionary with lists of samples

    Returns:
        Dictionary with concatenated numpy arrays
    """
    print("\nConverting to numpy arrays...")

    arrays = {}
    for key, value in samples_dict.items():
        if 'dates' in key:
            # Keep dates as list
            arrays[key] = value
        else:
            # Stack into single array
            arrays[key] = np.array(value)
            print(f"  {key}: {arrays[key].shape}")

    return arrays


def save_processed_data(data_dict: Dict, output_dir: str = "data/processed"):
    """
    Save processed data to parquet and pickle files.

    Args:
        data_dict: Dictionary with processed data
        output_dir: Output directory
    """
    print("\n" + "=" * 80)
    print("SAVING PROCESSED DATA")
    print("=" * 80)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save numpy arrays as pickle (efficient for arrays)
    for split in ['train', 'val', 'test']:
        X_key = f'X_{split}'
        y_key = f'y_{split}'
        dates_key = f'dates_{split}'

        if X_key in data_dict and len(data_dict[X_key]) > 0:
            # Save features (X)
            X_path = output_path / f"{split}_X.pkl"
            with open(X_path, 'wb') as f:
                pickle.dump(data_dict[X_key], f, protocol=4)
            print(f"  ✓ Saved {X_path} ({data_dict[X_key].nbytes / 1e6:.1f} MB)")

            # Save targets (y)
            y_path = output_path / f"{split}_y.pkl"
            with open(y_path, 'wb') as f:
                pickle.dump(data_dict[y_key], f, protocol=4)
            print(f"  ✓ Saved {y_path} ({data_dict[y_key].nbytes / 1e6:.1f} MB)")

            # Save dates
            dates_path = output_path / f"{split}_dates.pkl"
            with open(dates_path, 'wb') as f:
                pickle.dump(data_dict[dates_key], f, protocol=4)
            print(f"  ✓ Saved {dates_path}")

    # Save metadata
    metadata = {
        'n_samples_train': len(data_dict['X_train']),
        'n_samples_val': len(data_dict['X_val']),
        'n_samples_test': len(data_dict['X_test']) if len(data_dict['X_test']) > 0 else 0,
        'input_shape': data_dict['X_train'][0].shape,
        'output_shape': data_dict['y_train'][0].shape,
        'date_range_train': (str(data_dict['dates_train'][0]), str(data_dict['dates_train'][-1])),
        'date_range_val': (str(data_dict['dates_val'][0]), str(data_dict['dates_val'][-1])),
        'date_range_test': (str(data_dict['dates_test'][0]), str(data_dict['dates_test'][-1])) if len(data_dict['dates_test']) > 0 else None,
    }

    metadata_path = output_path / "metadata.pkl"
    with open(metadata_path, 'wb') as f:
        pickle.dump(metadata, f)
    print(f"\n  ✓ Saved {metadata_path}")

    # Print summary
    print("\n" + "=" * 80)
    print("DATA PROCESSING SUMMARY")
    print("=" * 80)
    print(f"Input shape:  {metadata['input_shape']} (lookback, factors, features)")
    print(f"Output shape: {metadata['output_shape']} (horizon, factors)")
    print(f"\nSamples:")
    print(f"  Training:   {metadata['n_samples_train']:,}")
    print(f"  Validation: {metadata['n_samples_val']:,}")
    print(f"  Test:       {metadata['n_samples_test']:,}")
    print(f"  Total:      {metadata['n_samples_train'] + metadata['n_samples_val'] + metadata['n_samples_test']:,}")
    print("=" * 80)


def main():
    """Main data processing pipeline."""
    print("\n" + "=" * 80)
    print("FACTOR DATA PROCESSING PIPELINE")
    print("=" * 80)
    print("\nPhilosophy: NO FALLBACKS, FAIL IS FAIL")
    print("  - All data must be valid")
    print("  - No synthetic data")
    print("  - Fail loud on errors")
    print("\n" + "=" * 80)

    # Step 1: Load data
    print("\nStep 1: Loading factor returns...")
    factor_returns, valid_mask = load_and_validate_data()

    # Step 2: Preprocess and create samples
    print("\nStep 2: Preprocessing and creating samples...")
    X_samples, y_samples, dates = preprocess_data(
        factor_returns,
        valid_mask,
        lookback_days=250,
        prediction_horizon=5,
    )

    # Step 3: Create train/val/test splits
    print("\nStep 3: Creating train/val/test splits...")
    splits = create_train_val_test_splits(
        X_samples,
        y_samples,
        dates,
        train_end_year=2015,
        val_end_year=2020,
    )

    # Step 4: Convert to arrays
    print("\nStep 4: Converting to numpy arrays...")
    arrays = convert_to_arrays(splits)

    # Step 5: Save processed data
    print("\nStep 5: Saving processed data...")
    save_processed_data(arrays)

    print("\n" + "=" * 80)
    print("✓ DATA PROCESSING COMPLETE!")
    print("=" * 80)
    print("\nProcessed data saved to: data/processed/")
    print("  - train_X.pkl, train_y.pkl, train_dates.pkl")
    print("  - val_X.pkl, val_y.pkl, val_dates.pkl")
    print("  - test_X.pkl, test_y.pkl, test_dates.pkl")
    print("  - metadata.pkl")
    print("\nReady for model training!")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ FAIL: Data processing failed!")
        print(f"Error: {e}")
        print("\nFAIL IS FAIL: Fix the error and try again.")
        sys.exit(1)
