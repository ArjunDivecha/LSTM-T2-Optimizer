"""
Process factor data and create walk-forward expanding window splits.

Philosophy: NO FALLBACKS, FAIL IS FAIL
- Walk-forward validation with expanding window
- Annual retraining (one model per year)
- No static splits - retrain on growing data each year
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from typing import Dict, List, Tuple
import sys

from loader import load_and_validate_data
from preprocessing import preprocess_data


def create_walk_forward_expanding_splits(
    X_samples: list,
    y_samples: list,
    dates: list,
    start_test_year: int = 2016,
    end_test_year: int = 2025,
) -> List[Dict]:
    """
    Create walk-forward splits with EXPANDING window.

    Annual retraining strategy:
    - 2016: Train on 2000-2015 → Test on 2016
    - 2017: Train on 2000-2016 → Test on 2017
    - 2018: Train on 2000-2017 → Test on 2018
    - ...
    - 2025: Train on 2000-2024 → Test on 2025

    Each year gets a separate model trained on ALL data up to that year.
    This is EXPANDING window (not rolling).

    Args:
        X_samples: List of input arrays
        y_samples: List of target arrays
        dates: List of sample dates
        start_test_year: First year to test on (default: 2016)
        end_test_year: Last year to test on (default: 2025)

    Returns:
        List of dictionaries, one per year, each containing train/test splits
    """
    print("\n" + "=" * 80)
    print("CREATING WALK-FORWARD EXPANDING WINDOW SPLITS")
    print("=" * 80)
    print("\nStrategy: Expanding Window + Annual Retraining")
    print("  - Each year trains on ALL data from 2000 to year-1")
    print("  - Training window expands each year (gets more data)")
    print("  - Retrain fresh model each year")
    print("\n" + "=" * 80)

    # Convert dates to years
    years = np.array([d.year for d in dates])

    all_splits = []

    for test_year in range(start_test_year, end_test_year + 1):
        print(f"\n{'='*80}")
        print(f"Year {test_year}:")
        print(f"{'='*80}")

        # EXPANDING WINDOW: Train on ALL data before test year
        train_mask = years < test_year
        test_mask = years == test_year

        # Create splits
        X_train = [X_samples[i] for i in range(len(X_samples)) if train_mask[i]]
        y_train = [y_samples[i] for i in range(len(y_samples)) if train_mask[i]]
        dates_train = [dates[i] for i in range(len(dates)) if train_mask[i]]

        X_test = [X_samples[i] for i in range(len(X_samples)) if test_mask[i]]
        y_test = [y_samples[i] for i in range(len(y_samples)) if test_mask[i]]
        dates_test = [dates[i] for i in range(len(dates)) if test_mask[i]]

        # Validate splits
        if len(X_train) == 0:
            raise ValueError(
                f"FAIL: Training set empty for year {test_year}!\n"
                "FAIL IS FAIL: Check date ranges."
            )

        if len(X_test) == 0:
            print(f"  ⚠️  WARNING: No test data for year {test_year}")
            print(f"      (Data may not extend to {test_year})")
            continue  # Skip this year

        # Report split
        train_start = dates_train[0].year
        train_end = dates_train[-1].year
        test_start = dates_test[0].date()
        test_end = dates_test[-1].date()

        print(f"  Training:   {len(X_train):,} samples ({train_start}-{train_end})")
        print(f"  Test:       {len(X_test):,} samples ({test_start} to {test_end})")
        print(f"  Train span: {train_end - train_start + 1} years (EXPANDING)")

        # Store split
        split = {
            'year': test_year,
            'X_train': X_train,
            'y_train': y_train,
            'dates_train': dates_train,
            'X_test': X_test,
            'y_test': y_test,
            'dates_test': dates_test,
        }

        all_splits.append(split)

    print("\n" + "=" * 80)
    print(f"✓ Created {len(all_splits)} walk-forward splits")
    print("=" * 80)

    return all_splits


def convert_to_arrays(split: Dict) -> Dict:
    """
    Convert lists of samples to numpy arrays for a single split.

    Args:
        split: Dictionary with lists of samples for one year

    Returns:
        Dictionary with numpy arrays
    """
    arrays = {}

    for key, value in split.items():
        if key == 'year':
            arrays[key] = value
        elif 'dates' in key:
            # Keep dates as list
            arrays[key] = value
        else:
            # Stack into single array
            arrays[key] = np.array(value)

    return arrays


def save_walk_forward_splits(splits: List[Dict], output_dir: str = "data/processed"):
    """
    Save walk-forward splits to disk.

    Creates separate directories for each year:
    - data/processed/2016/train_X.pkl, test_X.pkl, etc.
    - data/processed/2017/train_X.pkl, test_X.pkl, etc.
    - ...

    Args:
        splits: List of split dictionaries
        output_dir: Base output directory
    """
    print("\n" + "=" * 80)
    print("SAVING WALK-FORWARD SPLITS")
    print("=" * 80)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save each year's split
    for split in splits:
        year = split['year']
        year_dir = output_path / str(year)
        year_dir.mkdir(parents=True, exist_ok=True)

        print(f"\nYear {year}:")

        # Convert to arrays
        arrays = convert_to_arrays(split)

        # Save training data
        train_X_path = year_dir / "train_X.pkl"
        with open(train_X_path, 'wb') as f:
            pickle.dump(arrays['X_train'], f, protocol=4)
        print(f"  ✓ {train_X_path} ({arrays['X_train'].nbytes / 1e6:.1f} MB)")

        train_y_path = year_dir / "train_y.pkl"
        with open(train_y_path, 'wb') as f:
            pickle.dump(arrays['y_train'], f, protocol=4)
        print(f"  ✓ {train_y_path} ({arrays['y_train'].nbytes / 1e6:.1f} MB)")

        train_dates_path = year_dir / "train_dates.pkl"
        with open(train_dates_path, 'wb') as f:
            pickle.dump(arrays['dates_train'], f, protocol=4)
        print(f"  ✓ {train_dates_path}")

        # Save test data
        test_X_path = year_dir / "test_X.pkl"
        with open(test_X_path, 'wb') as f:
            pickle.dump(arrays['X_test'], f, protocol=4)
        print(f"  ✓ {test_X_path} ({arrays['X_test'].nbytes / 1e6:.1f} MB)")

        test_y_path = year_dir / "test_y.pkl"
        with open(test_y_path, 'wb') as f:
            pickle.dump(arrays['y_test'], f, protocol=4)
        print(f"  ✓ {test_y_path} ({arrays['y_test'].nbytes / 1e6:.1f} MB)")

        test_dates_path = year_dir / "test_dates.pkl"
        with open(test_dates_path, 'wb') as f:
            pickle.dump(arrays['dates_test'], f, protocol=4)
        print(f"  ✓ {test_dates_path}")

        # Save metadata for this year
        metadata = {
            'year': year,
            'n_train_samples': len(arrays['X_train']),
            'n_test_samples': len(arrays['X_test']),
            'input_shape': arrays['X_train'][0].shape if len(arrays['X_train']) > 0 else None,
            'output_shape': arrays['y_train'][0].shape if len(arrays['y_train']) > 0 else None,
            'train_date_range': (str(arrays['dates_train'][0]), str(arrays['dates_train'][-1])),
            'test_date_range': (str(arrays['dates_test'][0]), str(arrays['dates_test'][-1])),
        }

        metadata_path = year_dir / "metadata.pkl"
        with open(metadata_path, 'wb') as f:
            pickle.dump(metadata, f, protocol=4)
        print(f"  ✓ {metadata_path}")

    # Save overall summary
    summary = {
        'n_years': len(splits),
        'years': [s['year'] for s in splits],
        'strategy': 'walk_forward_expanding',
        'retrain_frequency': 'annual',
        'description': 'Expanding window: Each year trains on ALL data from 2000 to year-1',
    }

    summary_path = output_path / "walk_forward_summary.pkl"
    with open(summary_path, 'wb') as f:
        pickle.dump(summary, f, protocol=4)
    print(f"\n  ✓ {summary_path}")

    # Print summary
    print("\n" + "=" * 80)
    print("WALK-FORWARD SPLITS SUMMARY")
    print("=" * 80)
    print(f"Strategy: Expanding Window + Annual Retraining")
    print(f"Years: {len(splits)} models ({splits[0]['year']}-{splits[-1]['year']})")
    print(f"\nEach year:")
    print(f"  - Trains on expanding window (2000 to year-1)")
    print(f"  - Tests on that year")
    print(f"  - Separate model per year")
    print("=" * 80)


def main():
    """Main data processing pipeline with walk-forward expanding splits."""
    print("\n" + "=" * 80)
    print("FACTOR DATA PROCESSING - WALK-FORWARD EXPANDING")
    print("=" * 80)
    print("\nPhilosophy: NO FALLBACKS, FAIL IS FAIL")
    print("Strategy: Expanding Window + Annual Retraining")
    print("=" * 80)

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

    # Step 3: Create walk-forward expanding splits
    print("\nStep 3: Creating walk-forward expanding splits...")
    splits = create_walk_forward_expanding_splits(
        X_samples,
        y_samples,
        dates,
        start_test_year=2016,
        end_test_year=2025,
    )

    # Step 4: Save splits
    print("\nStep 4: Saving walk-forward splits...")
    save_walk_forward_splits(splits)

    print("\n" + "=" * 80)
    print("✓ DATA PROCESSING COMPLETE!")
    print("=" * 80)
    print("\nProcessed data saved to: data/processed/")
    print("  Structure:")
    print("    data/processed/2016/  (train on 2000-2015, test on 2016)")
    print("    data/processed/2017/  (train on 2000-2016, test on 2017)")
    print("    data/processed/2018/  (train on 2000-2017, test on 2018)")
    print("    ...")
    print("    data/processed/2025/  (train on 2000-2024, test on 2025)")
    print("\nEach directory contains:")
    print("  - train_X.pkl, train_y.pkl, train_dates.pkl")
    print("  - test_X.pkl, test_y.pkl, test_dates.pkl")
    print("  - metadata.pkl")
    print("\nReady for walk-forward model training!")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ FAIL: Data processing failed!")
        print(f"Error: {e}")
        print("\nFAIL IS FAIL: Fix the error and try again.")
        import traceback
        traceback.print_exc()
        sys.exit(1)
