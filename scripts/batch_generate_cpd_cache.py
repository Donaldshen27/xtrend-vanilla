"""Batch generate CPD regime cache files for all symbols.

This script pre-generates CPD segmentation cache files that were previously
created during training. This allows faster training iterations and easier
hyperparameter experimentation.

Usage:
    uv run python scripts/batch_generate_cpd_cache.py \
        --data-path data/bloomberg/processed \
        --cache-dir data/bloomberg/cpd_cache \
        --start 2000-01-01 \
        --end 2020-12-31 \
        --lookback 63 \
        --threshold 0.9 \
        --min-length 5 \
        --max-length 21 \
        --overwrite
"""
import argparse
import json
import pickle
import sys
import warnings
from datetime import datetime
from pathlib import Path

import pandas as pd
from tqdm import tqdm

try:
    from xtrend.cpd import CPDConfig, GPCPDSegmenter
    from xtrend.data.sources import BloombergParquetSource
except ImportError as e:
    print(f"ERROR: Failed to import required modules: {e}", file=sys.stderr)
    print("Make sure you're running with 'uv run python' in the project directory", file=sys.stderr)
    sys.exit(1)


def generate_cache_filename(symbol: str, start: pd.Timestamp, end: pd.Timestamp,
                            n_days: int, config: CPDConfig) -> str:
    """Generate cache filename matching train_xtrend.py format."""
    start_str = start.strftime("%Y%m%d")
    end_str = end.strftime("%Y%m%d")
    return (
        f"{symbol}_{start_str}_{end_str}_{n_days}_"
        f"lb{config.lookback}_th{config.threshold:.2f}_"
        f"min{config.min_length}_max{config.max_length}.pkl"
    )


def save_cached_regimes(path: Path, segments):
    """Save RegimeSegments to pickle file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as fh:
        pickle.dump({"segments": segments}, fh)


def batch_generate_cpd_cache(
    data_path: str,
    cache_dir: str,
    start_date: str,
    end_date: str,
    lookback: int,
    threshold: float,
    min_length: int,
    max_length: int,
    overwrite: bool = False,
    symbols: list = None,
):
    """Generate CPD cache files for all symbols.

    Args:
        data_path: Path to Bloomberg parquet data
        cache_dir: Directory to save cache files
        start_date: Start date for price data (YYYY-MM-DD)
        end_date: End date for price data (YYYY-MM-DD)
        lookback: CPD lookback window (days)
        threshold: Severity threshold for change-point detection
        min_length: Minimum regime length (days)
        max_length: Maximum regime length (days)
        overwrite: If True, regenerate even if cache exists
        symbols: Optional list of specific symbols to process (None = all)
    """
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)

    # Create CPD config
    config = CPDConfig(
        lookback=lookback,
        threshold=threshold,
        min_length=min_length,
        max_length=max_length
    )
    segmenter = GPCPDSegmenter(config)

    # Save hyperparameters to log file for reproducibility
    config_log = cache_path / f"config_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    config_dict = {
        'data_path': data_path,
        'start_date': start_date,
        'end_date': end_date,
        'lookback': lookback,
        'threshold': threshold,
        'min_length': min_length,
        'max_length': max_length,
        'timestamp': datetime.now().isoformat(),
    }
    with config_log.open('w') as f:
        json.dump(config_dict, f, indent=2)

    print(f"\nCPD Configuration:")
    print(f"  Lookback: {config.lookback} days")
    print(f"  Threshold: {config.threshold}")
    print(f"  Min length: {config.min_length} days")
    print(f"  Max length: {config.max_length} days")
    print(f"  Config saved to: {config_log}")
    print()

    # Load data source
    print(f"Loading data from {data_path}...")
    source = BloombergParquetSource(root_path=data_path)
    all_symbols = source.symbols()

    if symbols is None:
        symbols_to_process = all_symbols
    else:
        symbols_to_process = [s for s in symbols if s in all_symbols]
        if not symbols_to_process:
            raise ValueError(f"No valid symbols found. Available: {all_symbols[:10]}...")

    print(f"Processing {len(symbols_to_process)} symbols")

    # Track per-symbol results for CSV output
    results = []

    # Track statistics
    stats = {
        'total': len(symbols_to_process),
        'generated': 0,
        'skipped_cached': 0,
        'failed': 0,
        'total_segments': 0,
        'total_detected': 0,  # Segments with severity >= threshold
        'total_fallback': 0,  # Segments that hit fallback (severity < threshold)
    }

    # Process each symbol (load prices individually to avoid massive memory use)
    for symbol in tqdm(symbols_to_process, desc="Generating CPD caches"):
        symbol_start_time = datetime.now()

        try:
            # Load prices for this symbol only (memory efficient)
            tqdm.write(f"  Processing {symbol}...")
            prices_df = source.load_prices([symbol], start=start_date, end=end_date)
            price_series = prices_df[symbol].copy()

            # Skip if all NaN
            if price_series.isna().all():
                tqdm.write(f"  ⚠️  {symbol}: all NaN prices, skipping")
                stats['failed'] += 1
                results.append({
                    'symbol': symbol,
                    'status': 'failed_nan',
                    'error': 'All NaN prices',
                })
                continue

            # Fill NaN values causally
            price_series = price_series.ffill().bfill()

            # Use actual series length for this symbol
            n_days = len(price_series)
            start_ts = price_series.index[0]
            end_ts = price_series.index[-1]

            # Check if already cached
            filename = generate_cache_filename(symbol, start_ts, end_ts, n_days, config)
            cache_file = cache_path / filename

            if cache_file.exists() and not overwrite:
                stats['skipped_cached'] += 1
                tqdm.write(f"  ✓ {symbol}: cached ({cache_file.name})")
                results.append({
                    'symbol': symbol,
                    'status': 'cached',
                    'n_days': n_days,
                })
                continue

            # Run segmentation
            segments = segmenter.fit_segment(price_series)

            # Calculate statistics
            n_segments = len(segments.segments)
            severities = [seg.severity for seg in segments.segments]
            lengths = [seg.end_idx - seg.start_idx + 1 for seg in segments.segments]

            # Count detected vs fallback
            n_detected = sum(1 for s in severities if s >= threshold)
            n_fallback = n_segments - n_detected

            # Detection rate
            detection_rate = n_detected / n_segments if n_segments > 0 else 0

            # Check if all segments are max_length (likely fallback)
            n_max_length = sum(1 for l in lengths if l == max_length)
            fallback_fraction = n_max_length / n_segments if n_segments > 0 else 0

            # Duration
            duration = (datetime.now() - symbol_start_time).total_seconds()

            # Save to cache
            save_cached_regimes(cache_file, segments)

            # Update global statistics
            stats['generated'] += 1
            stats['total_segments'] += n_segments
            stats['total_detected'] += n_detected
            stats['total_fallback'] += n_fallback

            tqdm.write(
                f"  ✓ {symbol}: {n_segments} segments, "
                f"{detection_rate:.1%} detected, "
                f"{fallback_fraction:.1%} fallback ({duration:.1f}s)"
            )

            # Record results
            results.append({
                'symbol': symbol,
                'status': 'generated',
                'n_days': n_days,
                'n_segments': n_segments,
                'n_detected': n_detected,
                'n_fallback': n_fallback,
                'detection_rate': detection_rate,
                'fallback_fraction': fallback_fraction,
                'mean_severity': sum(severities) / len(severities) if severities else 0,
                'mean_length': sum(lengths) / len(lengths) if lengths else 0,
                'duration_seconds': duration,
                'cache_file': cache_file.name,
            })

        except Exception as exc:
            tqdm.write(f"  ✗ {symbol}: FAILED - {exc}")
            stats['failed'] += 1
            results.append({
                'symbol': symbol,
                'status': 'failed',
                'error': str(exc),
            })
            continue

    # Write results to CSV
    results_csv = cache_path / f"generation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    results_df = pd.DataFrame(results)
    results_df.to_csv(results_csv, index=False)

    # Print summary
    print("\n" + "="*60)
    print("BATCH GENERATION SUMMARY")
    print("="*60)
    print(f"Total symbols: {stats['total']}")
    print(f"Generated: {stats['generated']}")
    print(f"Skipped (cached): {stats['skipped_cached']}")
    print(f"Failed: {stats['failed']}")
    print()
    print(f"Total segments: {stats['total_segments']}")
    if stats['generated'] > 0:
        print(f"Average segments per symbol: {stats['total_segments'] / stats['generated']:.1f}")
    if stats['total_segments'] > 0:
        overall_detection_rate = stats['total_detected'] / stats['total_segments']
        overall_fallback_rate = stats['total_fallback'] / stats['total_segments']
        print(f"Overall detection rate: {overall_detection_rate:.1%}")
        print(f"  (segments with severity >= {threshold})")
        print(f"Overall fallback rate: {overall_fallback_rate:.1%}")
        print(f"  (segments likely from fallback branch)")
    print()
    print(f"Results saved to: {results_csv}")
    print(f"Cache directory: {cache_path}")
    print("="*60)


def main():
    parser = argparse.ArgumentParser(
        description="Batch generate CPD regime cache files",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--data-path',
        default='data/bloomberg/processed',
        help='Path to Bloomberg parquet data'
    )
    parser.add_argument(
        '--cache-dir',
        default='data/bloomberg/cpd_cache',
        help='Directory to save cache files'
    )
    parser.add_argument(
        '--start',
        default='2000-01-01',
        help='Start date (YYYY-MM-DD)'
    )
    parser.add_argument(
        '--end',
        default='2020-12-31',
        help='End date (YYYY-MM-DD)'
    )
    parser.add_argument(
        '--lookback',
        type=int,
        default=63,
        help='CPD lookback window (days)'
    )
    parser.add_argument(
        '--threshold',
        type=float,
        default=0.9,
        help='Severity threshold for change-point detection'
    )
    parser.add_argument(
        '--min-length',
        type=int,
        default=5,
        help='Minimum regime length (days)'
    )
    parser.add_argument(
        '--max-length',
        type=int,
        default=21,
        help='Maximum regime length (days)'
    )
    parser.add_argument(
        '--overwrite',
        action='store_true',
        help='Regenerate even if cache files exist'
    )
    parser.add_argument(
        '--symbols',
        nargs='+',
        help='Optional: specific symbols to process (default: all)'
    )

    args = parser.parse_args()

    batch_generate_cpd_cache(
        data_path=args.data_path,
        cache_dir=args.cache_dir,
        start_date=args.start,
        end_date=args.end,
        lookback=args.lookback,
        threshold=args.threshold,
        min_length=args.min_length,
        max_length=args.max_length,
        overwrite=args.overwrite,
        symbols=args.symbols,
    )


if __name__ == '__main__':
    main()
