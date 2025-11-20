"""Batch generate CPD regime cache files for all symbols.

This script pre-generates CPD segmentation cache files that were previously
created during training. This allows faster training iterations and easier
hyperparameter experimentation.

Defaults match paper's best X-Trend-Q configuration (2.70 Sharpe):
- Full backtest period: 1990-2023
- CPD lookback: 21 days
- CPD threshold: 0.95 (for 63-day max length)
- Max regime length: 63 days

Usage:
    # Generate cache with paper-optimal defaults
    uv run python scripts/batch_generate_cpd_cache.py \
        --data-path data/bloomberg/processed \
        --cache-dir data/bloomberg/cpd_cache

    # Or override specific parameters
    uv run python scripts/batch_generate_cpd_cache.py \
        --start 1990-01-01 \
        --end 2023-12-31 \
        --lookback 21 \
        --threshold 0.95 \
        --min-length 5 \
        --max-length 63 \
        --overwrite
"""
import argparse
import json
import os
import pickle
import sys
import warnings
from datetime import datetime
from pathlib import Path

import pandas as pd
from joblib import Parallel, delayed
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


def process_symbol(
    symbol: str,
    data_path: str,
    cache_path: Path,
    start_date: str,
    end_date: str,
    lookback: int,
    threshold: float,
    min_length: int,
    max_length: int,
    overwrite: bool,
):
    """Worker to load data, run CPD, and persist cache for one symbol."""
    warnings.filterwarnings(
        "ignore",
        message=r".*matches the stored training data.*",
    )
    start_time = datetime.now()

    try:
        # Re-create heavy objects per worker to avoid pickling issues.
        source = BloombergParquetSource(root_path=data_path)
        config = CPDConfig(
            lookback=lookback,
            threshold=threshold,
            min_length=min_length,
            max_length=max_length,
        )
        segmenter = GPCPDSegmenter(config)

        tqdm.write(f"[pid {os.getpid()}] processing {symbol}")

        prices_df = source.load_prices([symbol], start=start_date, end=end_date)
        price_series = prices_df[symbol].copy()

        if price_series.isna().all():
            return {
                'symbol': symbol,
                'status': 'failed_nan',
                'error': 'All NaN prices',
            }

        price_series = price_series.ffill().bfill()

        n_days = len(price_series)
        start_ts = price_series.index[0]
        end_ts = price_series.index[-1]

        filename = generate_cache_filename(symbol, start_ts, end_ts, n_days, config)
        cache_file = cache_path / filename

        if cache_file.exists() and not overwrite:
            return {
                'symbol': symbol,
                'status': 'cached',
                'n_days': n_days,
            }

        segments = segmenter.fit_segment(price_series)

        n_segments = len(segments.segments)
        severities = [seg.severity for seg in segments.segments]
        lengths = [seg.end_idx - seg.start_idx + 1 for seg in segments.segments]

        n_detected = sum(1 for s in severities if s >= threshold)
        n_fallback = n_segments - n_detected
        detection_rate = n_detected / n_segments if n_segments > 0 else 0
        n_max_length = sum(1 for l in lengths if l == max_length)
        fallback_fraction = n_max_length / n_segments if n_segments > 0 else 0

        save_cached_regimes(cache_file, segments)

        duration = (datetime.now() - start_time).total_seconds()

        return {
            'symbol': symbol,
            'status': 'generated',
            'worker_pid': os.getpid(),
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
        }

    except Exception as exc:  # pragma: no cover - defensive guardrail
        return {
            'symbol': symbol,
            'status': 'failed',
            'error': str(exc),
        }


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
    workers: int = -1,
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
        workers: Number of parallel workers (-1 uses all cores)
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

    print(f"Processing {len(symbols_to_process)} symbols with {workers} workers...")

    # Dispatch work in parallel across CPU cores.
    results = Parallel(n_jobs=workers, prefer="processes", verbose=10, batch_size=1)(
        delayed(process_symbol)(
            symbol,
            data_path,
            cache_path,
            start_date,
            end_date,
            lookback,
            threshold,
            min_length,
            max_length,
            overwrite,
        )
        for symbol in symbols_to_process
    )

    # Aggregate statistics
    stats = {
        'total': len(symbols_to_process),
        'generated': sum(1 for r in results if r['status'] == 'generated'),
        'skipped_cached': sum(1 for r in results if r['status'] == 'cached'),
        'failed': sum(1 for r in results if r['status'] not in ('generated', 'cached')),
        'total_segments': sum(r.get('n_segments', 0) for r in results),
        'total_detected': sum(r.get('n_detected', 0) for r in results),
        'total_fallback': sum(r.get('n_fallback', 0) for r in results),
    }

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
        default='1990-01-01',
        help='Start date (YYYY-MM-DD) - paper uses 1990'
    )
    parser.add_argument(
        '--end',
        default='2023-12-31',
        help='End date (YYYY-MM-DD) - paper uses 2023'
    )
    parser.add_argument(
        '--rolling-start',
        default=None,
        help='Optional: start of rolling end-date range (YYYY-MM-DD). If set, generates caches for multiple end dates.'
    )
    parser.add_argument(
        '--rolling-end',
        default=None,
        help='Optional: end of rolling end-date range (YYYY-MM-DD). Used only when --rolling-start is set.'
    )
    parser.add_argument(
        '--rolling-step-days',
        type=int,
        default=90,
        help='Step size in days between rolling end dates (default: 90). Used only when --rolling-start is set.'
    )
    parser.add_argument(
        '--lookback',
        type=int,
        default=21,
        help='CPD lookback window (days) - paper uses 21'
    )
    parser.add_argument(
        '--threshold',
        type=float,
        default=0.95,
        help='Severity threshold for change-point detection (paper: 0.95 for lmax=63)'
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
        default=63,
        help='Maximum regime length (days) - paper best: 63'
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
    parser.add_argument(
        '--workers',
        type=int,
        default=-1,
        help='Number of parallel workers (-1 = all cores)'
    )

    args = parser.parse_args()

    # Rolling mode: generate caches at multiple end dates
    if args.rolling_start:
        if not args.rolling_end:
            raise ValueError("--rolling-end must be provided when --rolling-start is set")
        rolling_dates = pd.date_range(
            start=args.rolling_start,
            end=args.rolling_end,
            freq=pd.DateOffset(days=args.rolling_step_days)
        )
        if rolling_dates[-1] != pd.Timestamp(args.rolling_end):
            # Ensure inclusive end even if step doesn't land exactly
            rolling_dates = rolling_dates.append(pd.DatetimeIndex([pd.Timestamp(args.rolling_end)]))

        print(f"\nRolling generation: {len(rolling_dates)} end dates from {rolling_dates[0].date()} to {rolling_dates[-1].date()} (step {args.rolling_step_days} days)\n")

        for idx, end_dt in enumerate(rolling_dates):
            end_str = end_dt.strftime('%Y-%m-%d')
            print(f"=== [{idx+1}/{len(rolling_dates)}] Generating caches up to {end_str} ===")
            batch_generate_cpd_cache(
                data_path=args.data_path,
                cache_dir=args.cache_dir,
                start_date=args.start,
                end_date=end_str,
                lookback=args.lookback,
                threshold=args.threshold,
                min_length=args.min_length,
                max_length=args.max_length,
                overwrite=args.overwrite,
                symbols=args.symbols,
                workers=args.workers,
            )
    else:
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
            workers=args.workers,
        )


if __name__ == '__main__':
    main()
