#!/usr/bin/env python3
"""
Training script for X-Trend models on Bloomberg futures data.

This implementation follows the X-Trend paper specification:
- Input features: 5 volatility-normalized returns + 3 MACD indicators
- Return timescales: [1, 21, 63, 126, 252] days
- MACD pairs: [(8,24), (16,28), (32,96)]
- Volatility: EWMA with span=60
- Normalization: r_hat = r / (σ_t * sqrt(t'))

Defaults now match paper's best X-Trend-Q configuration (2.70 Sharpe):
- Data: 1990-2023 (full backtest period)
- Context size: 10 sequences
- Context max length: 63 days
- CPD threshold: 0.95
- Learning rate: 1e-3
- Batch size: 64

Usage:
    # Train XTrendQ with paper-optimal defaults (2.70 Sharpe configuration)
    uv run python scripts/train_xtrend.py --model xtrendq

    # Train XTrendG
    uv run python scripts/train_xtrend.py --model xtrendg

    # Train baseline XTrend
    uv run python scripts/train_xtrend.py --model xtrend

    # Resume from checkpoint
    uv run python scripts/train_xtrend.py --model xtrendq --resume checkpoints/xtrend_q_epoch_10.pt
"""

import argparse
import json
import pickle
import warnings
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# X-Trend modules
from xtrend.models import (
    ModelConfig,
    EntityEmbedding,
    LSTMEncoder,
    XTrendCrossAttention,
    XTrend,
    XTrendG,
    XTrendQ,
    sharpe_loss,
    joint_gaussian_loss,
    joint_quantile_loss,
)
from xtrend.context import (
    sample_cpd_segmented,
    sample_final_hidden_state,
    sample_time_equivalent,
)
from xtrend.data.sources import BloombergParquetSource
from xtrend.data.features import compute_xtrend_features
from xtrend.cpd import CPDConfig, GPCPDSegmenter, RegimeSegment, RegimeSegments


class XTrendDataset(Dataset):
    """
    Dataset for X-Trend model training.

    Creates sequences of:
    - Target features (8 features × target_len time steps)
    - Context sequences sampled via Phase 4 samplers (cpd_segmented/time-aligned/final-state)
    - Target returns (for loss computation, EWMA-normalized with span=60)
    """

    def __init__(
        self,
        prices: pd.DataFrame,
        symbols: List[str],
        target_len: int = 126,
        context_size: int = 20,
        context_method: str = "cpd_segmented",
        context_max_length: int = 21,
        vol_target: float = 0.15,
        min_history: int = 252,
        cpd_config: Optional[CPDConfig] = None,
        seed: Optional[int] = None,
        cpd_cache_dir: Optional[str] = None,
        allow_future_regimes: bool = False,
        allow_cpd_recompute: bool = False,
        clip_value: float = 20.0,
    ):
        """
        Args:
            prices: DataFrame with prices (date x symbols)
            symbols: List of symbol names
            target_len: Target sequence length (default: 126 = ~6 months)
            context_size: Number of context sequences sampled per target (default: 20)
            context_method: Primary context sampling strategy (Phase 4)
            context_max_length: Maximum regime/context length (days)
            vol_target: Volatility target for normalization (default: 0.15)
            min_history: Minimum history needed (default: 252 days)
            cpd_config: Optional CPDConfig override
            seed: Optional RNG seed for deterministic sampling
            cpd_cache_dir: Optional directory for caching CPD regimes
            allow_future_regimes: If True, allow context regimes that end after
                the target date (paper-approved for training); keep False for
                validation/backtesting to preserve causality.
            allow_cpd_recompute: If False, require cached regimes and fail early
                when missing; if True, fall back to on-the-fly GP-CPD.
            clip_value: Clip absolute feature/return magnitudes to this value
                to guard against data gaps or near-zero prices creating
                huge normalized returns that explode the loss.
        """
        self.prices = prices
        self.dates = prices.index
        self.symbols = symbols
        self.target_len = target_len
        self.context_size = context_size
        self.context_method = context_method
        self.context_max_length = context_max_length
        self.vol_target = vol_target
        self.min_history = min_history
        self.seed = seed
        self._fallback_warned = False
        self.cpd_cache_dir = Path(cpd_cache_dir).expanduser() if cpd_cache_dir else None
        self.allow_future_regimes = allow_future_regimes
        self.allow_cpd_recompute = allow_cpd_recompute
        self.clip_value = clip_value
        if self.cpd_cache_dir:
            self.cpd_cache_dir.mkdir(parents=True, exist_ok=True)
        # Track whether we've already reported falling back to alternate cache names
        self._cache_fallback_notice = False
        # Pre-compute date -> index map for reindexing cached regimes
        self._date_to_idx = {pd.Timestamp(date): idx for idx, date in enumerate(self.dates)}
        # Map each symbol to its first real observation index (listing offset)
        self.listing_offsets = {}
        for symbol in symbols:
            first_valid = prices[symbol].first_valid_index()
            if first_valid is None:
                continue
            self.listing_offsets[symbol] = self.dates.get_loc(first_valid)

        # Precompute features and returns for all symbols
        print("Precomputing Phase 1 features and returns...")
        self.features = {}
        self.feature_tensors = {}
        self.returns = {}
        self.return_tensors = {}
        self.input_dim = None

        for symbol in tqdm(symbols, desc="Processing symbols"):
            raw_series = prices[symbol]
            if raw_series.isna().all():
                continue

            price_series = self._prepare_price_series(raw_series)
            
            # Compute features (8 features as per paper)
            self.features[symbol] = self._compute_features(price_series)
            if self.clip_value:
                self.features[symbol] = self.features[symbol].clip(
                    lower=-self.clip_value,
                    upper=self.clip_value
                )
            feature_tensor = torch.tensor(
                self.features[symbol].values,
                dtype=torch.float32
            )
            self.feature_tensors[symbol] = feature_tensor

            # Compute normalized target returns using EWMA
            # Paper: r_hat = r / σ_t where σ_t uses EWMA with span=60
            daily_rets = price_series.pct_change()

            # CRITICAL FOR TARGETS: Use lagged volatility to avoid lookahead bias
            # When predicting position at time t for return r[t+1]:
            # - We normalize r[t+1] by σ[t] (volatility known at time t)
            # - Using σ[t+1] would be lookahead (not known when making decision)
            #
            # Note: This is DIFFERENT from input features, which use concurrent σ[t]
            # because features represent the market state AT time t.
            sigma_t = daily_rets.ewm(span=60, min_periods=20).std().shift(1)

            # Clip to prevent division by zero, backfill first value
            sigma_t = sigma_t.clip(lower=1e-8).bfill()

            # Normalize: r_hat[t+1] = r[t+1] / σ[t]
            normalized_rets = daily_rets / sigma_t
            if self.clip_value:
                normalized_rets = normalized_rets.clip(
                    lower=-self.clip_value,
                    upper=self.clip_value
                )
            self.returns[symbol] = normalized_rets.fillna(0.0)
            self.return_tensors[symbol] = torch.tensor(
                self.returns[symbol].values,
                dtype=torch.float32
            )

            if self.input_dim is None:
                self.input_dim = feature_tensor.shape[-1]

        if self.input_dim is None:
            raise ValueError("No valid symbols with feature data were found.")

        # Phase 2: GP-CPD regime segmentation (optional fallback for context sampling)
        self.cpd_config = cpd_config or CPDConfig(max_length=context_max_length)
        self.regimes = {}

        if context_method == "cpd_segmented":
            print("Running GP-CPD segmentation for regime-aware context sampling...")
            missing_cpd = []
            segmenter = GPCPDSegmenter(self.cpd_config)
            for symbol in tqdm(symbols, desc="Segmenting regimes"):
                raw_series = prices[symbol]
                if raw_series.isna().all():
                    continue

                listing_offset = self.listing_offsets.get(symbol)
                if listing_offset is None:
                    continue  # No real observations

                price_series_full = self._prepare_price_series(raw_series)
                # Trim to real trading span to avoid feeding pre-listing NaNs/epsilons
                price_series = price_series_full.iloc[listing_offset:]
                if price_series.isna().any():
                    price_series = price_series.fillna(method="ffill").fillna(1e-8)

                # Composite cache loading: pick best cache, optionally gap-fill from others
                cache_candidates = self._list_cache_candidates(symbol)
                primary_cache = self._select_primary_cache(cache_candidates)
                extra_caches = [p for p in cache_candidates if p != primary_cache]

                aligned_primary = None
                if primary_cache and primary_cache.exists():
                    aligned_primary = self._load_and_align_cache(primary_cache, symbol)

                aligned_extras = []
                for path in extra_caches:
                    aligned = self._load_and_align_cache(path, symbol)
                    if aligned:
                        aligned_extras.append(aligned)

                if aligned_primary:
                    merged = self._merge_gapfill_segments(aligned_primary, aligned_extras)
                    self.regimes[symbol] = merged
                    if aligned_extras and not self._cache_fallback_notice:
                        warnings.warn(
                            "Using composite CPD caches for symbol "
                            f"{symbol} (primary: '{primary_cache.name}')."
                        )
                        self._cache_fallback_notice = True
                    continue

                if self.allow_cpd_recompute:
                    try:
                        warnings.warn(
                            f"No usable CPD cache for {symbol}; recomputing GP-CPD segmentation."
                        )
                        segmented = segmenter.fit_segment(price_series)
                        # Shift indices back to global calendar so samplers align with features
                        shifted_segments = [
                            RegimeSegment(
                                start_idx=seg.start_idx + listing_offset,
                                end_idx=seg.end_idx + listing_offset,
                                severity=seg.severity,
                                start_date=seg.start_date,
                                end_date=seg.end_date,
                            )
                            for seg in segmented.segments
                        ]
                        self.regimes[symbol] = RegimeSegments(
                            segments=shifted_segments, config=segmented.config
                        )
                        cache_path = self._regime_cache_path(symbol)
                        if cache_path:
                            self._save_cached_regimes(cache_path, self.regimes[symbol])
                    except Exception as exc:
                        warnings.warn(
                            f"CPD segmentation failed for {symbol}: {exc}. "
                            "Context sampler will fall back to alternate methods."
                        )
                else:
                    missing_cpd.append(symbol)

            if missing_cpd:
                raise ValueError(
                    "CPD cache required but missing for symbols: "
                    f"{missing_cpd}. "
                    "Run scripts/batch_generate_cpd_cache.py for these symbols "
                    "or enable --allow-cpd-recompute."
                )

        # Determine fallback order so training never blocks
        ordered_methods = [context_method]
        for candidate in ["cpd_segmented", "time_equivalent", "final_hidden_state"]:
            if candidate not in ordered_methods:
                ordered_methods.append(candidate)
        self._context_methods_order = ordered_methods

        # Create valid samples (target symbol + start index)
        self.samples = self._create_samples()
        print(f"Created {len(self.samples)} training samples")

    def _prepare_price_series(self, series: pd.Series) -> pd.Series:
        """Align prices to global index and fill missing values causally.

        IMPORTANT: Only forward-fills (ffill) to handle missing days within
        the asset's trading history. Does NOT backfill to avoid creating
        artificial flat price histories before the asset existed.

        _create_samples() now handles filtering out pre-listing samples.
        """
        # Treat 0.0 as missing data (prevents division by zero in features)
        series = series.replace(0.0, float('nan'))

        aligned = series.reindex(self.prices.index)
        # Only forward-fill (causal) - do NOT backfill (creates fake history)
        aligned = aligned.ffill()

        # Leave the pre-listing span as NaN so downstream samplers can filter it out.
        first_valid = aligned.first_valid_index()
        if first_valid is None:
            return aligned  # all NaN, asset will be skipped

        pre_listing_mask = aligned.index < first_valid
        if pre_listing_mask.any():
            aligned.loc[pre_listing_mask] = pd.NA

        # For any remaining NaNs inside the trading history, fall back to small epsilon
        post_listing_mask = ~pre_listing_mask
        aligned.loc[post_listing_mask] = aligned.loc[post_listing_mask].fillna(1e-8)

        return aligned

    def _compute_features(self, prices: pd.Series) -> pd.DataFrame:
        """Compute 8 features for a price series (matching paper)."""
        return compute_xtrend_features(prices)

    def _create_samples(self) -> list:
        """Create list of valid (symbol, start_idx) pairs.

        Only creates samples where the asset has REAL historical data,
        not backfilled synthetic data. This prevents training on artificial
        flat price histories for assets that didn't exist in early periods.
        """
        samples = []
        skipped_assets = []
        late_start_assets = []

        for symbol in self.symbols:
            if symbol not in self.feature_tensors:
                continue

            listing_offset = self.listing_offsets.get(symbol)
            if listing_offset is None:
                # Asset has no valid data at all, skip
                skipped_assets.append(symbol)
                continue

            first_valid_idx = self.dates[listing_offset]

            # Only start sampling after asset exists AND has min_history warmup
            # This ensures we have real data for feature calculation
            real_start_idx = listing_offset + self.min_history

            feat = self.feature_tensors[symbol]
            n = len(feat)

            # Need history plus one extra day for the future return at t+1
            required_len = self.target_len + self.min_history + 1

            if n < required_len:
                skipped_assets.append(symbol)
                continue

            # Ensure we don't go beyond the valid range
            if real_start_idx >= n - self.target_len:
                # Asset doesn't have enough real history after it started
                skipped_assets.append(symbol)
                continue

            # Track assets that started after dataset start (late listings)
            if listing_offset > self.min_history:
                years_late = (first_valid_idx - self.dates[0]).days / 365.25
                late_start_assets.append((symbol, first_valid_idx, years_late))

            # Create rolling windows from real data only
            for start_idx in range(real_start_idx, n - self.target_len):
                samples.append((symbol, start_idx))

        # Print diagnostic info
        if skipped_assets:
            warnings.warn(
                f"Skipped {len(skipped_assets)} assets with insufficient real data: "
                f"{skipped_assets[:5]}{'...' if len(skipped_assets) > 5 else ''}"
            )
        if late_start_assets:
            print(f"\nℹ️  Found {len(late_start_assets)} assets with late start dates:")
            for sym, start_date, years in sorted(late_start_assets, key=lambda x: x[2], reverse=True)[:10]:
                print(f"  {sym}: started {start_date.strftime('%Y-%m-%d')} "
                      f"(+{years:.1f} years after dataset start)")
            if len(late_start_assets) > 10:
                print(f"  ... and {len(late_start_assets) - 10} more")
            print(f"  → Training samples for these assets exclude backfilled synthetic history\n")

        return samples

    def _regime_cache_path(self, symbol: str) -> Optional[Path]:
        """Create deterministic cache path for a symbol/config/span."""
        if not self.cpd_cache_dir:
            return None
        start = pd.Timestamp(self.dates[0]).strftime("%Y%m%d")
        end = pd.Timestamp(self.dates[-1]).strftime("%Y%m%d")
        cfg = self.cpd_config
        token = (
            f"{start}_{end}_{len(self.dates)}_"
            f"lb{cfg.lookback}_th{cfg.threshold:.2f}_"
            f"min{cfg.min_length}_max{cfg.max_length}"
        )
        filename = f"{symbol}_{token}.pkl"
        return self.cpd_cache_dir / filename

    def _parse_cache_dates(self, path: Path) -> tuple[Optional[pd.Timestamp], Optional[pd.Timestamp]]:
        """Extract start/end dates from a cache filename."""
        try:
            tokens = path.stem.split("_")
            if len(tokens) < 3:
                return None, None
            # format: SYMBOL_START_END_LEN_lbXX_thYY_minZZ_maxWW
            start = pd.Timestamp(tokens[1])
            end = pd.Timestamp(tokens[2])
            return start, end
        except Exception:
            return None, None

    def _list_cache_candidates(self, symbol: str) -> list[Path]:
        """Return all cache files matching this symbol/config."""
        if not self.cpd_cache_dir:
            return []
        cfg = self.cpd_config
        pattern = (
            f"{symbol}_*_lb{cfg.lookback}_"
            f"th{cfg.threshold:.2f}_"
            f"min{cfg.min_length}_max{cfg.max_length}.pkl"
        )
        return sorted(self.cpd_cache_dir.glob(pattern))

    def _select_primary_cache(self, candidates: list[Path]) -> Optional[Path]:
        """Pick the cache that overlaps the dataset and ends closest to dataset_end.

        Priority:
            1) Any cache with positive date overlap with the dataset
               (max overlap length, tie-breaker: closest end to dataset_end).
            2) Otherwise, fallback to cache whose end is nearest (future first).
        """
        if not candidates:
            return None

        dataset_start = self.dates[0]
        dataset_end = self.dates[-1]

        def overlap_score(path: Path):
            start, end = self._parse_cache_dates(path)
            if start is None or end is None:
                return (-1, pd.Timedelta.max)  # non-overlapping/invalid
            # overlap length in days
            overlap_start = max(start, dataset_start)
            overlap_end = min(end, dataset_end)
            overlap_days = (overlap_end - overlap_start).days + 1
            if overlap_days <= 0:
                return (0, pd.Timedelta.max)
            # higher overlap wins; smaller end-gap to dataset_end ties
            return (overlap_days, abs(dataset_end - end))

        # Step 1: pick among overlapping caches
        overlapping = [c for c in candidates if overlap_score(c)[0] > 0]
        if overlapping:
            return max(overlapping, key=lambda p: (overlap_score(p)[0], -overlap_score(p)[1]))

        # Step 2: fallback to nearest end date (prefer future)
        def nearest_end(path: Path):
            _, end = self._parse_cache_dates(path)
            if end is None:
                return (pd.Timedelta.max, pd.Timedelta.max)
            if end >= dataset_end:
                return (end - dataset_end, pd.Timedelta(0))
            return (pd.Timedelta.max, dataset_end - end)

        return min(candidates, key=nearest_end)

    def _load_and_align_cache(self, path: Path, symbol: str) -> Optional[RegimeSegments]:
        """Load a cache file and align segments to current dates."""
        payload = self._load_cached_regimes(path, symbol)
        if payload is None:
            return None
        aligned = self._align_cached_segments(symbol, payload)
        return aligned

    def _merge_gapfill_segments(self, base: RegimeSegments, extras: list[RegimeSegments]) -> RegimeSegments:
        """Merge non-overlapping segments, filling gaps only."""
        merged = list(base.segments)
        if not merged:
            # No base; just take the first non-empty extras in order
            for reg in extras:
                if reg and reg.segments:
                    merged = list(reg.segments)
                    break

        # Track last end date to avoid overlaps
        last_end = merged[-1].end_date if merged else None

        # Iterate extras sorted by their last segment end date
        extras_sorted = sorted(
            [r for r in extras if r and r.segments],
            key=lambda r: r.segments[-1].end_date
        )
        for reg in extras_sorted:
            for seg in reg.segments:
                if last_end is not None and seg.start_date <= last_end:
                    continue  # skip overlaps
                merged.append(seg)
                last_end = seg.end_date

        merged.sort(key=lambda s: s.start_date)
        return RegimeSegments(segments=merged, config=self.cpd_config)

    def _load_cached_regimes(self, path: Path, symbol: str):
        """Load cached regimes for a symbol."""
        try:
            with path.open("rb") as fh:
                payload = pickle.load(fh)
            segments = payload.get("segments")
            if not segments:
                return None
            return self._align_cached_segments(symbol, segments)
        except Exception as exc:
            warnings.warn(f"Failed to load CPD cache {path}: {exc}")
            return None

    def _save_cached_regimes(self, path: Path, segments):
        """Persist regimes to cache for reuse."""
        try:
            with path.open("wb") as fh:
                pickle.dump({"segments": segments}, fh)
        except Exception as exc:
            warnings.warn(f"Failed to save CPD cache {path}: {exc}")

    def _align_cached_segments(
        self, symbol: str, segments: RegimeSegments
    ) -> Optional[RegimeSegments]:
        """Clip cached regime indices to the dataset's calendar."""
        dataset_start = self.dates[0]
        dataset_end = self.dates[-1]
        aligned = []
        listing_offset = self.listing_offsets.get(symbol, 0)
        listing_date = self.dates[listing_offset]

        for seg in segments.segments:
            if seg.end_date < dataset_start or seg.start_date > dataset_end:
                continue

            start_date = max(seg.start_date, dataset_start, listing_date)
            end_date = min(seg.end_date, dataset_end)

            start_idx = self._date_to_idx.get(start_date)
            end_idx = self._date_to_idx.get(end_date)
            if start_idx is None or end_idx is None or start_idx > end_idx:
                continue

            aligned.append(
                RegimeSegment(
                    start_idx=start_idx,
                    end_idx=end_idx,
                    severity=seg.severity,
                    start_date=start_date,
                    end_date=end_date,
                )
            )

        if not aligned:
            return None

        return RegimeSegments(segments=aligned, config=segments.config)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """
        Returns:
            target_features: (target_len, input_dim) tensor
            context_features: (context_size, context_max_length, input_dim) tensor
            context_padding_mask: (context_size, context_max_length) bool tensor
            context_entity_ids: (context_size,) tensor with entity indices (-1 for padding)
            target_returns: (target_len,) tensor
            entity_id: tensor scalar (index of target symbol)
        """
        target_symbol, start_idx = self.samples[idx]

        # Get target features (information available up to time t)
        target_feat = self.feature_tensors[target_symbol][
            start_idx:start_idx + self.target_len
        ]
        # Target returns are realized after the decision at time t, so use t+1
        target_ret = self.return_tensors[target_symbol][
            start_idx + 1:start_idx + self.target_len + 1
        ]

        context_batch = self._sample_context_batch(target_symbol, start_idx)
        context_features, context_mask, context_entity_ids = self._context_batch_to_tensor(
            context_batch
        )

        # Entity ID
        entity_id = self.symbols.index(target_symbol)

        return {
            'target_features': target_feat.clone(),
            'context_features': context_features,
            'context_padding_mask': context_mask,
            'context_entity_ids': context_entity_ids,
            'target_returns': target_ret.clone(),
            'entity_id': torch.tensor(entity_id, dtype=torch.long)
        }

    def _sample_context_batch(self, target_symbol: str, start_idx: int):
        """Sample context sequences with fallback ordering."""
        target_date = self.dates[start_idx]
        exclude = [target_symbol]
        last_error = None

        for method in self._context_methods_order:
            try:
                return self._sample_with_method(method, target_date, exclude)
            except ValueError as exc:
                last_error = exc
                if method == self.context_method and not self._fallback_warned:
                    warnings.warn(
                        f"Primary context method '{self.context_method}' unavailable "
                        f"at {target_date}. Falling back to alternate samplers."
                    )
                    self._fallback_warned = True
                continue

        raise last_error if last_error else RuntimeError("No context method succeeded")

    def _sample_with_method(self, method: str, target_date: pd.Timestamp, exclude: List[str]):
        """Run the requested context sampler."""
        if method == "cpd_segmented":
            if not self.regimes:
                raise ValueError("CPD regimes unavailable for segmentation.")
            return sample_cpd_segmented(
                features=self.feature_tensors,
                dates=self.dates,
                symbols=self.symbols,
                regimes=self.regimes,
                target_date=target_date,
                C=self.context_size,
                max_length=self.context_max_length,
                seed=self.seed,
                exclude_symbols=exclude,
                listing_offsets=self.listing_offsets,
                allow_future_regimes=self.allow_future_regimes,
            )
        if method == "time_equivalent":
            return sample_time_equivalent(
                features=self.feature_tensors,
                dates=self.dates,
                symbols=self.symbols,
                target_date=target_date,
                C=self.context_size,
                l_t=self.target_len,
                seed=self.seed,
                exclude_symbols=exclude,
                listing_offsets=self.listing_offsets,
            )
        if method == "final_hidden_state":
            return sample_final_hidden_state(
                features=self.feature_tensors,
                dates=self.dates,
                symbols=self.symbols,
                target_date=target_date,
                C=self.context_size,
                l_c=self.context_max_length,
                seed=self.seed,
                exclude_symbols=exclude,
                listing_offsets=self.listing_offsets,
            )
        raise ValueError(f"Unknown context method: {method}")

    def _context_batch_to_tensor(self, batch) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Pad/truncate context batch to (C, L, input_dim)."""
        features = torch.zeros(
            (self.context_size, self.context_max_length, self.input_dim),
            dtype=torch.float32
        )
        mask = torch.zeros(
            (self.context_size, self.context_max_length),
            dtype=torch.bool
        )
        entity_ids = torch.full(
            (self.context_size,),
            fill_value=-1,
            dtype=torch.long
        )

        for i, seq in enumerate(batch.sequences):
            seq_len = min(seq.length, self.context_max_length)
            features[i, :seq_len] = seq.features[:seq_len]
            mask[i, :seq_len] = seq.padding_mask[:seq_len]
            entity_ids[i] = seq.entity_id.long()

        return features, mask, entity_ids


def create_model(model_type: str, config: ModelConfig):
    """Create model based on type."""
    entity_embedding = EntityEmbedding(config)

    if model_type == 'xtrend':
        model = XTrend(config, entity_embedding=entity_embedding)
    elif model_type == 'xtrendg':
        model = XTrendG(config, entity_embedding=entity_embedding)
    elif model_type == 'xtrendq':
        model = XTrendQ(config, entity_embedding=entity_embedding, num_quantiles=13)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # Also return encoder and cross-attention (needed for forward pass)
    encoder = LSTMEncoder(config, use_entity=True, entity_embedding=entity_embedding)
    cross_attn = XTrendCrossAttention(config)

    return {
        'encoder': encoder,
        'cross_attn': cross_attn,
        'model': model,
        'entity_embedding': entity_embedding
    }


def encode_context_set(
    encoder: LSTMEncoder,
    context_features: torch.Tensor,
    context_entity_ids: torch.Tensor,
    context_padding_mask: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Encode context sequences so they match the hidden_dim expected by cross-attention.

    Args:
        encoder: Shared sequence encoder.
        context_features: (batch, C, L, input_dim) tensor.
        context_entity_ids: (batch, C) tensor with entity indices (-1 = padding).
        context_padding_mask: (batch, C, L) bool tensor (True = valid timestep).

    Returns:
        context_encoded: (batch, C*L, hidden_dim) tensor flattened across contexts/timesteps.
        context_mask: (batch, C*L) bool tensor indicating valid encoded positions.
    """
    batch_size, context_size, seq_len, input_dim = context_features.shape
    hidden_dim = encoder.config.hidden_dim

    context_valid = context_entity_ids >= 0  # (batch, C)
    timestep_mask = context_padding_mask & context_valid.unsqueeze(-1)

    # Flatten contexts to encode only valid ones in a single pass
    flat_features = context_features.view(batch_size * context_size, seq_len, input_dim)
    flat_entities = context_entity_ids.view(-1)
    flat_valid = (timestep_mask.any(dim=-1)).view(-1)

    encoded = context_features.new_zeros(batch_size * context_size, seq_len, hidden_dim)

    if flat_valid.any():
        valid_features = flat_features[flat_valid]
        valid_entities = flat_entities[flat_valid]
        encoded_outputs = encoder(valid_features, entity_ids=valid_entities).hidden_states
        encoded[flat_valid] = encoded_outputs

    encoded = encoded.view(batch_size, context_size, seq_len, hidden_dim)

    # Zero-out padded positions before flattening
    mask_float = timestep_mask.unsqueeze(-1).to(encoded.dtype)
    encoded = encoded * mask_float

    encoded = encoded.view(batch_size, context_size * seq_len, hidden_dim)
    context_mask = timestep_mask.view(batch_size, context_size * seq_len)

    return encoded, context_mask


def compute_loss(model_type, outputs, returns, quantile_levels=None):
    """Compute appropriate loss for model type."""
    warmup_steps = 63

    if model_type == 'xtrend':
        # Simple Sharpe loss
        return sharpe_loss(outputs, returns, warmup_steps=warmup_steps)

    elif model_type == 'xtrendg':
        # Joint Gaussian loss
        return joint_gaussian_loss(
            outputs['mean'],
            outputs['std'],
            outputs['positions'],
            returns,
            alpha=1.0,
            warmup_steps=warmup_steps
        )

    elif model_type == 'xtrendq':
        # Joint Quantile loss
        return joint_quantile_loss(
            outputs['quantiles'],
            quantile_levels,
            outputs['positions'],
            returns,
            alpha=5.0,
            warmup_steps=warmup_steps
        )


def train_epoch(
    models,
    dataloader,
    optimizer,
    model_type,
    device,
    quantile_levels=None
):
    """Train for one epoch."""
    encoder = models['encoder']
    cross_attn = models['cross_attn']
    model = models['model']

    encoder.train()
    cross_attn.train()
    model.train()

    total_loss = 0.0
    num_batches = 0

    pbar = tqdm(dataloader, desc="Training")
    for batch in pbar:
        # Move to device
        target_features = batch['target_features'].to(device)
        context_features = batch['context_features'].to(device)
        context_entity_ids = batch['context_entity_ids'].to(device)
        context_padding_mask = batch['context_padding_mask'].to(device)
        target_returns = batch['target_returns'].to(device)
        entity_ids = batch['entity_id'].to(device)

        # Forward pass
        optimizer.zero_grad()

        # Encode target
        target_encoded = encoder(target_features, entity_ids=entity_ids)

        # Encode context assets to hidden_dim before cross-attention
        context_encoded, context_mask = encode_context_set(
            encoder,
            context_features,
            context_entity_ids,
            context_padding_mask
        )

        # Cross-attention
        cross_attn_output = cross_attn(
            target_encoded.hidden_states,
            context_encoded,
            context_padding_mask=context_mask
        )

        # Decoder + prediction
        outputs = model(
            target_features,
            cross_attn_output.output,
            entity_ids=entity_ids
        )

        # Compute loss
        loss = compute_loss(model_type, outputs, target_returns, quantile_levels)

        # Backward
        loss.backward()
        # Paper specification: max_norm=10.0 (from x-trend-architecture skill)
        torch.nn.utils.clip_grad_norm_(
            list(encoder.parameters()) +
            list(cross_attn.parameters()) +
            list(model.parameters()),
            max_norm=10.0
        )
        optimizer.step()

        # Track
        total_loss += loss.item()
        num_batches += 1
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    return total_loss / num_batches


def validate(models, dataloader, model_type, device, quantile_levels=None):
    """Validate model."""
    encoder = models['encoder']
    cross_attn = models['cross_attn']
    model = models['model']

    encoder.eval()
    cross_attn.eval()
    model.eval()

    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validation"):
            target_features = batch['target_features'].to(device)
            context_features = batch['context_features'].to(device)
            context_entity_ids = batch['context_entity_ids'].to(device)
            context_padding_mask = batch['context_padding_mask'].to(device)
            target_returns = batch['target_returns'].to(device)
            entity_ids = batch['entity_id'].to(device)

            # Forward pass (same as training)
            target_encoded = encoder(target_features, entity_ids=entity_ids)
            context_encoded, context_mask = encode_context_set(
                encoder,
                context_features,
                context_entity_ids,
                context_padding_mask
            )
            cross_attn_output = cross_attn(
                target_encoded.hidden_states,
                context_encoded,
                context_padding_mask=context_mask
            )
            outputs = model(
                target_features,
                cross_attn_output.output,
                entity_ids=entity_ids
            )

            # Compute loss
            loss = compute_loss(model_type, outputs, target_returns, quantile_levels)

            total_loss += loss.item()
            num_batches += 1

    # Handle case with no validation samples gracefully
    if num_batches == 0:
        return float('nan')

    return total_loss / num_batches


def main():
    parser = argparse.ArgumentParser(description="Train X-Trend model")
    parser.add_argument('--model', type=str, default='xtrendq',
                       choices=['xtrend', 'xtrendg', 'xtrendq'],
                       help='Model variant to train')
    parser.add_argument('--data-path', type=str, default='data/bloomberg/processed',
                       help='Path to Bloomberg parquet files')
    parser.add_argument('--train-cutoff', type=str, default='2021-12-31',
                       help='Inclusive train end date; validation starts from this date')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=64,
                       help='Batch size (paper: 64 or 128)')
    parser.add_argument('--lr', type=float, default=1e-3,
                       help='Learning rate (paper: 1e-3)')
    parser.add_argument('--hidden-dim', type=int, default=128,
                       help='Hidden dimension')
    parser.add_argument('--num-heads', type=int, default=4,
                       help='Number of attention heads')
    # x-trend-architecture skill recommends 0.3-0.5 for regularization
    parser.add_argument('--dropout', type=float, default=0.3,
                       help='Dropout rate (paper default: 0.3)')
    parser.add_argument('--target-len', type=int, default=126,
                       help='Target sequence length (days)')
    parser.add_argument('--context-size', type=int, default=10,
                       help='Number of context sequences per target (paper best: 10)')
    parser.add_argument('--context-method', type=str, default='cpd_segmented',
                       choices=['cpd_segmented', 'time_equivalent', 'final_hidden_state'],
                       help='Primary Phase 4 context sampler')
    parser.add_argument('--context-max-length', type=int, default=None,
                       help='Max context length (days). Defaults to --cpd-max-length.')
    parser.add_argument('--min-history', type=int, default=252,
                       help='Minimum history (days) before sampling targets')
    parser.add_argument('--cpd-lookback', type=int, default=21,
                       help='GP-CPD lookback window (days)')
    parser.add_argument('--cpd-threshold', type=float, default=0.95,
                       help='GP-CPD severity threshold for change-points (paper: 0.95 for lmax=63)')
    parser.add_argument('--cpd-min-length', type=int, default=5,
                       help='Minimum GP-CPD regime length (days)')
    parser.add_argument('--cpd-max-length', type=int, default=63,
                       help='Maximum GP-CPD regime length (days) (paper best: 63)')
    parser.add_argument('--context-seed', type=int, default=None,
                       help='Optional RNG seed for context sampling')
    parser.add_argument('--cpd-cache-dir', type=str, default='data/bloomberg/cpd_cache',
                       help='Directory for caching GP-CPD regimes (set empty to disable)')
    parser.add_argument('--allow-cpd-recompute', action='store_true', default=False,
                       help='If set, recompute GP-CPD on the fly when cache is missing. '
                            'Default: False (require caches for consistency)')
    parser.add_argument('--return-clip', type=float, default=20.0,
                       help='Clip absolute normalized returns/features to this value to prevent '
                            'exploding losses from data gaps or placeholder prices')
    parser.add_argument('--resume', type=str, default=None,
                       help='Resume from checkpoint')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device to use')

    args = parser.parse_args()

    print("="*60)
    print(f"Training {args.model.upper()} on Bloomberg Futures Data")
    print("="*60)

    # Load data
    print(f"\nLoading data from {args.data_path}...")
    source = BloombergParquetSource(root_path=args.data_path)
    symbols = source.symbols()
    print(f"Found {len(symbols)} symbols: {symbols[:10]}...")

    # Load prices - full period including validation data (1990-2025)
    # Training uses data up to train_cutoff with lookahead-allowed caches
    # Validation uses data after train_cutoff with rolling non-lookahead caches
    prices = source.load_prices(symbols, start='1990-01-01', end='2025-12-31')
    print(f"Loaded prices: {prices.shape}")

    # Train/val split (time-based)
    train_cutoff = args.train_cutoff
    train_prices = prices.loc[:train_cutoff]
    val_prices = prices.loc[train_cutoff:]

    print(f"Train period: {train_prices.index[0]} to {train_prices.index[-1]}")
    print(f"Val period: {val_prices.index[0]} to {val_prices.index[-1]}")

    # CPD + context configuration
    context_max_length = (
        args.context_max_length
        if args.context_max_length is not None
        else args.cpd_max_length
    )
    cpd_config = CPDConfig(
        lookback=args.cpd_lookback,
        threshold=args.cpd_threshold,
        min_length=args.cpd_min_length,
        max_length=args.cpd_max_length
    )

    base_dataset_kwargs = dict(
        symbols=symbols,
        target_len=args.target_len,
        context_size=args.context_size,
        context_method=args.context_method,
        context_max_length=context_max_length,
        vol_target=0.15,
        min_history=args.min_history,
        cpd_config=cpd_config,
        seed=args.context_seed,
        cpd_cache_dir=args.cpd_cache_dir if args.cpd_cache_dir else None,
        allow_cpd_recompute=args.allow_cpd_recompute,
        clip_value=args.return_clip,
    )

    print("\nCreating datasets with episodic context sampling...")
    train_dataset = XTrendDataset(
        train_prices,
        allow_future_regimes=True,  # paper-approved hindsight for training
        **base_dataset_kwargs,
    )
    val_dataset = XTrendDataset(
        val_prices,
        allow_future_regimes=False,  # strict causality for validation/backtests
        **base_dataset_kwargs,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4
    )

    # Model configuration
    config = ModelConfig(
        input_dim=8,
        hidden_dim=args.hidden_dim,
        num_entities=len(symbols),
        num_attention_heads=args.num_heads,
        dropout=args.dropout
    )

    # Create model
    print(f"\nCreating {args.model} model...")
    models = create_model(args.model, config)

    # Move to device
    device = torch.device(args.device)
    for key in models:
        models[key].to(device)

    # Optimizer
    all_params = (
        list(models['encoder'].parameters()) +
        list(models['cross_attn'].parameters()) +
        list(models['model'].parameters())
    )
    # Deduplicate shared modules (entity embedding lives in encoder and decoder)
    unique_params = []
    seen = set()
    for p in all_params:
        if id(p) not in seen:
            unique_params.append(p)
            seen.add(id(p))
    optimizer = torch.optim.AdamW(unique_params, lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Quantile levels (for XTrendQ)
    quantile_levels = None
    if args.model == 'xtrendq':
        quantile_levels = torch.tensor(
            [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99],
            device=device
        )

    # Resume from checkpoint if specified
    start_epoch = 0
    best_val_loss = float('inf')

    if args.resume:
        print(f"\nResuming from {args.resume}...")
        checkpoint = torch.load(args.resume)
        models['encoder'].load_state_dict(checkpoint['encoder'])
        models['cross_attn'].load_state_dict(checkpoint['cross_attn'])
        models['model'].load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))

    # Training loop
    checkpoint_dir = Path('checkpoints')
    checkpoint_dir.mkdir(exist_ok=True)

    history = {
        'train_loss': [],
        'val_loss': [],
        'lr': []
    }

    print(f"\nStarting training for {args.epochs} epochs...")
    print(f"Device: {device}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")

    for epoch in range(start_epoch, args.epochs):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch + 1}/{args.epochs}")
        print(f"{'='*60}")

        # Train
        train_loss = train_epoch(
            models, train_loader, optimizer,
            args.model, device, quantile_levels
        )

        # Validate
        val_loss = validate(
            models, val_loader,
            args.model, device, quantile_levels
        )

        # Scheduler step
        scheduler.step()

        # Log
        current_lr = optimizer.param_groups[0]['lr']
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['lr'].append(current_lr)

        print(f"\nEpoch {epoch + 1} Summary:")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss:   {val_loss:.4f}")
        print(f"  LR:         {current_lr:.6f}")

        # Save checkpoint
        checkpoint = {
            'epoch': epoch,
            'encoder': models['encoder'].state_dict(),
            'cross_attn': models['cross_attn'].state_dict(),
            'model': models['model'].state_dict(),
            'optimizer': optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'config': config.__dict__,
            'model_type': args.model,
            'best_val_loss': best_val_loss
        }

        # Save latest
        torch.save(checkpoint, checkpoint_dir / f'{args.model}_latest.pt')

        # Save best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(checkpoint, checkpoint_dir / f'{args.model}_best.pt')
            print(f"  ✅ New best model! (val_loss: {val_loss:.4f})")

        # Save periodic
        if (epoch + 1) % 10 == 0:
            torch.save(checkpoint, checkpoint_dir / f'{args.model}_epoch_{epoch + 1}.pt')

    # Save training history
    history_file = checkpoint_dir / f'{args.model}_history.json'
    with open(history_file, 'w') as f:
        json.dump(history, f, indent=2)

    print("\n" + "="*60)
    print("Training complete!")
    print("="*60)
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Checkpoints saved to: {checkpoint_dir}")
    print(f"  - {args.model}_best.pt (best model)")
    print(f"  - {args.model}_latest.pt (latest)")
    print(f"  - {args.model}_history.json (training history)")


if __name__ == "__main__":
    main()
