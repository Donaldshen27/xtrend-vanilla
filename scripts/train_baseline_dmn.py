#!/usr/bin/env python3
"""Train the baseline Deep Momentum Network (DMN) without contexts.

This script mirrors the data handling in ``scripts/train_xtrend.py`` but
removes all context-set machinery so we can benchmark a pure single-regime
forecaster against X-Trend. It uses:

- Phase 1 features (5 vol‑scaled returns + 3 MACD indicators)
- EWMA σ (span=60, min_periods=20) for both feature normalization and
  the target return scaling r̂[t+1] = r[t+1] / σ[t] (lookahead-safe: σ is
  lagged by one day for the target)
- Sharpe-ratio loss (Equation 8) with a warmup period to avoid the
  burn‑in bias from the volatility estimator

Usage examples (run inside the uv environment):

    uv run python scripts/train_baseline_dmn.py \
        --data-path data/bloomberg/processed \
        --train-cutoff 2021-12-31 \
        --epochs 50 --batch-size 64 --hidden-dim 128

Checkpoints are written to ``checkpoints/`` and include the model state,
optimizer state, and basic training history.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Optional

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from xtrend.data.sources import BloombergParquetSource
from xtrend.data.features import compute_xtrend_features
from xtrend.models import ModelConfig, BaselineDMN, sharpe_loss


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class BaselineDMNDataset(Dataset):
    """Return-only dataset for the baseline DMN (no contexts).

    Each sample consists of a target feature window of length ``seq_len`` and
    the corresponding next-day normalized returns for the same window.

    Args:
        prices: Wide price DataFrame (date index, columns = symbols)
        symbols: Ordered list of symbols (used for stable entity IDs)
        seq_len: Input/target sequence length (paper uses 126)
        min_history: Minimum real history before sampling starts (paper uses 252)
        vol_span: EWMA span for σ_t (paper uses 60)
        clip_value: Clip abs(feature/return) to this value for stability
    """

    def __init__(
        self,
        prices: pd.DataFrame,
        symbols: List[str],
        seq_len: int = 126,
        min_history: int = 252,
        vol_span: int = 60,
        clip_value: float = 20.0,
    ):
        self.prices = prices
        self.dates = prices.index
        self.symbols = symbols
        self.seq_len = seq_len
        self.min_history = min_history
        self.vol_span = vol_span
        self.clip_value = clip_value

        # Book-keeping
        self.listing_offsets: dict[str, int] = {}
        self.features = {}
        self.feature_tensors = {}
        self.returns = {}
        self.return_tensors = {}

        # Pre-compute features and normalized returns per symbol
        for symbol in symbols:
            raw_series = prices.get(symbol)
            if raw_series is None or raw_series.isna().all():
                continue

            price_series = self._prepare_price_series(raw_series)
            first_valid = price_series.first_valid_index()
            if first_valid is None:
                continue

            self.listing_offsets[symbol] = self.dates.get_loc(first_valid)

            feats = compute_xtrend_features(price_series)
            if self.clip_value:
                feats = feats.clip(lower=-self.clip_value, upper=self.clip_value)
            self.features[symbol] = feats
            self.feature_tensors[symbol] = torch.tensor(
                feats.fillna(0.0).values, dtype=torch.float32
            )

            daily_rets = price_series.pct_change()
            sigma_t = (
                daily_rets.ewm(span=self.vol_span, min_periods=20)
                .std()
                .shift(1)  # lookahead-safe for targets
            )
            sigma_t = sigma_t.clip(lower=1e-8).bfill()
            normalized_rets = (daily_rets / sigma_t).fillna(0.0)
            if self.clip_value:
                normalized_rets = normalized_rets.clip(
                    lower=-self.clip_value, upper=self.clip_value
                )
            self.returns[symbol] = normalized_rets
            self.return_tensors[symbol] = torch.tensor(
                normalized_rets.values, dtype=torch.float32
            )

        self.samples = self._create_samples()
        if not self.samples:
            raise ValueError("No training samples available; check data coverage and min_history")

    # ------------------------------ utils ---------------------------------

    def _prepare_price_series(self, series: pd.Series) -> pd.Series:
        """Align prices to the dataset index and forward-fill causally.

        - Treat zeros as missing (avoids div-by-zero in returns)
        - Do *not* backfill pre-listing; leave as NaN so sampling can skip it
        - Fill intra-history gaps with ffill, then small epsilon
        """

        series = series.replace(0.0, float("nan"))
        aligned = series.reindex(self.dates).ffill()

        first_valid = aligned.first_valid_index()
        if first_valid is None:
            return aligned

        pre_listing = aligned.index < first_valid
        aligned.loc[pre_listing] = pd.NA
        aligned.loc[~pre_listing] = aligned.loc[~pre_listing].fillna(1e-8)
        return aligned

    def _create_samples(self) -> list[tuple[str, int]]:
        samples: list[tuple[str, int]] = []
        for symbol in self.symbols:
            if symbol not in self.feature_tensors:
                continue

            listing_offset = self.listing_offsets.get(symbol)
            if listing_offset is None:
                continue

            feat_len = len(self.feature_tensors[symbol])
            # Need seq_len features + 1 return for t+1
            max_start = feat_len - (self.seq_len + 1)
            start_min = listing_offset + self.min_history

            if max_start <= start_min:
                continue

            for start_idx in range(start_min, max_start):
                samples.append((symbol, start_idx))

        return samples

    # ------------------------------ dunder ---------------------------------

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.samples)

    def __getitem__(self, idx: int):  # type: ignore[override]
        symbol, start_idx = self.samples[idx]

        feats = self.feature_tensors[symbol][start_idx : start_idx + self.seq_len]
        rets = self.return_tensors[symbol][start_idx + 1 : start_idx + self.seq_len + 1]
        entity_id = torch.tensor(self.symbols.index(symbol), dtype=torch.long)

        return {
            "features": feats.clone(),
            "returns": rets.clone(),
            "entity_id": entity_id,
        }


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------


def train_epoch(model, dataloader, optimizer, device, warmup_steps: int):
    model.train()
    total_loss = 0.0
    n = 0

    for batch in tqdm(dataloader, desc="Train", leave=False):
        feats = batch["features"].to(device)
        rets = batch["returns"].to(device)
        entity_ids = batch["entity_id"].to(device)

        optimizer.zero_grad()
        positions = model(feats, entity_ids=entity_ids)
        loss = sharpe_loss(positions, rets, warmup_steps=warmup_steps)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        optimizer.step()

        total_loss += loss.item()
        n += 1

    return total_loss / max(n, 1)


def validate(model, dataloader, device, warmup_steps: int):
    model.eval()
    total_loss = 0.0
    n = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Val", leave=False):
            feats = batch["features"].to(device)
            rets = batch["returns"].to(device)
            entity_ids = batch["entity_id"].to(device)

            positions = model(feats, entity_ids=entity_ids)
            loss = sharpe_loss(positions, rets, warmup_steps=warmup_steps)

            total_loss += loss.item()
            n += 1

    if n == 0:
        return float("nan")
    return total_loss / n


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Train baseline DMN (no contexts) on Bloomberg futures data"
    )
    parser.add_argument("--data-path", type=str, default="data/bloomberg/processed",
                        help="Path to parquet prices (see scripts/convert_bloomberg_to_parquet.py)")
    parser.add_argument("--train-cutoff", type=str, default="2021-12-31",
                        help="Inclusive train end date; val starts after this date")
    parser.add_argument("--epochs", type=int, default=30, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--hidden-dim", type=int, default=128, help="Hidden size d_h")
    parser.add_argument("--dropout", type=float, default=0.3, help="Dropout rate")
    parser.add_argument("--target-len", type=int, default=126, help="Sequence length l_t")
    parser.add_argument("--min-history", type=int, default=252, help="Min history before sampling")
    parser.add_argument("--vol-span", type=int, default=60, help="EWMA span for σ_t")
    parser.add_argument("--warmup-steps", type=int, default=63, help="Sharpe warmup steps l_s")
    parser.add_argument("--return-clip", type=float, default=20.0,
                        help="Clip abs(feature/return) to this value")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=5e-4, help="AdamW weight decay")
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Compute device")
    parser.add_argument("--resume", type=str, default=None,
                        help="Optional checkpoint to resume from")
    parser.add_argument("--no-entity", action="store_true", default=False,
                        help="Disable entity embeddings (for zero-shot style regularization)")
    parser.add_argument("--early-stop-patience", type=int, default=3,
                        help="Early stop if val loss doesn't improve for this many epochs")
    parser.add_argument("--full-span-only", action="store_true", default=False,
                        help="Train only on symbols that have data across the full date range")
    parser.add_argument("--start-date", type=str, default="1990-01-01",
                        help="Earliest date to include in prices (filter symbols that start after this)")
    parser.add_argument("--coverage-grace-days", type=int, default=30,
                        help="Allow symbols whose first_valid is within this many days after --start-date")

    args = parser.parse_args()

    print("=" * 60)
    print("Baseline DMN training (no contexts)")
    print("=" * 60)

    if args.warmup_steps >= args.target_len:
        new_warmup = max(1, args.target_len - 1)
        print(f"⚠️  warmup_steps={args.warmup_steps} >= target_len={args.target_len}; "
              f"resetting warmup_steps to {new_warmup}")
        args.warmup_steps = new_warmup

    # Data loading
    source = BloombergParquetSource(root_path=args.data_path)
    symbols = source.symbols()
    prices = source.load_prices(symbols, start=args.start_date, end="2025-12-31")

    dataset_start, dataset_end = prices.index[0], prices.index[-1]

    if args.full_span_only:
        symbols = [
            sym for sym in symbols
            if (fv := prices[sym].first_valid_index()) is not None
            and (lv := prices[sym].last_valid_index()) is not None
            and fv <= dataset_start
            and lv >= dataset_end
        ]
        if not symbols:
            raise ValueError("No symbols have full-span coverage; disable --full-span-only.")
        prices = prices[symbols]
        print(f"Filtered to full-span symbols: {len(symbols)} remaining")

    # Optional: enforce coverage from start-date through train cutoff/end (e.g., 1995 start)
    coverage_filter_start = pd.Timestamp(args.start_date)
    grace = pd.Timedelta(days=args.coverage_grace_days)
    filtered = []
    for sym in symbols:
        series = prices[sym]
        fv, lv = series.first_valid_index(), series.last_valid_index()
        if fv is None or lv is None:
            continue
        if fv <= coverage_filter_start + grace and lv >= dataset_end:
            filtered.append(sym)

    print(f"Symbols before coverage filter: {len(symbols)}; after filter: {len(filtered)} "
          f"(start <= {coverage_filter_start.date()} + {args.coverage_grace_days}d, end >= {dataset_end.date()})")

    symbols = filtered
    if not symbols:
        raise ValueError(
            f"No symbols cover {coverage_filter_start.date()}..{dataset_end.date()}. "
            "Loosen --start-date or disable coverage filtering."
        )
    prices = prices[symbols]
    print(f"Filtered to symbols covering {coverage_filter_start.date()}..{dataset_end.date()}: {len(symbols)}")

    cutoff_ts = pd.Timestamp(args.train_cutoff)
    train_prices = prices.loc[: cutoff_ts]
    val_prices = prices.loc[cutoff_ts + pd.Timedelta(days=1) :]

    print(f"Train period: {train_prices.index[0]} → {train_prices.index[-1]}")
    print(f" Val period: {val_prices.index[0]} → {val_prices.index[-1]}")

    dataset_kwargs = dict(
        symbols=symbols,
        seq_len=args.target_len,
        min_history=args.min_history,
        vol_span=args.vol_span,
        clip_value=args.return_clip,
    )

    train_ds = BaselineDMNDataset(train_prices, **dataset_kwargs)
    val_ds = BaselineDMNDataset(val_prices, **dataset_kwargs)

    print(f"Train samples: {len(train_ds):,}  Val samples: {len(val_ds):,}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # Model
    config = ModelConfig(
        input_dim=8,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        num_entities=len(symbols),
    )
    model = BaselineDMN(config, use_entity=not args.no_entity).to(args.device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    start_epoch = 0
    best_val = float("inf")

    if args.resume:
        checkpoint = torch.load(args.resume, map_location=args.device)
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        start_epoch = checkpoint.get("epoch", 0) + 1
        best_val = checkpoint.get("best_val", float("inf"))
        scheduler.load_state_dict(checkpoint.get("scheduler_state", scheduler.state_dict()))
        print(f"Resumed from {args.resume} (epoch {start_epoch})")

    history = {"train_loss": [], "val_loss": [], "lr": []}
    ckpt_dir = Path("checkpoints")
    ckpt_dir.mkdir(exist_ok=True)

    patience = args.early_stop_patience
    bad_epochs = 0

    for epoch in range(start_epoch, args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")

        train_loss = train_epoch(model, train_loader, optimizer, args.device, args.warmup_steps)
        val_loss = validate(model, val_loader, args.device, args.warmup_steps)
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["lr"].append(optimizer.param_groups[0]["lr"])

        print(f"  train_loss: {train_loss:.4f}  val_loss: {val_loss:.4f}  lr: {optimizer.param_groups[0]['lr']:.6f}")

        ckpt = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "config": config.__dict__,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "best_val": best_val,
        }

        torch.save(ckpt, ckpt_dir / "baseline_dmn_latest.pt")
        if val_loss < best_val:
            best_val = val_loss
            torch.save(ckpt, ckpt_dir / "baseline_dmn_best.pt")
            print("  ✅ New best checkpoint")
            bad_epochs = 0
        else:
            bad_epochs += 1

        if patience and bad_epochs >= patience:
            print(f"Early stopping after {bad_epochs} worse epochs (patience={patience})")
            break

    with open(ckpt_dir / "baseline_dmn_history.json", "w") as f:
        json.dump(history, f, indent=2)

    print("\nTraining complete. Checkpoints saved to ./checkpoints/")


if __name__ == "__main__":
    main()
