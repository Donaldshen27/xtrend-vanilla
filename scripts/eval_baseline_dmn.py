#!/usr/bin/env python3
"""
Quick evaluation of a trained Baseline DMN on the validation window (2022–2025).

Loads a checkpoint, rebuilds the validation dataset with the same filtering
logic as the training script, runs the model, and reports realized annualized
Sharpe over the validation samples.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path
import pandas as pd
import torch
from torch.utils.data import DataLoader

from xtrend.data.sources import BloombergParquetSource
from xtrend.models import ModelConfig, BaselineDMN
from scripts.train_baseline_dmn import BaselineDMNDataset


def compute_sharpe(strategy_returns: torch.Tensor, eps: float = 1e-8) -> float:
    """Annualized Sharpe from a 1-D tensor of returns."""
    mean = strategy_returns.mean()
    std = strategy_returns.std(unbiased=False) + eps
    return float((mean / std) * math.sqrt(252.0))


def main():
    parser = argparse.ArgumentParser(description="Evaluate Baseline DMN Sharpe on validation window.")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/baseline_dmn_best.pt",
                        help="Path to checkpoint produced by train_baseline_dmn.py")
    parser.add_argument("--data-path", type=str, default="data/bloomberg/processed")
    parser.add_argument("--start-date", type=str, default="1995-01-01",
                        help="Earliest date to include for coverage filtering")
    parser.add_argument("--coverage-grace-days", type=int, default=30,
                        help="Allow first_valid within N days after start-date")
    parser.add_argument("--train-cutoff", type=str, default="2021-12-31",
                        help="Validation starts after this date (inclusive of next trading day)")
    parser.add_argument("--target-len", type=int, default=126)
    parser.add_argument("--min-history", type=int, default=252)
    parser.add_argument("--vol-span", type=int, default=60)
    parser.add_argument("--return-clip", type=float, default=20.0)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    # Load data
    source = BloombergParquetSource(root_path=args.data_path)
    symbols = source.symbols()
    prices = source.load_prices(symbols, start=args.start_date, end="2025-12-31")
    dataset_start, dataset_end = prices.index[0], prices.index[-1]

    coverage_start = pd.Timestamp(args.start_date)
    grace = pd.Timedelta(days=args.coverage_grace_days)
    symbols = [
        sym for sym in symbols
        if (fv := prices[sym].first_valid_index()) is not None
        and (lv := prices[sym].last_valid_index()) is not None
        and fv <= coverage_start + grace
        and lv >= dataset_end
    ]
    if not symbols:
        raise ValueError("No symbols pass coverage filter; relax start-date or grace.")

    prices = prices[symbols]

    # Val window: day after train_cutoff onward
    cutoff_ts = pd.Timestamp(args.train_cutoff)
    val_prices = prices.loc[cutoff_ts + pd.Timedelta(days=1):]

    # Adjust warmup if needed
    warmup_steps = min(args.target_len - 1, 63) if args.target_len <= 63 else 63

    # Build dataset/loader
    ds_kwargs = dict(
        symbols=symbols,
        seq_len=args.target_len,
        min_history=args.min_history,
        vol_span=args.vol_span,
        clip_value=args.return_clip,
    )
    val_ds = BaselineDMNDataset(val_prices, **ds_kwargs)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # Model
    config = ModelConfig(
        input_dim=8,
        hidden_dim=128,  # will be overwritten by checkpoint config if present
        dropout=0.3,
        num_entities=len(symbols),
    )
    model = BaselineDMN(config, use_entity=True).to(args.device)

    # Load checkpoint
    ckpt = torch.load(args.checkpoint, map_location=args.device)
    if "config" in ckpt:
        cfg = ckpt["config"]
        config.hidden_dim = cfg.get("hidden_dim", config.hidden_dim)
        config.dropout = cfg.get("dropout", config.dropout)
        config.num_entities = cfg.get("num_entities", config.num_entities)
    model = BaselineDMN(config, use_entity=True).to(args.device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    # Evaluate
    all_strategy_returns = []
    with torch.no_grad():
        for batch in val_loader:
            feats = batch["features"].to(args.device)
            rets = batch["returns"].to(args.device)
            entity_ids = batch["entity_id"].to(args.device)

            positions = model(feats, entity_ids=entity_ids)
            strategy_returns = positions[:, warmup_steps:] * rets[:, warmup_steps:]
            all_strategy_returns.append(strategy_returns.flatten().cpu())

    if not all_strategy_returns:
        raise RuntimeError("No validation strategy returns computed.")

    strategy_returns = torch.cat(all_strategy_returns)
    sharpe = compute_sharpe(strategy_returns)

    print("=" * 60)
    print(f"Validation Sharpe (annualized): {sharpe:.3f}")
    print(f"Samples used (post-warmup): {strategy_returns.numel()}")
    print("=" * 60)


if __name__ == "__main__":
    main()
