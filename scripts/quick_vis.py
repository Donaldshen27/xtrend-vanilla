#!/usr/bin/env python3
"""Quick visualization of XTrendQ predictions for a single symbol on the validation split.

Usage:
    uv run python scripts/quick_vis.py \
        --checkpoint checkpoints/xtrendq_best.pt \
        --symbol CL \
        --train-cutoff 2023-12-29 \
        --data-path data/bloomberg/processed \
        --cpd-cache-dir data/bloomberg/cpd_cache

Outputs PNGs to outputs/plots/
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

from scripts.train_xtrend import (
    XTrendDataset,
    ModelConfig,
    create_model,
    encode_context_set,
)
from xtrend.data.sources import BloombergParquetSource
from xtrend.cpd import CPDConfig


def load_batch_for_symbol(loader, symbols, target_symbol):
    for batch in loader:
        if symbols[batch["entity_id"].item()] == target_symbol:
            return batch
    raise ValueError(f"Symbol {target_symbol} not found in validation slice")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--train-cutoff", default="2023-12-29")
    parser.add_argument("--val-end", default=None, help="Optional end date for validation slice (default: dataset end)")
    parser.add_argument("--window-days", type=int, default=600, help="Days back from train_cutoff to include (ensures enough samples)")
    parser.add_argument("--data-path", default="data/bloomberg/processed")
    parser.add_argument("--cpd-cache-dir", default="data/bloomberg/cpd_cache")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--context-size", type=int, default=10)
    parser.add_argument("--target-len", type=int, default=126)
    parser.add_argument("--min-history", type=int, default=252)
    parser.add_argument("--cpd-lookback", type=int, default=21)
    parser.add_argument("--cpd-threshold", type=float, default=0.85)
    parser.add_argument("--cpd-min-length", type=int, default=5)
    parser.add_argument("--cpd-max-length", type=int, default=63)
    parser.add_argument("--allow-cpd-recompute", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data
    source = BloombergParquetSource(root_path=args.data_path)
    symbols = source.symbols()
    prices = source.load_prices(symbols, start="1990-01-01", end="2023-12-31")
    start_date = (torch.tensor([]),)  # placeholder to keep namespace clean
    start_date = pd.Timestamp(args.train_cutoff) - pd.Timedelta(days=args.window_days)
    end_date = pd.Timestamp(args.val_end) if args.val_end else prices.index[-1]
    val_prices = prices.loc[start_date:end_date]

    required_span = args.min_history + args.target_len + 1
    if len(val_prices) < required_span:
        raise ValueError(
            f"Validation window too short: {len(val_prices)} days "
            f"(need >= {required_span}). Increase --window-days or choose an earlier --train-cutoff."
        )

    cpd_config = CPDConfig(
        lookback=args.cpd_lookback,
        threshold=args.cpd_threshold,
        min_length=args.cpd_min_length,
        max_length=args.cpd_max_length,
    )

    dataset = XTrendDataset(
        val_prices,
        symbols=symbols,
        target_len=args.target_len,
        context_size=args.context_size,
        context_method="cpd_segmented",
        context_max_length=args.cpd_max_length,
        min_history=args.min_history,
        cpd_config=cpd_config,
        seed=0,
        cpd_cache_dir=args.cpd_cache_dir,
        allow_future_regimes=False,
        allow_cpd_recompute=args.allow_cpd_recompute,
    )

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    if len(dataset) == 0:
        raise ValueError(
            "No samples generated for the selected window. "
            "Increase --window-days or relax --min-history/--target-len."
        )

    # Model
    config = ModelConfig(
        input_dim=8,
        hidden_dim=args.hidden_dim,
        num_entities=len(symbols),
        num_attention_heads=args.num_heads,
        dropout=args.dropout,
    )
    models = create_model("xtrendq", config)
    for m in models.values():
        m.to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    models["encoder"].load_state_dict(ckpt["encoder"])
    models["cross_attn"].load_state_dict(ckpt["cross_attn"])
    models["model"].load_state_dict(ckpt["model"])
    for m in models.values():
        m.eval()

    quantile_levels = torch.tensor(
        [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99],
        device=device,
    )

    batch = load_batch_for_symbol(loader, symbols, args.symbol)

    tfeat = batch["target_features"].to(device)
    cfeat = batch["context_features"].to(device)
    cent = batch["context_entity_ids"].to(device)
    cmask = batch["context_padding_mask"].to(device)
    ent_id = batch["entity_id"].to(device)
    target_returns = batch["target_returns"].cpu().numpy()[0]

    with torch.no_grad():
        t_enc = models["encoder"](tfeat, entity_ids=ent_id)
        c_enc, c_mask = encode_context_set(models["encoder"], cfeat, cent, cmask)
        attn = models["cross_attn"](t_enc.hidden_states, c_enc, context_padding_mask=c_mask)
        out = models["model"](tfeat, attn.output, entity_ids=ent_id)

    positions = out["positions"].cpu().numpy()[0]
    quants = out["quantiles"].cpu().numpy()[0]  # (target_len, num_quantiles)

    Path("outputs/plots").mkdir(parents=True, exist_ok=True)

    # Plot positions
    plt.figure(figsize=(10, 4))
    plt.plot(positions, label="position")
    plt.axhline(0, color="gray", ls="--", lw=0.8)
    plt.title(f"{args.symbol} positions")
    plt.legend(); plt.tight_layout()
    plt.savefig("outputs/plots/pred_positions.png")

    # Plot a simple quantile fan (10/50/90)
    levels = [0.1, 0.5, 0.9]
    idxs = [quantile_levels.cpu().tolist().index(q) for q in levels]
    plt.figure(figsize=(10, 4))
    for lbl, i in zip(levels, idxs):
        plt.plot(quants[:, i], label=f"q{lbl}")
    plt.plot(target_returns, color="k", alpha=0.3, label="actual r_{t+1}")
    plt.axhline(0, color="gray", ls="--", lw=0.8)
    plt.title(f"{args.symbol} quantile fan (selected)")
    plt.legend(); plt.tight_layout()
    plt.savefig("outputs/plots/pred_quantiles.png")

    print("Saved plots:")
    print("  outputs/plots/pred_positions.png")
    print("  outputs/plots/pred_quantiles.png")


if __name__ == "__main__":
    main()
