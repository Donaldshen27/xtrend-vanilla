#!/usr/bin/env python3
"""
Training script for X-Trend models on Bloomberg futures data.

Usage:
    # Train XTrendQ (best performance from paper)
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
from xtrend.data.returns_vol import normalized_returns
from xtrend.cpd import CPDConfig, GPCPDSegmenter


class XTrendDataset(Dataset):
    """
    Dataset for X-Trend model training.

    Creates sequences of:
    - Target features (8 features × target_len time steps)
    - Context sequences sampled via Phase 4 samplers (cpd_segmented/time-aligned/final-state)
    - Target returns (for loss computation)
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

        # Precompute features and returns for all symbols
        print("Precomputing Phase 1 features and returns...")
        self.features = {}
        self.feature_tensors = {}
        self.returns = {}
        self.return_tensors = {}
        self.input_dim = None

        for symbol in tqdm(symbols, desc="Processing symbols"):
            price_series = self._prepare_price_series(prices[symbol])
            if price_series.isna().all():
                continue

            # Compute features (8 features as per paper)
            self.features[symbol] = self._compute_features(price_series)
            feature_tensor = torch.tensor(
                self.features[symbol].values,
                dtype=torch.float32
            )
            self.feature_tensors[symbol] = feature_tensor

            # Compute normalized returns
            rets = normalized_returns(
                price_series.to_frame(),
                scale=1,
                vol_window=252
            )
            self.returns[symbol] = rets[symbol].fillna(0.0)
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
            segmenter = GPCPDSegmenter(self.cpd_config)
            for symbol in tqdm(symbols, desc="Segmenting regimes"):
                price_series = self._prepare_price_series(prices[symbol])
                if price_series.isna().all():
                    continue
                try:
                    self.regimes[symbol] = segmenter.fit_segment(price_series)
                except Exception as exc:
                    warnings.warn(
                        f"CPD segmentation failed for {symbol}: {exc}. "
                        "Context sampler will fall back to alternate methods."
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
        """Align prices to global index and fill missing values causally."""
        aligned = series.reindex(self.prices.index)
        aligned = aligned.ffill().bfill()
        return aligned.fillna(0.0)

    def _compute_features(self, prices: pd.Series) -> pd.DataFrame:
        """Compute 8 features for a price series (matching paper)."""
        df = pd.DataFrame(index=prices.index)

        # Simple returns at multiple scales
        for scale in [1, 5, 21]:
            ret = prices.pct_change(scale)
            df[f'ret_{scale}d'] = ret

        # MACD-style momentum indicators
        df['macd_8_24'] = (prices.ewm(span=8).mean() - prices.ewm(span=24).mean()) / prices
        df['macd_16_48'] = (prices.ewm(span=16).mean() - prices.ewm(span=48).mean()) / prices

        # Volatility proxy
        df['vol_20d'] = prices.pct_change().rolling(20).std()

        # Price level (normalized)
        df['price_norm'] = (prices - prices.rolling(252).mean()) / prices.rolling(252).std()

        # Volume placeholder (if you have volume data, add it here)
        df['volume_proxy'] = 0.0

        return df.fillna(0.0)

    def _create_samples(self) -> list:
        """Create list of valid (symbol, start_idx) pairs."""
        samples = []

        for symbol in self.symbols:
            if symbol not in self.feature_tensors:
                continue
            feat = self.feature_tensors[symbol]
            n = len(feat)

            # Need target_len + min_history for features
            required_len = self.target_len + self.min_history

            if n < required_len:
                continue

            # Create rolling windows
            for start_idx in range(self.min_history, n - self.target_len + 1):
                samples.append((symbol, start_idx))

        return samples

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

        # Get target features and returns
        target_feat = self.feature_tensors[target_symbol][
            start_idx:start_idx + self.target_len
        ]
        target_ret = self.return_tensors[target_symbol][
            start_idx:start_idx + self.target_len
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
                exclude_symbols=exclude
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
                exclude_symbols=exclude
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
                exclude_symbols=exclude
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
        torch.nn.utils.clip_grad_norm_(
            list(encoder.parameters()) +
            list(cross_attn.parameters()) +
            list(model.parameters()),
            max_norm=1.0
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

    return total_loss / num_batches


def main():
    parser = argparse.ArgumentParser(description="Train X-Trend model")
    parser.add_argument('--model', type=str, default='xtrendq',
                       choices=['xtrend', 'xtrendg', 'xtrendq'],
                       help='Model variant to train')
    parser.add_argument('--data-path', type=str, default='data/bloomberg/processed',
                       help='Path to Bloomberg parquet files')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--hidden-dim', type=int, default=128,
                       help='Hidden dimension')
    parser.add_argument('--num-heads', type=int, default=4,
                       help='Number of attention heads')
    parser.add_argument('--dropout', type=float, default=0.1,
                       help='Dropout rate')
    parser.add_argument('--target-len', type=int, default=126,
                       help='Target sequence length (days)')
    parser.add_argument('--context-size', type=int, default=20,
                       help='Number of context sequences per target')
    parser.add_argument('--context-method', type=str, default='cpd_segmented',
                       choices=['cpd_segmented', 'time_equivalent', 'final_hidden_state'],
                       help='Primary Phase 4 context sampler')
    parser.add_argument('--context-max-length', type=int, default=None,
                       help='Max context length (days). Defaults to --cpd-max-length.')
    parser.add_argument('--min-history', type=int, default=252,
                       help='Minimum history (days) before sampling targets')
    parser.add_argument('--cpd-lookback', type=int, default=21,
                       help='GP-CPD lookback window (days)')
    parser.add_argument('--cpd-threshold', type=float, default=0.9,
                       help='GP-CPD severity threshold for change-points')
    parser.add_argument('--cpd-min-length', type=int, default=5,
                       help='Minimum GP-CPD regime length (days)')
    parser.add_argument('--cpd-max-length', type=int, default=21,
                       help='Maximum GP-CPD regime length (days)')
    parser.add_argument('--context-seed', type=int, default=None,
                       help='Optional RNG seed for context sampling')
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

    # Load prices
    prices = source.load_prices(symbols, start='2000-01-01', end='2023-12-31')
    print(f"Loaded prices: {prices.shape}")

    # Train/val split (time-based)
    train_cutoff = '2020-12-31'
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

    dataset_kwargs = dict(
        symbols=symbols,
        target_len=args.target_len,
        context_size=args.context_size,
        context_method=args.context_method,
        context_max_length=context_max_length,
        vol_target=0.15,
        min_history=args.min_history,
        cpd_config=cpd_config,
        seed=args.context_seed
    )

    print("\nCreating datasets with episodic context sampling...")
    train_dataset = XTrendDataset(train_prices, **dataset_kwargs)
    val_dataset = XTrendDataset(val_prices, **dataset_kwargs)

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
    optimizer = torch.optim.AdamW(all_params, lr=args.lr, weight_decay=1e-5)
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
