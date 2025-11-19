#!/usr/bin/env python3
"""
Phase 6 Visualization Script for X-Trend Project

Generates completion plots for Phase 6: Decoder & Loss Functions
1. Position predictions from all three model variants
2. Quantile predictions (XTrendQ) - fan chart
3. Gaussian distribution predictions (XTrendG)
4. Distribution snapshots at specific time points
5. Loss function comparison

Usage:
    uv run python scripts/phase6_visualizations.py
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

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


def setup_plotting_style():
    """Configure matplotlib for publication-quality plots."""
    plt.style.use('seaborn-v0_8-darkgrid')
    sns.set_palette("husl")
    plt.rcParams['figure.figsize'] = (14, 8)
    plt.rcParams['font.size'] = 11
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 10
    plt.rcParams['legend.fontsize'] = 10
    plt.rcParams['figure.titlesize'] = 16


def create_models(config):
    """Create all three model variants with shared components."""
    entity_embedding = EntityEmbedding(config)
    encoder = LSTMEncoder(config, use_entity=True, entity_embedding=entity_embedding)
    cross_attn = XTrendCrossAttention(config)

    model_xtrend = XTrend(config, entity_embedding=entity_embedding)
    model_xtrendg = XTrendG(config, entity_embedding=entity_embedding)
    model_xtrendq = XTrendQ(config, entity_embedding=entity_embedding, num_quantiles=13)

    # Set to eval mode
    encoder.eval()
    cross_attn.eval()
    model_xtrend.eval()
    model_xtrendg.eval()
    model_xtrendq.eval()

    return {
        'encoder': encoder,
        'cross_attn': cross_attn,
        'xtrend': model_xtrend,
        'xtrendg': model_xtrendg,
        'xtrendq': model_xtrendq,
        'entity_embedding': entity_embedding
    }


def generate_synthetic_data(target_len=126, input_dim=8, batch_size=1):
    """Generate realistic synthetic financial time series data."""
    # Simulate realistic patterns
    t = np.linspace(0, 4*np.pi, target_len)
    trend = 0.5 * np.sin(t / 5)  # Slow trend
    noise = np.random.randn(target_len) * 0.3

    # Create features
    target_features = torch.zeros(batch_size, target_len, input_dim)
    for i in range(input_dim):
        target_features[0, :, i] = torch.tensor(trend + noise + np.random.randn(target_len) * 0.1)

    # Simulate realistic returns
    returns = torch.randn(batch_size, target_len) * 0.01 + 0.0005

    entity_ids = torch.tensor([0])

    return target_features, returns, entity_ids


def run_models(models, target_features, entity_ids, context_size=20, config=None):
    """Run forward passes through all models."""
    with torch.no_grad():
        # Encode target
        target_encoded = models['encoder'](target_features, entity_ids=entity_ids)

        # Mock context
        batch_size = target_features.shape[0]
        context_encoded = torch.randn(batch_size, context_size, config.hidden_dim)

        # Cross-attention
        cross_attn_output = models['cross_attn'](
            target_encoded.hidden_states,
            context_encoded
        )

        # Get predictions from all models
        positions_xtrend = models['xtrend'](
            target_features,
            cross_attn_output.output,
            entity_ids=entity_ids
        )

        outputs_xtrendg = models['xtrendg'](
            target_features,
            cross_attn_output.output,
            entity_ids=entity_ids
        )

        outputs_xtrendq = models['xtrendq'](
            target_features,
            cross_attn_output.output,
            entity_ids=entity_ids
        )

    return {
        'xtrend': positions_xtrend,
        'xtrendg': outputs_xtrendg,
        'xtrendq': outputs_xtrendq
    }


def plot_position_predictions(outputs, target_len, output_dir):
    """Plot 1: Compare position predictions from all three models."""
    print("\n" + "="*60)
    print("PLOT 1: Position Predictions Over Time")
    print("="*60)

    fig, axes = plt.subplots(3, 1, figsize=(16, 10), sharex=True)
    time_steps = np.arange(target_len)

    # XTrend
    ax1 = axes[0]
    positions = outputs['xtrend'][0].numpy()
    ax1.plot(time_steps, positions, linewidth=2, color='steelblue', label='XTrend')
    ax1.fill_between(time_steps, -1, positions, alpha=0.2, color='green',
                     where=(positions > 0))
    ax1.fill_between(time_steps, -1, positions, alpha=0.2, color='red',
                     where=(positions <= 0))
    ax1.axhline(0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax1.set_ylabel('Position', fontweight='bold')
    ax1.set_title('XTrend: Direct Position Prediction', fontweight='bold')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(-1.1, 1.1)

    # XTrendG
    ax2 = axes[1]
    positions = outputs['xtrendg']['positions'][0].numpy()
    ax2.plot(time_steps, positions, linewidth=2, color='darkorange', label='XTrendG')
    ax2.fill_between(time_steps, -1, positions, alpha=0.2, color='green',
                     where=(positions > 0))
    ax2.fill_between(time_steps, -1, positions, alpha=0.2, color='red',
                     where=(positions <= 0))
    ax2.axhline(0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax2.set_ylabel('Position', fontweight='bold')
    ax2.set_title('XTrendG: Gaussian Distribution → PTP', fontweight='bold')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(-1.1, 1.1)

    # XTrendQ
    ax3 = axes[2]
    positions = outputs['xtrendq']['positions'][0].numpy()
    ax3.plot(time_steps, positions, linewidth=2, color='green', label='XTrendQ')
    ax3.fill_between(time_steps, -1, positions, alpha=0.2, color='green',
                     where=(positions > 0))
    ax3.fill_between(time_steps, -1, positions, alpha=0.2, color='red',
                     where=(positions <= 0))
    ax3.axhline(0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax3.set_ylabel('Position', fontweight='bold')
    ax3.set_xlabel('Time Step', fontweight='bold')
    ax3.set_title('XTrendQ: Quantile Distribution → PTP (Best Performance)', fontweight='bold')
    ax3.legend(loc='upper right')
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(-1.1, 1.1)

    plt.suptitle('Phase 6: Trading Position Predictions Over Time',
                 fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()

    output_path = output_dir / 'phase6_position_predictions.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {output_path}")

    plt.close()


def plot_quantile_predictions(outputs, returns, target_len, output_dir):
    """Plot 2: Quantile predictions (fan chart)."""
    print("\n" + "="*60)
    print("PLOT 2: Quantile Predictions (Fan Chart)")
    print("="*60)

    quantile_levels = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
    quantiles = outputs['xtrendq']['quantiles'][0].numpy()
    time_steps = np.arange(target_len)

    fig, ax = plt.subplots(figsize=(16, 8))

    # Plot median
    median_idx = 6
    ax.plot(time_steps, quantiles[:, median_idx], linewidth=2.5, color='black',
            label='Median (Q50)', zorder=10)

    # Plot confidence intervals
    colors = plt.cm.Blues(np.linspace(0.3, 0.9, 6))
    intervals = [(0, 12), (1, 11), (2, 10), (3, 9), (4, 8), (5, 7)]
    alphas = [0.15, 0.2, 0.25, 0.3, 0.35, 0.4]

    for idx, ((low_idx, high_idx), alpha) in enumerate(zip(intervals, alphas)):
        label = f'{int(quantile_levels[low_idx]*100)}-{int(quantile_levels[high_idx]*100)}%' if idx == 0 else None
        ax.fill_between(
            time_steps,
            quantiles[:, low_idx],
            quantiles[:, high_idx],
            alpha=alpha,
            color=colors[idx],
            label=label
        )

    # Actual returns
    ax.scatter(time_steps[::5], returns[0, ::5].numpy(), color='red', s=20,
              alpha=0.6, label='Actual Returns', zorder=5)

    ax.axhline(0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    ax.set_xlabel('Time Step', fontweight='bold')
    ax.set_ylabel('Predicted Return', fontweight='bold')
    ax.set_title('XTrendQ: Quantile Predictions (13 Quantiles)',
                fontweight='bold', fontsize=14)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = output_dir / 'phase6_quantile_predictions.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {output_path}")

    plt.close()


def plot_gaussian_predictions(outputs, returns, target_len, output_dir):
    """Plot 3: Gaussian predictions (mean ± std bands)."""
    print("\n" + "="*60)
    print("PLOT 3: Gaussian Distribution Predictions")
    print("="*60)

    mean = outputs['xtrendg']['mean'][0].numpy()
    std = outputs['xtrendg']['std'][0].numpy()
    time_steps = np.arange(target_len)

    fig, axes = plt.subplots(2, 1, figsize=(16, 10), sharex=True)

    # Plot 1: Mean ± confidence bands
    ax1 = axes[0]
    ax1.plot(time_steps, mean, linewidth=2, color='darkorange', label='Predicted Mean (μ)')
    ax1.fill_between(time_steps, mean - std, mean + std, alpha=0.3, color='orange',
                     label='±1σ (68% CI)')
    ax1.fill_between(time_steps, mean - 2*std, mean + 2*std, alpha=0.15, color='orange',
                     label='±2σ (95% CI)')
    ax1.scatter(time_steps[::5], returns[0, ::5].numpy(), color='red', s=20,
               alpha=0.6, label='Actual Returns', zorder=5)
    ax1.axhline(0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    ax1.set_ylabel('Predicted Return', fontweight='bold')
    ax1.set_title('XTrendG: Gaussian Mean & Uncertainty Bands', fontweight='bold')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)

    # Plot 2: Predicted standard deviation
    ax2 = axes[1]
    ax2.plot(time_steps, std * 100, linewidth=2, color='purple',
            label='Predicted Std Dev (σ)')
    ax2.fill_between(time_steps, 0, std * 100, alpha=0.3, color='purple')
    ax2.set_ylabel('Std Dev (%)', fontweight='bold')
    ax2.set_xlabel('Time Step', fontweight='bold')
    ax2.set_title('XTrendG: Predicted Uncertainty Over Time', fontweight='bold')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)

    plt.suptitle('Phase 6: Gaussian Distribution Predictions',
                fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    output_path = output_dir / 'phase6_gaussian_predictions.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {output_path}")

    plt.close()


def plot_distribution_snapshots(outputs, returns, output_dir):
    """Plot 4: Distribution snapshots at specific time points."""
    print("\n" + "="*60)
    print("PLOT 4: Distribution Snapshots")
    print("="*60)

    mean = outputs['xtrendg']['mean'][0].numpy()
    std = outputs['xtrendg']['std'][0].numpy()
    quantiles = outputs['xtrendq']['quantiles'][0].numpy()

    snapshot_times = [30, 60, 90, 120]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for idx, t in enumerate(snapshot_times):
        ax = axes[idx]

        # Gaussian distribution
        mu = mean[t]
        sigma = std[t]
        x_range = np.linspace(mu - 4*sigma, mu + 4*sigma, 200)
        gaussian_pdf = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x_range - mu) / sigma)**2)

        ax.plot(x_range, gaussian_pdf, linewidth=2, color='darkorange',
               label=f'Gaussian (μ={mu:.3f}, σ={sigma:.3f})')
        ax.fill_between(x_range, 0, gaussian_pdf, alpha=0.3, color='orange')

        # Mark quantiles
        for q_idx, q_level in enumerate([0.05, 0.5, 0.95]):
            if q_level == 0.05:
                q_val = quantiles[t, 1]
            elif q_level == 0.5:
                q_val = quantiles[t, 6]
            else:
                q_val = quantiles[t, 11]
            ax.axvline(q_val, color='blue', linestyle='--', linewidth=1.5, alpha=0.7)

        # Actual return
        actual = returns[0, t].item()
        ax.axvline(actual, color='red', linestyle='-', linewidth=2, alpha=0.8,
                  label=f'Actual: {actual:.3f}')

        ax.set_xlabel('Return', fontweight='bold')
        ax.set_ylabel('Density', fontweight='bold')
        ax.set_title(f'Time Step t={t}', fontweight='bold')
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.suptitle('Phase 6: Predicted Distributions at Different Time Points',
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    output_path = output_dir / 'phase6_distribution_snapshots.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {output_path}")

    plt.close()


def plot_loss_comparison(outputs, returns, output_dir):
    """Plot 5: Loss function comparison."""
    print("\n" + "="*60)
    print("PLOT 5: Loss Function Comparison")
    print("="*60)

    warmup_steps = 63

    # Compute losses
    loss_xtrend = sharpe_loss(outputs['xtrend'], returns, warmup_steps=warmup_steps)

    loss_xtrendg = joint_gaussian_loss(
        outputs['xtrendg']['mean'],
        outputs['xtrendg']['std'],
        outputs['xtrendg']['positions'],
        returns,
        alpha=1.0,
        warmup_steps=warmup_steps
    )

    quantile_levels_tensor = torch.tensor(
        [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
    )
    loss_xtrendq = joint_quantile_loss(
        outputs['xtrendq']['quantiles'],
        quantile_levels_tensor,
        outputs['xtrendq']['positions'],
        returns,
        alpha=5.0,
        warmup_steps=warmup_steps
    )

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))

    models = ['XTrend\n(Sharpe)', 'XTrendG\n(Gaussian+Sharpe)', 'XTrendQ\n(Quantile+Sharpe)']
    losses = [loss_xtrend.item(), loss_xtrendg.item(), loss_xtrendq.item()]
    colors = ['steelblue', 'darkorange', 'green']

    bars = ax.bar(models, losses, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)

    # Add value labels
    for bar, loss in zip(bars, losses):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{loss:.3f}',
                ha='center', va='bottom', fontweight='bold', fontsize=12)

    ax.set_ylabel('Loss Value', fontweight='bold', fontsize=12)
    ax.set_title('Phase 6: Loss Function Comparison (Untrained Models)',
                fontweight='bold', fontsize=14)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_axisbelow(True)

    plt.tight_layout()
    output_path = output_dir / 'phase6_loss_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {output_path}")

    plt.close()

    # Print summary
    print("\n" + "="*60)
    print("LOSS FUNCTION EVALUATION")
    print("="*60)
    print(f"XTrend (Sharpe Loss):          {loss_xtrend.item():.4f}")
    print(f"XTrendG (Joint Gaussian Loss): {loss_xtrendg.item():.4f}")
    print(f"XTrendQ (Joint Quantile Loss): {loss_xtrendq.item():.4f}")
    print("\nNote: Lower loss = better performance")
    print("(Untrained models - losses are arbitrary)")


def main():
    """Main execution function."""
    print("="*60)
    print("Phase 6 Visualization Script")
    print("X-Trend: Decoder & Loss Functions")
    print("="*60)

    # Set random seed
    torch.manual_seed(42)
    np.random.seed(42)

    # Setup plotting
    setup_plotting_style()

    # Create output directory
    output_dir = Path('outputs/plots')
    output_dir.mkdir(parents=True, exist_ok=True)

    # Model configuration
    print("\nCreating models...")
    config = ModelConfig(
        input_dim=8,
        hidden_dim=64,
        num_entities=50,
        num_attention_heads=4,
        dropout=0.1
    )

    models = create_models(config)
    print("✅ Models created successfully")

    # Generate data
    print("\nGenerating synthetic data...")
    target_len = 126
    target_features, returns, entity_ids = generate_synthetic_data(
        target_len=target_len,
        input_dim=config.input_dim
    )
    print(f"✅ Data generated: {target_len} time steps")

    # Run models
    print("\nRunning model forward passes...")
    outputs = run_models(models, target_features, entity_ids, config=config)
    print("✅ All models executed successfully")

    # Generate plots
    print("\n" + "="*60)
    print("Generating Phase 6 Completion Plots")
    print("="*60)

    plot_position_predictions(outputs, target_len, output_dir)
    plot_quantile_predictions(outputs, returns, target_len, output_dir)
    plot_gaussian_predictions(outputs, returns, target_len, output_dir)
    plot_distribution_snapshots(outputs, returns, output_dir)
    plot_loss_comparison(outputs, returns, output_dir)

    print("\n" + "="*60)
    print("✅ All Phase 6 visualizations complete!")
    print("="*60)
    print("\nPlots saved to: outputs/plots/")
    print("  1. phase6_position_predictions.png")
    print("  2. phase6_quantile_predictions.png")
    print("  3. phase6_gaussian_predictions.png")
    print("  4. phase6_distribution_snapshots.png")
    print("  5. phase6_loss_comparison.png")
    print("\n✨ Phase 6 completion criteria met! ✨")
    print("\nNext steps: Train models on real data to see actual performance!")


if __name__ == "__main__":
    main()
