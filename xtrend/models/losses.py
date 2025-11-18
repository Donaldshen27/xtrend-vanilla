"""Loss functions for X-Trend models."""
import torch
import torch.nn.functional as F


def sharpe_loss(
    positions: torch.Tensor,
    returns: torch.Tensor,
    warmup_steps: int = 63,
    eps: float = 1e-8
) -> torch.Tensor:
    """Sharpe ratio loss (Equation 8, Page 5).

    L_Sharpe = -√252 * mean(r * z) / std(r * z)

    Where:
    - r: Scaled returns (σ_tgt / σ_t * r_{t+1})
    - z: Predicted positions
    - √252: Annualization factor (trading days per year)
    - Batch Ω: All (i, t) pairs excluding warmup period

    Args:
        positions: Predicted positions (batch, seq_len) in (-1, 1)
        returns: Scaled returns (batch, seq_len)
        warmup_steps: Ignore first l_s steps (default: 63)
        eps: Small constant for numerical stability

    Returns:
        loss: Scalar, negative Sharpe ratio
    """
    # ✅ ADDED: Validate inputs (Issue #8)
    if warmup_steps >= positions.shape[1]:
        raise ValueError(
            f"warmup_steps ({warmup_steps}) must be < sequence length ({positions.shape[1]})"
        )

    # Remove warmup period
    if warmup_steps > 0:
        positions = positions[:, warmup_steps:]
        returns = returns[:, warmup_steps:]

    # ✅ ADDED: Ensure we have data left (Issue #8)
    if positions.numel() == 0:
        raise ValueError("No data remaining after warmup removal")

    # Compute strategy returns: r_t * z_t
    strategy_returns = returns * positions

    # Flatten to single batch Ω
    strategy_returns_flat = strategy_returns.flatten()

    # ✅ FIXED: Use population std (unbiased=False) per Issue #8
    mean_return = strategy_returns_flat.mean()
    std_return = strategy_returns_flat.std(unbiased=False) + eps

    sharpe_ratio = mean_return / std_return

    # Annualize and negate (minimize = maximize Sharpe)
    annualization_factor = torch.sqrt(torch.tensor(252.0, device=positions.device))
    loss = -annualization_factor * sharpe_ratio

    return loss


def gaussian_nll_loss(
    mean: torch.Tensor,
    std: torch.Tensor,
    target: torch.Tensor,
    warmup_steps: int = 63
) -> torch.Tensor:
    """Gaussian negative log-likelihood loss (Equation 20, Page 9).

    L_MLE = -1/|Ω| * Σ log p(r | μ, σ)

    Where p(r | μ, σ) = N(r; μ, σ) is Gaussian likelihood.

    Args:
        mean: Predicted mean (batch, seq_len)
        std: Predicted std dev (batch, seq_len), positive
        target: Scaled returns (batch, seq_len)
        warmup_steps: Ignore first l_s steps (default: 63)

    Returns:
        loss: Scalar, mean negative log-likelihood
    """
    # ✅ ADDED: Validate inputs (Issue #8)
    if warmup_steps >= mean.shape[1]:
        raise ValueError(
            f"warmup_steps ({warmup_steps}) must be < sequence length ({mean.shape[1]})"
        )

    # Remove warmup period
    if warmup_steps > 0:
        mean = mean[:, warmup_steps:]
        std = std[:, warmup_steps:]
        target = target[:, warmup_steps:]

    # ✅ FIXED: Clamp variance explicitly (Issue #5)
    variance = std.pow(2).clamp(min=1e-6)

    # Gaussian NLL: -log p(target | mean, std)
    loss = F.gaussian_nll_loss(mean, target, variance, reduction='mean')

    return loss


def quantile_loss(
    quantile_preds: torch.Tensor,
    target: torch.Tensor,
    quantile_levels: torch.Tensor,
    warmup_steps: int = 63
) -> torch.Tensor:
    """Quantile regression loss (Equation 22, Page 9).

    L_QRE = 1/(|Ω| * |H|) * Σ [η * (r - Q_η)_+ + (1-η) * (Q_η - r)_+]

    Where:
    - Q_η: Predicted quantile at level η
    - (·)_+ = max(0, ·): Pinball loss
    - H: Set of quantile levels [0.01, 0.05, 0.1, ..., 0.99]

    Args:
        quantile_preds: Predicted quantiles (batch, seq_len, num_quantiles)
        target: Scaled returns (batch, seq_len)
        quantile_levels: Quantile levels (num_quantiles,)
            e.g., [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
        warmup_steps: Ignore first l_s steps (default: 63)

    Returns:
        loss: Scalar, mean quantile loss
    """
    # ✅ ADDED: Validate inputs (Issue #8)
    if warmup_steps >= quantile_preds.shape[1]:
        raise ValueError(
            f"warmup_steps ({warmup_steps}) must be < sequence length ({quantile_preds.shape[1]})"
        )

    # Remove warmup period
    if warmup_steps > 0:
        quantile_preds = quantile_preds[:, warmup_steps:, :]
        target = target[:, warmup_steps:]

    # Expand target to match quantile predictions
    # target: (batch, seq_len) -> (batch, seq_len, num_quantiles)
    target_expanded = target.unsqueeze(-1).expand_as(quantile_preds)

    # Compute errors: r - Q_η
    errors = target_expanded - quantile_preds

    # Pinball loss: η * (r - Q_η)_+ + (1-η) * (Q_η - r)_+
    # Equivalent to: max(η * error, (η - 1) * error)
    quantile_levels = quantile_levels.to(quantile_preds.device)
    loss_per_quantile = torch.max(
        quantile_levels * errors,
        (quantile_levels - 1) * errors
    )

    # Mean over all dimensions
    loss = loss_per_quantile.mean()

    return loss


def joint_gaussian_loss(
    mean: torch.Tensor,
    std: torch.Tensor,
    positions: torch.Tensor,
    target: torch.Tensor,
    alpha: float = 1.0,
    warmup_steps: int = 63
) -> torch.Tensor:
    """Joint loss for X-Trend-G (Equation 21, Page 9).

    L_Joint^MLE = α * L_MLE + L_Sharpe^PTP_G

    Args:
        mean: Predicted mean from Gaussian head (batch, seq_len)
        std: Predicted std from Gaussian head (batch, seq_len)
        positions: Positions from PTP_G(mean, std) (batch, seq_len)
        target: Scaled returns (batch, seq_len)
        alpha: Weighting for MLE loss (default: 1.0, from Table 4)
        warmup_steps: Warmup period (default: 63)

    Returns:
        loss: Scalar, joint loss
    """
    # Gaussian NLL loss
    mle_loss = gaussian_nll_loss(mean, std, target, warmup_steps)

    # Sharpe loss on PTP positions
    sharpe = sharpe_loss(positions, target, warmup_steps)

    # Weighted combination
    loss = alpha * mle_loss + sharpe

    return loss


def joint_quantile_loss(
    quantile_preds: torch.Tensor,
    quantile_levels: torch.Tensor,
    positions: torch.Tensor,
    target: torch.Tensor,
    alpha: float = 5.0,
    warmup_steps: int = 63
) -> torch.Tensor:
    """Joint loss for X-Trend-Q (Equation 23, Page 9).

    L_Joint^QRE = α * L_QRE + L_Sharpe^PTP_Q

    Args:
        quantile_preds: Predicted quantiles (batch, seq_len, num_quantiles)
        quantile_levels: Quantile levels (num_quantiles,)
        positions: Positions from PTP_Q(quantiles) (batch, seq_len)
        target: Scaled returns (batch, seq_len)
        alpha: Weighting for QRE loss (default: 5.0, from Table 4)
        warmup_steps: Warmup period (default: 63)

    Returns:
        loss: Scalar, joint loss
    """
    # Quantile regression loss
    qre_loss = quantile_loss(quantile_preds, target, quantile_levels, warmup_steps)

    # Sharpe loss on PTP positions
    sharpe = sharpe_loss(positions, target, warmup_steps)

    # Weighted combination
    loss = alpha * qre_loss + sharpe

    return loss
