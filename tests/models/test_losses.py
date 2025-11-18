"""Tests for loss functions."""
import pytest
import torch

from xtrend.models.losses import (
    sharpe_loss,
    gaussian_nll_loss,
    quantile_loss,
    joint_gaussian_loss,
    joint_quantile_loss
)


class TestSharpeLoss:
    """Test Sharpe ratio loss (Equation 8)."""

    def test_sharpe_loss_basic(self):
        """Sharpe loss computes annualized risk-adjusted returns."""
        batch_size, seq_len = 2, 126

        # Deterministic positive PnL so Sharpe is positive → loss negative
        returns = torch.ones(batch_size, seq_len) * 0.01
        positions = torch.ones(batch_size, seq_len) * 0.5

        loss = sharpe_loss(positions, returns, warmup_steps=63)

        # Loss should be negative because we minimize (-√252 * Sharpe)
        assert loss.ndim == 0
        assert loss.item() < 0

    def test_sharpe_warmup_period(self):
        """Warmup period ignores first l_s steps."""
        batch_size, seq_len = 2, 126
        warmup = 63

        returns = torch.randn(batch_size, seq_len)
        positions = torch.randn(batch_size, seq_len)

        # Modify first warmup steps to extreme values
        returns[:, :warmup] = 1000.0
        positions[:, :warmup] = 1.0

        loss = sharpe_loss(positions, returns, warmup_steps=warmup)

        # Loss should ignore warmup (not explode from extreme values)
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)

    def test_sharpe_annualization(self):
        """Sharpe includes √252 annualization factor."""
        batch_size, seq_len = 1, 200

        # Constant positive returns
        returns = torch.ones(batch_size, seq_len) * 0.01
        positions = torch.ones(batch_size, seq_len) * 0.5

        loss = sharpe_loss(positions, returns, warmup_steps=0)

        # With constant returns, Sharpe should be large (low loss)
        # Loss = -√252 * (mean / std)
        # std ≈ 0 for constant, but we add epsilon
        assert loss.item() < 0  # Negative because good performance

    def test_sharpe_gradient_flow(self):
        """Gradients flow through Sharpe loss."""
        batch_size, seq_len = 2, 100

        returns = torch.randn(batch_size, seq_len)
        positions = torch.randn(batch_size, seq_len, requires_grad=True)

        loss = sharpe_loss(positions, returns, warmup_steps=20)

        # Backward pass
        loss.backward()

        # Check gradient exists
        assert positions.grad is not None
        assert not torch.isnan(positions.grad).any()

    # ✅ ADDED: Edge case test (Issue #8)
    def test_sharpe_invalid_warmup(self):
        """Sharpe loss validates warmup_steps < seq_len."""
        batch_size, seq_len = 2, 100

        returns = torch.randn(batch_size, seq_len)
        positions = torch.randn(batch_size, seq_len)

        # Warmup >= seq_len should raise error
        with pytest.raises(ValueError, match="warmup_steps.*must be.*sequence length"):
            sharpe_loss(positions, returns, warmup_steps=100)


class TestGaussianNLLLoss:
    """Test Gaussian NLL loss (Equation 20)."""

    def test_gaussian_nll_basic(self):
        """Gaussian NLL loss computes likelihood."""
        batch_size, seq_len = 2, 126

        mean = torch.randn(batch_size, seq_len)
        std = torch.rand(batch_size, seq_len) + 0.5  # Positive
        target = torch.randn(batch_size, seq_len)

        loss = gaussian_nll_loss(mean, std, target, warmup_steps=63)

        assert loss.ndim == 0
        assert loss.item() > 0  # NLL is positive


class TestQuantileLoss:
    """Test quantile regression loss (Equation 22)."""

    def test_quantile_loss_basic(self):
        """Quantile loss uses pinball loss."""
        batch_size, seq_len, num_quantiles = 2, 126, 13

        quantile_preds = torch.randn(batch_size, seq_len, num_quantiles).sort(dim=-1)[0]
        target = torch.randn(batch_size, seq_len)
        levels = torch.tensor([0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99])

        loss = quantile_loss(quantile_preds, target, levels, warmup_steps=63)

        assert loss.ndim == 0
        assert loss.item() > 0


class TestJointLosses:
    """Test joint losses (Equations 21, 23)."""

    def test_joint_gaussian_loss(self):
        """Joint Gaussian loss combines MLE and Sharpe."""
        batch_size, seq_len = 2, 126

        mean = torch.randn(batch_size, seq_len)
        std = torch.rand(batch_size, seq_len) + 0.5
        positions = torch.tanh(torch.randn(batch_size, seq_len))
        target = torch.randn(batch_size, seq_len)

        loss = joint_gaussian_loss(mean, std, positions, target, alpha=1.0, warmup_steps=63)

        assert loss.ndim == 0

    def test_joint_quantile_loss(self):
        """Joint Quantile loss combines QRE and Sharpe."""
        batch_size, seq_len, num_quantiles = 2, 126, 13

        quantile_preds = torch.randn(batch_size, seq_len, num_quantiles).sort(dim=-1)[0]
        levels = torch.tensor([0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99])
        positions = torch.tanh(torch.randn(batch_size, seq_len))
        target = torch.randn(batch_size, seq_len)

        loss = joint_quantile_loss(quantile_preds, levels, positions, target, alpha=5.0, warmup_steps=63)

        assert loss.ndim == 0
