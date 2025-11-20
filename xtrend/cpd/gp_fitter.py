"""GP model fitting and likelihood computation for CPD."""
import math
from typing import Optional, Tuple

import gpytorch
import torch
from torch import Tensor


class ExactGPModel(gpytorch.models.ExactGP):
    """Exact GP model for CPD.

    Uses Matérn kernel with nu=1.5 (once differentiable).
    """

    def __init__(self, train_x: Tensor, train_y: Tensor,
                 likelihood: gpytorch.likelihoods.Likelihood,
                 kernel: gpytorch.kernels.Kernel):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = kernel

    def forward(self, x: Tensor) -> gpytorch.distributions.MultivariateNormal:
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class GPFitter:
    """Fits GPs and computes marginal likelihoods for CPD."""

    def __init__(self, max_iter: int = 200, convergence_tol: float = 1e-3,
                 patience: int = 5, lr: float = 0.1,
                 grid_stride: Optional[int] = None,
                 grid_max_candidates: Optional[int] = None,
                 quick_iter: Optional[int] = None):
        """Initialize GP fitter.

        Args:
            max_iter: Maximum optimization iterations
            convergence_tol: Convergence tolerance for loss
            patience: Patience for early stopping
            lr: Learning rate for Adam optimizer
            grid_stride: Optional step size for CP grid search (defaults to n//30)
            grid_max_candidates: Optional cap on number of CP candidates to evaluate
            quick_iter: Optional cap for inner optimization steps per candidate
        """
        self.max_iter = max_iter
        self.convergence_tol = convergence_tol
        self.patience = patience
        self.lr = lr
        self.grid_stride = grid_stride
        self.grid_max_candidates = grid_max_candidates
        self.quick_iter = quick_iter

    def fit_stationary_gp(self, x: Tensor, y: Tensor) -> Tuple[ExactGPModel, float]:
        """Fit single Matérn GP (no change-point).

        Args:
            x: Time indices [N, 1]
            y: Observations [N]

        Returns:
            tuple: (fitted_model, log_marginal_likelihood)
                   Returns -inf for log_mll if optimization fails
        """
        # Normalize y to z-scores for numerical stability
        # GP kernels expect standardized inputs to avoid numerical issues
        y_mean = y.mean()
        # Use population std (unbiased=False) to avoid NaNs for very short windows.
        y_std = y.std(unbiased=False)
        if not torch.isfinite(y_std) or y_std < 1e-8:
            y_std = torch.tensor(1.0)  # Avoid division by zero for constant/degenerate sequences
        y_norm = (y - y_mean) / y_std

        # Create stationary kernel (Matérn nu=1.5)
        kernel = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.MaternKernel(nu=1.5)
        )
        likelihood = gpytorch.likelihoods.GaussianLikelihood()

        model = ExactGPModel(x, y_norm, likelihood, kernel)

        # Optimize with error handling
        model.train()
        likelihood.train()

        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

        prev_loss = float('inf')
        patience_count = 0

        try:
            for i in range(self.max_iter):
                optimizer.zero_grad()
                output = model(x)
                loss = -mll(output, y_norm)

                # Check for NaN/inf during training
                if not torch.isfinite(loss):
                    raise RuntimeError("Non-finite loss encountered during optimization")

                loss.backward()
                optimizer.step()

                # Convergence check
                if abs(loss.item() - prev_loss) < self.convergence_tol:
                    patience_count += 1
                    if patience_count >= self.patience:
                        break
                else:
                    patience_count = 0
                prev_loss = loss.item()

            # Compute final log marginal likelihood
            model.eval()
            likelihood.eval()
            with torch.no_grad():
                output = model(x)
                log_mll_value = mll(output, y_norm).item()

                # Final check for non-finite output
                if not math.isfinite(log_mll_value):
                    raise RuntimeError("Non-finite MLL output")

        except Exception:
            # Fallback for failed stationary fit
            # If stationary fit fails, it's a terrible explanation for the data
            # Return -inf so severity calculation works correctly:
            # severity = sigmoid(finite - (-inf)) = 1.0
            log_mll_value = float('-inf')

        return model, log_mll_value

    def fit_changepoint_gp(self, x: Tensor, y: Tensor,
                          stationary_model: ExactGPModel) -> Tuple[ExactGPModel, float, float]:
        """Fit change-point GP using grid search over candidate locations.

        For each candidate CP location, fits two separate GPs on either side
        and computes combined log marginal likelihood.

        Args:
            x: Time indices [N, 1]
            y: Observations [N]
            stationary_model: Fitted stationary GP for warm-start (not used in grid search)

        Returns:
            tuple: (stationary_model, log_marginal_likelihood, detected_changepoint_location)
        """
        # Normalize y to z-scores for numerical stability
        # Normalize BEFORE splitting to ensure consistent scaling
        y_mean = y.mean()
        # Use population std (unbiased=False) to avoid NaNs for very short windows.
        y_std = y.std(unbiased=False)
        if not torch.isfinite(y_std) or y_std < 1e-8:
            y_std = torch.tensor(1.0)  # Avoid division by zero for constant/degenerate sequences
        y_norm = (y - y_mean) / y_std

        n = len(x)
        min_segment_length = max(5, n // 10)  # At least 5 points or 10% of data

        best_log_mll = float('-inf')
        best_t_cp = n // 2  # Default to middle

        # Grid search over candidate change-point locations
        # Use configurable stride and optional cap for speed in tests
        step = self.grid_stride if self.grid_stride is not None else max(1, n // 30)
        candidates = list(range(min_segment_length, n - min_segment_length + 1, step))
        if self.grid_max_candidates is not None:
            candidates = candidates[:self.grid_max_candidates]

        for t_cp in candidates:
            # Split normalized data at candidate change-point
            x1, y1_norm = x[:t_cp], y_norm[:t_cp]
            x2, y2_norm = x[t_cp:], y_norm[t_cp:]

            # Fit two separate GPs on normalized data
            try:
                # GP for first segment
                kernel1 = gpytorch.kernels.ScaleKernel(
                    gpytorch.kernels.MaternKernel(nu=1.5)
                )
                likelihood1 = gpytorch.likelihoods.GaussianLikelihood()
                model1 = ExactGPModel(x1, y1_norm, likelihood1, kernel1)

                model1.train()
                likelihood1.train()
                optimizer1 = torch.optim.Adam(model1.parameters(), lr=self.lr)
                mll1 = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood1, model1)

                # Quick optimization (fewer iterations for grid search)
                inner_iter = min(self.quick_iter or 50, self.max_iter)
                for _ in range(inner_iter):
                    optimizer1.zero_grad()
                    output1 = model1(x1)
                    loss1 = -mll1(output1, y1_norm)
                    if not torch.isfinite(loss1):
                        raise RuntimeError("Non-finite loss in segment 1")
                    loss1.backward()
                    optimizer1.step()

                model1.eval()
                likelihood1.eval()
                with torch.no_grad():
                    output1 = model1(x1)
                    log_mll1 = mll1(output1, y1_norm).item()

                # GP for second segment
                kernel2 = gpytorch.kernels.ScaleKernel(
                    gpytorch.kernels.MaternKernel(nu=1.5)
                )
                likelihood2 = gpytorch.likelihoods.GaussianLikelihood()
                model2 = ExactGPModel(x2, y2_norm, likelihood2, kernel2)

                model2.train()
                likelihood2.train()
                optimizer2 = torch.optim.Adam(model2.parameters(), lr=self.lr)
                mll2 = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood2, model2)

                inner_iter = min(self.quick_iter or 50, self.max_iter)
                for _ in range(inner_iter):
                    optimizer2.zero_grad()
                    output2 = model2(x2)
                    loss2 = -mll2(output2, y2_norm)
                    if not torch.isfinite(loss2):
                        raise RuntimeError("Non-finite loss in segment 2")
                    loss2.backward()
                    optimizer2.step()

                model2.eval()
                likelihood2.eval()
                with torch.no_grad():
                    output2 = model2(x2)
                    log_mll2 = mll2(output2, y2_norm).item()

                # Combined log marginal likelihood
                combined_log_mll = log_mll1 + log_mll2

                if combined_log_mll > best_log_mll:
                    best_log_mll = combined_log_mll
                    best_t_cp = t_cp

            except Exception:
                # Skip this candidate if optimization fails
                continue

        # Return the stationary model (we don't keep the CP model for simplicity)
        # The best_log_mll is the combined likelihood at the best CP location
        return stationary_model, best_log_mll, float(best_t_cp)

    def compute_severity(self, log_mll_stationary: float,
                        log_mll_changepoint: float) -> float:
        """Compute severity using log Bayes factor.

        Uses CORRECT formula: sigmoid(Δ) where Δ = L_C - L_M

        Args:
            log_mll_stationary: Log marginal likelihood of stationary model
            log_mll_changepoint: Log marginal likelihood of change-point model

        Returns:
            Severity in [0, 1] where:
            - ≈ 0.5: No evidence for change-point (or both models failed)
            - ≥ 0.9: Strong evidence (Δ ≥ 2.2)
            - 1.0: Stationary model failed with reasonable CP fit
            - 0.0: CP model failed but stationary succeeded
        """
        # Handle edge cases from failed optimizations
        stat_fail = math.isinf(log_mll_stationary) and log_mll_stationary < 0
        cp_fail = math.isinf(log_mll_changepoint) and log_mll_changepoint < 0

        if stat_fail and cp_fail:
            # Both models failed - neutral evidence
            return 0.5

        if cp_fail and not stat_fail:
            # Changepoint failed but stationary succeeded - evidence against CP
            return 0.0

        if stat_fail and not cp_fail:
            # Stationary failed, CP succeeded - strong evidence for CP
            return 1.0

        delta = log_mll_changepoint - log_mll_stationary
        severity = torch.sigmoid(torch.tensor(delta)).item()
        return severity
