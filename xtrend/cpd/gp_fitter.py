"""GP model fitting and likelihood computation for CPD."""
from typing import Tuple

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
                 patience: int = 5, lr: float = 0.1):
        """Initialize GP fitter.

        Args:
            max_iter: Maximum optimization iterations
            convergence_tol: Convergence tolerance for loss
            patience: Patience for early stopping
            lr: Learning rate for Adam optimizer
        """
        self.max_iter = max_iter
        self.convergence_tol = convergence_tol
        self.patience = patience
        self.lr = lr

    def fit_stationary_gp(self, x: Tensor, y: Tensor) -> Tuple[ExactGPModel, float]:
        """Fit single Matérn GP (no change-point).

        Args:
            x: Time indices [N, 1]
            y: Observations [N]

        Returns:
            tuple: (fitted_model, log_marginal_likelihood)
        """
        # Create stationary kernel (Matérn nu=1.5)
        kernel = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.MaternKernel(nu=1.5)
        )
        likelihood = gpytorch.likelihoods.GaussianLikelihood()

        model = ExactGPModel(x, y, likelihood, kernel)

        # Optimize
        model.train()
        likelihood.train()

        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

        prev_loss = float('inf')
        patience_count = 0

        for i in range(self.max_iter):
            optimizer.zero_grad()
            output = model(x)
            loss = -mll(output, y)
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
            log_mll_value = mll(output, y).item()

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
        n = len(x)
        min_segment_length = max(5, n // 10)  # At least 5 points or 10% of data

        best_log_mll = float('-inf')
        best_t_cp = n // 2  # Default to middle

        # Grid search over candidate change-point locations
        # Use finer grid for better accuracy
        # Note: upper bound is inclusive to handle edge case when n = 2*min_segment_length
        candidates = range(min_segment_length, n - min_segment_length + 1, max(1, n // 30))

        for t_cp in candidates:
            # Split data at candidate change-point
            x1, y1 = x[:t_cp], y[:t_cp]
            x2, y2 = x[t_cp:], y[t_cp:]

            # Fit two separate GPs
            try:
                # GP for first segment
                kernel1 = gpytorch.kernels.ScaleKernel(
                    gpytorch.kernels.MaternKernel(nu=1.5)
                )
                likelihood1 = gpytorch.likelihoods.GaussianLikelihood()
                model1 = ExactGPModel(x1, y1, likelihood1, kernel1)

                model1.train()
                likelihood1.train()
                optimizer1 = torch.optim.Adam(model1.parameters(), lr=self.lr)
                mll1 = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood1, model1)

                # Quick optimization (fewer iterations for grid search)
                for _ in range(min(50, self.max_iter)):
                    optimizer1.zero_grad()
                    output1 = model1(x1)
                    loss1 = -mll1(output1, y1)
                    loss1.backward()
                    optimizer1.step()

                model1.eval()
                likelihood1.eval()
                with torch.no_grad():
                    output1 = model1(x1)
                    log_mll1 = mll1(output1, y1).item()

                # GP for second segment
                kernel2 = gpytorch.kernels.ScaleKernel(
                    gpytorch.kernels.MaternKernel(nu=1.5)
                )
                likelihood2 = gpytorch.likelihoods.GaussianLikelihood()
                model2 = ExactGPModel(x2, y2, likelihood2, kernel2)

                model2.train()
                likelihood2.train()
                optimizer2 = torch.optim.Adam(model2.parameters(), lr=self.lr)
                mll2 = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood2, model2)

                for _ in range(min(50, self.max_iter)):
                    optimizer2.zero_grad()
                    output2 = model2(x2)
                    loss2 = -mll2(output2, y2)
                    loss2.backward()
                    optimizer2.step()

                model2.eval()
                likelihood2.eval()
                with torch.no_grad():
                    output2 = model2(x2)
                    log_mll2 = mll2(output2, y2).item()

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
            - ≈ 0.5: No evidence for change-point
            - ≥ 0.9: Strong evidence (Δ ≥ 2.2)
        """
        delta = log_mll_changepoint - log_mll_stationary
        severity = torch.sigmoid(torch.tensor(delta)).item()
        return severity
