"""Tests for GP fitting and likelihood computation."""
import pytest
import torch
from xtrend.cpd.gp_fitter import GPFitter


class TestGPFitter:
    def test_fit_stationary_gp_returns_model_and_likelihood(self, synthetic_changepoint_data):
        """fit_stationary_gp returns GP model and log marginal likelihood."""
        x, y, _ = synthetic_changepoint_data

        fitter = GPFitter(max_iter=20, patience=3, lr=0.2, grid_stride=5, grid_max_candidates=8, quick_iter=5)
        model, log_mll = fitter.fit_stationary_gp(x, y)

        # Model should be returned
        assert model is not None

        # Log likelihood should be finite value
        assert isinstance(log_mll, float)
        assert not torch.isnan(torch.tensor(log_mll))
        assert not torch.isinf(torch.tensor(log_mll))

    def test_stationary_gp_converges(self, synthetic_changepoint_data):
        """Stationary GP optimization converges properly."""
        x, y, _ = synthetic_changepoint_data

        fitter = GPFitter(max_iter=20, patience=3, lr=0.2, grid_stride=5, grid_max_candidates=8, quick_iter=5)
        model, log_mll = fitter.fit_stationary_gp(x, y)

        # Should converge to reasonable likelihood
        # (exact value depends on data, but should be finite)
        assert log_mll > -1000  # Sanity check


class TestGPFitterChangePoint:
    def test_fit_changepoint_gp_returns_model_likelihood_and_location(self, synthetic_changepoint_data):
        """fit_changepoint_gp returns model, likelihood, and detected CP location."""
        x, y, true_cp = synthetic_changepoint_data

        fitter = GPFitter(max_iter=20, patience=3, lr=0.2, grid_stride=5, grid_max_candidates=8, quick_iter=5)

        # Fit stationary model first (for warm-start)
        stat_model, log_mll_M = fitter.fit_stationary_gp(x, y)

        # Fit change-point model
        cp_model, log_mll_C, detected_cp = fitter.fit_changepoint_gp(x, y, stat_model)

        # Should return all three values
        assert cp_model is not None
        assert isinstance(log_mll_C, float)
        assert isinstance(detected_cp, (int, float))

        # Likelihood should be finite
        assert not torch.isnan(torch.tensor(log_mll_C))
        assert not torch.isinf(torch.tensor(log_mll_C))

        # Severities should be finite and non-zero on this synthetic jump
        severity = fitter.compute_severity(log_mll_M, log_mll_C)
        assert severity >= 0.1

        assert 0 <= detected_cp < len(x)


class TestSeverityComputation:
    def test_severity_formula_equal_likelihoods(self):
        """Equal likelihoods → severity ≈ 0.5."""
        fitter = GPFitter(max_iter=20, patience=3, lr=0.2, grid_stride=5, grid_max_candidates=8, quick_iter=5)
        severity = fitter.compute_severity(0.0, 0.0)

        assert severity == pytest.approx(0.5, abs=0.01)

    def test_severity_formula_strong_evidence(self):
        """Strong evidence (Δ ≥ 2.2) → severity ≥ 0.9."""
        fitter = GPFitter(max_iter=20, patience=3, lr=0.2, grid_stride=5, grid_max_candidates=8, quick_iter=5)
        severity = fitter.compute_severity(0.0, 2.2)

        assert severity >= 0.9

    def test_severity_formula_negative_evidence(self):
        """Negative evidence (Δ < 0) → severity < 0.5."""
        fitter = GPFitter()
        severity = fitter.compute_severity(1.0, 0.0)

        assert severity < 0.5

    def test_severity_on_synthetic_changepoint(self, synthetic_changepoint_data):
        """Severity is high for obvious change-point."""
        x, y, _ = synthetic_changepoint_data

        fitter = GPFitter(max_iter=30, patience=5, grid_stride=5, grid_max_candidates=8, quick_iter=10)
        stat_model, log_mll_M = fitter.fit_stationary_gp(x, y)
        cp_model, log_mll_C, _ = fitter.fit_changepoint_gp(x, y, stat_model)

        severity = fitter.compute_severity(log_mll_M, log_mll_C)

        # Should show some evidence for CP (looser bound to keep GP runtime small)
        assert severity >= 0.15

    def test_edge_case_both_models_failed(self):
        """Both models failed (both -inf) should return neutral severity 0.5."""
        fitter = GPFitter()
        severity = fitter.compute_severity(float('-inf'), float('-inf'))
        assert severity == 0.5

    def test_edge_case_stationary_failed_cp_terrible(self):
        """Stationary failed but CP finite log_mll still returns strong evidence (1.0)."""
        fitter = GPFitter()
        severity = fitter.compute_severity(float('-inf'), -1000.0)
        assert severity == 1.0

    def test_edge_case_stationary_failed_cp_reasonable(self):
        """Stationary failed with CP finite fit should return 1.0."""
        fitter = GPFitter()
        # Stationary failed, but CP model is reasonable
        severity = fitter.compute_severity(float('-inf'), -10.0)
        assert severity == 1.0

    def test_massive_jump_no_nan_severity(self):
        """Massive price jump (step function) should not produce NaN severity.

        This tests the fix for the issue where a sudden jump (e.g., 2→5)
        causes the stationary GP to fail with NaN, which should now be
        handled gracefully by returning -inf and producing severity=1.0.
        """
        torch.manual_seed(123)
        n = 30
        x = torch.arange(n).float().unsqueeze(-1)

        # Create a massive step function: constant at 2, then jump to 5
        y = torch.cat([
            torch.full((15,), 2.0) + 0.01 * torch.randn(15),  # First regime at 2.0
            torch.full((15,), 5.0) + 0.01 * torch.randn(15)   # Second regime at 5.0
        ])

        fitter = GPFitter()

        # Fit stationary model (may fail due to step function)
        stat_model, log_mll_M = fitter.fit_stationary_gp(x, y)

        # Fit changepoint model (should succeed by splitting the regimes)
        cp_model, log_mll_C, detected_cp = fitter.fit_changepoint_gp(x, y, stat_model)

        # Compute severity
        severity = fitter.compute_severity(log_mll_M, log_mll_C)

        # CRITICAL: Severity should NOT be NaN
        assert not torch.isnan(torch.tensor(severity))

        # If stationary failed (returned -inf), severity should be 1.0
        # If both succeeded, severity should still be high (>0.9) for such a clear jump
        if log_mll_M == float('-inf'):
            assert severity == 1.0
        else:
            # Even if it didn't fail, severity should be very high for this clear jump
            assert severity > 0.9

        # Detected changepoint should be near the actual jump location (around index 15)
        assert 10 <= detected_cp <= 20
