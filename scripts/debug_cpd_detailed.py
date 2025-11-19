"""Debug GP-CPD grid search to see why it's failing."""
import gpytorch
import pandas as pd
import torch

from xtrend.cpd.gp_fitter import ExactGPModel
from xtrend.data.sources import BloombergParquetSource

# Load ES data for 2019-2020
source = BloombergParquetSource(root_path="data/bloomberg/processed")
prices_df = source.load_prices(["ES"], start="2019-01-01", end="2020-12-31")
price_series = prices_df["ES"].ffill().bfill()

# Test window
lookback = 63
window_start = 200
window_end = window_start + lookback - 1
window = price_series.iloc[window_start:window_end + 1]

# Convert to tensors
x = torch.arange(len(window)).float().unsqueeze(-1)
y = torch.tensor(window.values).float()

print(f"Testing window: {len(window)} days")
print(f"x range: [{x.min():.0f}, {x.max():.0f}]")
print(f"y range: [{y.min():.2f}, {y.max():.2f}]")
print()

# Try grid search manually
n = len(x)
min_segment_length = max(5, n // 10)
candidates = list(range(min_segment_length, n - min_segment_length + 1, max(1, n // 30)))

print(f"Grid search:")
print(f"  min_segment_length: {min_segment_length}")
print(f"  candidates: {candidates}")
print(f"  total candidates: {len(candidates)}")
print()

successful = 0
failed = 0
best_log_mll = float('-inf')
best_t_cp = n // 2

for i, t_cp in enumerate(candidates):
    try:
        # Split data
        x1, y1 = x[:t_cp], y[:t_cp]
        x2, y2 = x[t_cp:], y[t_cp:]

        print(f"Candidate {i+1}/{len(candidates)}: t_cp={t_cp}, "
              f"seg1_len={len(x1)}, seg2_len={len(x2)}", end="")

        # Fit GP1
        kernel1 = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(nu=1.5))
        likelihood1 = gpytorch.likelihoods.GaussianLikelihood()
        model1 = ExactGPModel(x1, y1, likelihood1, kernel1)

        model1.train()
        likelihood1.train()
        optimizer1 = torch.optim.Adam(model1.parameters(), lr=0.1)
        mll1 = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood1, model1)

        for _ in range(50):
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

        # Fit GP2
        kernel2 = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(nu=1.5))
        likelihood2 = gpytorch.likelihoods.GaussianLikelihood()
        model2 = ExactGPModel(x2, y2, likelihood2, kernel2)

        model2.train()
        likelihood2.train()
        optimizer2 = torch.optim.Adam(model2.parameters(), lr=0.1)
        mll2 = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood2, model2)

        for _ in range(50):
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

        # Combined
        combined_log_mll = log_mll1 + log_mll2

        print(f" → mll1={log_mll1:.2f}, mll2={log_mll2:.2f}, combined={combined_log_mll:.2f}")

        if combined_log_mll > best_log_mll:
            best_log_mll = combined_log_mll
            best_t_cp = t_cp

        successful += 1

    except Exception as e:
        print(f" → FAILED: {type(e).__name__}: {e}")
        failed += 1

print()
print("="*60)
print(f"Grid search completed:")
print(f"  Successful: {successful}/{len(candidates)}")
print(f"  Failed: {failed}/{len(candidates)}")
print(f"  Best log MLL: {best_log_mll:.4f}")
print(f"  Best t_cp: {best_t_cp}")
