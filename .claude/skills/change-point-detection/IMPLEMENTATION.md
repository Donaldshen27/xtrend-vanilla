# Change-Point Detection Implementation

## Table of Contents

1. [Gaussian Process Kernels](#gaussian-process-kernels)
2. [GP-CPD Algorithm](#gp-cpd-algorithm)
3. [Segmentation Algorithm](#segmentation-algorithm)
4. [Practical Implementation](#practical-implementation)
5. [Context Set Construction](#context-set-construction)
6. [Visualization](#visualization)

---

## Gaussian Process Kernels

### Matérn 3/2 Kernel

```python
def matern_32_kernel(x1, x2, length_scale=1.0, variance=1.0):
    """
    Matérn 3/2 kernel for GP.

    k(r) = σ² * (1 + √3*r/ℓ) * exp(-√3*r/ℓ)

    where r = |x1 - x2|

    Args:
        length_scale (ℓ): How quickly correlation decays
        variance (σ²): Overall scale of the function

    Properties:
    - Once differentiable (smoother than OU process)
    - Not infinitely smooth (realistic for financial data)
    - Better than RBF for capturing financial dynamics
    """
    r = np.abs(x1 - x2)
    sqrt3_r = np.sqrt(3.0) * r / length_scale

    return variance * (1.0 + sqrt3_r) * np.exp(-sqrt3_r)
```

### Change-Point Kernel

```python
def changepoint_kernel(x1, x2, t_cp, k1, k2, sigma=1.0):
    """
    Change-point covariance kernel.

    Assumes two different GPs (with kernels k1 and k2) before
    and after change-point t_cp, with soft sigmoid transition.

    k_CP(x1, x2) = σ(x1) * σ(x2) * k1(x1, x2)
                 + (1-σ(x1)) * (1-σ(x2)) * k2(x1, x2)

    where σ(x) = sigmoid((x - t_cp) / sigma) is transition function

    Args:
        t_cp: Location of change-point
        k1: Kernel before change-point
        k2: Kernel after change-point
        sigma: Softness of transition (lower = sharper)
    """
    # Sigmoid transition
    s1 = sigmoid((x1 - t_cp) / sigma)
    s2 = sigmoid((x2 - t_cp) / sigma)

    # Weighted combination
    return (s1 * s2 * k1(x1, x2) +
            (1 - s1) * (1 - s2) * k2(x1, x2))
```

## GP-CPD Algorithm

```python
def gp_cpd(prices, lookback=21, threshold=0.9):
    """
    Detect change-points using GP marginal likelihood comparison.

    Algorithm:
    1. Fit GP with Matérn kernel → compute L_M
    2. Fit GP with Change-Point kernel → compute L_C (optimize t_cp)
    3. Compare: severity = L_C / (L_M + L_C)
    4. If severity ≥ threshold, declare change-point

    Args:
        prices: Price series
        lookback: Window size to check for changes (21 days ≈ 1 month)
        threshold: Severity threshold ν (0.9-0.95 typical)

    Returns:
        changepoint_time: Location of detected change-point (or None)
        severity: L_C / (L_M + L_C)
    """
    # 1. Fit Matérn GP (no change-point)
    gp_matern = fit_gp(prices[-lookback:], kernel='matern32')
    L_M = gp_matern.log_marginal_likelihood()

    # 2. Fit Change-Point GP (optimize location)
    gp_cp, t_cp = fit_gp_changepoint(prices[-lookback:])
    L_C = gp_cp.log_marginal_likelihood()

    # 3. Calculate severity
    severity = L_C / (L_M + L_C)

    # 4. Test threshold
    if severity >= threshold:
        return t_cp, severity
    else:
        return None, severity
```

**Severity Interpretation:**
- `severity = 0.5`: No evidence for change-point (models equally good)
- `severity = 0.9`: Strong evidence for change-point
- `severity = 0.95`: Very strong evidence for change-point

## Segmentation Algorithm

```python
def segment_timeseries(prices, lookback=21, threshold=0.9,
                      min_length=5, max_length=63):
    """
    Segment time-series into regime periods.

    Args:
        lookback: Window for CPD (21 days)
        threshold: Severity threshold ν
        min_length: Minimum regime length (5 days)
        max_length: Maximum regime length (63 days for 3-month)

    Returns:
        regimes: List of (start, end) tuples for each regime
    """
    regimes = []
    t = len(prices) - 1
    regime_end = t

    while t >= 0:
        # Check for change-point in lookback window
        window = prices[max(0, t-lookback):t+1]
        cp_location, severity = gp_cpd(window, lookback, threshold)

        if severity >= threshold:
            # Change-point detected
            regime_start = t - lookback + cp_location

            # Validate regime length
            if regime_end - regime_start >= min_length:
                regimes.append((regime_start, regime_end))

            # Move before change-point (avoid corrupting representation)
            t = regime_start - 1
            regime_end = t

        else:
            # No change-point, keep looking back
            t = t - 1

            # Enforce maximum regime length
            if regime_end - t > max_length:
                regime_start = regime_end - max_length
                regimes.append((regime_start, regime_end))
                t = regime_start - 1
                regime_end = t

    # Add final regime if it exists
    if regime_end - 0 >= min_length:
        regimes.append((0, regime_end))

    return list(reversed(regimes))  # Chronological order
```

## Practical Implementation

```python
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern

class FinancialCPD:
    """Gaussian Process Change-Point Detection for finance."""

    def __init__(self, lookback=21, threshold=0.9):
        self.lookback = lookback
        self.threshold = threshold

    def fit_matern_gp(self, prices):
        """Fit GP with Matérn 3/2 kernel."""
        X = np.arange(len(prices)).reshape(-1, 1)
        y = prices.reshape(-1, 1)

        kernel = Matern(nu=1.5)  # nu=1.5 → Matérn 3/2
        gp = GaussianProcessRegressor(kernel=kernel)
        gp.fit(X, y)

        return gp, gp.log_marginal_likelihood_value_

    def fit_changepoint_gp(self, prices):
        """Fit GP with change-point kernel (simplified)."""
        X = np.arange(len(prices)).reshape(-1, 1)
        y = prices.reshape(-1, 1)

        best_cp = None
        best_likelihood = -np.inf

        # Search for best change-point location
        for t_cp in range(5, len(prices) - 5):  # Avoid edges
            # Fit two separate GPs before/after t_cp
            gp1 = GaussianProcessRegressor(kernel=Matern(nu=1.5))
            gp2 = GaussianProcessRegressor(kernel=Matern(nu=1.5))

            gp1.fit(X[:t_cp], y[:t_cp])
            gp2.fit(X[t_cp:], y[t_cp:])

            # Combined likelihood
            likelihood = (gp1.log_marginal_likelihood_value_ +
                         gp2.log_marginal_likelihood_value_)

            if likelihood > best_likelihood:
                best_likelihood = likelihood
                best_cp = t_cp

        return best_cp, best_likelihood

    def detect_changepoint(self, prices):
        """Detect change-point in price window."""
        # Fit both models
        gp_m, L_M = self.fit_matern_gp(prices)
        t_cp, L_C = self.fit_changepoint_gp(prices)

        # Calculate severity
        severity = L_C / (L_M + L_C)

        if severity >= self.threshold:
            return t_cp, severity
        else:
            return None, severity

    def segment(self, prices, min_len=5, max_len=63):
        """Segment entire time-series into regimes."""
        regimes = []
        t = len(prices) - 1
        regime_end = t

        while t >= 0:
            # Get window
            start_idx = max(0, t - self.lookback + 1)
            window = prices[start_idx:t+1]

            # Detect change-point
            cp_loc, severity = self.detect_changepoint(window)

            if cp_loc is not None:
                # Found change-point
                regime_start = start_idx + cp_loc

                if regime_end - regime_start >= min_len:
                    regimes.append({
                        'start': regime_start,
                        'end': regime_end,
                        'severity': severity,
                        'prices': prices[regime_start:regime_end+1]
                    })

                t = regime_start - 1
                regime_end = t

            else:
                # No change-point
                t -= 1

                # Max length constraint
                if regime_end - t > max_len:
                    regime_start = regime_end - max_len
                    regimes.append({
                        'start': regime_start,
                        'end': regime_end,
                        'severity': 0.0,
                        'prices': prices[regime_start:regime_end+1]
                    })
                    t = regime_start - 1
                    regime_end = t

        return list(reversed(regimes))
```

## Context Set Construction

```python
def cpd_context_set(assets, target_time, size=20, lookback=21,
                   threshold=0.9):
    """
    Create context set using CPD-segmented regimes.

    X-Trend paper shows this improves Sharpe by 11.3% vs random sampling.

    Strategy:
    1. Segment each asset's history with CPD
    2. Sample random regime segments as context
    3. Ensure all context is before target_time (causality)
    """
    cpd = FinancialCPD(lookback=lookback, threshold=threshold)
    context_set = []

    for _ in range(size):
        # Sample random asset
        asset = random.choice(assets)
        prices = load_prices(asset, end=target_time)

        # Segment into regimes
        regimes = cpd.segment(prices, min_len=5, max_len=21)

        if len(regimes) > 0:
            # Sample random regime
            regime = random.choice(regimes)

            context_set.append({
                'asset': asset,
                'start': regime['start'],
                'end': regime['end'],
                'features': extract_features(regime['prices']),
                'returns': calculate_returns(regime['prices'])
            })

    return context_set
```

## Visualization

```python
import matplotlib.pyplot as plt

def visualize_regimes(prices, regimes):
    """Visualize detected regimes."""
    plt.figure(figsize=(15, 6))

    # Plot price series
    plt.plot(prices.index, prices.values, color='gray', alpha=0.5)

    # Color each regime differently
    colors = plt.cm.tab10(np.linspace(0, 1, len(regimes)))

    for i, regime in enumerate(regimes):
        start, end = regime['start'], regime['end']
        plt.plot(
            prices.index[start:end+1],
            prices.values[start:end+1],
            color=colors[i],
            linewidth=2,
            label=f"Regime {i+1} (sev={regime['severity']:.2f})"
        )

        # Mark change-points
        if i < len(regimes) - 1:
            plt.axvline(prices.index[end], color='red',
                       linestyle='--', alpha=0.3)

    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.title('Regime Segmentation with GP-CPD')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()
```

## Use Cases

### Momentum Crash Detection

```python
def detect_momentum_crash(prices, positions):
    """
    Identify regime changes that cause momentum losses.

    Momentum crashes occur when:
    - Market reverses rapidly (change-point detected)
    - Existing position is wrong direction
    """
    cpd = FinancialCPD(lookback=21, threshold=0.92)
    regimes = cpd.segment(prices)

    crashes = []
    for i in range(1, len(regimes)):
        prev_regime = regimes[i-1]
        curr_regime = regimes[i]

        # Check if position was wrong
        prev_trend = np.sign(prev_regime['prices'][-1] -
                            prev_regime['prices'][0])
        curr_trend = np.sign(curr_regime['prices'][-1] -
                            curr_regime['prices'][0])

        if prev_trend != curr_trend:
            crashes.append({
                'time': curr_regime['start'],
                'severity': curr_regime['severity'],
                'reversal': True
            })

    return crashes
```

### Adaptive Strategy Selection

```python
def select_strategy_by_regime(prices, strategies):
    """
    Choose trading strategy based on current regime characteristics.

    Different regimes may require different strategies:
    - Trending regimes: Momentum strategies
    - Mean-reverting regimes: Contrarian strategies
    - High-volatility regimes: Risk-off strategies
    """
    cpd = FinancialCPD(lookback=21)
    current_regime = cpd.segment(prices[-63:])[-1]  # Latest regime

    # Classify regime
    trend = (current_regime['prices'][-1] /
             current_regime['prices'][0] - 1)
    volatility = np.std(current_regime['prices'])

    if abs(trend) > 0.05 and volatility < 0.02:
        return strategies['momentum']
    elif abs(trend) < 0.02:
        return strategies['mean_reversion']
    elif volatility > 0.04:
        return strategies['defensive']
    else:
        return strategies['balanced']
```

---

**Last Updated**: March 2024
**Reference Type**: Implementation Details
