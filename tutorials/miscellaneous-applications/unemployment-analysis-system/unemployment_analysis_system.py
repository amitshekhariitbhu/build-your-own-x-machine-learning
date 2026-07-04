import numpy as np


class UnemploymentForecaster:
    """Autoregressive unemployment-rate forecaster from scratch.

    One-step-ahead model of a monthly unemployment rate. Each month is a
    ridge-regularized linear function of:
      - autoregressive lags of the rate (persistence),
      - a LAGGED leading indicator (layoff-announcement index) that moves
        before the rate does,
      - sin/cos seasonal terms (hiring cycles within a year),
      - a slow trend term (the business cycle).
    Solved in closed form via ridge normal equations; the bias is left
    unpenalized. Beats naive persistence by exploiting seasonality + the
    leading indicator that plain "predict last month" cannot see.
    """

    def __init__(self, lags=(1, 2, 3, 12), lead=2, ridge=1.0):
        self.lags = tuple(lags)
        self.lead = lead              # months the layoff index leads the rate
        self.ridge = ridge
        self.w = None
        self.mu = None
        self.sigma = None

    def build_features(self, rate, layoffs):
        """Vectorized design matrix for one-step-ahead prediction.

        Returns (X, y, idx): row t predicts rate[t] from information available
        strictly before t. idx holds the month index of each row.
        """
        rate = np.asarray(rate, dtype=float)
        layoffs = np.asarray(layoffs, dtype=float)
        T = rate.shape[0]
        start = max(max(self.lags), self.lead)
        idx = np.arange(start, T)
        cols = [rate[idx - L] for L in self.lags]      # autoregressive lags
        cols.append(layoffs[idx - self.lead])          # leading indicator
        cols.append(np.sin(2 * np.pi * idx / 12.0))    # seasonal
        cols.append(np.cos(2 * np.pi * idx / 12.0))
        cols.append(idx / float(T))                    # normalized trend
        X = np.column_stack(cols)
        y = rate[idx]
        return X, y, idx

    def _augment(self, X):
        Xs = (X - self.mu) / self.sigma
        return np.hstack([Xs, np.ones((Xs.shape[0], 1))])  # bias column

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        self.mu = X.mean(axis=0)
        self.sigma = X.std(axis=0) + 1e-8
        Xb = self._augment(X)
        d = Xb.shape[1]
        R = self.ridge * np.eye(d)
        R[-1, -1] = 0.0                                # do not penalize bias
        self.w = np.linalg.solve(Xb.T @ Xb + R, Xb.T @ np.asarray(y, float))
        return self

    def predict(self, X):
        return self._augment(np.asarray(X, dtype=float)) @ self.w


def make_unemployment_data(T=240, seed=0):
    """Synthetic monthly unemployment rate with planted latent structure.

    Ingredients (the signal a good model must recover):
      trend    - slow business cycle (multi-year up/down swings),
      season   - within-year hiring seasonality,
      layoffs  - a volatile leading indicator that pushes the rate up ~2 months
                 later (predictable change naive persistence cannot anticipate),
      AR       - persistence of deviations around the trend.
    """
    rng = np.random.RandomState(seed)
    t = np.arange(T)
    lead = 2

    # Leading indicator: mean-zero AR(1) layoff-announcement index.
    layoffs = np.zeros(T)
    shock = rng.normal(0.0, 1.0, size=T)
    for i in range(1, T):
        layoffs[i] = 0.7 * layoffs[i - 1] + shock[i]

    trend = 5.5 + 0.004 * t                              # slow upward drift
    season = 1.3 * np.sin(2 * np.pi * t / 12.0 + 0.4)    # strong yearly swing

    # Deviation from trend+season is AR(1) pushed by the lagged layoff index;
    # this is exactly the persistence + leading-signal a model must recover.
    dev = np.zeros(T)
    noise = rng.normal(0.0, 0.10, size=T)
    for i in range(T):
        exo = 0.8 * layoffs[i - lead] if i >= lead else 0.0
        dev[i] = 0.5 * dev[i - 1] + exo + noise[i] if i > 0 else exo + noise[i]
    rate = trend + season + dev
    return rate, layoffs


def rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


if __name__ == "__main__":
    np.random.seed(0)

    rate, layoffs = make_unemployment_data(T=240, seed=0)

    model = UnemploymentForecaster(lags=(1, 2, 3, 12), lead=2, ridge=1.0)
    X, y, idx = model.build_features(rate, layoffs)

    # Held-out split: train on the earlier months, forecast the recent tail.
    cut = int(0.75 * len(y))
    Xtr, ytr, Xte, yte = X[:cut], y[:cut], X[cut:], y[cut:]
    test_idx = idx[cut:]

    model.fit(Xtr, ytr)
    pred = model.predict(Xte)

    # Baselines on the SAME test months.
    naive = rate[test_idx - 1]            # persistence: predict last month
    seasonal = rate[test_idx - 12]        # seasonal naive: predict a year ago
    mean_pred = np.full_like(yte, ytr.mean())

    model_rmse = rmse(yte, pred)
    naive_rmse = rmse(yte, naive)
    seasonal_rmse = rmse(yte, seasonal)
    mean_rmse = rmse(yte, mean_pred)
    improve = (naive_rmse - model_rmse) / naive_rmse * 100.0

    print("=== Unemployment Analysis System (from scratch) ===")
    print(f"months total          : {len(rate)}   test months: {len(yte)}")
    print(f"test rate range       : {yte.min():.2f}% - {yte.max():.2f}%")
    print(f"baseline mean         : RMSE={mean_rmse:.3f}")
    print(f"baseline seasonal     : RMSE={seasonal_rmse:.3f}")
    print(f"baseline naive (last) : RMSE={naive_rmse:.3f}")
    print(f"AR forecaster         : RMSE={model_rmse:.3f}")
    print(f"improvement vs naive  : {improve:.1f}% lower RMSE")
    assert model_rmse < naive_rmse, "model should beat naive persistence"
