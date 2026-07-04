import numpy as np

# Crop Yield Prediction as time-series forecasting, from scratch.
# Yield each growing season is driven by a slow TECHNOLOGY TREND, an AR(1)
# soil-carryover term, and exogenous WEATHER (rainfall & temperature) whose
# effects are quadratic -- there is an optimum, too little/too much hurts.
# We fit an ARX model (autoregression + regression on lags & weather) by
# ridge normal equations, then forecast a held-out tail of future seasons.


def make_seasons(T=140, seed=0):
    # Synthetic per-season crop yield (tons/ha) with planted, recoverable structure.
    rng = np.random.RandomState(seed)
    t = np.arange(T)
    tech = 0.018 * t                              # technology/genetics uplift over time
    # Weather regressors: periodic climate cycles + slow warming + noise.
    rain = 5.5 + 1.6 * np.sin(2 * np.pi * t / 11) + rng.normal(0, 0.6, T)   # rain index
    temp = 22.0 + 0.02 * t + 2.0 * np.sin(2 * np.pi * t / 7) + rng.normal(0, 0.8, T)
    ropt, topt = 6.0, 23.0                        # agronomic optima
    y = np.zeros(T)
    y[0] = 4.0
    for k in range(1, T):
        w = (-0.28 * (rain[k] - ropt) ** 2        # rainfall: quadratic, peak at ropt
             - 0.05 * (temp[k] - topt) ** 2)      # heat stress: quadratic penalty
        y[k] = (2.2 + tech[k]                      # base + rising trend
                + 0.35 * y[k - 1]                  # soil-moisture / residue carryover (AR1)
                + w + rng.normal(0, 0.25))         # weather effect + shock
    return y, rain, temp


class CropYieldModel:
    """ARX forecaster: y_t ~ c + a*t + phi*y_{t-1} + weather (rain, rain^2, temp, temp^2)."""

    def __init__(self, lam=1e-2):
        self.lam = lam

    def _design(self, y, rain, temp, idx):
        # Build feature rows for the given target indices (each uses y_{t-1} & weather_t).
        t = idx.astype(float)
        return np.column_stack([
            np.ones_like(t), t, y[idx - 1],
            rain[idx], rain[idx] ** 2,
            temp[idx], temp[idx] ** 2,
        ])

    def fit(self, y, rain, temp, idx):
        Z = self._design(y, rain, temp, idx)
        R = self.lam * np.eye(Z.shape[1])
        R[0, 0] = 0.0                              # don't penalize the intercept
        self.w = np.linalg.solve(Z.T @ Z + R, Z.T @ y[idx])
        return self

    def predict(self, y, rain, temp, idx):
        # One-step-ahead: uses the TRUE previous yield and known season weather.
        return self._design(y, rain, temp, idx) @ self.w

    def forecast(self, y0, t0, rain_fut, temp_fut):
        # Recursive multi-step: feed each predicted yield back in as next y_{t-1}.
        yp, prev, out = y0, y0, []
        for h, (r, tp) in enumerate(zip(rain_fut, temp_fut)):
            t = t0 + h
            feat = np.array([1.0, t, prev, r, r ** 2, tp, tp ** 2])
            yp = feat @ self.w
            out.append(yp)
            prev = yp
        return np.array(out)


def fit_ar1(y, idx):
    # AR(1)+trend only (no weather) -> shows the exogenous weather features add value.
    Z = np.column_stack([np.ones(len(idx)), idx.astype(float), y[idx - 1]])
    w = np.linalg.solve(Z.T @ Z + 1e-2 * np.eye(3), Z.T @ y[idx])
    return w


def rmse(a, b):
    return np.sqrt(np.mean((a - b) ** 2))


def mae(a, b):
    return np.mean(np.abs(a - b))


if __name__ == "__main__":
    np.random.seed(0)

    y, rain, temp = make_seasons(T=140, seed=0)
    split = 112                                    # held-out tail = last 28 seasons
    tr = np.arange(1, split)                       # start at 1 (needs y_{t-1})
    te = np.arange(split, len(y))

    model = CropYieldModel(lam=1e-2).fit(y, rain, temp, tr)
    ar1 = fit_ar1(y, tr)

    pred = model.predict(y, rain, temp, te)        # ARX one-step forecasts
    Zar = np.column_stack([np.ones(len(te)), te.astype(float), y[te - 1]])
    pred_ar = Zar @ ar1                            # AR1+trend baseline
    naive = y[te - 1]                              # persistence (last season) baseline
    mean_base = np.full(len(te), y[tr].mean())     # predict-mean baseline
    true = y[te]

    print("Crop yield series: %d seasons   train=%d  test=%d   (tons/ha)"
          % (len(y), len(tr), len(te)))
    print("-" * 60)
    print("ARX  (trend+AR+weather)  RMSE: %.4f   MAE: %.4f" % (rmse(pred, true), mae(pred, true)))
    print("AR1+trend  (no weather)  RMSE: %.4f   MAE: %.4f" % (rmse(pred_ar, true), mae(pred_ar, true)))
    print("Naive last-season base   RMSE: %.4f   MAE: %.4f" % (rmse(naive, true), mae(naive, true)))
    print("Predict-mean baseline    RMSE: %.4f   MAE: %.4f" % (rmse(mean_base, true), mae(mean_base, true)))
    print("-" * 60)

    fc = model.forecast(y[split - 1], split, rain[te][:5], temp[te][:5])
    print("Recursive 5-season forecast from train end:")
    for i, (p, tval) in enumerate(zip(fc, true[:5]), 1):
        print("   season+%d  pred=%.3f   true=%.3f" % (i, p, tval))
    print("-" * 60)
    beats = rmse(pred, true) < min(rmse(naive, true), rmse(mean_base, true), rmse(pred_ar, true))
    print("ARX beats naive, mean & weather-free AR baselines: %s" % beats)
