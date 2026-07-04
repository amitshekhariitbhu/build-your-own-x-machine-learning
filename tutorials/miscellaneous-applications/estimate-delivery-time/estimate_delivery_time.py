import numpy as np


class DeliveryTimeRegressor:
    """Linear-regression delivery-time estimator trained from scratch.

    Standardizes the input features, learns weights + bias by full-batch
    gradient descent on the L2-regularized mean-squared-error loss, and
    predicts the delivery duration in minutes. No ML libraries -- just numpy.
    """

    def __init__(self, lr=0.1, n_iter=5000, l2=1e-3):
        self.lr = lr
        self.n_iter = n_iter
        self.l2 = l2  # L2 regularization strength
        self.w = None
        self.b = 0.0
        self.mu = None
        self.sigma = None

    def _standardize(self, X):
        return (X - self.mu) / self.sigma

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1)
        # scaling stats computed on training data only (no test leakage)
        self.mu = X.mean(axis=0)
        self.sigma = X.std(axis=0) + 1e-8
        Xs = self._standardize(X)
        n, d = Xs.shape
        self.w = np.zeros(d)
        self.b = float(y.mean())  # start bias at the mean duration
        for _ in range(self.n_iter):
            pred = Xs @ self.w + self.b
            err = pred - y
            grad_w = Xs.T @ err / n + self.l2 * self.w
            grad_b = err.mean()
            self.w -= self.lr * grad_w
            self.b -= self.lr * grad_b
        return self

    def predict(self, X):
        Xs = self._standardize(np.asarray(X, dtype=float))
        return Xs @ self.w + self.b


def make_delivery_data(n=1200):
    """Synthetic food-delivery orders with a planted duration signal.

    Raw signals (delivery inspired):
      distance_km, prep_time, traffic (0-1), rain (0/1),
      items, courier_speed (km/min), rush_hour (0/1).
    A physically-motivated rule sets the true minutes: travel time is
    distance / effective_speed where traffic and rain slow the courier,
    plus kitchen prep and per-item handling. Nonlinear ratio features are
    added so a linear model can recover the divide-by-speed structure.
    """
    distance = np.random.uniform(0.5, 12.0, n)           # km to customer
    prep_time = np.random.uniform(5, 25, n)              # kitchen minutes
    traffic = np.random.uniform(0, 1, n)                 # congestion index
    rain = np.random.binomial(1, 0.25, n)                # 1 = raining
    items = np.random.randint(1, 8, n)                   # order size
    courier_speed = np.random.uniform(0.3, 0.7, n)       # base km/min
    rush_hour = np.random.binomial(1, 0.35, n)           # 1 = peak time

    # effective speed drops with traffic, rain and rush hour
    eff_speed = courier_speed * (1.0 - 0.5 * traffic) * (1.0 - 0.2 * rain)
    eff_speed *= (1.0 - 0.15 * rush_hour)
    eff_speed = np.clip(eff_speed, 0.08, None)

    # planted delivery time (minutes)
    minutes = (prep_time
               + distance / eff_speed           # travel time dominates
               + 1.2 * items                     # per-item handling
               + 3.0 * rush_hour                 # dispatch backlog at peak
               + 2.0)                            # fixed handoff overhead
    minutes *= np.random.normal(1.0, 0.04, n)   # +-4% multiplicative noise
    minutes += np.random.normal(0, 1.0, n)      # small additive noise

    # engineered features expose the divide-by-speed / interaction terms
    inv_speed = 1.0 / courier_speed
    dist_traffic = distance * traffic
    dist_rain = distance * rain
    X = np.column_stack([distance, prep_time, traffic, rain, items,
                         courier_speed, rush_hour,
                         inv_speed, dist_traffic, dist_rain])
    feats = ["distance", "prep_time", "traffic", "rain", "items",
             "courier_speed", "rush_hour",
             "inv_speed", "dist*traffic", "dist*rain"]
    return X, minutes, feats


if __name__ == "__main__":
    np.random.seed(0)

    X, y, feats = make_delivery_data(1200)

    # held-out train/test split
    idx = np.random.permutation(len(y))
    cut = int(0.75 * len(y))
    tr, te = idx[:cut], idx[cut:]
    Xtr, ytr, Xte, yte = X[tr], y[tr], X[te], y[te]

    model = DeliveryTimeRegressor(lr=0.1, n_iter=5000, l2=1e-3).fit(Xtr, ytr)
    pred = model.predict(Xte)

    # model error
    rmse = np.sqrt(np.mean((pred - yte) ** 2))
    mae = np.mean(np.abs(pred - yte))

    # mean-duration baseline (predict the training mean for every order)
    base_pred = ytr.mean()
    base_rmse = np.sqrt(np.mean((base_pred - yte) ** 2))
    base_mae = np.mean(np.abs(base_pred - yte))

    # R^2 (fraction of duration variance explained on held-out data)
    ss_res = np.sum((yte - pred) ** 2)
    ss_tot = np.sum((yte - yte.mean()) ** 2)
    r2 = 1.0 - ss_res / ss_tot

    print("test orders           :", len(yte))
    print("mean delivery (min)   :", round(float(yte.mean()), 2))
    print("baseline RMSE (mean)  :", round(float(base_rmse), 2))
    print("model RMSE            :", round(float(rmse), 2))
    print("baseline MAE (mean)   :", round(float(base_mae), 2))
    print("model MAE             :", round(float(mae), 2))
    print("R^2 (held-out)        :", round(float(r2), 3))
    print("within +-3 min        :", round(float(np.mean(np.abs(pred - yte) <= 3.0)), 3))
    print("feature weights (std) :", dict(zip(feats, np.round(model.w, 2))))
    print("BEATS baseline        :", bool(rmse < base_rmse))
