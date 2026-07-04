import numpy as np


class WineQualityRegressor:
    """Ridge-regularized linear regression trained from scratch.

    Standardizes the physicochemical wine features, then learns weights + bias
    by full-batch gradient descent on the L2-penalized mean-squared-error loss.
    Predicts a continuous quality score (real data uses integer scores 3..8).
    No ML libraries -- just numpy math.
    """

    def __init__(self, lr=0.1, n_iter=2000, l2=1e-2):
        self.lr = lr
        self.n_iter = n_iter
        self.l2 = l2  # L2 (ridge) strength
        self.w = None
        self.b = None
        self.mu = None
        self.sigma = None

    def _standardize(self, X):
        return (X - self.mu) / self.sigma

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1)
        # scaling stats from training data only (no test leakage)
        self.mu = X.mean(axis=0)
        self.sigma = X.std(axis=0) + 1e-8
        Xs = self._standardize(X)
        n, d = Xs.shape
        self.w = np.zeros(d)
        self.b = 0.0
        for _ in range(self.n_iter):
            pred = Xs @ self.w + self.b
            err = pred - y                       # residuals
            grad_w = Xs.T @ err / n + self.l2 * self.w
            grad_b = err.mean()
            self.w -= self.lr * grad_w
            self.b -= self.lr * grad_b
        return self

    def predict(self, X):
        Xs = self._standardize(np.asarray(X, dtype=float))
        return Xs @ self.w + self.b


def make_wine_data(n=1400):
    """Synthetic red-wine physicochemistry with a planted quality signal.

    11 features mirror the classic wine dataset.  A latent quality score is
    driven mostly by alcohol (+) and sulphates/citric_acid (+), and lowered by
    volatile_acidity (-) and chlorides (-), plus tasting noise so it overlaps.
    Scores are clipped to the familiar 3..8 integer range.
    """
    fixed_acidity = np.random.uniform(4.6, 15.0, n)
    volatile_acidity = np.random.uniform(0.12, 1.20, n)
    citric_acid = np.random.uniform(0.0, 1.0, n)
    residual_sugar = np.random.uniform(0.9, 15.0, n)
    chlorides = np.random.uniform(0.01, 0.40, n)
    free_so2 = np.random.uniform(1, 70, n)
    total_so2 = free_so2 + np.random.uniform(5, 200, n)
    density = np.random.uniform(0.990, 1.004, n)
    pH = np.random.uniform(2.8, 4.0, n)
    sulphates = np.random.uniform(0.33, 2.0, n)
    alcohol = np.random.uniform(8.0, 14.0, n)
    X = np.column_stack([fixed_acidity, volatile_acidity, citric_acid,
                         residual_sugar, chlorides, free_so2, total_so2,
                         density, pH, sulphates, alcohol])

    # planted latent quality: alcohol dominates (like the real dataset)
    score = (5.0
             + 0.55 * (alcohol - 10.5)          # more alcohol -> better
             - 1.60 * (volatile_acidity - 0.5)  # vinegary -> worse
             + 0.90 * (sulphates - 0.65)         # antioxidant -> better
             + 0.60 * citric_acid                # freshness -> better
             - 2.50 * (chlorides - 0.08)         # salty -> worse
             - 0.30 * (fixed_acidity - 8.0) * 0.1)
    score += np.random.normal(0, 0.55, n)        # tasting overlap noise
    y = np.clip(np.rint(score), 3, 8)            # integer quality 3..8
    return X, y


if __name__ == "__main__":
    np.random.seed(0)

    X, y = make_wine_data(1400)

    # held-out train/test split
    idx = np.random.permutation(len(y))
    cut = int(0.75 * len(y))
    tr, te = idx[:cut], idx[cut:]
    Xtr, ytr, Xte, yte = X[tr], y[tr], X[te], y[te]

    model = WineQualityRegressor(lr=0.1, n_iter=2000).fit(Xtr, ytr)
    pred = model.predict(Xte)

    # regression signal: RMSE / MAE vs a mean-predictor baseline
    rmse = float(np.sqrt(np.mean((pred - yte) ** 2)))
    mae = float(np.mean(np.abs(pred - yte)))
    base = float(ytr.mean())                     # predict training mean
    base_rmse = float(np.sqrt(np.mean((base - yte) ** 2)))
    base_mae = float(np.mean(np.abs(base - yte)))

    # ordinal check: rounded score within +/-1 of true quality
    within1 = float(np.mean(np.abs(np.rint(pred) - yte) <= 1))

    # derived good/bad task: quality >= 6 -> "good", accuracy vs majority
    good = (yte >= 6).astype(int)
    pred_good = (pred >= 5.5).astype(int)
    acc = float(np.mean(pred_good == good))
    majority = int(np.bincount((ytr >= 6).astype(int)).argmax())
    base_acc = float(np.mean(good == majority))

    feats = ["fixed_acidity", "volatile_acidity", "citric_acid",
             "residual_sugar", "chlorides", "free_so2", "total_so2",
             "density", "pH", "sulphates", "alcohol"]
    # |standardized weight| ranks feature influence -> alcohol on top
    importance = dict(zip(feats, np.round(np.abs(model.w), 3)))
    top = max(importance, key=importance.get)

    print("test samples            :", len(yte))
    print("quality range           : 3..8")
    print("mean-baseline RMSE      :", round(base_rmse, 3))
    print("model RMSE              :", round(rmse, 3))
    print("mean-baseline MAE       :", round(base_mae, 3))
    print("model MAE               :", round(mae, 3))
    print("within +/-1 of true     :", round(within1, 3))
    print("good/bad majority acc   :", round(base_acc, 3))
    print("good/bad model acc      :", round(acc, 3))
    print("top feature (|w|)       :", top)
    print("feature importance      :", importance)
    print("BEATS baseline (RMSE)   :", bool(rmse < base_rmse))
