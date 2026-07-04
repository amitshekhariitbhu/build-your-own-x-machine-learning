import numpy as np

# Industry sectors and their planted base net worth (in $B). The whole
# analysis system exists to recover this latent wealth structure from data.
INDUSTRIES = np.array(["Tech", "Finance", "Retail", "Energy", "RealEstate"])
IND_BASE = np.array([14.0, 9.0, 6.0, 7.5, 4.0])          # $B baseline per sector
REGIONS = np.array(["NorthAmerica", "Europe", "Asia", "Other"])
REGION_BONUS = np.array([3.0, 1.0, 2.0, 0.0])            # additive $B per region


class BillionaireAnalyzer:
    """Ridge-regression model that predicts a billionaire's net worth (in $B)
    from profile features, learned from scratch via the normal equations.

    Continuous features are standardized; a bias term is kept unregularized.
    The learned weights expose which factors drive extreme wealth.
    """

    def __init__(self, l2=1.0):
        self.l2 = l2          # ridge strength (guards against collinear one-hots)
        self.w = None         # [bias, feature weights...]
        self.mu = None
        self.sigma = None

    def _standardize(self, X):
        return (X - self.mu) / self.sigma

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1)
        self.mu = X.mean(axis=0)
        self.sigma = X.std(axis=0) + 1e-8
        Xb = np.column_stack([np.ones(len(X)), self._standardize(X)])  # add bias
        d = Xb.shape[1]
        R = self.l2 * np.eye(d)
        R[0, 0] = 0.0                               # never regularize the bias
        # closed-form ridge solution: w = (X'X + R)^-1 X'y
        self.w = np.linalg.solve(Xb.T @ Xb + R, Xb.T @ y)
        return self

    def predict(self, X):
        Xb = np.column_stack([np.ones(len(X)), self._standardize(np.asarray(X, float))])
        return Xb @ self.w


def make_billionaire_data(n=900):
    """Synthetic worldwide-billionaire records with a planted wealth signal.

    Each record: one-hot industry (5) + one-hot region (4) + age, years_active,
    num_companies. Net worth is a true linear function of these plus noise, so a
    correct model must recover the sector/region/experience premiums.
    """
    ind = np.random.randint(0, len(INDUSTRIES), n)
    reg = np.random.randint(0, len(REGIONS), n)
    age = np.clip(np.random.normal(63, 12, n), 30, 95)
    years_active = np.clip(np.random.normal(30, 10, n), 1, 70)
    num_companies = np.clip(np.random.poisson(3, n), 1, 15)

    # planted ground-truth net worth (in $B)
    net_worth = (IND_BASE[ind] + REGION_BONUS[reg]
                 + 0.18 * years_active
                 + 0.9 * num_companies
                 + 0.04 * (age - 60)
                 + np.random.normal(0, 2.0, n))
    net_worth = np.maximum(net_worth, 1.0)           # billionaires: at least $1B

    ind_oh = np.eye(len(INDUSTRIES))[ind]
    reg_oh = np.eye(len(REGIONS))[reg]
    X = np.column_stack([ind_oh, reg_oh, age, years_active, num_companies])
    return X, net_worth, ind


if __name__ == "__main__":
    np.random.seed(0)

    X, y, ind = make_billionaire_data(900)

    # held-out train/test split
    idx = np.random.permutation(len(y))
    cut = int(0.75 * len(y))
    tr, te = idx[:cut], idx[cut:]

    model = BillionaireAnalyzer(l2=1.0).fit(X[tr], y[tr])
    pred = model.predict(X[te])

    # regression quality vs a predict-the-mean baseline
    rmse = np.sqrt(np.mean((pred - y[te]) ** 2))
    mae = np.mean(np.abs(pred - y[te]))
    base_pred = y[tr].mean()
    base_rmse = np.sqrt(np.mean((base_pred - y[te]) ** 2))
    ss_res = np.sum((y[te] - pred) ** 2)
    ss_tot = np.sum((y[te] - y[te].mean()) ** 2)
    r2 = 1.0 - ss_res / ss_tot

    # ANALYSIS: recover the richest-industry ranking from the data itself
    obs_means = np.array([y[ind == k].mean() for k in range(len(INDUSTRIES))])
    recovered_order = np.argsort(-obs_means)
    planted_order = np.argsort(-IND_BASE)
    ranking_ok = bool(np.array_equal(recovered_order, planted_order))

    print("test billionaires       :", len(te))
    print("mean net worth (train)  : $%.2fB" % base_pred)
    print("baseline RMSE (mean)    : $%.2fB" % base_rmse)
    print("model RMSE              : $%.2fB" % rmse)
    print("model MAE               : $%.2fB" % mae)
    print("model R^2               :", round(float(r2), 3))
    print("richest->poorest sector :", INDUSTRIES[recovered_order].tolist())
    print("planted sector ranking  :", INDUSTRIES[planted_order].tolist())
    print("ranking recovered       :", ranking_ok)
    print("BEATS baseline (RMSE)   :", bool(rmse < base_rmse))
