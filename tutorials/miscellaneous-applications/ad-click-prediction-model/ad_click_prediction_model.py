import numpy as np


class AdClickPredictor:
    """Logistic-regression CTR (click-through rate) model trained from scratch.

    Online ad-click prediction mixes CATEGORICAL context (device, ad position,
    time-of-day) with NUMERIC signals (user age, historical CTR, ad relevance).
    Categoricals are one-hot encoded, numerics are standardized, and weights +
    bias are learned by full-batch gradient descent on the binary cross-entropy
    (log-loss) objective -- the metric that actually matters for CTR ranking.
    """

    def __init__(self, lr=0.3, n_iter=4000, l2=1e-3):
        self.lr = lr
        self.n_iter = n_iter
        self.l2 = l2  # L2 regularization strength
        self.w = None
        self.b = 0.0
        self.mu = None
        self.sigma = None

    @staticmethod
    def _sigmoid(z):
        return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))

    def _standardize(self, X):
        return (X - self.mu) / self.sigma

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1)
        # scaling stats learned from the training set only (avoids leakage)
        self.mu = X.mean(axis=0)
        self.sigma = X.std(axis=0) + 1e-8
        Xs = self._standardize(X)
        n, d = Xs.shape
        self.w = np.zeros(d)
        self.b = 0.0
        for _ in range(self.n_iter):
            p = self._sigmoid(Xs @ self.w + self.b)  # predicted click probs
            err = p - y
            grad_w = Xs.T @ err / n + self.l2 * self.w
            grad_b = err.mean()
            self.w -= self.lr * grad_w
            self.b -= self.lr * grad_b
        return self

    def predict_proba(self, X):
        Xs = self._standardize(np.asarray(X, dtype=float))
        return self._sigmoid(Xs @ self.w + self.b)

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)


def _one_hot(idx, k):
    """Encode integer category codes (0..k-1) as one-hot columns."""
    oh = np.zeros((len(idx), k))
    oh[np.arange(len(idx)), idx] = 1.0
    return oh


def make_ad_data(n=4000):
    """Synthetic ad impressions with a planted click-vs-no-click rule.

    Raw log fields per impression:
      device    : 0=mobile 1=desktop 2=tablet   (mobile clicks a bit more)
      position  : 0=top 1=side 2=bottom         (top slots get far more clicks)
      daypart   : 0=night 1=morning 2=evening   (evening browsing clicks more)
      user_age  : viewer age in years
      hist_ctr  : that user's past click-through rate (0..1)
      relevance : ad<->query relevance score (0..1)
      bid       : advertiser bid (higher bid ~ better-targeted creative)
    Whether a user clicks depends mostly on ad position, relevance and the
    user's own history -- exactly the latent signal the model must recover.
    """
    device = np.random.randint(0, 3, n)
    position = np.random.randint(0, 3, n)
    daypart = np.random.randint(0, 3, n)
    user_age = np.random.randint(18, 65, n)
    hist_ctr = np.clip(np.random.beta(2, 5, n), 0, 1)
    relevance = np.clip(np.random.beta(2, 2, n), 0, 1)
    bid = np.abs(np.random.normal(1.5, 0.6, n))

    # planted click logit: position/relevance/history dominate
    pos_effect = np.array([1.4, 0.0, -1.2])[position]     # top >> side > bottom
    dev_effect = np.array([0.4, 0.0, -0.2])[device]       # mobile clicks more
    day_effect = np.array([-0.3, 0.1, 0.5])[daypart]      # evening clicks more
    z = (pos_effect + dev_effect + day_effect
         + 3.0 * relevance + 2.5 * hist_ctr + 0.4 * bid
         - 0.02 * (user_age - 35) - 3.2)
    prob = 1.0 / (1.0 + np.exp(-z))
    y = (np.random.rand(n) < prob).astype(int)  # sample real Bernoulli clicks

    # build the model feature matrix: one-hot categoricals + numeric columns
    X = np.column_stack([
        _one_hot(device, 3), _one_hot(position, 3), _one_hot(daypart, 3),
        user_age, hist_ctr, relevance, bid,
    ])
    return X, y


def roc_auc(y_true, scores):
    """AUC via the rank-sum (Mann-Whitney U) identity, computed from scratch."""
    order = np.argsort(scores)
    ranks = np.empty(len(scores), dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1)  # 1-based ranks
    pos = y_true == 1
    n_pos, n_neg = pos.sum(), (~pos).sum()
    if n_pos == 0 or n_neg == 0:
        return 0.5
    return (ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def log_loss(y_true, p):
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return -np.mean(y_true * np.log(p) + (1 - y_true) * np.log(1 - p))


if __name__ == "__main__":
    np.random.seed(0)

    X, y = make_ad_data(4000)

    # held-out train/test split
    idx = np.random.permutation(len(y))
    cut = int(0.75 * len(y))
    tr, te = idx[:cut], idx[cut:]
    Xtr, ytr, Xte, yte = X[tr], y[tr], X[te], y[te]

    model = AdClickPredictor(lr=0.3, n_iter=4000, l2=1e-3).fit(Xtr, ytr)
    proba = model.predict_proba(Xte)
    pred = (proba >= 0.5).astype(int)

    acc = np.mean(pred == yte)
    majority = int(round(ytr.mean()))                 # predict most common class
    base_acc = np.mean(yte == majority)
    base_rate = ytr.mean()                            # constant CTR baseline
    auc = roc_auc(yte, proba)
    ll_model = log_loss(yte, proba)
    ll_base = log_loss(yte, np.full(len(yte), base_rate))

    tp = np.sum((pred == 1) & (yte == 1))
    fp = np.sum((pred == 1) & (yte == 0))
    fn = np.sum((pred == 0) & (yte == 1))
    precision = tp / (tp + fp + 1e-9)
    recall = tp / (tp + fn + 1e-9)
    f1 = 2 * precision * recall / (precision + recall + 1e-9)

    print("test impressions       :", len(yte))
    print("click rate (test)      :", round(float(yte.mean()), 3))
    print("majority baseline acc  :", round(float(base_acc), 3))
    print("model accuracy         :", round(float(acc), 3))
    print("precision / recall     :", round(float(precision), 3), "/", round(float(recall), 3))
    print("F1 score               :", round(float(f1), 3))
    print("ROC AUC  (0.5=random)  :", round(float(auc), 3))
    print("log-loss constant-CTR  :", round(float(ll_base), 4))
    print("log-loss model (lower) :", round(float(ll_model), 4))
    print("BEATS baseline         :", bool(acc > base_acc and auc > 0.5 and ll_model < ll_base))
