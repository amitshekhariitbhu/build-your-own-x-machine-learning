import numpy as np


class PurchaseIntentionClassifier:
    """Class-weighted logistic regression for online-shopping purchase intent.

    A web-store session logs behaviour (pages viewed, time on site, bounce/exit
    rates, the running "page value" the analytics tag assigns) plus context
    (visitor type, weekend, closeness to a special-sale day). Only a minority of
    sessions end in a purchase, so the data is IMBALANCED. We standardize the
    numerics, one-hot the categoricals, and learn weights + bias by full-batch
    gradient descent on a CLASS-WEIGHTED binary cross-entropy -- up-weighting the
    rare "buyer" class so the model recovers recall/F1 instead of just predicting
    "nobody buys". np.linalg is not needed; plain vectorized numpy does it all.
    """

    def __init__(self, lr=0.2, n_iter=4000, l2=1e-3):
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
        # scaling stats learned from the training split only (no leakage)
        self.mu = X.mean(axis=0)
        self.sigma = X.std(axis=0) + 1e-8
        Xs = self._standardize(X)
        n, d = Xs.shape
        # per-sample weights that balance the two classes (rare buyers count more)
        pos_rate = y.mean()
        wpos = 0.5 / max(pos_rate, 1e-6)
        wneg = 0.5 / max(1 - pos_rate, 1e-6)
        sw = np.where(y == 1, wpos, wneg)
        self.w = np.zeros(d)
        self.b = 0.0
        for _ in range(self.n_iter):
            p = self._sigmoid(Xs @ self.w + self.b)  # predicted buy probs
            err = sw * (p - y)
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


def make_session_data(n=4000):
    """Synthetic web-store sessions with a planted buy-vs-leave rule.

    Raw session fields:
      product_pages : # product-related pages viewed (engagement)
      duration      : total product browsing time (seconds)
      bounce_rate   : avg bounce rate of visited pages (0..1, high=bad)
      exit_rate     : avg exit rate of visited pages (0..1, high=bad)
      page_value    : analytics page-value of the session (dollars; top signal)
      special_day   : closeness to a promo/holiday (0..1)
      visitor_type  : 0=returning 1=new 2=other  (returning buy more)
      weekend       : 1 if the session happened on a weekend
    Buying is driven mostly by high page_value + engagement and LOW exit/bounce
    -- exactly the latent structure the classifier must recover.
    """
    product_pages = np.random.poisson(18, n).astype(float)
    duration = product_pages * np.random.uniform(30, 90, n)
    bounce_rate = np.clip(np.random.beta(2, 8, n), 0, 1)
    exit_rate = np.clip(bounce_rate + np.random.beta(2, 6, n) * 0.3, 0, 1)
    page_value = np.abs(np.random.normal(0, 12, n)) * (np.random.rand(n) < 0.4)
    special_day = np.where(np.random.rand(n) < 0.25, np.random.rand(n), 0.0)
    visitor_type = np.random.choice([0, 1, 2], n, p=[0.55, 0.4, 0.05])
    weekend = (np.random.rand(n) < 0.3).astype(int)

    vis_effect = np.array([0.6, -0.2, 0.0])[visitor_type]  # returning >> new
    # planted purchase logit: page_value dominates, exit/bounce push away
    z = (0.28 * page_value
         + 0.05 * product_pages
         - 6.0 * exit_rate
         - 2.5 * bounce_rate
         + 1.4 * special_day
         + 0.5 * weekend
         + vis_effect
         - 1.6)
    prob = 1.0 / (1.0 + np.exp(-z))
    y = (np.random.rand(n) < prob).astype(int)  # real Bernoulli purchases

    # model feature matrix: numerics + one-hot visitor type
    X = np.column_stack([
        product_pages, duration, bounce_rate, exit_rate, page_value,
        special_day, weekend, _one_hot(visitor_type, 3),
    ])
    return X, y


def roc_auc(y_true, scores):
    """AUC via the rank-sum (Mann-Whitney U) identity, from scratch."""
    order = np.argsort(scores)
    ranks = np.empty(len(scores), dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1)  # 1-based ranks
    pos = y_true == 1
    n_pos, n_neg = pos.sum(), (~pos).sum()
    if n_pos == 0 or n_neg == 0:
        return 0.5
    return (ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


if __name__ == "__main__":
    np.random.seed(0)

    X, y = make_session_data(4000)

    # held-out train/test split
    idx = np.random.permutation(len(y))
    cut = int(0.75 * len(y))
    tr, te = idx[:cut], idx[cut:]
    Xtr, ytr, Xte, yte = X[tr], y[tr], X[te], y[te]

    model = PurchaseIntentionClassifier(lr=0.2, n_iter=4000, l2=1e-3).fit(Xtr, ytr)
    proba = model.predict_proba(Xte)
    pred = (proba >= 0.5).astype(int)

    acc = np.mean(pred == yte)
    majority = int(round(ytr.mean()))            # predict most common class
    base_acc = np.mean(yte == majority)          # majority-class baseline

    tp = np.sum((pred == 1) & (yte == 1))
    fp = np.sum((pred == 1) & (yte == 0))
    fn = np.sum((pred == 0) & (yte == 1))
    precision = tp / (tp + fp + 1e-9)
    recall = tp / (tp + fn + 1e-9)
    f1 = 2 * precision * recall / (precision + recall + 1e-9)
    auc = roc_auc(yte, proba)

    print("test sessions          :", len(yte))
    print("purchase rate (test)   :", round(float(yte.mean()), 3))
    print("majority baseline acc  :", round(float(base_acc), 3), "(F1 = 0.0)")
    print("model accuracy         :", round(float(acc), 3))
    print("precision / recall     :", round(float(precision), 3), "/", round(float(recall), 3))
    print("model F1 score         :", round(float(f1), 3))
    print("ROC AUC  (0.5=random)  :", round(float(auc), 3))
    print("BEATS baseline         :", bool(acc > base_acc and f1 > 0.4 and auc > 0.5))
