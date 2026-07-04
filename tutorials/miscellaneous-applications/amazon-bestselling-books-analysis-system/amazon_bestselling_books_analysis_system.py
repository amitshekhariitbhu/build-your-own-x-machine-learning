import numpy as np


class BooksAnalyzer:
    """From-scratch logistic-regression analysis of Amazon bestsellers.

    The core task: from a book's catalog features (price, review volume,
    rating, page count, publication year) decide whether it is Fiction or
    Non-Fiction. Features are standardized, then a linear model is trained
    with full-batch gradient descent on the L2-regularized log-loss:
        p = sigmoid(Xw + b),  grad = X^T (p - y) / n + l2 * w
    predict_proba() / predict() classify held-out books, and the learned
    weights double as an "analysis" of which features drive each genre.
    """

    def __init__(self, lr=0.3, l2=1e-3, epochs=400):
        self.lr = lr           # gradient-descent step size
        self.l2 = l2           # ridge penalty on weights (not bias)
        self.epochs = epochs
        self.w = None
        self.b = 0.0
        self.mu = None
        self.sigma = None

    @staticmethod
    def _sigmoid(z):
        return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))

    def _standardize(self, X, fit=False):
        if fit:
            self.mu = X.mean(axis=0)
            self.sigma = X.std(axis=0) + 1e-8
        return (X - self.mu) / self.sigma

    def fit(self, X, y):
        X = self._standardize(np.asarray(X, dtype=float), fit=True)
        y = np.asarray(y, dtype=float).reshape(-1)
        n, d = X.shape
        self.w = np.zeros(d)
        self.b = 0.0
        for _ in range(self.epochs):
            p = self._sigmoid(X @ self.w + self.b)
            err = p - y
            self.w -= self.lr * (X.T @ err / n + self.l2 * self.w)
            self.b -= self.lr * err.mean()
        return self

    def predict_proba(self, X):
        X = self._standardize(np.asarray(X, dtype=float))
        return self._sigmoid(X @ self.w + self.b)

    def predict(self, X):
        return (self.predict_proba(X) >= 0.5).astype(int)


def make_books_data(n=800, seed=0):
    """Synthetic Amazon bestseller catalog with a planted genre signal.

    Each book is Non-Fiction (label 1) or Fiction (label 0). Genre drives
    the feature means with overlap + noise, so the boundary is learnable but
    not trivial:
      price        - Non-Fiction (textbooks/business) costs more
      log_reviews  - Fiction blockbusters rack up more reviews
      rating       - both high (bestsellers), Fiction slightly higher
      pages        - Non-Fiction runs longer
      year         - mild recency skew
    Returns (X, y, genre_names).
    """
    rng = np.random.RandomState(seed)
    y = rng.randint(0, 2, n)                    # 1 = Non-Fiction, 0 = Fiction
    nf = y == 1

    price = np.where(nf, rng.normal(18, 5, n), rng.normal(11, 4, n))
    price = np.clip(price, 2, 60)
    log_reviews = np.where(nf, rng.normal(8.2, 0.9, n),
                               rng.normal(9.1, 0.9, n))      # ln(#reviews)
    rating = np.where(nf, rng.normal(4.5, 0.25, n),
                          rng.normal(4.6, 0.22, n))
    rating = np.clip(rating, 3.0, 5.0)
    pages = np.where(nf, rng.normal(360, 90, n), rng.normal(300, 80, n))
    pages = np.clip(pages, 60, 900)
    year = rng.randint(2009, 2020, n).astype(float)

    X = np.column_stack([price, log_reviews, rating, pages, year])
    return X, y, ("Fiction", "Non-Fiction")


def metrics(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    acc = float((y_true == y_pred).mean())
    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return acc, prec, rec, f1


if __name__ == "__main__":
    np.random.seed(0)

    X, y, names = make_books_data(n=800, seed=0)
    feat_names = ["price", "log_reviews", "rating", "pages", "year"]

    n_train = int(0.7 * len(y))
    Xtr, Xte = X[:n_train], X[n_train:]
    ytr, yte = y[:n_train], y[n_train:]

    model = BooksAnalyzer(lr=0.3, l2=1e-3, epochs=400).fit(Xtr, ytr)
    pred = model.predict(Xte)
    acc, prec, rec, f1 = metrics(yte, pred)

    # Baseline: always guess the majority genre in the training set.
    majority = int(round(ytr.mean()))
    base_pred = np.full_like(yte, majority)
    base_acc, _, _, base_f1 = metrics(yte, base_pred)

    print("=== Amazon Bestselling Books Analysis System (from scratch) ===")
    print(f"books: {len(y)}  (train {n_train} / test {len(yte)})   "
          f"classes: {names}")
    print(f"majority baseline accuracy : {base_acc:.3f}  (F1 {base_f1:.3f})")
    print(f"logistic model  accuracy   : {acc:.3f}  (F1 {f1:.3f})")
    print(f"  precision {prec:.3f}  recall {rec:.3f}")
    print(f"improvement over baseline  : +{(acc - base_acc) * 100:4.1f} accuracy pts")
    order = np.argsort(-np.abs(model.w))
    drivers = ", ".join(f"{feat_names[i]}({model.w[i]:+.2f})" for i in order[:3])
    print(f"top genre drivers (weights): {drivers}")

    assert acc > base_acc + 0.10, "model must clearly beat majority baseline"
    assert f1 > 0.6, "model must have a healthy F1"
    print("PASS: genre classifier beats majority baseline on held-out books")
