import numpy as np


class DecisionTreeClassifier:
    """CART decision tree grown from scratch using Gini impurity.

    Greedily searches every feature/threshold for the split that most reduces
    Gini impurity, recurses until max_depth / min_samples, and predicts the
    majority class of the leaf a sample falls into.
    """

    def __init__(self, max_depth=4, min_samples=8):
        self.max_depth = max_depth
        self.min_samples = min_samples
        self.tree = None

    @staticmethod
    def _gini(y):
        if len(y) == 0:
            return 0.0
        p = np.bincount(y, minlength=2) / len(y)
        return 1.0 - np.sum(p ** 2)

    def _best_split(self, X, y):
        n, d = X.shape
        parent = self._gini(y)
        best = None
        best_gain = 1e-12
        for f in range(d):
            # candidate thresholds = midpoints between sorted unique values
            vals = np.unique(X[:, f])
            if len(vals) < 2:
                continue
            thresholds = (vals[:-1] + vals[1:]) / 2.0
            for t in thresholds:
                left = X[:, f] <= t
                nl = left.sum()
                if nl < self.min_samples or n - nl < self.min_samples:
                    continue
                gini = (nl * self._gini(y[left]) +
                        (n - nl) * self._gini(y[~left])) / n
                gain = parent - gini
                if gain > best_gain:
                    best_gain = gain
                    best = (f, t)
        return best

    def _build(self, X, y, depth):
        # leaf if pure, too deep, or too few samples to split
        if (depth >= self.max_depth or len(y) < 2 * self.min_samples
                or len(np.unique(y)) == 1):
            return {"leaf": True, "pred": int(round(y.mean()))}
        split = self._best_split(X, y)
        if split is None:
            return {"leaf": True, "pred": int(round(y.mean()))}
        f, t = split
        left = X[:, f] <= t
        return {
            "leaf": False, "feature": f, "thresh": t,
            "left": self._build(X[left], y[left], depth + 1),
            "right": self._build(X[~left], y[~left], depth + 1),
        }

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int).reshape(-1)
        self.tree = self._build(X, y, 0)
        return self

    def _predict_one(self, x, node):
        while not node["leaf"]:
            node = node["left"] if x[node["feature"]] <= node["thresh"] else node["right"]
        return node["pred"]

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        return np.array([self._predict_one(x, self.tree) for x in X])


def make_titanic_data(n=900):
    """Synthetic Titanic passenger manifest with planted survival structure.

    Features: pclass (1/2/3), sex (0=male,1=female), age, fare, family_size.
    Survival follows the historical "women and children first" + class rule:
    women, children, higher class and higher fare survive more often.
    """
    pclass = np.random.choice([1, 2, 3], size=n, p=[0.25, 0.25, 0.5])
    sex = np.random.randint(0, 2, size=n)                 # 0=male, 1=female
    age = np.clip(np.random.normal(30, 14, n), 0.5, 80)
    # fare correlates (inversely) with class, with noise
    fare = np.clip(90 / pclass + np.random.normal(0, 12, n), 5, 200)
    family_size = np.random.poisson(0.9, n)

    # planted survival logit: female++, child++, upper class++, higher fare+
    z = (-1.4
         + 2.6 * sex
         + 1.3 * (age < 12)
         + 0.9 * (pclass == 1) - 0.9 * (pclass == 3)
         + 0.01 * (fare - 40)
         - 0.15 * (family_size >= 4))
    prob = 1.0 / (1.0 + np.exp(-z))
    y = (prob > np.random.rand(n)).astype(int)
    X = np.column_stack([pclass, sex, age, fare, family_size])
    return X, y, ["pclass", "sex", "age", "fare", "family_size"]


if __name__ == "__main__":
    np.random.seed(0)

    X, y, names = make_titanic_data(900)

    # held-out train/test split
    idx = np.random.permutation(len(y))
    cut = int(0.75 * len(y))
    tr, te = idx[:cut], idx[cut:]
    Xtr, ytr, Xte, yte = X[tr], y[tr], X[te], y[te]

    model = DecisionTreeClassifier(max_depth=4, min_samples=10).fit(Xtr, ytr)
    pred = model.predict(Xte)

    acc = np.mean(pred == yte)
    majority = int(round(ytr.mean()))
    base_acc = np.mean(yte == majority)

    tp = np.sum((pred == 1) & (yte == 1))
    fp = np.sum((pred == 1) & (yte == 0))
    fn = np.sum((pred == 0) & (yte == 1))
    precision = tp / (tp + fp + 1e-9)
    recall = tp / (tp + fn + 1e-9)
    f1 = 2 * precision * recall / (precision + recall + 1e-9)

    # recovered survival rates for the classic "women and children first" story
    female = Xte[:, 1] == 1
    child = Xte[:, 2] < 12
    first = Xte[:, 0] == 1

    print("test samples            :", len(yte))
    print("survival rate (test)    :", round(float(yte.mean()), 3))
    print("majority baseline acc   :", round(float(base_acc), 3))
    print("tree accuracy           :", round(float(acc), 3))
    print("precision / recall      :", round(float(precision), 3), "/", round(float(recall), 3))
    print("F1 score                :", round(float(f1), 3))
    print("survival female / male  :", round(float(yte[female].mean()), 3), "/",
          round(float(yte[~female].mean()), 3))
    print("survival child / adult  :", round(float(yte[child].mean()), 3), "/",
          round(float(yte[~child].mean()), 3))
    print("survival 1st / other cls:", round(float(yte[first].mean()), 3), "/",
          round(float(yte[~first].mean()), 3))
    print("BEATS baseline          :", bool(acc > base_acc))
