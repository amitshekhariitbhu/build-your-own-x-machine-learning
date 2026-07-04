import numpy as np

# A gender + age "detection" system on synthetic facial-descriptor features.
# One shared hidden layer feeds two heads: a gender classifier (sigmoid + BCE)
# and an age regressor (linear + MSE). Trained jointly by manual backprop, so
# the shared representation must serve both tasks at once (multi-task learning).
class GenderAgeNet:
    def __init__(self, hidden=16, lr=0.05, epochs=400, age_weight=1.0):
        self.hidden = hidden
        self.lr = lr
        self.epochs = epochs
        self.age_weight = age_weight  # trade-off between the two losses

    @staticmethod
    def _sigmoid(z):
        return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))

    def fit(self, X, g, a):
        n, d = X.shape
        # Standardize inputs and age target (kept for de-normalizing later).
        self.x_mu, self.x_sd = X.mean(0), X.std(0) + 1e-8
        self.a_mu, self.a_sd = a.mean(), a.std() + 1e-8
        Xn = (X - self.x_mu) / self.x_sd
        an = ((a - self.a_mu) / self.a_sd).reshape(-1, 1)
        gv = g.reshape(-1, 1).astype(float)

        rng = np.random.randn
        h = self.hidden
        self.W1 = rng(d, h) * 0.3   # shared layer
        self.b1 = np.zeros(h)
        self.Wg = rng(h, 1) * 0.3   # gender head
        self.bg = 0.0
        self.Wa = rng(h, 1) * 0.3   # age head
        self.ba = 0.0

        for _ in range(self.epochs):
            # Forward pass.
            Z1 = Xn @ self.W1 + self.b1
            H = np.maximum(0, Z1)              # ReLU shared features
            pg = self._sigmoid(H @ self.Wg + self.bg)   # gender probability
            ya = H @ self.Wa + self.ba                  # normalized age

            # Gradients of (mean BCE) + age_weight * (mean MSE) w.r.t. logits.
            dzg = (pg - gv) / n
            dya = self.age_weight * 2.0 * (ya - an) / n

            dWg = H.T @ dzg
            dbg = dzg.sum()
            dWa = H.T @ dya
            dba = dya.sum()

            # Backprop into the shared layer through both heads.
            dH = dzg @ self.Wg.T + dya @ self.Wa.T
            dZ1 = dH * (Z1 > 0)
            dW1 = Xn.T @ dZ1
            db1 = dZ1.sum(0)

            for p, gp in [(self.W1, dW1), (self.b1, db1), (self.Wg, dWg),
                          (self.Wa, dWa)]:
                p -= self.lr * gp
            self.bg -= self.lr * dbg
            self.ba -= self.lr * dba
        return self

    def _features(self, X):
        Xn = (X - self.x_mu) / self.x_sd
        return np.maximum(0, Xn @ self.W1 + self.b1)

    def predict(self, X):
        H = self._features(X)
        gender = (self._sigmoid(H @ self.Wg + self.bg).ravel() >= 0.5).astype(int)
        age = (H @ self.Wa + self.ba).ravel() * self.a_sd + self.a_mu
        return gender, age


def make_faces(n=700):
    # Latent identity: gender in {0,1} and a real age. Features mimic facial
    # descriptors driven by gender, by age, by both, plus one pure distractor.
    g = (np.random.rand(n) < 0.5).astype(int)
    a = np.random.uniform(18, 70, n)
    noise = lambda s: np.random.randn(n) * s
    f0 = 1.6 * g + 0.015 * a + noise(0.4)            # gender-dominant
    f1 = -1.3 * g + noise(0.5)                        # gender-dominant
    f2 = 0.05 * a - 1.0 + noise(0.4)                  # age-dominant
    f3 = 0.001 * (a - 45) ** 2 + noise(0.4)           # nonlinear age cue
    f4 = 0.04 * a + 0.9 * g + noise(0.5)              # mixed
    f5 = 0.8 * g - 0.012 * a + noise(0.4)             # mixed
    f6 = 0.03 * a + noise(0.3)                        # age-dominant
    f7 = noise(1.0)                                   # distractor
    X = np.column_stack([f0, f1, f2, f3, f4, f5, f6, f7])
    return X, g, a


if __name__ == "__main__":
    np.random.seed(0)
    X, g, a = make_faces(700)
    perm = np.random.permutation(len(g))
    X, g, a = X[perm], g[perm], a[perm]
    cut = int(0.7 * len(g))
    Xtr, Xte = X[:cut], X[cut:]
    gtr, gte = g[:cut], g[cut:]
    atr, ate = a[:cut], a[cut:]

    net = GenderAgeNet(hidden=16, lr=0.05, epochs=400).fit(Xtr, gtr, atr)
    g_pred, a_pred = net.predict(Xte)

    # Gender: accuracy vs majority-class baseline.
    gender_acc = np.mean(g_pred == gte)
    majority = max(gte.mean(), 1 - gte.mean())
    # Age: RMSE vs predict-the-mean baseline (uses the training mean).
    rmse = lambda p, t: np.sqrt(np.mean((p - t) ** 2))
    age_rmse = rmse(a_pred, ate)
    base_rmse = rmse(np.full_like(ate, atr.mean()), ate)

    print("Test faces               :", len(gte))
    print("Gender majority baseline : {:.3f}".format(majority))
    print("Gender model accuracy    : {:.3f}".format(gender_acc))
    print("Age mean-predictor RMSE  : {:.2f} years".format(base_rmse))
    print("Age model RMSE           : {:.2f} years".format(age_rmse))
    print("Gender beats majority    :", bool(gender_acc > majority))
    print("Age beats mean-predictor :", bool(age_rmse < base_rmse))
