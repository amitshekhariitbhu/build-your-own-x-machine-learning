import numpy as np


class GradeRegressor:
    """Ridge linear-regression student-grade predictor trained from scratch.

    Standardizes the input features, learns weights + bias by full-batch
    gradient descent on the L2-regularized mean-squared-error loss, and
    predicts the final exam grade (0-100). No ML libraries -- just numpy.
    """

    def __init__(self, lr=0.1, n_iter=3000, l2=1e-3):
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
        self.b = float(y.mean())  # start bias at the mean grade
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
        return np.clip(Xs @ self.w + self.b, 0.0, 100.0)


def make_student_data(n=800):
    """Synthetic student records with a planted final-grade signal.

    Features (academic inspired):
      study_hours/week, attendance %, past_gpa, sleep_hours,
      absences, tutoring (0/1), screen_time hours.
    A true rule sets the grade: studying, attendance and prior GPA lift it,
    absences and excessive screen time drag it down, tutoring gives a bump,
    plus noise so the mapping is learnable but not perfectly fit.
    """
    study = np.random.uniform(0, 25, n)          # hours studied / week
    attendance = np.random.uniform(50, 100, n)   # class attendance %
    past_gpa = np.random.uniform(2.0, 4.0, n)    # prior GPA
    sleep = np.random.normal(7, 1.2, n)          # nightly sleep hours
    absences = np.random.randint(0, 20, n)       # days absent
    tutoring = np.random.binomial(1, 0.3, n)     # 1 = attends tutoring
    screen = np.random.uniform(0, 8, n)          # recreational screen hours

    X = np.column_stack([study, attendance, past_gpa, sleep,
                         absences, tutoring, screen])

    # planted grade: prior GPA and study time dominate, attendance helps,
    # too little/much sleep and heavy screen time hurt, absences penalize.
    grade = (18.0
             + 1.35 * study
             + 0.30 * attendance
             + 9.0 * past_gpa
             - 2.0 * np.abs(sleep - 7.5)      # sweet-spot around 7.5h
             - 0.9 * absences
             + 4.0 * tutoring
             - 1.1 * screen)
    grade += np.random.normal(0, 3.0, n)         # exam-day noise
    grade = np.clip(grade, 0.0, 100.0)
    return X, grade


if __name__ == "__main__":
    np.random.seed(0)

    X, y = make_student_data(800)

    # held-out train/test split
    idx = np.random.permutation(len(y))
    cut = int(0.75 * len(y))
    tr, te = idx[:cut], idx[cut:]
    Xtr, ytr, Xte, yte = X[tr], y[tr], X[te], y[te]

    model = GradeRegressor(lr=0.1, n_iter=3000, l2=1e-3).fit(Xtr, ytr)
    pred = model.predict(Xte)

    # model error
    rmse = np.sqrt(np.mean((pred - yte) ** 2))
    mae = np.mean(np.abs(pred - yte))

    # mean-grade baseline (predict the training mean for everyone)
    base_pred = ytr.mean()
    base_rmse = np.sqrt(np.mean((base_pred - yte) ** 2))
    base_mae = np.mean(np.abs(base_pred - yte))

    # R^2 (fraction of grade variance explained on held-out data)
    ss_res = np.sum((yte - pred) ** 2)
    ss_tot = np.sum((yte - yte.mean()) ** 2)
    r2 = 1.0 - ss_res / ss_tot

    # pass/fail (>=60) accuracy as an interpretable classification check
    pass_acc = np.mean((pred >= 60) == (yte >= 60))
    maj = max(np.mean(yte >= 60), np.mean(yte < 60))  # majority-class rate

    feats = ["study", "attendance", "past_gpa", "sleep",
             "absences", "tutoring", "screen"]

    print("test samples          :", len(yte))
    print("mean grade (test)     :", round(float(yte.mean()), 1))
    print("baseline RMSE (mean)  :", round(float(base_rmse), 2))
    print("model RMSE            :", round(float(rmse), 2))
    print("baseline MAE (mean)   :", round(float(base_mae), 2))
    print("model MAE             :", round(float(mae), 2))
    print("R^2 (held-out)        :", round(float(r2), 3))
    print("pass/fail majority    :", round(float(maj), 3))
    print("pass/fail accuracy    :", round(float(pass_acc), 3))
    print("feature weights (std) :", dict(zip(feats, np.round(model.w, 2))))
    print("BEATS baseline        :", bool(rmse < base_rmse))
