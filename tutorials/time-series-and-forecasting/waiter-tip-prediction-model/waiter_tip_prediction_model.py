import numpy as np

# Waiter Tip Prediction as regression-over-time, from scratch.
# We simulate one waiter's tables served across a season, ordered in time.
# Each table carries a total_bill, party size, day-of-week, and lunch/dinner
# shift. The tip is mostly a slice of the bill, nudged up on weekends (more
# generous crowds), at dinner, by larger parties, plus a slow upward TREND
# (the waiter's rising reputation) and random noise. We fit a ridge linear
# regression on those features via the normal equations and forecast the tip
# on a held-out TIME TAIL, beating predict-mean and last-value baselines.

DAYS = ["Thu", "Fri", "Sat", "Sun"]


def make_tips(T=480, seed=0):
    # Synthetic table-by-table log with planted, recoverable structure ($).
    rng = np.random.RandomState(seed)
    t = np.arange(T)
    day = rng.randint(0, 4, T)                     # 0=Thu 1=Fri 2=Sat 3=Sun
    time = (rng.rand(T) < 0.6).astype(float)       # 1=dinner (more frequent)
    size = rng.randint(1, 6, T) + 1                # party of 2..6 people
    # Bigger parties and dinner runs push the bill up; clip to a sane minimum.
    bill = 15.0 * size + 12.0 * time + rng.normal(0, 6.0, T)
    bill = np.clip(bill, 6.0, None)

    day_bonus = np.array([0.0, 0.4, 1.1, 0.9])     # weekend crowds tip extra
    trend = 0.0025 * t                             # reputation lifts tips slowly
    tip = (0.15 * bill                             # ~15% of the bill (main driver)
           + 0.25 * size                           # a little per head
           + 0.80 * time                           # dinner beats lunch
           + day_bonus[day]                        # day-of-week effect
           + trend                                 # slow upward drift
           + rng.normal(0, 0.5, T))                # irreducible noise
    return dict(t=t, bill=bill, size=size.astype(float),
                time=time, day=day, tip=tip)


class TipForecaster:
    """Ridge regression: tip ~ c + a*bill + b*size + d*dinner + day(one-hot) + g*t."""

    def __init__(self, lam=1e-2):
        self.lam = lam

    def _design(self, d, idx):
        oh = np.eye(4)[d["day"][idx]][:, 1:]       # Thu is the reference day
        return np.column_stack([
            np.ones(len(idx)),
            d["bill"][idx], d["size"][idx], d["time"][idx],
            oh, d["t"][idx].astype(float),
        ])

    def fit(self, d, idx):
        Z = self._design(d, idx)
        R = self.lam * np.eye(Z.shape[1])
        R[0, 0] = 0.0                              # don't penalize the intercept
        self.w = np.linalg.solve(Z.T @ Z + R, Z.T @ d["tip"][idx])
        return self

    def predict(self, d, idx):
        return self._design(d, idx) @ self.w


def rmse(a, b):
    return np.sqrt(np.mean((a - b) ** 2))


def mae(a, b):
    return np.mean(np.abs(a - b))


if __name__ == "__main__":
    np.random.seed(0)

    d = make_tips(T=480, seed=0)
    split = 384                                    # held-out tail = last 96 tables
    tr = np.arange(1, split)                       # start at 1 (last-value needs t-1)
    te = np.arange(split, len(d["t"]))

    model = TipForecaster(lam=1e-2).fit(d, tr)

    pred = model.predict(d, te)
    true = d["tip"][te]
    mean_base = np.full(len(te), d["tip"][tr].mean())   # predict-mean baseline
    last_base = d["tip"][te - 1]                         # last-value (persistence)

    print("Waiter tips: %d tables   train=%d  test=%d   (tip in $)"
          % (len(d["t"]), len(tr), len(te)))
    print("-" * 60)
    print("Regression (bill+size+shift+day+trend) RMSE: %6.3f  MAE: %6.3f"
          % (rmse(pred, true), mae(pred, true)))
    print("Predict-mean baseline                  RMSE: %6.3f  MAE: %6.3f"
          % (rmse(mean_base, true), mae(mean_base, true)))
    print("Last-value (persistence) baseline      RMSE: %6.3f  MAE: %6.3f"
          % (rmse(last_base, true), mae(last_base, true)))
    print("-" * 60)

    names = ["intercept", "bill", "size", "dinner", "Fri", "Sat", "Sun", "trend"]
    print("Recovered coefficients (true bill~0.150, size~0.250, dinner~0.800):")
    for nm, wv in zip(names, model.w):
        print("   %-9s % .4f" % (nm, wv))
    print("-" * 60)

    print("Sample held-out predictions (table: pred vs true tip):")
    for k in te[:6]:
        print("   #%3d  bill=%5.1f size=%d %-6s pred=$%5.2f  true=$%5.2f"
              % (k, d["bill"][k], int(d["size"][k]),
                 DAYS[d["day"][k]], model.predict(d, np.array([k]))[0], d["tip"][k]))
    print("-" * 60)
    beats = rmse(pred, true) < min(rmse(mean_base, true), rmse(last_base, true))
    print("Regression beats predict-mean & last-value baselines: %s" % beats)
