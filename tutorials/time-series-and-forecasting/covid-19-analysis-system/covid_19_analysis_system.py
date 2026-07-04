import numpy as np

# COVID-19 Analysis System (from scratch)
# ---------------------------------------
# An epidemic wave has an S-shaped CUMULATIVE curve, so daily new cases follow a
# bell shape (rise -> peak -> decline). We fit a logistic growth curve
#     C(t) = L / (1 + exp(-k (t - t0)))            (L=final size, k=rate, t0=peak)
# to the cumulative count via Levenberg-Marquardt (Gauss-Newton + damping), all
# by hand. From the fit we forecast future DAILY NEW cases and, as extra
# analysis, report a 7-day smoothed curve and an Rt (reproduction) estimate.


class Covid19AnalysisSystem:
    """Fit a logistic epidemic curve to cumulative cases and forecast new cases."""

    def __init__(self, n_iter=200, lam=1e-2):
        self.n_iter, self.lam0 = n_iter, lam

    def _model(self, t, p):
        # Logistic curve in normalized (t, C) units; p = [L, k, t0].
        L, k, t0 = p
        return L / (1.0 + np.exp(-k * (t - t0)))

    def _jacobian(self, t, p):
        # Numeric Jacobian of the model w.r.t. each parameter (central differences).
        J = np.zeros((len(t), 3))
        for i in range(3):
            step = 1e-5 * (abs(p[i]) + 1e-6)
            dp = np.zeros(3); dp[i] = step
            J[:, i] = (self._model(t, p + dp) - self._model(t, p - dp)) / (2 * step)
        return J

    def fit(self, cumulative):
        C = np.asarray(cumulative, float)
        self.n_train = len(C)
        # Scale t and C to O(1) for a well-conditioned nonlinear least-squares.
        self.Cs, self.ts = C.max(), float(len(C))
        y, tn = C / self.Cs, np.arange(len(C)) / self.ts
        # Init: asymptote just above current total, midpoint at steepest daily rise.
        t0 = (np.argmax(np.diff(C)) + 1) / self.ts if len(C) > 1 else 0.5
        p = np.array([1.3, 12.0, t0])
        lam = self.lam0
        r = y - self._model(tn, p); cost = r @ r
        for _ in range(self.n_iter):
            J = self._jacobian(tn, p)
            JTJ, g = J.T @ J, J.T @ r
            A = JTJ + lam * np.diag(np.diag(JTJ) + 1e-12)   # LM damping
            try:
                dp = np.linalg.solve(A, g)
            except np.linalg.LinAlgError:
                break
            p_new = p + dp
            r_new = y - self._model(tn, p_new); cost_new = r_new @ r_new
            if cost_new < cost:                              # accept -> less damping
                p, r, cost, lam = p_new, r_new, cost_new, max(lam * 0.7, 1e-9)
            else:                                            # reject -> more damping
                lam = min(lam * 2.5, 1e9)
        self.p_ = p
        # Recovered parameters back in ORIGINAL units.
        self.L_ = p[0] * self.Cs
        self.k_ = p[1] / self.ts
        self.t0_ = p[2] * self.ts
        return self

    def predict_cumulative(self, t):
        # Predicted cumulative cases at raw day index/indices t.
        t = np.asarray(t, float)
        return self._model(t / self.ts, self.p_) * self.Cs

    def forecast_new_cases(self, h):
        # Forecast h days of NEW cases past training end (new_j = C(j) - C(j-1)).
        j = np.arange(self.n_train, self.n_train + h)
        return self.predict_cumulative(j) - self.predict_cumulative(j - 1)

    @staticmethod
    def moving_average(x, w=7):
        # 7-day trailing average to smooth weekday reporting noise.
        x = np.asarray(x, float); out = np.full(len(x), np.nan)
        for i in range(w - 1, len(x)):
            out[i] = x[i - w + 1:i + 1].mean()
        return out

    @staticmethod
    def estimate_rt(new_cases, w=7):
        # Simple Rt proxy: cases in last window / cases in the prior window.
        s = np.asarray(new_cases, float)
        if len(s) < 2 * w:
            return np.nan
        return s[-w:].sum() / max(s[-2 * w:-w].sum(), 1e-9)


def make_epidemic(n=180, seed=0):
    """Synthetic COVID-like wave: logistic cumulative -> bell-shaped daily cases
    with weekday-reporting seasonality and noise."""
    np.random.seed(seed)
    t = np.arange(n)
    L, k, t0 = 1.0e5, 0.09, 85.0                 # planted latent structure
    cum_true = L / (1.0 + np.exp(-k * (t - t0)))
    new_true = np.diff(np.concatenate([[0.0], cum_true]))   # daily new (bell)
    weekday = 1.0 + 0.25 * np.sin(2 * np.pi * t / 7.0)      # reporting seasonality
    noise = np.random.normal(1.0, 0.06, size=n)
    new_obs = np.clip(new_true * weekday * noise, 0, None)
    return np.cumsum(new_obs), new_obs


def rmse(a, b): return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))
def mae(a, b):  return float(np.mean(np.abs(np.asarray(a) - np.asarray(b))))


if __name__ == "__main__":
    np.random.seed(0)
    cumulative, new_cases = make_epidemic(n=180)

    h = 21                                        # forecast the held-out tail
    split = len(cumulative) - h
    train_cum, test_new = cumulative[:split], new_cases[split:]

    sys = Covid19AnalysisSystem().fit(train_cum)
    pred_new = sys.forecast_new_cases(h)

    # Baselines for daily-new-case forecasting on the held-out tail.
    naive = np.full(h, new_cases[split - 1])      # carry-forward last observed day
    mean_base = np.full(h, new_cases[:split].mean())

    print("COVID-19 Analysis System - logistic epidemic fit (from scratch)")
    print("-" * 64)
    print("days=%d  train=%d  held-out tail=%d" % (len(cumulative), split, h))
    print("recovered  L=%8.0f  k=%.4f  t0=%.1f   (true L=100000 k=0.0900 t0=85.0)"
          % (sys.L_, sys.k_, sys.t0_))
    print("latest Rt estimate (train): %.3f  (>1 growing, <1 declining)"
          % sys.estimate_rt(new_cases[:split]))
    print("-" * 64)
    print("Naive (last value)  RMSE=%9.1f  MAE=%9.1f" % (rmse(test_new, naive), mae(test_new, naive)))
    print("Mean baseline       RMSE=%9.1f  MAE=%9.1f" % (rmse(test_new, mean_base), mae(test_new, mean_base)))
    print("Logistic forecast   RMSE=%9.1f  MAE=%9.1f" % (rmse(test_new, pred_new), mae(test_new, pred_new)))
    print("-" * 64)
    best_base = min(rmse(test_new, naive), rmse(test_new, mean_base))
    model_rmse = rmse(test_new, pred_new)
    print("best baseline RMSE=%9.1f -> logistic RMSE=%9.1f  (%.1f%% lower)"
          % (best_base, model_rmse, 100.0 * (best_base - model_rmse) / best_base))
    print("sample forecasts vs truth (held-out tail):")
    for j in range(0, h, max(1, h // 6)):
        print("  day %3d  pred=%8.0f  true=%8.0f  naive=%8.0f"
              % (split + j, pred_new[j], test_new[j], naive[j]))
    print("RESULT:", "PASS - beats baseline" if model_rmse < best_base else "FAIL")
