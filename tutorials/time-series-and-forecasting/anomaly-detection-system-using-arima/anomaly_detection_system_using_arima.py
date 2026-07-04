import numpy as np

# Anomaly Detection System using ARIMA (from scratch)
# ---------------------------------------------------
# Fit an ARIMA(p, d, q) model to a series, then flag time points whose
# one-step-ahead forecast error is far larger than the model's typical error.
#   ARIMA = AR(p) autoregression + I(d) differencing + MA(q) moving average.
# We estimate the ARMA coefficients on the d-times-differenced series with the
# two-stage Hannan-Rissanen least-squares method (no optimizer libraries):
#   stage 1 -- a long AR fit gives an estimate of the innovations (residuals);
#   stage 2 -- regress w_t on p own-lags + q lagged innovations via OLS.
# A well-fit ARIMA leaves ~white-noise residuals, so a genuine anomaly stands
# out as a residual many robust-sigmas from zero. Differencing is linear, so
# the ARIMA one-step error at time t equals the ARMA error on the diff series.


def difference(y, d):
    for _ in range(d):
        y = np.diff(y)
    return y


class ARIMA:
    """From-scratch ARIMA(p, d, q) via Hannan-Rissanen least squares."""

    def __init__(self, p=2, d=1, q=1, ar_order=12):
        self.p, self.d, self.q = p, d, q
        self.ar_order = ar_order            # order of the stage-1 long AR

    @staticmethod
    def _ols(X, y):
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        return beta

    @staticmethod
    def _design(w, resid, p, q, start):
        """Rows [w_{t-1..t-p}, e_{t-1..t-q}] and targets w_t for t>=start."""
        X, target = [], []
        for t in range(start, len(w)):
            row = [w[t - 1 - i] for i in range(p)] + [resid[t - 1 - j] for j in range(q)]
            X.append(row)
            target.append(w[t])
        return np.asarray(X, float), np.asarray(target, float)

    def fit(self, y):
        y = np.asarray(y, float)
        w = difference(y, self.d)
        self.mu_ = w.mean()
        wc = w - self.mu_                    # work on the centered stationary series

        # --- stage 1: long AR(m) to approximate the unobserved innovations ---
        m = self.ar_order
        Xa, ya = self._design(wc, np.zeros_like(wc), m, 0, m)
        phi = self._ols(Xa, ya)
        resid = np.zeros_like(wc)
        resid[m:] = ya - Xa @ phi

        # --- stage 2: OLS of w_t on p own-lags + q lagged innovations ---
        start = max(self.p, self.q, m)
        X, target = self._design(wc, resid, self.p, self.q, start)
        coef = self._ols(X, target)
        self.ar_, self.ma_ = coef[:self.p], coef[self.p:]

        # --- in-sample one-step residuals, mapped back to original time ---
        e = np.zeros_like(wc)
        e[start:] = target - X @ coef
        self.resid_ = np.zeros(len(y))
        self.resid_[start + self.d:] = e[start:]   # ARIMA error(t) == ARMA error(t-d)
        self.valid_from_ = start + self.d
        return self

    def anomaly_scores(self):
        """Robust |residual| in MAD-sigma units (outlier-resistant scale)."""
        v = self.valid_from_
        seg = self.resid_[v:]
        med = np.median(seg)
        sigma = 1.4826 * np.median(np.abs(seg - med)) + 1e-9
        scores = np.zeros_like(self.resid_)
        scores[v:] = np.abs(seg - med) / sigma
        return scores

    def detect(self, k=3.5):
        scores = self.anomaly_scores()
        return np.where(scores > k)[0], scores


# ---------------------------- evaluation helpers ----------------------------
def match_metrics(true_idx, flags, tol=1):
    """Precision/recall/F1 with a +/-tol match window (diff spreads a spike)."""
    true_idx, flags = np.asarray(true_idx), np.asarray(flags)
    if len(flags) == 0:
        return 0.0, 0.0, 0.0
    tp = sum(any(abs(f - i) <= tol for i in true_idx) for f in flags)
    precision = tp / len(flags)
    detected = sum(any(abs(f - i) <= tol for f in flags) for i in true_idx)
    recall = detected / len(true_idx)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def random_baseline(true_idx, n_flags, n, tol=1, trials=300, seed=1):
    rng = np.random.RandomState(seed)
    ps, rs, fs = [], [], []
    for _ in range(trials):
        flags = rng.choice(n, size=n_flags, replace=False)
        p, r, f = match_metrics(true_idx, flags, tol)
        ps.append(p); rs.append(r); fs.append(f)
    return np.mean(ps), np.mean(rs), np.mean(fs)


def auc(labels, scores):
    """Rank-based ROC AUC (Mann-Whitney): P(score[pos] > score[neg])."""
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores)); ranks[order] = np.arange(1, len(scores) + 1)
    pos = labels == 1
    n_pos, n_neg = int(pos.sum()), int((~pos).sum())
    return (ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


if __name__ == "__main__":
    np.random.seed(0)

    # --- synthetic ARIMA(1,1,1) series: drifting integrated ARMA process ---
    n = 320
    phi, theta, s, drift = 0.5, 0.3, 1.0, 0.20
    innov = np.random.normal(0, s, n)                 # white-noise innovations
    e = np.zeros(n)
    for t in range(1, n):
        e[t] = phi * e[t - 1] + innov[t] + theta * innov[t - 1]   # ARMA(1,1)
    y = np.cumsum(e) + drift * np.arange(n) + 50.0    # integrate once -> I(1) + drift

    # --- plant additive anomalies (spikes) at KNOWN indices ---
    n_anom = 12
    true_idx = np.sort(np.random.choice(np.arange(20, n - 2), size=n_anom, replace=False))
    spikes = np.random.choice([-1.0, 1.0], n_anom) * np.random.uniform(6, 9, n_anom)
    y[true_idx] += spikes

    # --- fit ARIMA from scratch and detect anomalies from its residuals ---
    model = ARIMA(p=2, d=1, q=1, ar_order=12).fit(y)
    flags, scores = model.detect(k=3.5)

    prec, rec, f1 = match_metrics(true_idx, flags, tol=1)
    labels = np.zeros(n); labels[true_idx] = 1
    roc = auc(labels[model.valid_from_:], scores[model.valid_from_:])

    r_prec, r_rec, r_f1 = random_baseline(true_idx, len(flags), n, tol=1)

    print("Anomaly Detection with a from-scratch ARIMA(2,1,1)")
    print("-" * 60)
    print("series length=%d   planted anomalies=%d   flagged=%d"
          % (n, n_anom, len(flags)))
    print("fitted AR coef=%s  MA coef=%s"
          % (np.round(model.ar_, 3), np.round(model.ma_, 3)))
    print("-" * 60)
    print("ARIMA detector   Precision=%.3f  Recall=%.3f  F1=%.3f  AUC=%.3f"
          % (prec, rec, f1, roc))
    print("Random detector  Precision=%.3f  Recall=%.3f  F1=%.3f  AUC=0.500"
          % (r_prec, r_rec, r_f1))
    print("-" * 60)
    print("F1 lift over random: %.3f -> %.3f" % (r_f1, f1))
    print("RESULT:", "PASS - beats random baseline"
          if (f1 > r_f1 + 0.3 and roc > 0.8) else "FAIL")
