import numpy as np

def sigmoid(z):
    # Sigmoid activation, maps values to (0, 1); gate outputs live here.
    return 1 / (1 + np.exp(-z))

class LSTMForecaster:
    """Many-to-one LSTM for one-step time-series forecasting (manual forward + BPTT)."""
    def __init__(self, input_dim=1, hidden_dim=24, lr=0.01, seed=0):
        rng = np.random.RandomState(seed)
        H, Z = hidden_dim, input_dim + hidden_dim  # gates read concat[x_t, h_{t-1}]
        s = 1.0 / np.sqrt(Z)
        self.D, self.H, self.lr = input_dim, hidden_dim, lr
        self.p = {
            'Wf': rng.randn(Z, H) * s, 'bf': np.ones(H),   # forget-bias=1 aids gradient flow
            'Wi': rng.randn(Z, H) * s, 'bi': np.zeros(H),
            'Wg': rng.randn(Z, H) * s, 'bg': np.zeros(H),
            'Wo': rng.randn(Z, H) * s, 'bo': np.zeros(H),
            'Wy': rng.randn(H, 1) * s, 'by': np.zeros(1),
        }
        self.m = {k: np.zeros_like(v) for k, v in self.p.items()}  # Adam moments
        self.v = {k: np.zeros_like(v) for k, v in self.p.items()}
        self.t = 0

    def forward(self, X):
        # X: (N, T, D). Roll the cell over T steps; predict from the final hidden state.
        N, T, _ = X.shape
        H, p = self.H, self.p
        h, c = np.zeros((N, H)), np.zeros((N, H))
        cache = []
        for t in range(T):
            z = np.concatenate([X[:, t, :], h], axis=1)   # (N, D+H)
            f = sigmoid(z @ p['Wf'] + p['bf'])            # forget gate
            i = sigmoid(z @ p['Wi'] + p['bi'])            # input gate
            g = np.tanh(z @ p['Wg'] + p['bg'])            # candidate cell
            o = sigmoid(z @ p['Wo'] + p['bo'])            # output gate
            c_prev = c
            c = f * c_prev + i * g                        # cell update
            tc = np.tanh(c)
            h = o * tc                                    # hidden state
            cache.append((z, f, i, g, o, c_prev, tc))
        y = h @ p['Wy'] + p['by']
        return y, (cache, h)

    def backward(self, X, y_pred, y_true, packed):
        # Backprop-through-time for MSE loss taken at the final step.
        cache, h_last = packed
        N, T, D = X.shape
        H, p = self.H, self.p
        g = {k: np.zeros_like(v) for k, v in p.items()}
        dy = (y_pred - y_true) * (2.0 / N)               # dMSE/dy
        g['Wy'] = h_last.T @ dy
        g['by'] = dy.sum(axis=0)
        dh_next = dy @ p['Wy'].T
        dc_next = np.zeros((N, H))
        for t in reversed(range(T)):
            z, f, i, gt, o, c_prev, tc = cache[t]
            dh = dh_next
            do = dh * tc
            dc = dc_next + dh * o * (1 - tc ** 2)         # through h = o * tanh(c)
            df, di, dg = dc * c_prev, dc * gt, dc * i
            dc_next = dc * f                              # cell grad to previous step
            fr, ir, gr, orr = df*f*(1-f), di*i*(1-i), dg*(1-gt**2), do*o*(1-o)
            for name, dr in (('f', fr), ('i', ir), ('g', gr), ('o', orr)):
                g['W' + name] += z.T @ dr
                g['b' + name] += dr.sum(axis=0)
            dz = fr @ p['Wf'].T + ir @ p['Wi'].T + gr @ p['Wg'].T + orr @ p['Wo'].T
            dh_next = dz[:, D:]                           # split off grad w.r.t h_{t-1}
        return g

    def step(self, grads, clip=5.0):
        # Adam update with gradient clipping to tame exploding BPTT gradients.
        self.t += 1
        b1, b2, eps = 0.9, 0.999, 1e-8
        for k in self.p:
            gr = np.clip(grads[k], -clip, clip)
            self.m[k] = b1 * self.m[k] + (1 - b1) * gr
            self.v[k] = b2 * self.v[k] + (1 - b2) * gr * gr
            mh = self.m[k] / (1 - b1 ** self.t)
            vh = self.v[k] / (1 - b2 ** self.t)
            self.p[k] -= self.lr * mh / (np.sqrt(vh) + eps)

    def fit(self, X, y, epochs=250):
        for _ in range(epochs):
            y_pred, packed = self.forward(X)
            self.step(self.backward(X, y_pred, y, packed))
        return self

    def predict(self, X):
        return self.forward(X)[0]

def make_series(n=420, seed=0):
    # Synthetic signal: linear trend + two seasonal cycles + small noise (recoverable structure).
    np.random.seed(seed)
    t = np.arange(n)
    trend = 0.03 * t
    seasonal = 3.0 * np.sin(2 * np.pi * t / 25) + 1.2 * np.sin(2 * np.pi * t / 7)
    noise = np.random.randn(n) * 0.3
    return trend + seasonal + noise

def windowize(series, T):
    # Build (N,T,1) windows -> next-value targets; return target index per window.
    X = np.array([series[i:i + T] for i in range(len(series) - T)])[:, :, None]
    y = np.array([series[i + T] for i in range(len(series) - T)])[:, None]
    idx = np.arange(T, len(series))
    return X, y, idx

if __name__ == "__main__":
    np.random.seed(0)
    T, test_len = 20, 60
    series = make_series()
    split = len(series) - test_len                       # forecast the held-out tail

    # Standardize using ONLY training statistics (no leakage from the future).
    mu, sd = series[:split].mean(), series[:split].std()
    scaled = (series - mu) / sd

    X, y, idx = windowize(scaled, T)
    train, test = idx < split, idx >= split              # split windows by target position

    model = LSTMForecaster(hidden_dim=24, lr=0.01, seed=0)
    model.fit(X[train], y[train], epochs=250)

    # One-step-ahead forecasts over the held-out tail (true history in each window).
    pred = model.predict(X[test]) * sd + mu              # back to original units
    true = y[test] * sd + mu
    naive = series[idx[test] - 1][:, None]               # baseline: predict previous value

    rmse = lambda a, b: float(np.sqrt(np.mean((a - b) ** 2)))
    mae = lambda a, b: float(np.mean(np.abs(a - b)))
    lstm_rmse, base_rmse = rmse(pred, true), rmse(naive, true)
    lstm_mae, base_mae = mae(pred, true), mae(naive, true)

    print("LSTM one-step time-series forecasting (manual BPTT)")
    print(f"  series={len(series)}  window={T}  hidden={model.H}  test_tail={test_len}")
    print(f"  naive (last-value) RMSE: {base_rmse:.3f}   MAE: {base_mae:.3f}")
    print(f"  LSTM  forecast     RMSE: {lstm_rmse:.3f}   MAE: {lstm_mae:.3f}")
    print(f"  RMSE improvement over baseline: {100 * (1 - lstm_rmse / base_rmse):.1f}%")
    print(f"  MAE  improvement over baseline: {100 * (1 - lstm_mae / base_mae):.1f}%")
    print(f"  result: {'LSTM BEATS naive baseline' if lstm_rmse < base_rmse else 'baseline wins'}")
    print("  sample forecasts vs truth (held-out tail):")
    for j in range(0, len(true), max(1, len(true) // 6)):
        print(f"    pred={pred[j,0]:7.3f}   true={true[j,0]:7.3f}   naive={naive[j,0]:7.3f}")
