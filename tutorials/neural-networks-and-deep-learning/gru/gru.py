import numpy as np

def sigmoid(z):
    # Sigmoid activation, maps values to (0, 1)
    return 1 / (1 + np.exp(-z))

class GRU:
    """GRU for many-to-one sequence regression, hand-rolled forward + BPTT."""
    def __init__(self, input_dim=1, hidden_dim=16, output_dim=1, lr=0.01, seed=0):
        rng = np.random.RandomState(seed)
        H, Z = hidden_dim, input_dim + hidden_dim  # gates act on concat[x_t, h_{t-1}]
        s = 1.0 / np.sqrt(Z)
        self.D, self.H, self.lr = input_dim, hidden_dim, lr
        self.p = {
            'Wz': rng.randn(Z, H) * s, 'bz': np.zeros(H),  # update gate
            'Wr': rng.randn(Z, H) * s, 'br': np.zeros(H),  # reset gate
            'Wh': rng.randn(Z, H) * s, 'bh': np.zeros(H),  # candidate state
            'Wy': rng.randn(H, output_dim) * s, 'by': np.zeros(output_dim),
        }
        # Adam optimizer state
        self.m = {k: np.zeros_like(v) for k, v in self.p.items()}
        self.v = {k: np.zeros_like(v) for k, v in self.p.items()}
        self.t = 0

    def forward(self, X):
        # X: (N, T, D). Runs the GRU cell over T steps; output from final hidden state.
        N, T, _ = X.shape
        H, p = self.H, self.p
        h = np.zeros((N, H))
        cache = []
        for t in range(T):
            zc = np.concatenate([X[:, t, :], h], axis=1)       # (N, D+H)
            z = sigmoid(zc @ p['Wz'] + p['bz'])                # update gate
            r = sigmoid(zc @ p['Wr'] + p['br'])                # reset gate
            ac = np.concatenate([X[:, t, :], r * h], axis=1)   # reset-masked context
            ht = np.tanh(ac @ p['Wh'] + p['bh'])               # candidate state
            h_prev = h
            h = (1 - z) * h_prev + z * ht                      # interpolate old vs candidate
            cache.append((zc, ac, z, r, ht, h_prev))
        y = h @ p['Wy'] + p['by']
        return y, (cache, h)

    def backward(self, X, y_pred, y_true, packed):
        # Backprop through time for MSE loss applied at the final step.
        cache, h_last = packed
        N, T, D = X.shape
        H, p = self.H, self.p
        g = {k: np.zeros_like(v) for k, v in p.items()}
        dy = (y_pred - y_true) * (2.0 / N)                    # dMSE/dy
        g['Wy'] = h_last.T @ dy
        g['by'] = dy.sum(axis=0)
        dh_next = dy @ p['Wy'].T                              # grad into final hidden state
        for t in reversed(range(T)):
            zc, ac, z, r, ht, h_prev = cache[t]
            dh = dh_next                                      # only recurrence feeds inner steps
            dz = dh * (ht - h_prev)                           # through h interpolation
            dht = dh * z
            dh_prev = dh * (1 - z)                            # direct carry of old state
            # candidate: ht = tanh(ac @ Wh + bh)
            dhr = dht * (1 - ht ** 2)
            g['Wh'] += ac.T @ dhr
            g['bh'] += dhr.sum(axis=0)
            dac = dhr @ p['Wh'].T
            drh = dac[:, D:]                                  # grad w.r.t. r * h_prev
            dr = drh * h_prev
            dh_prev += drh * r
            # update/reset gate pre-activations
            zr = dz * z * (1 - z)
            rr = dr * r * (1 - r)
            g['Wz'] += zc.T @ zr
            g['bz'] += zr.sum(axis=0)
            g['Wr'] += zc.T @ rr
            g['br'] += rr.sum(axis=0)
            dh_prev += (zr @ p['Wz'].T)[:, D:] + (rr @ p['Wr'].T)[:, D:]
            dh_next = dh_prev                                 # grad to previous step
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

    def fit(self, X, y, epochs=300):
        losses = []
        for _ in range(epochs):
            y_pred, packed = self.forward(X)
            losses.append(np.mean((y_pred - y) ** 2))
            self.step(self.backward(X, y_pred, y, packed))
        return losses

    def predict(self, X):
        return self.forward(X)[0]

# Example usage
if __name__ == "__main__":
    np.random.seed(0)
    # Task: predict the next value of a sine wave from a short window (many-to-one).
    T = 20
    t = np.linspace(0, 6 * np.pi, 160)
    wave = np.sin(t)
    X = np.array([wave[i:i + T] for i in range(len(wave) - T)])[:, :, None]  # (N, T, 1)
    Y = np.array([wave[i + T] for i in range(len(wave) - T)])[:, None]       # (N, 1)

    model = GRU(input_dim=1, hidden_dim=16, output_dim=1, lr=0.01, seed=0)
    losses = model.fit(X, Y, epochs=300)

    start, final = losses[0], losses[-1]
    pred = model.predict(X)
    rmse = np.sqrt(np.mean((pred - Y) ** 2))
    print("Sine-wave next-value prediction (GRU, manual BPTT)")
    print(f"  sequences={len(X)}  timesteps={T}  hidden={model.H}")
    print(f"  start MSE : {start:.5f}")
    print(f"  final MSE : {final:.5f}")
    print(f"  reduction : {100 * (1 - final / start):.1f}%")
    print(f"  final RMSE: {rmse:.5f}  (targets lie in [-1, 1])")
    print("  sample predictions vs truth:")
    for j in range(0, len(X), len(X) // 5):
        print(f"    pred={pred[j, 0]:+.3f}   true={Y[j, 0]:+.3f}")
