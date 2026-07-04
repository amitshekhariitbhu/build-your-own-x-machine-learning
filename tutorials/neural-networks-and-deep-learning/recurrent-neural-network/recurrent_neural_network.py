import numpy as np

class RNN:
    """Vanilla (Elman) RNN for char-level next-char prediction, hand-rolled forward + BPTT."""
    def __init__(self, vocab_size, hidden_dim=64, lr=0.05, seed=0):
        rng = np.random.RandomState(seed)
        V, H = vocab_size, hidden_dim
        self.V, self.H, self.lr = V, H, lr
        self.p = {
            'Wxh': rng.randn(V, H) * (1.0 / np.sqrt(V)),  # input -> hidden
            'Whh': rng.randn(H, H) * (1.0 / np.sqrt(H)),  # hidden -> hidden (recurrence)
            'Why': rng.randn(H, V) * (1.0 / np.sqrt(H)),  # hidden -> logits
            'bh': np.zeros(H), 'by': np.zeros(V),
        }
        # Adam optimizer state
        self.m = {k: np.zeros_like(v) for k, v in self.p.items()}
        self.v = {k: np.zeros_like(v) for k, v in self.p.items()}
        self.t = 0

    def forward(self, xs, h0):
        # xs: list of T int token ids. Runs the tanh recurrence; returns per-step logits + cache.
        p, hs, ps = self.p, {-1: h0}, {}
        for t, x in enumerate(xs):
            # h_t = tanh(x_t Wxh + h_{t-1} Whh + bh)
            hs[t] = np.tanh(p['Wxh'][x] + hs[t - 1] @ p['Whh'] + p['bh'])
            logit = hs[t] @ p['Why'] + p['by']
            logit -= logit.max()                           # softmax over the vocabulary
            e = np.exp(logit)
            ps[t] = e / e.sum()
        return hs, ps

    def backward(self, xs, ys, hs, ps):
        # Backprop through time for softmax cross-entropy summed over all steps.
        p = self.p
        g = {k: np.zeros_like(v) for k, v in p.items()}
        dh_next = np.zeros(self.H)
        for t in reversed(range(len(xs))):
            dlog = ps[t].copy()
            dlog[ys[t]] -= 1                               # dL/dlogit = softmax - onehot
            g['Why'] += np.outer(hs[t], dlog)
            g['by'] += dlog
            dh = dlog @ p['Why'].T + dh_next               # grad into h_t
            draw = (1 - hs[t] ** 2) * dh                   # through tanh
            g['bh'] += draw
            g['Wxh'][xs[t]] += draw                        # one-hot input picks a row
            g['Whh'] += np.outer(hs[t - 1], draw)
            dh_next = draw @ p['Whh'].T                    # grad to previous hidden state
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

    def fit(self, xs, ys, epochs=300):
        # Train on one sequence: inputs xs, targets ys = xs shifted by one char.
        losses = []
        for _ in range(epochs):
            h0 = np.zeros(self.H)
            hs, ps = self.forward(xs, h0)
            loss = -np.mean([np.log(ps[t][ys[t]] + 1e-12) for t in range(len(xs))])
            losses.append(loss)
            self.step(self.backward(xs, ys, hs, ps))
        return losses

    def predict(self, xs):
        # Greedy next-char prediction at every step (argmax of the softmax).
        _, ps = self.forward(xs, np.zeros(self.H))
        return [int(np.argmax(ps[t])) for t in range(len(xs))]

    def sample(self, seed_id, n):
        # Autoregressively generate n tokens starting from seed_id.
        p, h, x, out = self.p, np.zeros(self.H), seed_id, []
        for _ in range(n):
            h = np.tanh(p['Wxh'][x] + h @ p['Whh'] + p['bh'])
            logit = h @ p['Why'] + p['by']
            x = int(np.argmax(logit))
            out.append(x)
        return out

# Example usage
if __name__ == "__main__":
    np.random.seed(0)
    # Task: char-level next-char prediction on a short string (many-to-many).
    text = "hello world, a tiny recurrent neural network learns to spell. "
    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    itos = {i: c for i, c in enumerate(chars)}
    ids = [stoi[c] for c in text]
    xs, ys = ids[:-1], ids[1:]                             # predict char t+1 from char t

    model = RNN(vocab_size=len(chars), hidden_dim=64, lr=0.05, seed=0)
    losses = model.fit(xs, ys, epochs=300)

    start, final = losses[0], losses[-1]
    pred = model.predict(xs)
    acc = np.mean([pred[t] == ys[t] for t in range(len(xs))])
    gen = "".join(itos[i] for i in model.sample(stoi[text[0]], len(text) - 1))
    print("Char-level next-char prediction (vanilla Elman RNN, manual BPTT)")
    print(f"  chars={len(text)}  vocab={len(chars)}  hidden={model.H}")
    print(f"  start loss    : {start:.5f}  (random ~= {np.log(len(chars)):.5f})")
    print(f"  final loss    : {final:.5f}")
    print(f"  reduction     : {100 * (1 - final / start):.1f}%")
    print(f"  next-char acc : {100 * acc:.1f}%")
    print(f"  seed char     : '{text[0]}'")
    print(f"  generated     : '{text[0]}{gen}'")
    print(f"  target text   : '{text}'")
