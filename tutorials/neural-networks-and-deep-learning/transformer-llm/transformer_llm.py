import numpy as np

# Minimal char-level Transformer language model, forward AND backward by hand.
# Pre-norm blocks: token+positional embeddings -> [LN, causal multi-head
# attention, LN, feed-forward] x n_layers -> final LN -> linear head.


def split_heads(x, H):
    # (B, T, D) -> (B, H, T, Dh)
    B, T, D = x.shape
    return x.reshape(B, T, H, D // H).transpose(0, 2, 1, 3)


def merge_heads(x):
    # (B, H, T, Dh) -> (B, T, D)
    B, H, T, Dh = x.shape
    return x.transpose(0, 2, 1, 3).reshape(B, T, H * Dh)


def layernorm_forward(x, gamma, beta, eps=1e-5):
    mu = x.mean(-1, keepdims=True)
    xc = x - mu
    std = np.sqrt((xc ** 2).mean(-1, keepdims=True) + eps)
    xhat = xc / std
    return gamma * xhat + beta, (xhat, std, gamma)


def layernorm_backward(dout, cache):
    xhat, std, gamma = cache
    D = xhat.shape[-1]
    dgamma = (dout * xhat).reshape(-1, D).sum(0)
    dbeta = dout.reshape(-1, D).sum(0)
    dxhat = dout * gamma
    dx = (dxhat - dxhat.mean(-1, keepdims=True)
          - xhat * (dxhat * xhat).mean(-1, keepdims=True)) / std
    return dx, dgamma, dbeta


class TransformerLM:
    def __init__(self, vocab_size, d_model=32, n_heads=2, n_layers=1,
                 d_ff=64, seq_len=16):
        self.V, self.D, self.H = vocab_size, d_model, n_heads
        self.n_layers, self.seq_len = n_layers, seq_len
        r = lambda *s: np.random.randn(*s) * 0.02
        p = {'tok_emb': r(vocab_size, d_model), 'pos_emb': r(seq_len, d_model),
             'gf': np.ones(d_model), 'bf': np.zeros(d_model),
             'Whead': r(d_model, vocab_size), 'bhead': np.zeros(vocab_size)}
        for l in range(n_layers):
            p[f'l{l}_g1'], p[f'l{l}_b1'] = np.ones(d_model), np.zeros(d_model)
            p[f'l{l}_g2'], p[f'l{l}_b2'] = np.ones(d_model), np.zeros(d_model)
            p[f'l{l}_Wq'], p[f'l{l}_Wk'] = r(d_model, d_model), r(d_model, d_model)
            p[f'l{l}_Wv'], p[f'l{l}_Wo'] = r(d_model, d_model), r(d_model, d_model)
            p[f'l{l}_W1'], p[f'l{l}_b1f'] = r(d_model, d_ff), np.zeros(d_ff)
            p[f'l{l}_W2'], p[f'l{l}_b2f'] = r(d_ff, d_model), np.zeros(d_model)
        self.p = p
        # Causal mask: allow position i to attend only to j <= i.
        self.mask = np.triu(np.ones((seq_len, seq_len)), 1) * -1e9
        self.m = {k: np.zeros_like(v) for k, v in p.items()}
        self.v = {k: np.zeros_like(v) for k, v in p.items()}
        self.t = 0

    def _attn_forward(self, x, l):
        p, H = self.p, self.H
        B, T, D = x.shape
        Dh = D // H
        scale = 1.0 / np.sqrt(Dh)
        Q, K, Vv = x @ p[f'l{l}_Wq'], x @ p[f'l{l}_Wk'], x @ p[f'l{l}_Wv']
        Qh, Kh, Vh = split_heads(Q, H), split_heads(K, H), split_heads(Vv, H)
        scores = (Qh @ Kh.transpose(0, 1, 3, 2)) * scale + self.mask[:T, :T]
        scores -= scores.max(-1, keepdims=True)
        e = np.exp(scores)
        attn = e / e.sum(-1, keepdims=True)
        ctx_m = merge_heads(attn @ Vh)
        out = ctx_m @ p[f'l{l}_Wo']
        return out, (x, Qh, Kh, Vh, attn, ctx_m, scale)

    def _attn_backward(self, dout, cache, l, g):
        p, H = self.p, self.H
        x, Qh, Kh, Vh, attn, ctx_m, scale = cache
        B, T, D = dout.shape
        g[f'l{l}_Wo'] = ctx_m.reshape(B * T, D).T @ dout.reshape(B * T, D)
        dctx = split_heads(dout @ p[f'l{l}_Wo'].T, H)
        dattn = dctx @ Vh.transpose(0, 1, 3, 2)
        dVh = attn.transpose(0, 1, 3, 2) @ dctx
        # softmax backward (masked entries have attn=0, so they stay 0)
        dscores = attn * (dattn - (dattn * attn).sum(-1, keepdims=True)) * scale
        dQ = merge_heads(dscores @ Kh)
        dK = merge_heads(dscores.transpose(0, 1, 3, 2) @ Qh)
        dV = merge_heads(dVh)
        xf = x.reshape(B * T, D)
        g[f'l{l}_Wq'] = xf.T @ dQ.reshape(B * T, D)
        g[f'l{l}_Wk'] = xf.T @ dK.reshape(B * T, D)
        g[f'l{l}_Wv'] = xf.T @ dV.reshape(B * T, D)
        return dQ @ p[f'l{l}_Wq'].T + dK @ p[f'l{l}_Wk'].T + dV @ p[f'l{l}_Wv'].T

    def _ff_forward(self, x, l):
        p = self.p
        z1 = x @ p[f'l{l}_W1'] + p[f'l{l}_b1f']
        a1 = np.maximum(z1, 0)
        return a1 @ p[f'l{l}_W2'] + p[f'l{l}_b2f'], (x, z1, a1)

    def _ff_backward(self, dout, cache, l, g):
        p = self.p
        x, z1, a1 = cache
        B, T, D = dout.shape
        F = z1.shape[-1]
        g[f'l{l}_W2'] = a1.reshape(B * T, F).T @ dout.reshape(B * T, D)
        g[f'l{l}_b2f'] = dout.reshape(B * T, D).sum(0)
        dz1 = (dout @ p[f'l{l}_W2'].T) * (z1 > 0)
        g[f'l{l}_W1'] = x.reshape(B * T, D).T @ dz1.reshape(B * T, F)
        g[f'l{l}_b1f'] = dz1.reshape(B * T, F).sum(0)
        return dz1 @ p[f'l{l}_W1'].T

    def forward(self, X, Y=None):
        p = self.p
        B, T = X.shape
        h = p['tok_emb'][X] + p['pos_emb'][:T]
        caches = []
        for l in range(self.n_layers):
            ln1, c1 = layernorm_forward(h, p[f'l{l}_g1'], p[f'l{l}_b1'])
            a, ca = self._attn_forward(ln1, l)
            h_mid = h + a
            ln2, c2 = layernorm_forward(h_mid, p[f'l{l}_g2'], p[f'l{l}_b2'])
            f, cf = self._ff_forward(ln2, l)
            h = h_mid + f
            caches.append((c1, ca, c2, cf))
        lnf, clf = layernorm_forward(h, p['gf'], p['bf'])
        logits = lnf @ p['Whead'] + p['bhead']
        logits -= logits.max(-1, keepdims=True)
        e = np.exp(logits)
        probs = e / e.sum(-1, keepdims=True)
        self.cache = (X, Y, caches, clf, lnf, probs)
        if Y is None:
            return probs
        N = B * T
        return -np.log(probs.reshape(N, -1)[np.arange(N), Y.reshape(N)] + 1e-9).mean()

    def backward(self):
        X, Y, caches, clf, lnf, probs = self.cache
        p = self.p
        B, T = X.shape
        V, N = self.V, B * T
        g = {k: np.zeros_like(v) for k, v in p.items()}
        # cross-entropy + softmax gradient
        dlogits = probs.reshape(N, V).copy()
        dlogits[np.arange(N), Y.reshape(N)] -= 1
        dlogits /= N
        g['Whead'] = lnf.reshape(N, -1).T @ dlogits
        g['bhead'] = dlogits.sum(0)
        dh, g['gf'], g['bf'] = layernorm_backward(
            (dlogits @ p['Whead'].T).reshape(B, T, -1), clf)
        for l in reversed(range(self.n_layers)):
            c1, ca, c2, cf = caches[l]
            dln2 = self._ff_backward(dh, cf, l, g)
            dx2, g[f'l{l}_g2'], g[f'l{l}_b2'] = layernorm_backward(dln2, c2)
            dh_mid = dh + dx2
            dln1 = self._attn_backward(dh_mid, ca, l, g)
            dx1, g[f'l{l}_g1'], g[f'l{l}_b1'] = layernorm_backward(dln1, c1)
            dh = dh_mid + dx1
        g['pos_emb'][:T] += dh.sum(0)
        np.add.at(g['tok_emb'], X, dh)
        return g

    def step(self, g, lr=0.01, b1=0.9, b2=0.999, eps=1e-8):
        self.t += 1
        for k in self.p:
            self.m[k] = b1 * self.m[k] + (1 - b1) * g[k]
            self.v[k] = b2 * self.v[k] + (1 - b2) * g[k] ** 2
            mh = self.m[k] / (1 - b1 ** self.t)
            vh = self.v[k] / (1 - b2 ** self.t)
            self.p[k] -= lr * mh / (np.sqrt(vh) + eps)

    def generate(self, start_ids, n):
        ids = list(start_ids)
        for _ in range(n):
            ctx = np.array(ids[-self.seq_len:])[None, :]
            ids.append(int(self.forward(ctx)[0, -1].argmax()))
        return ids


if __name__ == "__main__":
    np.random.seed(0)

    text = "hello world " * 60
    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    itos = {i: c for i, c in enumerate(chars)}
    data = np.array([stoi[c] for c in text])

    seq_len = 16
    model = TransformerLM(len(chars), d_model=32, n_heads=2, n_layers=1,
                          d_ff=64, seq_len=seq_len)

    # overlapping (input, next-char target) windows
    starts = np.arange(len(data) - seq_len - 1)
    B, iters = 16, 400
    first = last = None
    for it in range(iters):
        idx = np.random.choice(starts, B)
        X = np.stack([data[i:i + seq_len] for i in idx])
        Y = np.stack([data[i + 1:i + seq_len + 1] for i in idx])
        loss = model.forward(X, Y)
        model.step(model.backward(), lr=0.01)
        if it == 0:
            first = loss
        last = loss

    seed = [stoi[c] for c in "hello "]
    out = "".join(itos[i] for i in model.generate(seed, 40))

    print("vocab size:", len(chars), " params:",
          sum(v.size for v in model.p.values()))
    print("start loss: {:.4f}".format(first))
    print("final loss: {:.4f}".format(last))
    print("loss dropped {:.1f}x".format(first / last))
    print("generated:  {!r}".format(out))
